"""
Profile the VLLM_FT_TP_CORRECT TCPStore exchange overhead at production
shapes (Qwen3-30B-A3B: H=2048, 48 layers, LoRA r=16, TP=2).

Measures, per exchange size:
  - symmetric exchange latency (both ranks publish/collect, barrier-aligned)
  - phase breakdown for one representative size (GPU->CPU cast, store.set,
    wait+get+sum, CPU->GPU)
  - the fallback paths: peer-absent timeout (production 50ms cap) and
    round-mismatch (should be ~instant)

Then prints an aggregate per-training-round overhead model from the measured
means, for the exchange inventory as of 2026-06-11:

  per forward round : 1x embed [T,H], 48x o_total [T,H],
                      (+48x moefwd [T,H] on the non-C+D_batch path only)
  per backward round: 1x lmh_stats [3,T-1], 1x lmh_grad [T-1,H],
                      48x moebwd [T,H], 48x attnbwd [T,H]
  per optimizer step: 1x grad pack [48*4*16*2048] f32 (~25 MB)

Run:
    torchrun --nproc_per_node=2 bench_tp_correct_overhead.py
"""
import os

# Bench with a huge timeout so we measure true completion time, not the
# production 50ms cap (the cap is benchmarked separately as the fallback).
os.environ["VLLM_FT_TP_CORRECT_TIMEOUT_MS"] = "30000"

import sys
import time

sys.path.insert(0, '/mnt/nfs/home/ramya/vllm')

import torch
import torch.distributed as dist

import bubbletea.trainer as blt

H       = 2048    # Qwen3-30B-A3B hidden size
LAYERS  = 48
LORA_R  = 16
WARMUP  = 5
ITERS   = 30


def _stats(times):
    s = sorted(times)
    n = len(s)
    return (sum(s) / n, s[n // 2], s[int(n * 0.95) - 1], s[-1])


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 2

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(local_rank)
        device, act_dtype = torch.device("cuda", local_rank), torch.bfloat16
    else:
        device, act_dtype = torch.device("cpu"), torch.float32

    # gloo default pg: barriers stay on CPU, no NCCL anywhere near the bench
    dist.init_process_group(backend="gloo")
    store = dist.distributed_c10d._get_default_store()

    tr = object.__new__(blt.BubbleTeaLoRATrainer)
    tr._tp_correct_store = store
    tr._tp_rank, tr._tp_size = rank, 2

    grad_pack_elems = LAYERS * 4 * LORA_R * H   # q/k/v lora_A + o lora_B
    cases = []
    act_bytes = 2 if act_dtype == torch.bfloat16 else 4
    for T in (64, 128, 256, 512):
        cases.append((f"act_T{T}", f"[{T},{H}] {str(act_dtype).split('.')[-1]} "
                       f"({T*H*act_bytes/1e6:.2f} MB wire)",
                       torch.randn(T, H, device=device, dtype=act_dtype)))
    cases.append(("lmh_stats", f"[3,127] f32 ({3*127*4/1e3:.1f} KB wire)",
                   torch.randn(3, 127, device=device, dtype=torch.float32)))
    cases.append(("grad_pack", f"[{grad_pack_elems}] f32 "
                   f"({grad_pack_elems*4/1e6:.1f} MB wire)",
                   torch.randn(grad_pack_elems, device=device,
                               dtype=torch.float32)))

    # ── Correctness sanity (ones -> 2s) ─────────────────────────────────────
    ones = torch.ones(8, device=device)
    out = tr._tp_correct_exchange_sum(ones, "bench_sanity", 0)
    assert torch.allclose(out, torch.full_like(ones, 2.0)), "exchange broken"
    dist.barrier()

    # ── Symmetric exchange latency per size ─────────────────────────────────
    results = {}
    for tag, desc, x in cases:
        times = []
        for it in range(WARMUP + ITERS):
            dist.barrier()
            t0 = time.perf_counter()
            tr._tp_correct_exchange_sum(x, f"bench_{tag}", it)
            dt = (time.perf_counter() - t0) * 1e3
            if it >= WARMUP:
                times.append(dt)
        results[tag] = (desc, _stats(times))

    # ── Phase breakdown at T=128 ────────────────────────────────────────────
    x = torch.randn(128, H, device=device, dtype=act_dtype)
    phases = {k: [] for k in ("d2h_cast", "store_set", "wait_get_sum", "h2d")}
    import struct as _struct
    for it in range(WARMUP + ITERS):
        dist.barrier()
        rid = 10_000 + it
        t0 = time.perf_counter()
        cpu_t = x.detach().to('cpu', dtype=torch.float32).contiguous()
        t1 = time.perf_counter()
        my_key = f"tpcorrect/bench_break/{rank}"
        store.set(my_key, blt._TP_CORRECT_HDR.pack(rid) + cpu_t.numpy().tobytes())
        t2 = time.perf_counter()
        peer_key = f"tpcorrect/bench_break/{1 - rank}"
        import datetime as _dt
        store.wait([peer_key], _dt.timedelta(seconds=30))
        buf = store.get(peer_key)
        peer = torch.frombuffer(bytearray(buf[blt._TP_CORRECT_HDR.size:]),
                                 dtype=torch.float32).reshape(cpu_t.shape)
        total = cpu_t + peer
        t3 = time.perf_counter()
        total = total.to(device, dtype=x.dtype)
        if cuda:
            torch.cuda.synchronize()
        t4 = time.perf_counter()
        if it >= WARMUP:
            phases["d2h_cast"].append((t1 - t0) * 1e3)
            phases["store_set"].append((t2 - t1) * 1e3)
            phases["wait_get_sum"].append((t3 - t2) * 1e3)
            phases["h2d"].append((t4 - t3) * 1e3)

    # ── Fallback costs (production timeout) ────────────────────────────────
    blt._TP_CORRECT_TIMEOUT_MS = 50
    # (a) peer never publishes for this tag -> timeout fallback
    timeout_ms = None
    if rank == 0:
        t0 = time.perf_counter()
        tr._tp_correct_exchange_sum(x, "bench_absent", 0)
        timeout_ms = (time.perf_counter() - t0) * 1e3
    dist.barrier()
    # (b) peer published, but for a different round -> immediate fallback
    tr._tp_correct_exchange_sum(x, "bench_stale", rank)  # ranks publish 0 / 1
    dist.barrier()
    mismatch_ms = None
    if rank == 0:
        t0 = time.perf_counter()
        tr._tp_correct_exchange_sum(x, "bench_stale", 0)   # peer key has rid=1
        mismatch_ms = (time.perf_counter() - t0) * 1e3
    blt._TP_CORRECT_TIMEOUT_MS = 30000
    dist.barrier()

    # ── Pipelined (async publish/consume) exposed-latency simulation ───────
    # The production call sites now publish via _tp_correct_exchange_sum_async
    # and consume the future one-or-more sub-ops later; `gap_ms` simulates the
    # work (other sub-ops + inter-bubble time) between publish and consume.
    tr._tp_correct = True
    async_results = {}
    for T in (128, 512):
        x = torch.randn(T, H, device=device, dtype=act_dtype)
        for gap_ms in (0, 1, 2, 5, 10, 20):
            pub, exposed = [], []
            for it in range(WARMUP + ITERS):
                dist.barrier()
                t0 = time.perf_counter()
                fut = tr._tp_correct_exchange_sum_async(
                    x, f"bench_async_T{T}_{gap_ms}", it)
                t1 = time.perf_counter()
                if gap_ms:
                    time.sleep(gap_ms / 1e3)
                t2 = time.perf_counter()
                fut.result()
                t3 = time.perf_counter()
                if it >= WARMUP:
                    pub.append((t1 - t0) * 1e3)
                    exposed.append((t3 - t2) * 1e3)
            async_results[f"T{T}/gap{gap_ms}"] = (_stats(pub)[0], _stats(exposed))

    # ── Streamed per-layer grad exchange (items 3-5) simulation ────────────
    # 48 publishes of [3*r*H] f32 (q/v lora_A + o lora_B) spaced ~1ms apart
    # (the rest of the backward sweep), all consumed at optimizer time.
    layer_flat = torch.randn(3 * LORA_R * H, device=device, dtype=torch.float32)
    stream_pub, stream_exposed = [], []
    for it in range(3 + 10):
        dist.barrier()
        futs, t_pub = [], 0.0
        for li in range(LAYERS):
            t0 = time.perf_counter()
            futs.append(tr._tp_correct_exchange_sum_async(
                layer_flat, f"bench_g{li}", it))
            t_pub += time.perf_counter() - t0
            time.sleep(0.001)
        t0 = time.perf_counter()
        for f in futs:
            f.result()
        t_cons = (time.perf_counter() - t0) * 1e3
        if it >= 3:
            stream_pub.append(t_pub * 1e3)
            stream_exposed.append(t_cons)

    # ── Gather + report on rank 0 ───────────────────────────────────────────
    payload = {"results": results,
               "phases": {k: _stats(v) for k, v in phases.items()},
               "async": async_results,
               "stream": (_stats(stream_pub), _stats(stream_exposed)),
               "timeout_ms": timeout_ms, "mismatch_ms": mismatch_ms}
    gathered = [None, None]
    dist.all_gather_object(gathered, payload)

    if rank == 0:
        print(f"\n{'='*74}")
        print(f"TCPStore exchange latency, symmetric, barrier-aligned "
              f"(ms; {ITERS} iters)")
        print(f"{'='*74}")
        print(f"{'case':12s} {'payload':34s} "
              f"{'mean':>7s} {'p50':>7s} {'p95':>7s} {'max':>7s}")
        for tag, (desc, _) in gathered[0]["results"].items():
            for r in (0, 1):
                m, p50, p95, mx = gathered[r]["results"][tag][1]
                label = tag if r == 0 else ""
                d = desc if r == 0 else f"  (rank {r})"
                print(f"{label:12s} {d:34s} "
                      f"{m:7.2f} {p50:7.2f} {p95:7.2f} {mx:7.2f}")

        print(f"\nPhase breakdown, [128,{H}] exchange (rank0 mean ms): " +
              "  ".join(f"{k}={v[0]:.2f}" for k, v in
                        gathered[0]["phases"].items()))
        print(f"Fallback: peer-absent timeout (50ms cap) = "
              f"{gathered[0]['timeout_ms']:.1f} ms; "
              f"round-mismatch = {gathered[0]['mismatch_ms']:.2f} ms")

        print(f"\n{'='*74}")
        print("Pipelined async exchange: publish cost + EXPOSED consume wait")
        print("(gap = simulated work between publish and consume)")
        print(f"{'='*74}")
        print(f"{'case':16s} {'publish':>8s} {'exposed mean':>13s} "
              f"{'p50':>7s} {'p95':>7s}   (per rank)")
        for key in gathered[0]["async"]:
            for r in (0, 1):
                pub, (m, p50, p95, _) = gathered[r]["async"][key]
                label = key if r == 0 else ""
                print(f"{label:16s} {pub:8.2f} {m:13.2f} {p50:7.2f} {p95:7.2f}")
        for r in (0, 1):
            (pm, *_), (em, ep50, ep95, _) = gathered[r]["stream"]
            print(f"\nStreamed 48-layer grad exchange (rank {r}): "
                  f"total publish={pm:.1f} ms (~{pm/48:.2f} ms/layer, hidden "
                  f"in backward), exposed consume at optimizer="
                  f"{em:.1f} ms (p95 {ep95:.1f})")

        # ── Aggregate per-round model from measured means ──────────────────
        print(f"\n{'='*74}")
        print("Aggregate per-training-round overhead model (rank0 means)")
        print(f"{'='*74}")
        for T in (128, 512):
            act = gathered[0]["results"][f"act_T{T}"][1][0]
            # lmh_grad ~ [T-1, H] ~ same as act_T; lmh_stats tiny
            lmh_stats = gathered[0]["results"]["lmh_stats"][1][0]
            grad_pack = gathered[0]["results"]["grad_pack"][1][0]
            pre  = (48 + 1) * act + lmh_stats + act + grad_pack   # items 1-7,10
            new  = (48 + 48) * act                                 # items 9 + attnbwd
            moefwd = 48 * act                                      # non-C+D only
            print(f"  T_ft={T}:")
            print(f"    pre-existing (embed + 48x o_total + lm_head CE + "
                  f"grad pack): {pre:8.1f} ms")
            print(f"    new this session (48x moebwd + 48x attnbwd):        "
                  f"     {new:8.1f} ms")
            print(f"    non-C+D_batch path additionally (48x moefwd):       "
                  f"     {moefwd:8.1f} ms")
            print(f"    TOTAL per round (C+D_batch path): {pre + new:8.1f} ms")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
