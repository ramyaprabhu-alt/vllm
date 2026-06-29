#!/usr/bin/env python3
"""
Measure TTFT overhead when backward runs in a separate subprocess with no
bubble awareness — the simplest possible overlap approach.

Spawns one backward worker per GPU (to match the TP=2 setup), each running
Qwen3-30B-A3B backward ops in a tight loop.  This is what MPS-based overlap
looks like before SM partitioning: full contention, no scheduling.

With MPS and CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=P, the subprocess is limited
to P% of SMs, which reduces (but does not eliminate) the interference.

Usage:
    # Terminal 1 — inference server (no scheduler):
    bash run_qwen3_30b_a3b.sh --enforce-eager --no-enable-chunked-prefill \
        --max-num-batched-tokens 32768 --no-enable-prefix-caching

    # Terminal 2 — this test (no backward subprocess):
    python mps_overlap_test.py --mode baseline --reps 20

    # Terminal 2 — this test (with backward subprocess):
    python mps_overlap_test.py --mode subprocess --reps 20

    # With MPS (requires nvidia-cuda-mps-control -d first):
    python mps_overlap_test.py --mode mps --mps-pct 10 --reps 20
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import time

import requests
from transformers import AutoTokenizer

MODEL  = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
DEVICE = "cuda"

# ── Backward worker ───────────────────────────────────────────────────────────

def _backward_worker(gpu_id: int, stop_event: mp.Event, mps_pct: int | None):
    """Run Qwen3 backward ops in a tight loop on gpu_id until stop_event.

    Uses a single layer's worth of tensors (reused each iteration) to stay
    within the ~5GB free memory left after vLLM's 92% allocation.
    """
    if mps_pct is not None:
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_pct)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from bubbletea.backward_subops import (
        _attn_bwd_op, _moe_chunk_bwd_op,
        H, Q_DIM, KV_DIM, Q_HEADS, KV_HEADS, HEAD_DIM, N_EXPERTS, MOE_INT, DTYPE
    )
    import torch, torch.nn.functional as F

    device = 0  # remapped via CUDA_VISIBLE_DEVICES
    dev    = f"cuda:{device}"
    T_FT   = 128
    CHUNK  = 8   # experts per MoE chunk

    # Pre-allocate ONE layer's worth of tensors — reused every iteration.
    # This keeps memory usage to ~100MB vs 5GB for 48 layers.
    attn_op  = _attn_bwd_op(T_FT, device)
    moe_ops  = [_moe_chunk_bwd_op(T_FT, CHUNK, device) for _ in range(N_EXPERTS // CHUNK)]

    # Warm up
    attn_op()
    for op in moe_ops:
        op()
    torch.cuda.synchronize()

    # Tight loop: simulate one full backward layer per iteration
    while not stop_event.is_set():
        attn_op()
        for op in moe_ops:
            op()
        torch.cuda.synchronize()


# ── TTFT measurement ──────────────────────────────────────────────────────────

_DIVERSE = (
    "The mitochondria generates ATP through oxidative phosphorylation. "
    "def merge_sort(arr): return arr if len(arr)<=1 else merge(merge_sort(arr[:len(arr)//2]),merge_sort(arr[len(arr)//2:])). "
    "SELECT name, COUNT(*) FROM employees GROUP BY department. "
    "The Riemann hypothesis concerns the non-trivial zeros of the zeta function. "
)


def build_prompt(tokenizer, n_tokens):
    base = tokenizer.encode(_DIVERSE, add_special_tokens=False)
    pad  = tokenizer.encode("the ", add_special_tokens=False)[:1]
    ids  = base + pad * max(0, n_tokens - len(base))
    return tokenizer.decode(ids[:n_tokens])


def measure_ttft(port, tokenizer, t_in, reps):
    results = []
    for i in range(reps):
        prompt = build_prompt(tokenizer, t_in) + f" [mps-test-{i}-{time.time_ns()}]"
        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"http://localhost:{port}/v1/chat/completions",
                json={"model": MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 4, "temperature": 0,
                      "chat_template_kwargs": {"enable_thinking": False}},
                timeout=120)
            r.raise_for_status()
            ms = (time.perf_counter() - t0) * 1e3
            results.append(ms)
            print(f"    rep {i:2d}: {ms:.1f} ms")
        except Exception as e:
            print(f"    rep {i}: error {e}")
    return results


def report(label, ttfts, baseline_mean=None):
    if not ttfts:
        print(f"  {label}: no data"); return
    s = sorted(ttfts)
    n = len(s)
    mean = statistics.mean(s)
    p50  = s[n // 2]
    p95  = s[int(.95 * n)]
    overhead = f"  overhead vs baseline: {mean - baseline_mean:+.1f}ms mean" if baseline_mean else ""
    print(f"\n  {label}  n={n}  mean={mean:.1f}ms  p50={p50:.1f}ms  p95={p95:.1f}ms{overhead}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode",    choices=["baseline", "subprocess", "mps"],
                        default="subprocess")
    parser.add_argument("--port",    type=int, default=8000)
    parser.add_argument("--t-in",   type=int, default=16384)
    parser.add_argument("--reps",   type=int, default=20)
    parser.add_argument("--mps-pct",type=int, default=10,
                        help="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for the backward subprocess")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print(f"\nMPS overlap test — mode={args.mode}  T_in={args.t_in}  reps={args.reps}")

    # Warm up
    print(f"  Warming up ({args.warmup} requests)...")
    for _ in range(args.warmup):
        p = build_prompt(tok, args.t_in) + f" [wu-{time.time_ns()}]"
        requests.post(f"http://localhost:{args.port}/v1/chat/completions",
                      json={"model": MODEL, "messages": [{"role": "user", "content": p}],
                            "max_tokens": 4, "temperature": 0,
                            "chat_template_kwargs": {"enable_thinking": False}},
                      timeout=120)

    if args.mode == "baseline":
        print(f"\nBaseline (no backward subprocess):")
        ttfts = measure_ttft(args.port, tok, args.t_in, args.reps)
        report("baseline", ttfts)
        return

    # Start backward subprocesses on both GPUs
    mps_pct = args.mps_pct if args.mode == "mps" else None
    if args.mode == "mps":
        print(f"\nStarting backward subprocesses with MPS_PCT={mps_pct}% on GPU 0 and 1")
    else:
        print(f"\nStarting backward subprocesses (no SM limit) on GPU 0 and 1")

    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    workers = []
    for gpu_id in range(2):
        p = ctx.Process(target=_backward_worker, args=(gpu_id, stop, mps_pct), daemon=True)
        p.start()
        workers.append(p)

    # Give workers time to reach steady state
    time.sleep(3)
    print(f"  Workers running.  Measuring TTFT...")

    ttfts = measure_ttft(args.port, tok, args.t_in, args.reps)

    stop.set()
    for p in workers:
        p.join(timeout=5)

    label = f"subprocess MPS={mps_pct}%" if mps_pct else "subprocess (no MPS)"
    report(label, ttfts)

    print(f"\n  Reference points (same server, T_in={args.t_in}):")
    print(f"    baseline (no backward):        ~872ms mean")
    print(f"    bubble-sched trigger_rank=1:   ~913ms mean  (+41ms)")
    print(f"    bubble-sched both ranks:        ~922ms mean  (+50ms)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
