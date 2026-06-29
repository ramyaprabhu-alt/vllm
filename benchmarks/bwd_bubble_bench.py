#!/usr/bin/env python3
"""
Benchmark individual backward-pass kernels against rank-1 imbalance bubbles
in vLLM Qwen3-30B-A3B EP (no EPLB, 2× A100-80GB).

Per-rank shapes (TP=2):
  hidden_size H   = 2048
  Q heads/rank    = 16  (full: 32)   → Q_dim  = 16×128 = 2048
  KV heads/rank   =  2  (full:  4)   → KV_dim =  2×128 =  256
  local experts   = 64  (full: 128)
  moe_intermediate_size = 768

Bubble data (rank-1, no EPLB, repeated-token prompts):
  T_in   | total bubble | max/layer
  -------+-------------+----------
   4 096 |   17.6 ms   |  1.32 ms   (20/48 layers exploitable)
   8 192 |   34.3 ms   |  2.58 ms   (24/48)
  16 384 |   67.5 ms   |  4.84 ms   (25/48)
  24 576 |  104.1 ms   |  7.31 ms   (28/48)
  32 000 |  137.4 ms   |  9.33 ms   (31/48)

Usage:
    python bwd_bubble_bench.py [--tft T [T ...]] [--reps N]
"""
import argparse
import statistics
import torch
import torch.nn.functional as F
import torch.distributed as dist

# ── Qwen3-30B-A3B per-rank constants (TP=2) ──────────────────────────────────
H         = 2048   # hidden size (full, replicated across TP ranks)
Q_DIM     = 2048   # Q output dim per rank  (16 heads × 128)
KV_DIM    =  256   # K or V output dim per rank (2 heads × 128)
Q_HEADS   =   16   # Q attention heads per rank
KV_HEADS  =    2   # KV heads per rank (GQA)
HEAD_DIM  =  128
TOP_K     =    8   # active experts per token
N_EXPERTS =   64   # local experts per rank
MOE_INT   =  768   # moe_intermediate_size

DTYPE  = torch.bfloat16
DEVICE = "cuda"

# ── Bubble reference (from find_bubbles.py, repeated-token prompts) ──────────
BUBBLES = {
     4096: dict(total=17.6, max_layer=1.32, n_layers=20, n_moe=48),
     8192: dict(total=34.3, max_layer=2.58, n_layers=24, n_moe=48),
    16384: dict(total=67.5, max_layer=4.84, n_layers=25, n_moe=48),
    24576: dict(total=104.1,max_layer=7.31, n_layers=28, n_moe=48),
    32000: dict(total=137.4,max_layer=9.33, n_layers=31, n_moe=48),
}

# ── Estimated all_reduce time (from measured forward TP all_reduce data) ─────
# At T_in=32k forward: mean ar = 1.5ms for [32000, H] bf16 ≈ 128 MB
# Backward ar: [T_ft, H] bf16 = T_ft × H × 2 bytes
_AR_BW_MBPS = 128.0 / 1.5   # ~85 MB/ms  (NVLink ~150 GB/s bidirectional)

def ar_est_ms(n_tokens: int) -> float:
    return (n_tokens * H * 2 / 1e6) / _AR_BW_MBPS


# ── Timing harness ────────────────────────────────────────────────────────────
def time_op(fn, warmup: int = 10, reps: int = 50) -> tuple[float, float]:
    """Returns (mean_ms, stdev_ms) using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for s, e in zip(starts, ends):
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return statistics.mean(ms), statistics.stdev(ms) if reps > 1 else 0.0


# ── Individual backward kernels ───────────────────────────────────────────────

def bench_q_proj_bwd(T_ft: int):
    """dW_Q = grad_out.T @ x,  dx = grad_out @ W_Q"""
    x    = torch.randn(T_ft, H,     device=DEVICE, dtype=DTYPE)
    W    = torch.randn(Q_DIM, H,    device=DEVICE, dtype=DTYPE)
    g    = torch.randn(T_ft, Q_DIM, device=DEVICE, dtype=DTYPE)
    def fn():
        dx = g @ W          # [T_ft, H]
        dW = g.t() @ x      # [Q_DIM, H]
        return dx, dW
    return time_op(fn)


def bench_k_proj_bwd(T_ft: int):
    """dW_K = grad_out.T @ x,  dx = grad_out @ W_K"""
    x    = torch.randn(T_ft, H,      device=DEVICE, dtype=DTYPE)
    W    = torch.randn(KV_DIM, H,    device=DEVICE, dtype=DTYPE)
    g    = torch.randn(T_ft, KV_DIM, device=DEVICE, dtype=DTYPE)
    def fn():
        dx = g @ W
        dW = g.t() @ x
        return dx, dW
    return time_op(fn)


def bench_v_proj_bwd(T_ft: int):
    """Same shape as K."""
    return bench_k_proj_bwd(T_ft)


def bench_o_proj_bwd(T_ft: int):
    """O proj: [T_ft, Q_DIM] → [T_ft, H].  dW_O = x.T @ grad,  dx = grad @ W_O"""
    x    = torch.randn(T_ft, Q_DIM, device=DEVICE, dtype=DTYPE)
    W    = torch.randn(H, Q_DIM,    device=DEVICE, dtype=DTYPE)   # weight [H, Q_DIM]
    g    = torch.randn(T_ft, H,     device=DEVICE, dtype=DTYPE)
    def fn():
        dx = g @ W          # [T_ft, H] @ [H, Q_DIM] → [T_ft, Q_DIM]  (uses W rows)
        dW = x.t() @ g      # [Q_DIM, T_ft] @ [T_ft, H] → [Q_DIM, H]
        return dx, dW
    return time_op(fn)


def bench_attn_bwd(T_ft: int):
    """FlashAttention / SDPA backward (GQA: 16 Q heads, 2 KV heads)."""
    q = torch.randn(T_ft, Q_HEADS,  HEAD_DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(T_ft, KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(T_ft, KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    g = torch.randn(T_ft, Q_HEADS,  HEAD_DIM, device=DEVICE, dtype=DTYPE)

    # Expand KV for GQA: [T_ft, KV_HEADS, d] → [T_ft, Q_HEADS, d]
    GQA_FACTOR = Q_HEADS // KV_HEADS   # = 8

    def fn():
        # Detach and re-require grad on every call to isolate backward
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        # Transpose to [heads, seq, dim] for SDPA
        qt = q_.transpose(0, 1)                             # [Q_H, T, d]
        kt = k_.transpose(0, 1).repeat_interleave(GQA_FACTOR, dim=0)  # [Q_H, T, d]
        vt = v_.transpose(0, 1).repeat_interleave(GQA_FACTOR, dim=0)
        out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
        out.backward(g.transpose(0, 1))

    return time_op(fn, warmup=5, reps=20)


def bench_allreduce(T_ft: int):
    """TP all_reduce on [T_ft, H] bf16 — estimated from measured forward data."""
    est = ar_est_ms(T_ft)
    # Also time local memcpy of same size as lower bound
    x = torch.randn(T_ft, H, device=DEVICE, dtype=DTYPE)
    def fn():
        _ = x.clone()   # memcpy — lower bound
    mem_ms, _ = time_op(fn)
    return est, mem_ms   # (estimate_ms, memcpy_lower_bound_ms)


def bench_moe_expert_bwd(T_ft: int):
    """MoE expert backward.

    With TOP_K=8 and T_ft tokens split across N_EXPERTS=64 local experts:
    avg tokens/expert = T_ft × TOP_K / N_EXPERTS.

    Each expert has:
      gate_up: [T/exp, MOE_INT×2] grad → weight [MOE_INT×2, H] and input dx [T/exp, H]
      down:    [T/exp, H] grad → weight [H, MOE_INT] and input dx [T/exp, MOE_INT]
    """
    tpe = max(1, (T_ft * TOP_K) // N_EXPERTS)   # avg tokens per expert

    # Fused gate_up backward (single expert, then ×64)
    W_up  = torch.randn(MOE_INT * 2, H,   device=DEVICE, dtype=DTYPE)
    x_up  = torch.randn(tpe, H,           device=DEVICE, dtype=DTYPE)
    g_up  = torch.randn(tpe, MOE_INT * 2, device=DEVICE, dtype=DTYPE)
    W_dn  = torch.randn(H, MOE_INT,       device=DEVICE, dtype=DTYPE)
    x_dn  = torch.randn(tpe, MOE_INT,     device=DEVICE, dtype=DTYPE)
    g_dn  = torch.randn(tpe, H,           device=DEVICE, dtype=DTYPE)

    # Single expert
    def one_exp_bwd():
        dx_up = g_up @ W_up          # [tpe, H]
        dW_up = g_up.t() @ x_up      # [MOE_INT×2, H]
        # SiLU activation backward (element-wise, negligible) skipped
        dx_dn = g_dn @ W_dn          # [tpe, H] @ [H, MOE_INT] → [tpe, MOE_INT]
        dW_dn = x_dn.t() @ g_dn      # [MOE_INT, tpe] @ [tpe, H] → [MOE_INT, H]
        return dx_up, dW_up, dx_dn, dW_dn

    single_ms, _ = time_op(one_exp_bwd)

    # All 64 experts (sequential, as DeltaServe does)
    def all_exp_bwd():
        for _ in range(N_EXPERTS):
            dx_up = g_up @ W_up
            dW_up = g_up.t() @ x_up
            dx_dn = g_dn @ W_dn
            dW_dn = x_dn.t() @ g_dn

    all_ms, _ = time_op(all_exp_bwd, warmup=5, reps=20)
    return single_ms, all_ms


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tft", type=int, nargs="+",
                        default=[32, 64, 128, 256, 512])
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Dtype:  {DTYPE}")
    print(f"Qwen3-30B-A3B per-rank shapes: H={H}, Q_DIM={Q_DIM}, KV_DIM={KV_DIM}, "
          f"Q_HEADS={Q_HEADS}, KV_HEADS={KV_HEADS}, MOE_INT={MOE_INT}, "
          f"N_EXPERTS={N_EXPERTS}\n")

    # ── Benchmark table ───────────────────────────────────────────────────────
    ops = ["q_proj", "k_proj", "v_proj", "o_proj", "attn_sdpa",
           "ar_est", "moe_1exp", "moe_all64"]
    results = {op: {} for op in ops}

    for T_ft in args.tft:
        print(f"Benchmarking T_ft={T_ft}...", end=" ", flush=True)

        results["q_proj"][T_ft],   _ = bench_q_proj_bwd(T_ft)
        results["k_proj"][T_ft],   _ = bench_k_proj_bwd(T_ft)
        results["v_proj"][T_ft],   _ = bench_v_proj_bwd(T_ft)
        results["o_proj"][T_ft],   _ = bench_o_proj_bwd(T_ft)
        results["attn_sdpa"][T_ft],_ = bench_attn_bwd(T_ft)
        results["ar_est"][T_ft], _   = bench_allreduce(T_ft)

        single, all64 = bench_moe_expert_bwd(T_ft)
        results["moe_1exp"][T_ft]  = single
        results["moe_all64"][T_ft] = all64

        print("done")

    # ── Print op timing table ─────────────────────────────────────────────────
    print("\n" + "═"*90)
    print("BACKWARD SUB-OP TIMINGS (ms)  —  per attention/MoE layer")
    print("═"*90)
    header = f"  {'Operation':<22}" + "".join(f"  T_ft={t:>4}" for t in args.tft)
    print(header)
    print("─"*len(header))

    labels = {
        "q_proj":   "Q_proj bwd",
        "k_proj":   "K_proj bwd",
        "v_proj":   "V_proj bwd",
        "o_proj":   "O_proj bwd",
        "attn_sdpa":"Attn SDPA bwd",
        "ar_est":   "TP all_reduce (est)",
        "moe_1exp": "MoE 1-expert bwd",
        "moe_all64":"MoE all-64 bwd",
    }
    for op, label in labels.items():
        row = f"  {label:<22}" + "".join(f"  {results[op].get(t, 0.0):>9.3f}" for t in args.tft)
        print(row)

    # composite: full attn bwd per layer (Q+K+V+O+SDPA+AR)
    print("─"*len(header))
    for T_ft in args.tft:
        full = (results["q_proj"][T_ft] + results["k_proj"][T_ft] +
                results["v_proj"][T_ft] + results["o_proj"][T_ft] +
                results["attn_sdpa"][T_ft] + results["ar_est"][T_ft])
        results.setdefault("full_attn", {})[T_ft] = full
    row = f"  {'full attn bwd/layer':<22}" + "".join(f"  {results['full_attn'].get(t,0):>9.3f}" for t in args.tft)
    print(row)

    # ── Fit analysis ──────────────────────────────────────────────────────────
    print("\n" + "═"*90)
    print("BUBBLE FIT ANALYSIS — which sub-ops fit in rank-1 imbalance window?")
    print("  ✓ = fits in p50 bubble   ◑ = fits in max bubble only   ✗ = too large")
    print("  Bubble = rank-1 idle time while waiting at TP all_reduce for rank-0 FFN")
    print("═"*90)

    fit_ops = ["q_proj", "k_proj", "v_proj", "o_proj", "attn_sdpa", "ar_est",
               "moe_1exp", "moe_all64", "full_attn"]

    for t_in, b in BUBBLES.items():
        p50_bubble  = b["total"] / b["n_layers"] if b["n_layers"] > 0 else 0
        max_bubble  = b["max_layer"]
        total_avail = b["total"]

        print(f"\n  T_in={t_in:>6}  "
              f"total_bubble={total_avail:.1f}ms  "
              f"p50/layer={p50_bubble:.2f}ms  "
              f"max/layer={max_bubble:.2f}ms  "
              f"exploitable_layers={b['n_layers']}/{b['n_moe']}")

        for T_ft in args.tft:
            cells = []
            for op in fit_ops:
                t = results[op].get(T_ft, 0.0)
                if t <= p50_bubble:
                    sym = "✓"
                elif t <= max_bubble:
                    sym = "◑"
                else:
                    sym = "✗"
                cells.append(f"{sym}{t:.2f}")

            print(f"    T_ft={T_ft:>4}  " +
                  "  ".join(f"{labels.get(op, op)[:12]:>12}:{cells[i]}"
                             for i, op in enumerate(fit_ops)))

    # ── Strategy summary ──────────────────────────────────────────────────────
    print("\n" + "═"*90)
    print("SCHEDULING STRATEGY SUMMARY")
    print("═"*90)
    print("""
  Approach: run backward sub-ops on rank-1's low-priority CUDA stream during
  the rank-1 imbalance bubble (rank-1 idles at TP all_reduce barrier while
  rank-0 finishes its extra FFN compute).

  The bubble is per-layer and uneven: ~half of 48 MoE layers have >0.5ms,
  a few peak at 4-9ms depending on context length.

  Recommended decomposition (one sub-op per bubble slot):
    Slot A (smallest bubble, ~p25):  q_proj bwd  OR  k_proj bwd  OR  v_proj bwd
    Slot B (medium bubble, ~p50):    o_proj bwd  (larger Q_DIM×H matmul)
    Slot C (large bubble, p75-max):  attn_sdpa bwd  OR  moe_1exp bwd

  Limitation: all_reduce and moe_all64 bwd are too large for individual layer
  slots — they span multiple layers or must use C_thread approach instead.
    """)


if __name__ == "__main__":
    main()
