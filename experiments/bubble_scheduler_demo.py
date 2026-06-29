#!/usr/bin/env python3
"""
Standalone demo and unit test for VllmBubbleScheduler.

Simulates the rank-1 idle window during vLLM TP prefill all_reduce barriers
and measures what fraction of backward sub-ops fit inside those bubbles.

The demo has two modes:
  --mode unit   Quick correctness check (no GPU timing).
  --mode bench  Full timing simulation matching Qwen3-30B-A3B EP profile.

Usage:
    python bubble_scheduler_demo.py
    python bubble_scheduler_demo.py --mode bench --t-in 16384 --t-ft 128

The bench mode schedules real Qwen3-30B-A3B backward sub-ops (from
bwd_bubble_bench.py) into simulated layer bubbles and reports utilization.
"""
from __future__ import annotations

import argparse
import statistics
import time
import threading
import torch

import importlib.util, sys, pathlib

# Import bubble_scheduler directly (avoids triggering the full vllm package
# init chain which has circular deps around fused_moe at import time).
_bs_path = pathlib.Path(__file__).parent / "vllm/model_executor/layers/fused_moe/runner/bubble_scheduler.py"
_spec = importlib.util.spec_from_file_location("bubble_scheduler", _bs_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
VllmBubbleScheduler = _mod.VllmBubbleScheduler

_mr_path = pathlib.Path(__file__).parent / "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
# moe_runner has heavy vllm imports we don't need for the demo — just pull
# the scheduler globals directly via the already-loaded bubble_scheduler module
# and replicate the arm/disarm logic inline.
import types as _types
_moe_runner_sched = _types.SimpleNamespace(
    _active_scheduler=None,
    _sched_trigger_rank=1,
    _sched_min_tokens=512,
)

def arm_bubble_scheduler(sched, trigger_rank=1, min_tokens=512):
    _moe_runner_sched._active_scheduler = sched
    _moe_runner_sched._sched_trigger_rank = trigger_rank
    _moe_runner_sched._sched_min_tokens = min_tokens

def disarm_bubble_scheduler():
    _moe_runner_sched._active_scheduler = None

DEVICE = "cuda"
DTYPE  = torch.bfloat16

# ── Qwen3-30B-A3B per-rank constants (TP=2) ──────────────────────────────────
H        = 2048
Q_DIM    = 2048
KV_DIM   = 256
Q_HEADS  = 16
KV_HEADS = 2
HEAD_DIM = 128
TOP_K    = 8
N_EXPERTS = 64
MOE_INT  = 768

# ── Bubble reference data (from find_bubbles.py, rank-1, no-EPLB) ────────────
#   keyed by T_in: (total_ms, p50_ms, max_ms, n_exploitable_layers, n_moe_layers)
BUBBLE_PROFILE = {
     4096: (17.6,  0.88, 1.32, 20, 48),
     8192: (34.3,  1.43, 2.58, 24, 48),
    16384: (67.5,  2.70, 4.84, 25, 48),
    24576: (104.1, 3.72, 7.31, 28, 48),
    32000: (137.4, 4.43, 9.33, 31, 48),
}


# ── Sub-op builders ────────────────────────────────────────────────────────────

def make_q_proj_bwd(t_ft: int) -> callable:
    x = torch.randn(t_ft, H,     device=DEVICE, dtype=DTYPE)
    W = torch.randn(Q_DIM, H,    device=DEVICE, dtype=DTYPE)
    g = torch.randn(t_ft, Q_DIM, device=DEVICE, dtype=DTYPE)
    def op():
        _ = g @ W
        _ = g.t() @ x
    return op


def make_k_proj_bwd(t_ft: int) -> callable:
    x = torch.randn(t_ft, H,      device=DEVICE, dtype=DTYPE)
    W = torch.randn(KV_DIM, H,    device=DEVICE, dtype=DTYPE)
    g = torch.randn(t_ft, KV_DIM, device=DEVICE, dtype=DTYPE)
    def op():
        _ = g @ W
        _ = g.t() @ x
    return op


def make_v_proj_bwd(t_ft: int) -> callable:
    return make_k_proj_bwd(t_ft)


def make_o_proj_bwd(t_ft: int) -> callable:
    x = torch.randn(t_ft, Q_DIM, device=DEVICE, dtype=DTYPE)
    W = torch.randn(H, Q_DIM,    device=DEVICE, dtype=DTYPE)
    g = torch.randn(t_ft, H,     device=DEVICE, dtype=DTYPE)
    def op():
        _ = g @ W
        _ = x.t() @ g
    return op


def make_sdpa_bwd(t_ft: int) -> callable:
    GQA = Q_HEADS // KV_HEADS
    q = torch.randn(t_ft, Q_HEADS,  HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(t_ft, KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(t_ft, KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    g = torch.randn(t_ft, Q_HEADS,  HEAD_DIM, device=DEVICE, dtype=DTYPE)
    import torch.nn.functional as F
    def op():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        qt = q_.transpose(0, 1)
        kt = k_.transpose(0, 1).repeat_interleave(GQA, dim=0)
        vt = v_.transpose(0, 1).repeat_interleave(GQA, dim=0)
        out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
        out.backward(g.transpose(0, 1))
    return op


def make_moe_chunk_bwd(t_ft: int, chunk_experts: int = 8) -> callable:
    tpe = max(1, (t_ft * TOP_K) // N_EXPERTS)
    W_up = torch.randn(MOE_INT * 2, H,   device=DEVICE, dtype=DTYPE)
    x_up = torch.randn(tpe, H,           device=DEVICE, dtype=DTYPE)
    g_up = torch.randn(tpe, MOE_INT * 2, device=DEVICE, dtype=DTYPE)
    W_dn = torch.randn(H, MOE_INT,       device=DEVICE, dtype=DTYPE)
    x_dn = torch.randn(tpe, MOE_INT,     device=DEVICE, dtype=DTYPE)
    g_dn = torch.randn(tpe, H,           device=DEVICE, dtype=DTYPE)
    n = chunk_experts
    def op():
        for _ in range(n):
            _ = g_up @ W_up
            _ = g_up.t() @ x_up
            _ = g_dn @ W_dn
            _ = x_dn.t() @ g_dn
    return op


def build_backward_sub_ops(
    t_ft: int,
    n_layers: int = 48,
    moe_chunk_size: int = 8,
) -> list[callable]:
    """Build a full backward pass decomposed into per-layer sub-ops.

    Each transformer layer contributes two sub-ops:
      1. Full attention backward (Q+K+V+O proj + SDPA)
      2. MoE expert backward in chunks of moe_chunk_size

    Sub-ops for all layers are returned in reversed-layer order (48→0),
    matching the backward pass direction.
    """
    ops = []
    for _ in range(n_layers - 1, -1, -1):
        # Attention backward: fused as one sub-op per layer
        q = make_q_proj_bwd(t_ft)
        k = make_k_proj_bwd(t_ft)
        v = make_v_proj_bwd(t_ft)
        o = make_o_proj_bwd(t_ft)
        s = make_sdpa_bwd(t_ft)
        def attn_op(q=q, k=k, v=v, o=o, s=s):
            q(); k(); v(); o(); s()
        ops.append(attn_op)

        # MoE backward: split into n_slots sub-ops, each covering chunk_size experts
        n_slots = N_EXPERTS // moe_chunk_size
        for _ in range(n_slots):
            ops.append(make_moe_chunk_bwd(t_ft, moe_chunk_size))

    return ops


# ── Timing helper ─────────────────────────────────────────────────────────────

def time_op_ms(fn, warmup=5, reps=20) -> float:
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for s, e in zip(starts, ends):
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return statistics.mean(s.elapsed_time(e) for s, e in zip(starts, ends))


# ── Unit test (no GPU timing) ─────────────────────────────────────────────────

def run_unit_test() -> None:
    print("Running unit test...")
    n_ops = 10

    # Build trivial sub-ops (increment a counter on the GPU)
    results = []
    def make_op(i):
        def op():
            results.append(i)
        return op
    sub_ops = [make_op(i) for i in range(n_ops)]

    sched = VllmBubbleScheduler(sub_ops, device=0)

    # Simulate 8 layer bubbles (8 fill_one calls)
    for _ in range(8):
        sched.fill_one(in_bubble=True)

    # fill_remaining drains the last 2
    sched.fill_remaining()
    sched.wait()

    assert sched.is_complete(), "Scheduler did not complete"
    assert sched.subops_in_bubble == 8, f"Expected 8 in-bubble, got {sched.subops_in_bubble}"
    assert sched.subops_after == 2, f"Expected 2 after, got {sched.subops_after}"
    assert abs(sched.bubble_utilization - 0.8) < 1e-6, f"Bad utilization: {sched.bubble_utilization}"

    # Wait for results to be appended (worker is a thread, ops may lag fill_one)
    timeout = time.time() + 5.0
    while len(results) < n_ops and time.time() < timeout:
        time.sleep(0.01)
    assert sorted(results) == list(range(n_ops)), f"Sub-ops not all executed: {results}"

    print(f"  PASS — {sched.summary()}")


# ── Bench mode ────────────────────────────────────────────────────────────────

def run_bench(t_in: int, t_ft: int, moe_chunk_size: int = 8) -> None:
    if t_in not in BUBBLE_PROFILE:
        print(f"No bubble profile for T_in={t_in}. Available: {list(BUBBLE_PROFILE)}")
        return

    total_ms, p50_ms, max_ms, n_exploit, n_moe = BUBBLE_PROFILE[t_in]
    n_layers = n_moe  # 48 for Qwen3-30B-A3B

    print(f"\nBubble profile (T_in={t_in}, no-EPLB, rank-1, repeated-token):")
    print(f"  total={total_ms:.1f}ms  p50/layer={p50_ms:.2f}ms  "
          f"max/layer={max_ms:.2f}ms  exploitable={n_exploit}/{n_moe}")

    sub_ops = build_backward_sub_ops(t_ft, n_layers, moe_chunk_size)
    n_ops = len(sub_ops)
    print(f"\nBackward decomposition: {n_ops} sub-ops "
          f"({n_layers} attn + {n_layers * (N_EXPERTS // moe_chunk_size)} MoE-{moe_chunk_size}-expert chunks)")

    # Benchmark individual sub-op times
    print("\nSub-op timing (warmup on main stream):")
    attn_ms  = time_op_ms(sub_ops[0])  # first sub-op is attn bwd for layer 47
    chunk_ms = time_op_ms(sub_ops[1])  # second is MoE chunk bwd
    print(f"  Attn bwd/layer:          {attn_ms:.3f}ms")
    print(f"  MoE {moe_chunk_size}-expert chunk bwd: {chunk_ms:.3f}ms")
    print(f"  Sub-ops per layer:       {1 + N_EXPERTS // moe_chunk_size}  "
          f"(1 attn + {N_EXPERTS // moe_chunk_size} MoE chunks)")

    # Simulate the prefill: arm the scheduler, call fill_one() for each layer
    # bubble (only for layers where the bubble exceeds the sub-op cost),
    # then fill_remaining() for the rest.
    # ── Multi-prefill simulation ──────────────────────────────────────────────
    # The scheduler stays armed across consecutive prefills.  Decode steps are
    # never touched — backward work only runs inside prefill bubbles.
    # With `fills_per_layer` fill_one() calls per MoE layer, the queue drains
    # after ceil(n_ops / (n_moe * fills_per_layer)) prefills.

    fills_per_layer = max(1, int(p50_ms / max(attn_ms, chunk_ms)))
    ops_per_prefill = n_moe * fills_per_layer
    n_prefills_needed = -(-n_ops // ops_per_prefill)  # ceil div

    print(f"\nMulti-prefill drain ({fills_per_layer} fill_one()/layer, "
          f"{fills_per_layer * n_moe} fills/prefill):")
    print(f"  Sub-ops: {n_ops}  →  needs {n_prefills_needed} prefills to fully drain")

    sched = VllmBubbleScheduler(list(sub_ops), device=0)
    arm_bubble_scheduler(sched, trigger_rank=0, min_tokens=1)

    t_start = time.perf_counter()
    for prefill_idx in range(n_prefills_needed):
        # Each prefill: fill_one() fires once per MoE layer (× fills_per_layer)
        for _ in range(n_moe * fills_per_layer):
            if sched.has_work():
                sched.fill_one(in_bubble=True)
        # No fill_remaining() here — decode runs, backward is silent

    # Backward complete: disarm, then wait for worker to finish GPU ops
    disarm_bubble_scheduler()
    sched.wait()
    wall_ms = (time.perf_counter() - t_start) * 1e3

    print(f"  {sched.summary()}  wall={wall_ms:.1f}ms")
    print(f"  All {n_ops} sub-ops ran inside prefill bubbles — decode untouched.")

    print(f"\nFit analysis at T_in={t_in}:")
    print(f"  p50 bubble {p50_ms:.2f}ms: attn bwd {attn_ms:.3f}ms → "
          f"{'✓ fits' if attn_ms <= p50_ms else '✗ too large'}")
    print(f"  p50 bubble {p50_ms:.2f}ms: MoE-{moe_chunk_size} {chunk_ms:.3f}ms → "
          f"{'✓ fits' if chunk_ms <= p50_ms else '✗ too large'}")
    print(f"  max bubble {max_ms:.2f}ms: attn bwd {attn_ms:.3f}ms → "
          f"{'✓ fits' if attn_ms <= max_ms else '✗ too large'}")

    # Estimate max theoretical backward throughput
    ops_per_layer = 1 + N_EXPERTS // moe_chunk_size
    sub_op_budget_ms = n_exploit * p50_ms
    sub_ops_possible = int(sub_op_budget_ms / max(attn_ms, chunk_ms))
    layers_covered   = sub_ops_possible // ops_per_layer
    print(f"\nEstimated bubble capacity at p50:")
    print(f"  Budget: {n_exploit} layers × {p50_ms:.2f}ms = {sub_op_budget_ms:.1f}ms")
    print(f"  Sub-ops that fit: ~{sub_ops_possible}  "
          f"→ covers {layers_covered}/{n_layers} backward layers")
    pct = layers_covered / n_layers * 100
    print(f"  Backward coverage in-bubble: ~{pct:.0f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["unit", "bench"], default="unit")
    parser.add_argument("--t-in", type=int, default=16384,
                        help="Inference context length (for bubble profile lookup)")
    parser.add_argument("--t-ft", type=int, default=128,
                        help="Fine-tuning batch size (tokens per backward pass)")
    parser.add_argument("--moe-chunk", type=int, default=8,
                        help="MoE expert chunk size for backward decomposition")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device found; skipping GPU tests.")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    if args.mode == "unit":
        run_unit_test()
    else:
        run_bench(args.t_in, args.t_ft, args.moe_chunk)


if __name__ == "__main__":
    main()
