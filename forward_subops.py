"""
forward_subops.py — Qwen3-30B-A3B forward sub-op factory.

Builds callable forward sub-ops for one TP rank.  Each callable performs
the local, allreduce-free GPU work for one sub-op unit and is safe to run
on a separate CUDA stream (or synchronously) concurrently with inference.

Per-rank shapes (TP=2, no EPLB):
    hidden_size H   = 2048
    Q heads/rank    = 16   → Q_dim  = 16 × 128 = 2048
    KV heads/rank   =  2   → KV_dim =  2 × 128 =  256
    local experts   = 64   (rank 0: experts  0–63,  rank 1: experts 64–127)
    moe_intermediate = 768

No gradient computation — sub-ops are inference-mode-safe.  They produce
the activations a real FT forward would produce, with matching FLOP count,
but discard outputs immediately.

Usage:
    from forward_subops import build_qwen3_fwd_subops
    sub_ops = build_qwen3_fwd_subops(t_ft=128, device=0)
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
        submit_forward_job)
    submit_forward_job(sub_ops, fills_per_trigger=9, device=0)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# ── Qwen3-30B-A3B per-rank constants (TP=2) ──────────────────────────────────
H         = 2048
Q_DIM     = 2048    # 16 heads × 128
KV_DIM    = 256     # 2 heads  × 128 (GQA)
Q_HEADS   = 16
KV_HEADS  = 2
HEAD_DIM  = 128
GQA_RATIO = Q_HEADS // KV_HEADS   # 8
TOP_K     = 8
N_EXPERTS = 64      # local experts per rank
MOE_INT   = 768     # moe_intermediate_size
DTYPE     = torch.bfloat16


# ── Individual sub-op builders ────────────────────────────────────────────────

def _attn_fwd_op(t_ft: int, device: int) -> callable:
    """Full attention forward for one layer: Q+K+V proj + SDPA + O proj.

    All ops are allreduce-free (column-parallel Q/K/V and row-parallel O are
    local to this rank).  No gradient computation — safe in inference mode.
    """
    dev = f"cuda:{device}"
    x  = torch.randn(t_ft, H,      device=dev, dtype=DTYPE)
    Wq = torch.randn(Q_DIM,  H,    device=dev, dtype=DTYPE)
    Wk = torch.randn(KV_DIM, H,    device=dev, dtype=DTYPE)
    Wv = torch.randn(KV_DIM, H,    device=dev, dtype=DTYPE)
    Wo = torch.randn(H, Q_DIM,     device=dev, dtype=DTYPE)

    def op():
        q = x @ Wq.t()                      # (t_ft, Q_DIM)
        k = x @ Wk.t()                      # (t_ft, KV_DIM)
        v = x @ Wv.t()                      # (t_ft, KV_DIM)

        # Reshape for SDPA: (heads, t_ft, head_dim)
        qt = q.view(t_ft, Q_HEADS,  HEAD_DIM).transpose(0, 1)
        kt = k.view(t_ft, KV_HEADS, HEAD_DIM).transpose(0, 1)
        vt = v.view(t_ft, KV_HEADS, HEAD_DIM).transpose(0, 1)

        # Expand KV for GQA
        kt = kt.repeat_interleave(GQA_RATIO, dim=0)
        vt = vt.repeat_interleave(GQA_RATIO, dim=0)

        attn_out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
        attn_out = attn_out.transpose(0, 1).reshape(t_ft, Q_DIM)

        _ = attn_out @ Wo.t()               # (t_ft, H) — discard

    return op


def _moe_chunk_fwd_op(t_ft: int, n_experts: int, device: int) -> callable:
    """MoE expert forward for a chunk of n_experts (allreduce-free).

    Each rank owns its experts outright.  gate_up → silu-gating → down.
    """
    dev = f"cuda:{device}"
    tpe = max(1, (t_ft * TOP_K) // N_EXPERTS)   # avg tokens per expert
    W_gate_up = torch.randn(MOE_INT * 2, H,   device=dev, dtype=DTYPE)
    W_down     = torch.randn(H, MOE_INT,       device=dev, dtype=DTYPE)
    x          = torch.randn(tpe, H,           device=dev, dtype=DTYPE)

    def op():
        for _ in range(n_experts):
            gate_up = x @ W_gate_up.t()                # (tpe, MOE_INT*2)
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up              # (tpe, MOE_INT)
            _ = activated @ W_down.t()                 # (tpe, H) — discard

    return op


# ── Public factory ────────────────────────────────────────────────────────────

def build_qwen3_fwd_subops(
    t_ft: int,
    n_layers: int = 48,
    moe_chunk_size: int = 8,
    device: int = 0,
) -> list[callable]:
    """Build the full forward sub-op list for one TP rank.

    Sub-ops are ordered as the forward pass executes: layer 0 → 47,
    with MoE sub-ops split into chunks so each fits in one bubble slot.

    Layout per layer (0 → 47):
        1 × attn_fwd      (~0.3 ms at T_ft=128)
        N × moe_chunk_fwd (~0.2 ms each, N = N_EXPERTS / moe_chunk_size)

    Parameters
    ----------
    t_ft : Fine-tuning batch size (tokens per forward pass).
    n_layers : Number of transformer layers (48 for Qwen3-30B-A3B).
    moe_chunk_size : Experts per MoE forward chunk (8 → ~0.2 ms, 16 → ~0.4 ms).
    device : CUDA device index for this rank.
    """
    sub_ops = []
    n_moe_chunks = N_EXPERTS // moe_chunk_size

    for _ in range(n_layers):
        sub_ops.append(_attn_fwd_op(t_ft, device))
        for _ in range(n_moe_chunks):
            sub_ops.append(_moe_chunk_fwd_op(t_ft, moe_chunk_size, device))

    return sub_ops


def subops_info(t_ft: int, n_layers: int = 48, moe_chunk_size: int = 8) -> dict:
    """Return counts and estimated timing for a forward sub-op list."""
    n_moe_chunks = N_EXPERTS // moe_chunk_size
    n_attn = n_layers
    n_moe  = n_layers * n_moe_chunks
    total  = n_attn + n_moe
    return {
        "n_attn_ops":     n_attn,
        "n_moe_ops":      n_moe,
        "n_total_ops":    total,
        "ops_per_layer":  1 + n_moe_chunks,
        "moe_chunk_size": moe_chunk_size,
        "t_ft":           t_ft,
    }
