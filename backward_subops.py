"""
backward_subops.py — Qwen3-30B-A3B backward sub-op factory.

Builds callable backward sub-ops for one TP rank.  Each callable performs
the local, allreduce-free GPU work for one sub-op unit and is safe to run
on a separate CUDA stream concurrently with inference.

Per-rank shapes (TP=2, no EPLB):
    hidden_size H   = 2048
    Q heads/rank    = 16   → Q_dim  = 16 × 128 = 2048
    KV heads/rank   =  2   → KV_dim =  2 × 128 =  256
    local experts   = 64   (rank 0: experts  0–63,  rank 1: experts 64–127)
    moe_intermediate = 768

Gradient allreduce (needed for column/row-parallel attention weight grads)
is NOT included here — it must be coordinated across ranks after local GPU
work finishes.  Pass a post_complete_fn to VllmBubbleScheduler to trigger
the allreduce once this rank's backward stream has synchronised.

Usage:
    from backward_subops import build_qwen3_subops
    sub_ops = build_qwen3_subops(
        t_ft=128, n_layers=48, moe_chunk_size=8, device=0)
    sched = VllmBubbleScheduler(sub_ops, device=0,
                                post_complete_fn=my_allreduce_fn)
    submit_backward_job(sub_ops, fills_per_trigger=2, post_complete_fn=..., device=0)
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

def _attn_bwd_op(t_ft: int, device: int) -> callable:
    """Full attention backward for one layer: Q+K+V+O proj + SDPA.

    All ops are allreduce-free for the weight gradients (column-parallel Q/K/V
    and row-parallel O grads are local to this rank).  The input gradient
    dx for O requires an allreduce if passed upstream — handle in post_fn.
    """
    dev = f"cuda:{device}"
    x    = torch.randn(t_ft, H,      device=dev, dtype=DTYPE)
    gq   = torch.randn(t_ft, Q_DIM,  device=dev, dtype=DTYPE)
    gkv  = torch.randn(t_ft, KV_DIM, device=dev, dtype=DTYPE)
    go   = torch.randn(t_ft, H,      device=dev, dtype=DTYPE)
    Wq   = torch.randn(Q_DIM,  H,    device=dev, dtype=DTYPE)
    Wk   = torch.randn(KV_DIM, H,    device=dev, dtype=DTYPE)
    Wv   = torch.randn(KV_DIM, H,    device=dev, dtype=DTYPE)
    Wo   = torch.randn(H, Q_DIM,     device=dev, dtype=DTYPE)
    q    = torch.randn(t_ft, Q_HEADS,  HEAD_DIM, device=dev, dtype=DTYPE)
    k    = torch.randn(t_ft, KV_HEADS, HEAD_DIM, device=dev, dtype=DTYPE)
    v    = torch.randn(t_ft, KV_HEADS, HEAD_DIM, device=dev, dtype=DTYPE)
    gsdpa= torch.randn(t_ft, Q_HEADS,  HEAD_DIM, device=dev, dtype=DTYPE)

    def op():
        # Projection weight grads (allreduce-free: local column/row parallel)
        _ = gq @ Wq;      _ = gq.t() @ x    # Q
        _ = gkv @ Wk;     _ = gkv.t() @ x   # K
        _ = gkv @ Wv;     _ = gkv.t() @ x   # V
        _ = go @ Wo;      _ = x.t() @ go    # O

        # SDPA backward (GQA: expand KV heads)
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        qt = q_.transpose(0, 1)
        kt = k_.transpose(0, 1).repeat_interleave(GQA_RATIO, dim=0)
        vt = v_.transpose(0, 1).repeat_interleave(GQA_RATIO, dim=0)
        out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
        out.backward(gsdpa.transpose(0, 1))

    return op


def _moe_chunk_bwd_op(t_ft: int, n_experts: int, device: int) -> callable:
    """MoE expert backward for a chunk of n_experts (allreduce-free).

    Each rank owns its experts outright — no cross-rank communication needed
    for expert weight gradients.
    """
    dev = f"cuda:{device}"
    tpe = max(1, (t_ft * TOP_K) // N_EXPERTS)   # avg tokens per expert
    W_up = torch.randn(MOE_INT * 2, H,   device=dev, dtype=DTYPE)
    x_up = torch.randn(tpe, H,           device=dev, dtype=DTYPE)
    g_up = torch.randn(tpe, MOE_INT * 2, device=dev, dtype=DTYPE)
    W_dn = torch.randn(H, MOE_INT,       device=dev, dtype=DTYPE)
    x_dn = torch.randn(tpe, MOE_INT,     device=dev, dtype=DTYPE)
    g_dn = torch.randn(tpe, H,           device=dev, dtype=DTYPE)

    def op():
        for _ in range(n_experts):
            _ = g_up @ W_up;  _ = g_up.t() @ x_up   # gate_up bwd
            _ = g_dn @ W_dn;  _ = x_dn.t() @ g_dn   # down bwd

    return op


# ── Public factory ────────────────────────────────────────────────────────────

def build_qwen3_subops(
    t_ft: int,
    n_layers: int = 48,
    moe_chunk_size: int = 8,
    device: int = 0,
) -> list[callable]:
    """Build the full backward sub-op list for one TP rank.

    Sub-ops are ordered as the backward pass would execute: reversed layer
    order, with FFN sub-ops split into chunks so each fits in one bubble slot.

    Layout per layer (reversed 47 → 0):
        1 × attn_bwd      (~1.1 ms at T_ft=128)
        N × moe_chunk_bwd (~0.4 ms each, N = N_EXPERTS / moe_chunk_size)

    Parameters
    ----------
    t_ft : Fine-tuning batch size (tokens per backward pass).
    n_layers : Number of transformer layers (48 for Qwen3-30B-A3B).
    moe_chunk_size : Experts per MoE backward chunk (8 → 0.4 ms, 16 → 0.8 ms).
    device : CUDA device index for this rank.
    """
    sub_ops = []
    n_moe_chunks = N_EXPERTS // moe_chunk_size

    for _ in range(n_layers - 1, -1, -1):
        sub_ops.append(_attn_bwd_op(t_ft, device))
        for _ in range(n_moe_chunks):
            sub_ops.append(_moe_chunk_bwd_op(t_ft, moe_chunk_size, device))

    return sub_ops


def subops_info(t_ft: int, n_layers: int = 48, moe_chunk_size: int = 8) -> dict:
    """Return counts and estimated timing for a backward sub-op list."""
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
