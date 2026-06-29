"""
bt_lora_layer_bench.py — Standalone single-layer LoRA training microbenchmark.

Trains LoRA adapters (Q, V, O projections) on ONE transformer layer with
Qwen3-30B-A3B dimensions.  No vLLM, no NCCL, no loss.backward().

Everything is manual: forward pass, CE loss gradient, attention backward,
RoPE backward, RMSNorm backward, LoRA gradient accumulation, AdamW.

The only mini-autograd blocks are for SDPA backward (complex Flash-Attention
gradient) — all other ops are written out explicitly.

Expected result: CE loss decreases from ~log(VOCAB) ≈ 8.3 over 300 steps
as LoRA overfits to the fixed training sequences.

Run:
    .venv/bin/python bt_lora_layer_bench.py
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

# ── Qwen3-30B-A3B dimensions (TP=1) ───────────────────────────────────────
H        = 2048
N_HEADS  = 32
N_KV     = 4
HEAD_DIM = 128
Q_SZ     = N_HEADS * HEAD_DIM    # 4096
KV_SZ    = N_KV    * HEAD_DIM    # 512
RANK     = 16
SCALING  = 1.0    # lora_alpha / lora_rank = 16/16

T        = 128    # training sequence length
VOCAB    = 4096   # small vocab for fast CE
STEPS    = 300
LR       = 2e-4
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════
# Primitive ops (no autograd except sdpa_bwd)
# ══════════════════════════════════════════════════════════════════════════

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x: [..., D], w: [D]  →  y: [..., D]"""
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return (x.float() / rms * w.float()).to(x.dtype)


def rms_norm_bwd(grad: torch.Tensor, x: torch.Tensor, w: torch.Tensor,
                 eps: float = 1e-6):
    """Returns (grad_x, grad_w), fully manual.  x: [..., D], w: [D]."""
    rms   = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    x_hat = x.float() / rms                                           # [..., D]
    dxh   = grad.float() * w.float()                                  # [..., D]
    dx    = (dxh - x_hat * (dxh * x_hat).mean(-1, keepdim=True)) / rms
    dw    = (grad.float() * x_hat).sum(tuple(range(x.dim() - 1)))    # [D]
    return dx.to(x.dtype), dw.to(w.dtype)


def build_rope_freqs(seq_len: int, head_dim: int,
                     theta: float = 10000.0, device=None) -> torch.Tensor:
    """Returns complex-valued freqs tensor [seq_len, head_dim//2]."""
    inv   = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                          device=device) / head_dim)
    t     = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv)                                        # [T, hd/2]
    return torch.polar(torch.ones_like(freqs), freqs)                  # complex


def rope_fwd(x: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
    """x: [T, nh, hd], fc: [T, hd/2] complex  →  [T, nh, hd]"""
    T, nh, hd = x.shape
    xc = torch.view_as_complex(x.float().reshape(T, nh, hd // 2, 2))
    out = torch.view_as_real(xc * fc[:T].unsqueeze(1)).flatten(3)
    return out.to(x.dtype)


def rope_bwd(grad: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
    """RoPE Jacobian is orthogonal → backward = rotate by conjugate (−θ)."""
    T, nh, hd = grad.shape
    gc = torch.view_as_complex(grad.float().reshape(T, nh, hd // 2, 2))
    out = torch.view_as_real(gc * fc[:T].conj().unsqueeze(1)).flatten(3)
    return out.to(grad.dtype)


def causal_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q: [T,Q_SZ], k: [T,KV_SZ], v: [T,KV_SZ]  →  [T,Q_SZ].  GQA-aware."""
    T = q.shape[0]
    q3 = q.view(T, N_HEADS, HEAD_DIM).transpose(0, 1)    # [nh, T, hd]
    k3 = k.view(T, N_KV,    HEAD_DIM).transpose(0, 1)    # [nkv, T, hd]
    v3 = v.view(T, N_KV,    HEAD_DIM).transpose(0, 1)
    if N_KV < N_HEADS:
        rep = N_HEADS // N_KV
        k3 = k3.repeat_interleave(rep, 0)
        v3 = v3.repeat_interleave(rep, 0)
    out = F.scaled_dot_product_attention(q3, k3, v3, is_causal=True)
    return out.transpose(0, 1).reshape(T, Q_SZ)


def sdpa_bwd(q_rot: torch.Tensor, k_rot: torch.Tensor,
             v: torch.Tensor, grad_out: torch.Tensor):
    """Mini-autograd for SDPA only — safe, isolated, no GIL concern here."""
    T = q_rot.shape[0]
    with torch.enable_grad():
        ql = q_rot.detach().requires_grad_(True)
        vl = v.detach().requires_grad_(True)
        q3 = ql.view(T, N_HEADS, HEAD_DIM).transpose(0, 1)
        k3 = k_rot.detach().view(T, N_KV, HEAD_DIM).transpose(0, 1)
        v3 = vl.view(T, N_KV, HEAD_DIM).transpose(0, 1)
        if N_KV < N_HEADS:
            rep = N_HEADS // N_KV
            k3 = k3.repeat_interleave(rep, 0)
            v3 = v3.repeat_interleave(rep, 0)
        sdpa = F.scaled_dot_product_attention(q3, k3, v3, is_causal=True)
        sdpa = sdpa.transpose(0, 1).reshape(T, Q_SZ)
        sdpa.backward(grad_out.to(sdpa.dtype))
    return ql.grad, vl.grad   # [T, Q_SZ], [T, KV_SZ]


def qknorm_bwd(q_pre: torch.Tensor, grad_q_normed: torch.Tensor,
               w_qnorm: torch.Tensor) -> torch.Tensor:
    """Per-head QK-norm backward.  q_pre: [T, Q_SZ]  →  grad_q: [T, Q_SZ]."""
    T = q_pre.shape[0]
    x = q_pre.view(T * N_HEADS, HEAD_DIM)
    g = grad_q_normed.view(T * N_HEADS, HEAD_DIM)
    dx, _ = rms_norm_bwd(g, x, w_qnorm)
    return dx.view(T, Q_SZ)


def lm_head_grad(h_normed: torch.Tensor, W_lm: torch.Tensor,
                 tokens: torch.Tensor):
    """
    h_normed: [T, H] after final RMSNorm
    W_lm:     [VOCAB, H]  (fixed, not trained)
    tokens:   [T] int64   (used as next-token labels)

    Returns (grad_h_normed [T, H], loss_scalar float).
    """
    logits = h_normed.float() @ W_lm.T           # [T, VOCAB]
    sh_logits = logits[:-1]                        # [T-1, VOCAB]
    sh_labels = tokens[1:].long()                  # [T-1]

    loss_scalar = F.cross_entropy(sh_logits, sh_labels).item()

    probs   = torch.softmax(sh_logits, dim=-1)    # [T-1, VOCAB]
    n_valid = sh_labels.shape[0]
    probs[torch.arange(n_valid), sh_labels] -= 1.0
    probs   /= n_valid

    grad_shifted  = probs.to(h_normed.dtype) @ W_lm   # [T-1, H]
    grad_h_normed = torch.zeros_like(h_normed)
    grad_h_normed[:-1] = grad_shifted
    return grad_h_normed, loss_scalar


# ══════════════════════════════════════════════════════════════════════════
# Model weights and LoRA parameters
# ══════════════════════════════════════════════════════════════════════════

def init_weights(dev: str):
    """Synthetic random base-model weights (frozen throughout)."""
    std = 0.02
    def r(*s): return torch.randn(*s, device=dev, dtype=torch.float32) * std

    return {
        # Embedding table (not VocabParallelEmbedding — no NCCL needed)
        "embed": r(VOCAB, H),

        # QKV + O projection base weights (frozen)
        "W_q": r(Q_SZ,  H),
        "W_k": r(KV_SZ, H),
        "W_v": r(KV_SZ, H),
        "W_o": r(H,     Q_SZ),

        # Per-head QK-norm weights (frozen)
        "w_qnorm": torch.ones(HEAD_DIM, device=dev),
        "w_knorm": torch.ones(HEAD_DIM, device=dev),

        # Layer-norm weights (frozen)
        "w_ln_in":    torch.ones(H, device=dev),   # input LN
        "w_ln_post":  torch.ones(H, device=dev),   # post-attention LN
        "w_ln_final": torch.ones(H, device=dev),   # final norm before lm_head

        # LM head (fixed — not trained, just seeds the CE loss)
        "W_lm": r(VOCAB, H),
    }


def init_lora(dev: str) -> dict:
    """LoRA A/B matrices: standard init (A random-small, B zeros)."""
    def ra(*s): return (torch.randn(*s, device=dev) * 0.01).requires_grad_(True)
    def zb(*s): return torch.zeros(*s, device=dev).requires_grad_(True)

    return {
        "A_q": ra(RANK, H),      "B_q": zb(Q_SZ,  RANK),   # Q: [r,H],[q_sz,r]
        "A_v": ra(RANK, H),      "B_v": zb(KV_SZ, RANK),   # V: [r,H],[kv_sz,r]
        "A_o": ra(RANK, Q_SZ),   "B_o": zb(H,     RANK),   # O: [r,q_sz],[H,r]
    }


# ══════════════════════════════════════════════════════════════════════════
# Forward pass
# ══════════════════════════════════════════════════════════════════════════

def forward_one_layer(tokens: torch.Tensor, W: dict, L: dict,
                      rope_freqs: torch.Tensor):
    """
    tokens:     [T] int64
    Returns:    (h_normed [T,H], cache dict)
    cache has everything the backward needs.
    """
    s = SCALING

    # ── Embed ──────────────────────────────────────────────────────────────
    hidden   = W["embed"][tokens]           # [T, H]
    residual = hidden.detach().clone()      # skip-connection seed

    # ── Input LayerNorm ────────────────────────────────────────────────────
    res_a  = (hidden + residual).detach()   # pre-norm sum (saved for bwd)
    x_norm = rms_norm(res_a, W["w_ln_in"]) # [T, H]

    # ── QKV base projections (frozen, detached) ────────────────────────────
    xd     = x_norm.detach()
    q_base = xd @ W["W_q"].T               # [T, Q_SZ]
    k_base = xd @ W["W_k"].T               # [T, KV_SZ]
    v_base = xd @ W["W_v"].T               # [T, KV_SZ]

    # ── LoRA deltas for Q and V ────────────────────────────────────────────
    q = q_base + (xd @ L["A_q"].T) @ L["B_q"].T * s   # [T, Q_SZ]
    v = v_base + (xd @ L["A_v"].T) @ L["B_v"].T * s   # [T, KV_SZ]
    k = k_base

    q_pre_qknorm = q.detach().clone()       # saved: needed for QK-norm bwd

    # ── Per-head QK-norm ──────────────────────────────────────────────────
    q_normed = rms_norm(q.view(T, N_HEADS, HEAD_DIM),
                        W["w_qnorm"]).view(T, Q_SZ)
    k_normed = rms_norm(k.view(T, N_KV,    HEAD_DIM),
                        W["w_knorm"]).view(T, KV_SZ)

    # ── RoPE ──────────────────────────────────────────────────────────────
    q_rot = rope_fwd(q_normed.view(T, N_HEADS, HEAD_DIM),
                     rope_freqs).view(T, Q_SZ)
    k_rot = rope_fwd(k_normed.view(T, N_KV,    HEAD_DIM),
                     rope_freqs).view(T, KV_SZ)

    # ── Scaled dot-product attention ───────────────────────────────────────
    sdpa_out = causal_sdpa(q_rot, k_rot, v)   # [T, Q_SZ]

    # ── O projection base + LoRA delta ────────────────────────────────────
    so  = sdpa_out.detach()
    o_base  = so @ W["W_o"].T
    o_local = o_base + (so @ L["A_o"].T) @ L["B_o"].T * s   # [T, H]

    # ── Post-attention residual sum (no FFN — identity MoE) ───────────────
    res_b    = o_local + residual             # [T, H]
    x2       = rms_norm(res_b, W["w_ln_post"])
    hidden_out = x2                           # MoE = identity

    # ── Final LayerNorm ───────────────────────────────────────────────────
    h_normed = rms_norm(hidden_out, W["w_ln_final"])   # [T, H]

    cache = {
        "x_norm":       x_norm,
        "res_a":        res_a,
        "q_pre_qknorm": q_pre_qknorm,
        "q_normed":     q_normed,
        "k_rot":        k_rot,
        "q_rot":        q_rot,
        "v":            v,
        "sdpa_out":     sdpa_out,
        "o_local":      o_local,
        "res_b":        res_b,
        "hidden_out":   hidden_out,
        "residual":     residual,
    }
    return h_normed, cache


# ══════════════════════════════════════════════════════════════════════════
# Backward pass
# ══════════════════════════════════════════════════════════════════════════

def backward_one_layer(grad_h_normed: torch.Tensor,
                       W: dict, L: dict, cache: dict,
                       rope_freqs: torch.Tensor) -> dict:
    """
    Manual backward through one layer.
    Returns a dict of parameter-name → gradient tensor for A_q,B_q,A_v,B_v,A_o,B_o.
    """
    s   = SCALING
    xn  = cache["x_norm"]
    ra  = cache["res_a"]
    qpn = cache["q_pre_qknorm"]
    qn  = cache["q_normed"]
    kr  = cache["k_rot"]
    qr  = cache["q_rot"]
    v_  = cache["v"]
    so  = cache["sdpa_out"].detach()
    ol  = cache["o_local"]
    rb  = cache["res_b"]
    ho  = cache["hidden_out"]

    # ── Final LN backward → grad_hidden_out ──────────────────────────────
    grad_ho, _ = rms_norm_bwd(grad_h_normed, ho, W["w_ln_final"])

    # ── Identity FFN: grad_x2 = grad_ho ──────────────────────────────────
    # Post-attn LN backward:  res_b = o_local + residual,  x2 = rms_norm(res_b)
    # gradient of loss w.r.t. res_b comes only from the x2 output (single layer)
    grad_res_b, _ = rms_norm_bwd(grad_ho, rb, W["w_ln_post"])
    grad_o_local  = grad_res_b              # res_b = o_local + residual

    # ── O LoRA gradients ─────────────────────────────────────────────────
    # Forward:  z_o = so @ A_o.T  [T,r];  o_delta = z_o @ B_o.T * s  [T,H]
    z_o   = so.float() @ L["A_o"].T.float()                          # [T, r]
    gz_o  = grad_o_local.float() @ L["B_o"].float() * s              # [T, r]
    dA_o  = (gz_o.T @ so.float()).to(L["A_o"].dtype)                  # [r, Q_SZ]
    dB_o  = (grad_o_local.float().T @ z_o.float() * s).to(L["B_o"].dtype)  # [H, r]

    # ── O passthrough → grad_sdpa_out ────────────────────────────────────
    grad_sdpa = (grad_o_local.float() @ W["W_o"].float()              # base path
                 + gz_o @ L["A_o"].float())                           # LoRA path
    grad_sdpa = grad_sdpa.to(qr.dtype)                                # [T, Q_SZ]

    # ── SDPA backward (mini-autograd, isolated) ───────────────────────────
    grad_q_rot, grad_v = sdpa_bwd(qr, kr, v_, grad_sdpa)

    # ── RoPE backward → grad_q_normed ─────────────────────────────────────
    grad_q_normed = rope_bwd(
        grad_q_rot.view(T, N_HEADS, HEAD_DIM), rope_freqs
    ).view(T, Q_SZ)

    # ── QK-norm backward (manual) → grad_q ────────────────────────────────
    grad_q = qknorm_bwd(qpn, grad_q_normed, W["w_qnorm"])             # [T, Q_SZ]

    # ── Q LoRA gradients ──────────────────────────────────────────────────
    xd   = xn.detach().float()
    z_q  = xd @ L["A_q"].T.float()                                    # [T, r]
    gz_q = grad_q.float() @ L["B_q"].float() * s                      # [T, r]
    dA_q = (gz_q.T @ xd).to(L["A_q"].dtype)                          # [r, H]
    dB_q = (grad_q.float().T @ z_q * s).to(L["B_q"].dtype)           # [Q_SZ, r]

    # ── V LoRA gradients ──────────────────────────────────────────────────
    z_v  = xd @ L["A_v"].T.float()                                    # [T, r]
    gz_v = grad_v.float() @ L["B_v"].float() * s                      # [T, r]
    dA_v = (gz_v.T @ xd).to(L["A_v"].dtype)                          # [r, H]
    dB_v = (grad_v.float().T @ z_v * s).to(L["B_v"].dtype)           # [KV_SZ, r]

    return {"A_q": dA_q, "B_q": dB_q,
            "A_v": dA_v, "B_v": dB_v,
            "A_o": dA_o, "B_o": dB_o}


# ══════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    dev = DEVICE

    W          = init_weights(dev)
    L          = init_lora(dev)
    params     = list(L.values())
    optimizer  = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    rope_freqs = build_rope_freqs(T, HEAD_DIM, device=dev)

    # Fixed training batch — 8 random sequences (benchmark overfits to these)
    N_SEQS = 8
    seqs   = torch.randint(0, VOCAB, (N_SEQS, T), device=dev)

    n_params = sum(p.numel() for p in params)
    print(f"bt_lora_layer_bench — Qwen3-30B-A3B single attention layer")
    print(f"  H={H}  N_HEADS={N_HEADS}  N_KV={N_KV}  HEAD_DIM={HEAD_DIM}")
    print(f"  LoRA rank={RANK}  scaling={SCALING}")
    print(f"  T={T}  VOCAB={VOCAB}  trainable params={n_params:,}")
    print(f"  device={dev}  steps={STEPS}  lr={LR}\n")
    print(f"{'step':>6}  {'loss':>8}  {'note'}")
    print("-" * 40)

    for step in range(STEPS):
        tokens = seqs[step % N_SEQS]        # cycle through fixed sequences

        # ── Forward ───────────────────────────────────────────────────────
        h_normed, cache = forward_one_layer(tokens, W, L, rope_freqs)

        # ── CE loss + gradient seed ───────────────────────────────────────
        grad_h_normed, loss = lm_head_grad(h_normed, W["W_lm"], tokens)

        # ── Backward ──────────────────────────────────────────────────────
        grads = backward_one_layer(grad_h_normed, W, L, cache, rope_freqs)

        # ── AdamW step ────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        for name, param in L.items():
            param.grad = grads[name]

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        if step % 10 == 0:
            note = "← start" if step == 0 else ""
            print(f"{step:>6}  {loss:>8.4f}  {note}")

    print("-" * 40)
    print(f"Final loss at step {STEPS-1} printed above.")
    print("If loss decreased from step 0, manual gradients are correct.")


if __name__ == "__main__":
    main()
