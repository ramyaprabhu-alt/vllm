"""
qwen3_moe_lora_microbench.py — Standalone single-layer MoE LoRA training
microbenchmark for Qwen3-30B-A3B.

Trains LoRA adapters on MoE expert layers (gate_proj, up_proj, down_proj)
and/or attention projections (Q, V, O) on ONE transformer layer with
Qwen3-30B-A3B dimensions.  No vLLM, no NCCL.

Uses synthetic random base weights (no model loading), manual forward/backward
(no autograd except SDPA backward), and SwiGLU-activated MoE with top-k
routing — matching the production _training_moe_forward / _moe_backward_passthrough
code paths in bt_lora_trainer.py.

Single adapter:
    .venv/bin/python qwen3_moe_lora_microbench.py
    .venv/bin/python qwen3_moe_lora_microbench.py --mode moe_only

Multi-adapter (compares naive loop vs Punica-style batched):
    .venv/bin/python qwen3_moe_lora_microbench.py --num-adapters 4 --multi-adapter naive
    .venv/bin/python qwen3_moe_lora_microbench.py --num-adapters 4 --multi-adapter batched
    .venv/bin/python qwen3_moe_lora_microbench.py --num-adapters 4 --multi-adapter fused
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ══════════════════════════════════════════════════════════════════════════
# Qwen3-30B-A3B defaults (overridable via CLI)
# ══════════════════════════════════════════════════════════════════════════

H        = 2048
N_HEADS  = 32
N_KV     = 4
HEAD_DIM = 128
Q_SZ     = N_HEADS * HEAD_DIM   # 4096
KV_SZ    = N_KV * HEAD_DIM      # 512
MOE_INT  = 768
N_EXPERTS = 128
TOP_K    = 8
RANK     = 16
SCALING  = 1.0   # lora_alpha / lora_rank = 16/16
VOCAB    = 4096
T        = 128


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank",             type=int,   default=16)
    ap.add_argument("--num-experts",      type=int,   default=128)
    ap.add_argument("--top-k",            type=int,   default=8)
    ap.add_argument("--moe-intermediate", type=int,   default=768)
    ap.add_argument("--seq-len",          type=int,   default=128,  help="training seq len")
    ap.add_argument("--steps",            type=int,   default=300,  help="timed training steps")
    ap.add_argument("--warmup",           type=int,   default=20,   help="untimed warmup steps")
    ap.add_argument("--lr",               type=float, default=2e-4)
    ap.add_argument("--moe-lora-targets", type=str,   default="gate,up,down",
                     help="comma-separated: gate,up,down")
    ap.add_argument("--n-lora-experts",   type=int,   default=None,
                     help="experts with LoRA (default: all)")
    ap.add_argument("--mode",  choices=["both", "attn_only", "moe_only"], default="both")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num-adapters",  type=int, default=1,
                     help="number of concurrent LoRA adapters to train")
    ap.add_argument("--multi-adapter", choices=["naive", "batched", "fused"], default="naive",
                     help="naive=loop; batched=gather+bmm; fused=Triton kernels")
    return ap.parse_args()


def apply_cfg(cfg):
    """Set module-level constants from parsed args."""
    global H, N_HEADS, N_KV, HEAD_DIM, Q_SZ, KV_SZ
    global MOE_INT, N_EXPERTS, TOP_K, RANK, SCALING, VOCAB, T
    RANK      = cfg.rank
    SCALING   = 1.0
    N_EXPERTS = cfg.num_experts
    TOP_K     = cfg.top_k
    MOE_INT   = cfg.moe_intermediate
    T         = cfg.seq_len
    Q_SZ      = N_HEADS * HEAD_DIM
    KV_SZ     = N_KV * HEAD_DIM
    if cfg.n_lora_experts is None:
        cfg.n_lora_experts = cfg.num_experts
    cfg.moe_targets = set(cfg.moe_lora_targets.split(","))
    cfg.torch_dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32


# ══════════════════════════════════════════════════════════════════════════
# Primitive ops (reused from bt_lora_layer_bench.py)
# ══════════════════════════════════════════════════════════════════════════

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return (x.float() / rms * w.float()).to(x.dtype)


def rms_norm_bwd(grad: torch.Tensor, x: torch.Tensor, w: torch.Tensor,
                 eps: float = 1e-6):
    rms   = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    x_hat = x.float() / rms
    dxh   = grad.float() * w.float()
    dx    = (dxh - x_hat * (dxh * x_hat).mean(-1, keepdim=True)) / rms
    dw    = (grad.float() * x_hat).sum(tuple(range(x.dim() - 1)))
    return dx.to(x.dtype), dw.to(w.dtype)


def build_rope_freqs(seq_len: int, head_dim: int,
                     theta: float = 10000.0, device=None) -> torch.Tensor:
    inv   = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                          device=device) / head_dim)
    t     = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_fwd(x: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
    Tl, nh, hd = x.shape
    xc = torch.view_as_complex(x.float().reshape(Tl, nh, hd // 2, 2))
    out = torch.view_as_real(xc * fc[:Tl].unsqueeze(1)).flatten(3)
    return out.to(x.dtype)


def rope_bwd(grad: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
    Tl, nh, hd = grad.shape
    gc = torch.view_as_complex(grad.float().reshape(Tl, nh, hd // 2, 2))
    out = torch.view_as_real(gc * fc[:Tl].conj().unsqueeze(1)).flatten(3)
    return out.to(grad.dtype)


def causal_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    Tl = q.shape[0]
    q3 = q.view(Tl, N_HEADS, HEAD_DIM).transpose(0, 1)
    k3 = k.view(Tl, N_KV,    HEAD_DIM).transpose(0, 1)
    v3 = v.view(Tl, N_KV,    HEAD_DIM).transpose(0, 1)
    if N_KV < N_HEADS:
        rep = N_HEADS // N_KV
        k3 = k3.repeat_interleave(rep, 0)
        v3 = v3.repeat_interleave(rep, 0)
    out = F.scaled_dot_product_attention(q3, k3, v3, is_causal=True)
    return out.transpose(0, 1).reshape(Tl, Q_SZ)


def sdpa_bwd(q_rot: torch.Tensor, k_rot: torch.Tensor,
             v: torch.Tensor, grad_out: torch.Tensor):
    Tl = q_rot.shape[0]
    with torch.enable_grad():
        ql = q_rot.detach().requires_grad_(True)
        vl = v.detach().requires_grad_(True)
        q3 = ql.view(Tl, N_HEADS, HEAD_DIM).transpose(0, 1)
        k3 = k_rot.detach().view(Tl, N_KV, HEAD_DIM).transpose(0, 1)
        v3 = vl.view(Tl, N_KV, HEAD_DIM).transpose(0, 1)
        if N_KV < N_HEADS:
            rep = N_HEADS // N_KV
            k3 = k3.repeat_interleave(rep, 0)
            v3 = v3.repeat_interleave(rep, 0)
        sdpa = F.scaled_dot_product_attention(q3, k3, v3, is_causal=True)
        sdpa = sdpa.transpose(0, 1).reshape(Tl, Q_SZ)
        sdpa.backward(grad_out.to(sdpa.dtype))
    return ql.grad, vl.grad


def qknorm_bwd(q_pre: torch.Tensor, grad_q_normed: torch.Tensor,
               w_qnorm: torch.Tensor) -> torch.Tensor:
    Tl = q_pre.shape[0]
    x = q_pre.view(Tl * N_HEADS, HEAD_DIM)
    g = grad_q_normed.view(Tl * N_HEADS, HEAD_DIM)
    dx, _ = rms_norm_bwd(g, x, w_qnorm)
    return dx.view(Tl, Q_SZ)


def lm_head_grad(h_normed: torch.Tensor, W_lm: torch.Tensor,
                 tokens: torch.Tensor):
    logits = h_normed.float() @ W_lm.float().T
    sh_logits = logits[:-1]
    sh_labels = tokens[1:].long()
    loss_scalar = F.cross_entropy(sh_logits, sh_labels).item()
    probs   = torch.softmax(sh_logits, dim=-1)
    n_valid = sh_labels.shape[0]
    probs[torch.arange(n_valid, device=probs.device), sh_labels] -= 1.0
    probs   /= n_valid
    grad_shifted  = probs.to(h_normed.dtype) @ W_lm.to(h_normed.dtype)
    grad_h_normed = torch.zeros_like(h_normed)
    grad_h_normed[:-1] = grad_shifted
    return grad_h_normed, loss_scalar


# ══════════════════════════════════════════════════════════════════════════
# Weight and LoRA initialization
# ══════════════════════════════════════════════════════════════════════════

def init_weights(dev: str, dtype: torch.dtype):
    std = 0.02
    def r(*s): return (torch.randn(*s, device=dev, dtype=torch.float32) * std).to(dtype)

    return {
        "embed":      r(VOCAB, H),
        "W_q":        r(Q_SZ,  H),
        "W_k":        r(KV_SZ, H),
        "W_v":        r(KV_SZ, H),
        "W_o":        r(H,     Q_SZ),
        "w_qnorm":    torch.ones(HEAD_DIM, device=dev, dtype=dtype),
        "w_knorm":    torch.ones(HEAD_DIM, device=dev, dtype=dtype),
        "w_ln_in":    torch.ones(H, device=dev, dtype=dtype),
        "w_ln_post":  torch.ones(H, device=dev, dtype=dtype),
        "w_ln_final": torch.ones(H, device=dev, dtype=dtype),
        "W_lm":       r(VOCAB, H),
        # MoE weights
        "gate_weight": r(N_EXPERTS, H),
        "w13":         r(N_EXPERTS, 2 * MOE_INT, H),
        "w2":          r(N_EXPERTS, H, MOE_INT),
    }


def init_attn_lora(dev: str, dtype: torch.dtype) -> dict:
    def ra(*s): return (torch.randn(*s, device=dev, dtype=torch.float32) * 0.01).to(dtype).requires_grad_(True)
    def zb(*s): return torch.zeros(*s, device=dev, dtype=dtype).requires_grad_(True)
    return {
        "A_q": ra(RANK, H),      "B_q": zb(Q_SZ,  RANK),
        "A_v": ra(RANK, H),      "B_v": zb(KV_SZ, RANK),
        "A_o": ra(RANK, Q_SZ),   "B_o": zb(H,     RANK),
    }


def init_moe_lora(dev: str, dtype: torch.dtype, cfg) -> dict:
    n = cfg.n_lora_experts
    targets = cfg.moe_targets
    def ra(*s): return (torch.randn(*s, device=dev, dtype=torch.float32) * 0.01).to(dtype).requires_grad_(True)
    def zb(*s): return torch.zeros(*s, device=dev, dtype=dtype).requires_grad_(True)
    d = {}
    if "gate" in targets:
        d["A_gate"] = ra(n, RANK, H)
        d["B_gate"] = zb(n, MOE_INT, RANK)
    if "up" in targets:
        d["A_up"] = ra(n, RANK, H)
        d["B_up"] = zb(n, MOE_INT, RANK)
    if "down" in targets:
        d["A_down"] = ra(n, RANK, MOE_INT)
        d["B_down"] = zb(n, H, RANK)
    return d


# ══════════════════════════════════════════════════════════════════════════
# MoE forward with LoRA
# ══════════════════════════════════════════════════════════════════════════

def moe_forward(x2: torch.Tensor, W: dict, M: dict, cfg) -> tuple:
    """MoE forward with optional LoRA on expert projections.

    Returns (output, moe_cache) where moe_cache stores per-expert
    intermediates needed for the backward pass.
    """
    Tl, Hl = x2.shape
    dev, dtype = x2.device, x2.dtype
    s = SCALING
    targets = cfg.moe_targets
    n_lora = cfg.n_lora_experts
    has_moe_lora = cfg.mode in ("both", "moe_only")

    router_logits = x2.detach() @ W["gate_weight"].T
    scores = torch.softmax(router_logits.float(), dim=-1).to(dtype)
    topk_scores, topk_ids = torch.topk(scores, TOP_K, dim=-1)
    topk_scores = topk_scores / (topk_scores.sum(-1, keepdim=True) + 1e-9)

    w13 = W["w13"]
    w2  = W["w2"]
    output = torch.zeros(Tl, Hl, device=dev, dtype=dtype)
    moe_cache = {
        "topk_scores": topk_scores,
        "topk_ids": topk_ids,
        "experts": {},
    }

    for e in range(N_EXPERTS):
        token_mask = (topk_ids == e).any(-1)
        if not token_mask.any():
            continue
        tokens = x2[token_mask]
        gate_up = tokens @ w13[e].T

        if has_moe_lora and e < n_lora:
            if "gate" in targets:
                gate_up[:, :MOE_INT] += (tokens @ M["A_gate"][e].T) @ M["B_gate"][e].T * s
            if "up" in targets:
                gate_up[:, MOE_INT:] += (tokens @ M["A_up"][e].T) @ M["B_up"][e].T * s

        gate = gate_up[:, :MOE_INT]
        up   = gate_up[:, MOE_INT:]
        act  = F.silu(gate) * up

        out_e = act @ w2[e].T
        if has_moe_lora and e < n_lora and "down" in targets:
            out_e = out_e + (act @ M["A_down"][e].T) @ M["B_down"][e].T * s

        exp_mask = (topk_ids[token_mask] == e)
        score_e  = (topk_scores[token_mask] * exp_mask.to(dtype)).sum(-1, keepdim=True)
        output[token_mask] += out_e * score_e

        moe_cache["experts"][e] = {
            "tokens": tokens.detach(),
            "gate_up": gate_up.detach(),
            "act": act.detach(),
            "token_mask": token_mask,
            "score_e": score_e.detach(),
        }

    return output, moe_cache


# ══════════════════════════════════════════════════════════════════════════
# MoE backward with LoRA gradients
# ══════════════════════════════════════════════════════════════════════════

def moe_backward(grad_hidden: torch.Tensor, W: dict, M: dict,
                 moe_cache: dict, cfg) -> tuple:
    """Manual backward through MoE with LoRA gradient computation.

    Returns (grad_x2, moe_grads) where moe_grads maps param name to
    accumulated gradient tensor.
    """
    dtype = grad_hidden.dtype
    s = SCALING
    targets = cfg.moe_targets
    n_lora = cfg.n_lora_experts
    has_moe_lora = cfg.mode in ("both", "moe_only")

    w13 = W["w13"]
    w2  = W["w2"]
    topk_scores = moe_cache["topk_scores"]
    topk_ids    = moe_cache["topk_ids"]

    grad_x2 = torch.zeros_like(grad_hidden)

    moe_grads = {}
    if has_moe_lora:
        for key in M:
            moe_grads[key] = torch.zeros_like(M[key])

    for e, ecache in moe_cache["experts"].items():
        tokens     = ecache["tokens"]
        gate_up    = ecache["gate_up"]
        act        = ecache["act"]
        token_mask = ecache["token_mask"]
        score_e    = ecache["score_e"]

        gate = gate_up[:, :MOE_INT]
        up   = gate_up[:, MOE_INT:]

        grad_out_e = grad_hidden[token_mask] * score_e

        # Backward through down_proj: out_e = act @ w2[e].T [+ LoRA]
        grad_act = grad_out_e @ w2[e]

        if has_moe_lora and e < n_lora and "down" in targets:
            z_down  = act.float() @ M["A_down"][e].T.float()
            gz_down = grad_out_e.float() @ M["B_down"][e].float() * s
            moe_grads["A_down"][e] = (gz_down.T @ act.float()).to(dtype)
            moe_grads["B_down"][e] = (grad_out_e.float().T @ z_down * s).to(dtype)
            grad_act = grad_act + (gz_down @ M["A_down"][e].float()).to(dtype)

        # SwiGLU backward: act = silu(gate) * up
        sig        = torch.sigmoid(gate.float())
        silu_deriv = sig * (1.0 + gate.float() * (1.0 - sig))
        grad_gate  = (grad_act.float() * silu_deriv * up.float()).to(dtype)
        grad_up    = (grad_act.float() * F.silu(gate.float())).to(dtype)
        grad_gate_up = torch.cat([grad_gate, grad_up], dim=-1)

        # Backward through gate_up_proj: gate_up = tokens @ w13[e].T [+ LoRA]
        grad_x2[token_mask] += grad_gate_up @ w13[e]

        if has_moe_lora and e < n_lora:
            if "gate" in targets:
                z_gate  = tokens.float() @ M["A_gate"][e].T.float()
                gz_gate = grad_gate.float() @ M["B_gate"][e].float() * s
                moe_grads["A_gate"][e] = (gz_gate.T @ tokens.float()).to(dtype)
                moe_grads["B_gate"][e] = (grad_gate.float().T @ z_gate * s).to(dtype)
                grad_x2[token_mask] += (gz_gate @ M["A_gate"][e].float()).to(dtype)

            if "up" in targets:
                z_up  = tokens.float() @ M["A_up"][e].T.float()
                gz_up = grad_up.float() @ M["B_up"][e].float() * s
                moe_grads["A_up"][e] = (gz_up.T @ tokens.float()).to(dtype)
                moe_grads["B_up"][e] = (grad_up.float().T @ z_up * s).to(dtype)
                grad_x2[token_mask] += (gz_up @ M["A_up"][e].float()).to(dtype)

    return grad_x2, moe_grads


# ══════════════════════════════════════════════════════════════════════════
# Full single-layer forward / backward
# ══════════════════════════════════════════════════════════════════════════

def forward_one_layer(tokens: torch.Tensor, W: dict, L: dict, M: dict,
                      rope_freqs: torch.Tensor, cfg):
    s = SCALING
    has_attn_lora = cfg.mode in ("both", "attn_only")

    # Embed
    hidden   = W["embed"][tokens]
    residual = hidden.detach().clone()

    # Input LayerNorm
    res_a  = (hidden + residual).detach()
    x_norm = rms_norm(res_a, W["w_ln_in"])

    # QKV base projections (frozen)
    xd     = x_norm.detach()
    q_base = xd @ W["W_q"].T
    k_base = xd @ W["W_k"].T
    v_base = xd @ W["W_v"].T

    # LoRA deltas for Q and V
    if has_attn_lora:
        q = q_base + (xd @ L["A_q"].T) @ L["B_q"].T * s
        v = v_base + (xd @ L["A_v"].T) @ L["B_v"].T * s
    else:
        q = q_base
        v = v_base
    k = k_base
    q_pre_qknorm = q.detach().clone()

    # Per-head QK-norm
    Tl = tokens.shape[0]
    q_normed = rms_norm(q.view(Tl, N_HEADS, HEAD_DIM), W["w_qnorm"]).view(Tl, Q_SZ)
    k_normed = rms_norm(k.view(Tl, N_KV,    HEAD_DIM), W["w_knorm"]).view(Tl, KV_SZ)

    # RoPE
    q_rot = rope_fwd(q_normed.view(Tl, N_HEADS, HEAD_DIM), rope_freqs).view(Tl, Q_SZ)
    k_rot = rope_fwd(k_normed.view(Tl, N_KV,    HEAD_DIM), rope_freqs).view(Tl, KV_SZ)

    # SDPA
    sdpa_out = causal_sdpa(q_rot, k_rot, v)

    # O projection + LoRA
    so = sdpa_out.detach()
    o_base = so @ W["W_o"].T
    if has_attn_lora:
        o_local = o_base + (so @ L["A_o"].T) @ L["B_o"].T * s
    else:
        o_local = o_base

    # Post-attention residual + LN
    res_b = o_local + residual
    x2    = rms_norm(res_b, W["w_ln_post"])

    # MoE forward
    moe_out, moe_cache = moe_forward(x2, W, M, cfg)

    # Final LN
    hidden_out = moe_out
    h_normed   = rms_norm(hidden_out, W["w_ln_final"])

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
        "x2":           x2,
        "hidden_out":   hidden_out,
        "residual":     residual,
        "moe_cache":    moe_cache,
    }
    return h_normed, cache


def backward_one_layer(grad_h_normed: torch.Tensor,
                       W: dict, L: dict, M: dict, cache: dict,
                       rope_freqs: torch.Tensor, cfg) -> tuple:
    s   = SCALING
    has_attn_lora = cfg.mode in ("both", "attn_only")
    xn  = cache["x_norm"]
    qpn = cache["q_pre_qknorm"]
    kr  = cache["k_rot"]
    qr  = cache["q_rot"]
    v_  = cache["v"]
    so  = cache["sdpa_out"].detach()
    rb  = cache["res_b"]
    ho  = cache["hidden_out"]
    x2  = cache["x2"]
    Tl  = xn.shape[0]

    # Final LN backward
    grad_ho, _ = rms_norm_bwd(grad_h_normed, ho, W["w_ln_final"])

    # MoE backward
    grad_x2, moe_grads = moe_backward(grad_ho, W, M, cache["moe_cache"], cfg)

    # Post-attn LN backward
    grad_res_b, _   = rms_norm_bwd(grad_x2, rb, W["w_ln_post"])
    grad_o_local = grad_res_b

    attn_grads = {}
    if has_attn_lora:
        # O LoRA gradients
        z_o  = so.float() @ L["A_o"].T.float()
        gz_o = grad_o_local.float() @ L["B_o"].float() * s
        attn_grads["A_o"] = (gz_o.T @ so.float()).to(L["A_o"].dtype)
        attn_grads["B_o"] = (grad_o_local.float().T @ z_o * s).to(L["B_o"].dtype)

        # O passthrough
        grad_sdpa = (grad_o_local.float() @ W["W_o"].float()
                     + gz_o @ L["A_o"].float()).to(qr.dtype)
    else:
        grad_sdpa = (grad_o_local.float() @ W["W_o"].float()).to(qr.dtype)

    # SDPA backward
    grad_q_rot, grad_v = sdpa_bwd(qr, kr, v_, grad_sdpa)

    # RoPE backward
    grad_q_normed = rope_bwd(
        grad_q_rot.view(Tl, N_HEADS, HEAD_DIM), rope_freqs
    ).view(Tl, Q_SZ)

    # QK-norm backward
    grad_q = qknorm_bwd(qpn, grad_q_normed, W["w_qnorm"])

    if has_attn_lora:
        # Q LoRA gradients
        xd   = xn.detach().float()
        z_q  = xd @ L["A_q"].T.float()
        gz_q = grad_q.float() @ L["B_q"].float() * s
        attn_grads["A_q"] = (gz_q.T @ xd).to(L["A_q"].dtype)
        attn_grads["B_q"] = (grad_q.float().T @ z_q * s).to(L["B_q"].dtype)

        # V LoRA gradients
        z_v  = xd @ L["A_v"].T.float()
        gz_v = grad_v.float() @ L["B_v"].float() * s
        attn_grads["A_v"] = (gz_v.T @ xd).to(L["A_v"].dtype)
        attn_grads["B_v"] = (grad_v.float().T @ z_v * s).to(L["B_v"].dtype)

    return attn_grads, moe_grads


# ══════════════════════════════════════════════════════════════════════════
# Multi-adapter: Punica-style batched MoE forward/backward
#
# Instead of looping over K adapters × E experts (naive), this loops over
# E experts once, reading base weights once per expert.  Per-token adapter
# indexing uses gather + bmm — the same pattern as vLLM's Punica BGMV
# kernels (vllm/lora/ops/torch_ops/lora_ops.py bgmv_shrink/bgmv_expand).
#
# Optimizations vs. naive approach:
#   1. Precompute expert→token index lists (eliminates per-expert nonzero)
#   2. All LoRA math in working dtype (no .float() upcasts per expert)
#   3. Pre-gather adapter LoRA weights by expert (avoids repeated fancy index)
#   4. Scatter gradients via scatter_add_ (avoids K-loop boolean masking)
# ══════════════════════════════════════════════════════════════════════════

def _precompute_expert_assignment(topk_ids: torch.Tensor,
                                  topk_scores: torch.Tensor,
                                  adapter_ids: torch.Tensor | None,
                                  num_experts: int, dtype: torch.dtype):
    """Build per-expert token index lists on GPU in one pass.

    Returns dict[expert_id] → {indices, scores} where indices are
    positions in the original [N_tok] dimension.
    """
    N_tok = topk_ids.shape[0]
    # Flatten top-k: each (token, slot) pair → one row
    flat_ids   = topk_ids.reshape(-1)                   # [N_tok * top_k]
    token_idx  = torch.arange(N_tok, device=topk_ids.device).unsqueeze(1).expand_as(topk_ids).reshape(-1)
    flat_scores = topk_scores.reshape(-1)

    # Sort by expert id — one argsort replaces N_EXPERTS nonzero calls
    order = torch.argsort(flat_ids, stable=True)
    sorted_expert = flat_ids[order]
    sorted_token  = token_idx[order]
    sorted_score  = flat_scores[order]

    # Find boundaries per expert via searchsorted
    expert_range = torch.arange(num_experts + 1, device=topk_ids.device)
    boundaries = torch.searchsorted(sorted_expert.contiguous(), expert_range)

    expert_map = {}
    for e in range(num_experts):
        start, end = boundaries[e].item(), boundaries[e + 1].item()
        if start == end:
            continue
        toks = sorted_token[start:end]
        # Aggregate per-token score for this expert (a token may appear
        # multiple times if top-k selected the same expert in multiple slots)
        unique_toks, inv = torch.unique(toks, return_inverse=True)
        score_agg = torch.zeros(unique_toks.shape[0], 1, device=topk_ids.device, dtype=dtype)
        score_agg.scatter_add_(0, inv.unsqueeze(1), sorted_score[start:end].unsqueeze(1).to(dtype))
        entry = {"indices": unique_toks, "scores": score_agg}
        if adapter_ids is not None:
            entry["aids"] = adapter_ids[unique_toks]
        expert_map[e] = entry
    return expert_map


def moe_forward_batched(x2_cat: torch.Tensor, W: dict,
                        adapters_moe: list[dict],
                        adapter_ids: torch.Tensor, cfg) -> tuple:
    """Expert-major MoE forward with per-token adapter gather."""
    K = len(adapters_moe)
    N_tok, Hl = x2_cat.shape
    dev, dtype = x2_cat.device, x2_cat.dtype
    s = SCALING
    targets = cfg.moe_targets
    n_lora = cfg.n_lora_experts
    has_moe_lora = cfg.mode in ("both", "moe_only")

    stacked = {}
    if has_moe_lora and adapters_moe:
        for key in adapters_moe[0]:
            stacked[key] = torch.stack([m[key] for m in adapters_moe])

    router_logits = x2_cat.detach() @ W["gate_weight"].T
    scores = torch.softmax(router_logits.float(), dim=-1).to(dtype)
    topk_scores, topk_ids = torch.topk(scores, TOP_K, dim=-1)
    topk_scores = topk_scores / (topk_scores.sum(-1, keepdim=True) + 1e-9)

    expert_map = _precompute_expert_assignment(
        topk_ids, topk_scores, adapter_ids, N_EXPERTS, dtype)

    w13 = W["w13"]
    w2  = W["w2"]
    output = torch.zeros(N_tok, Hl, device=dev, dtype=dtype)
    moe_cache = {"expert_map": expert_map, "experts": {}}

    for e, emap in expert_map.items():
        idx     = emap["indices"]
        score_e = emap["scores"]
        aids    = emap["aids"]
        tokens  = x2_cat[idx]                           # [n, H]

        gate_up = tokens @ w13[e].T                     # [n, 2*MOE_INT]

        if has_moe_lora and e < n_lora:
            if "gate" in targets:
                A_sel = stacked["A_gate"][aids, e]
                B_sel = stacked["B_gate"][aids, e]
                z = torch.bmm(A_sel, tokens.unsqueeze(2)).squeeze(2)
                gate_up[:, :MOE_INT] += torch.bmm(B_sel, z.unsqueeze(2)).squeeze(2) * s
            if "up" in targets:
                A_sel = stacked["A_up"][aids, e]
                B_sel = stacked["B_up"][aids, e]
                z = torch.bmm(A_sel, tokens.unsqueeze(2)).squeeze(2)
                gate_up[:, MOE_INT:] += torch.bmm(B_sel, z.unsqueeze(2)).squeeze(2) * s

        gate = gate_up[:, :MOE_INT]
        up   = gate_up[:, MOE_INT:]
        act  = F.silu(gate) * up

        out_e = act @ w2[e].T
        if has_moe_lora and e < n_lora and "down" in targets:
            A_sel = stacked["A_down"][aids, e]
            B_sel = stacked["B_down"][aids, e]
            z = torch.bmm(A_sel, act.unsqueeze(2)).squeeze(2)
            out_e = out_e + torch.bmm(B_sel, z.unsqueeze(2)).squeeze(2) * s

        output[idx] += out_e * score_e

        moe_cache["experts"][e] = {
            "tokens": tokens.detach(), "gate_up": gate_up.detach(),
            "act": act.detach(), "idx": idx, "score_e": score_e.detach(),
            "aids": aids,
        }

    return output, moe_cache


def moe_backward_batched(grad_hidden: torch.Tensor, W: dict,
                         adapters_moe: list[dict],
                         moe_cache: dict, cfg) -> tuple:
    """Expert-major backward with per-adapter gradient scatter.

    Returns (grad_x2, list_of_moe_grads) where list_of_moe_grads[k] maps
    param name to gradient tensor for adapter k.
    """
    K = len(adapters_moe)
    dtype = grad_hidden.dtype
    s = SCALING
    targets = cfg.moe_targets
    n_lora = cfg.n_lora_experts
    has_moe_lora = cfg.mode in ("both", "moe_only")

    w13 = W["w13"]
    w2  = W["w2"]

    grad_x2 = torch.zeros_like(grad_hidden)

    per_adapter_grads = []
    for k in range(K):
        g = {}
        if has_moe_lora:
            for key in adapters_moe[k]:
                g[key] = torch.zeros_like(adapters_moe[k][key])
        per_adapter_grads.append(g)

    stacked = {}
    if has_moe_lora and adapters_moe:
        for key in adapters_moe[0]:
            stacked[key] = torch.stack([m[key] for m in adapters_moe])

    for e, ecache in moe_cache["experts"].items():
        tokens  = ecache["tokens"]
        gate_up = ecache["gate_up"]
        act     = ecache["act"]
        idx     = ecache["idx"]
        score_e = ecache["score_e"]
        aids    = ecache["aids"]

        gate = gate_up[:, :MOE_INT]
        up   = gate_up[:, MOE_INT:]
        grad_out_e = grad_hidden[idx] * score_e

        grad_act = grad_out_e @ w2[e]

        if has_moe_lora and e < n_lora and "down" in targets:
            A_sel = stacked["A_down"][aids, e]
            B_sel = stacked["B_down"][aids, e]
            z_down = torch.bmm(A_sel, act.unsqueeze(2)).squeeze(2)
            gz_down = torch.bmm(B_sel.transpose(1, 2),
                                grad_out_e.unsqueeze(2)).squeeze(2) * s
            # Scatter gradients via scatter_add_ (no K-loop boolean masking)
            # dA[k,e] = gz_down[t].T @ act[t] summed over tokens t in adapter k
            dA_per = gz_down.unsqueeze(2) * act.unsqueeze(1)      # [n, r, MOE_INT]
            dB_per = grad_out_e.unsqueeze(2) * z_down.unsqueeze(1) * s  # [n, H, r]
            aids_a = aids.view(-1, 1, 1).expand_as(dA_per)
            aids_b = aids.view(-1, 1, 1).expand_as(dB_per)
            for k in range(K):
                per_adapter_grads[k]["A_down"][e] = torch.zeros(RANK, MOE_INT, device=act.device, dtype=dtype)
                per_adapter_grads[k]["B_down"][e] = torch.zeros(H, RANK, device=act.device, dtype=dtype)
            grad_A_all = torch.zeros(K, RANK, MOE_INT, device=act.device, dtype=dtype)
            grad_B_all = torch.zeros(K, H, RANK, device=act.device, dtype=dtype)
            grad_A_all.scatter_add_(0, aids_a, dA_per)
            grad_B_all.scatter_add_(0, aids_b, dB_per)
            for k in range(K):
                per_adapter_grads[k]["A_down"][e] = grad_A_all[k]
                per_adapter_grads[k]["B_down"][e] = grad_B_all[k]

            grad_act = grad_act + torch.bmm(gz_down.unsqueeze(1), A_sel).squeeze(1)

        # SwiGLU backward — stay in working dtype
        sig        = torch.sigmoid(gate)
        silu_deriv = sig * (1.0 + gate * (1.0 - sig))
        grad_gate  = grad_act * silu_deriv * up
        grad_up    = grad_act * F.silu(gate)
        grad_gate_up = torch.cat([grad_gate, grad_up], dim=-1)

        grad_x2[idx] += grad_gate_up @ w13[e]

        if has_moe_lora and e < n_lora:
            if "gate" in targets:
                A_sel = stacked["A_gate"][aids, e]
                B_sel = stacked["B_gate"][aids, e]
                z_gate = torch.bmm(A_sel, tokens.unsqueeze(2)).squeeze(2)
                gz_gate = torch.bmm(B_sel.transpose(1, 2),
                                    grad_gate.unsqueeze(2)).squeeze(2) * s
                dA_per = gz_gate.unsqueeze(2) * tokens.unsqueeze(1)
                dB_per = grad_gate.unsqueeze(2) * z_gate.unsqueeze(1) * s
                aids_a = aids.view(-1, 1, 1).expand_as(dA_per)
                aids_b = aids.view(-1, 1, 1).expand_as(dB_per)
                grad_A_all = torch.zeros(K, RANK, H, device=tokens.device, dtype=dtype)
                grad_B_all = torch.zeros(K, MOE_INT, RANK, device=tokens.device, dtype=dtype)
                grad_A_all.scatter_add_(0, aids_a, dA_per)
                grad_B_all.scatter_add_(0, aids_b, dB_per)
                for k in range(K):
                    per_adapter_grads[k]["A_gate"][e] = grad_A_all[k]
                    per_adapter_grads[k]["B_gate"][e] = grad_B_all[k]
                grad_x2[idx] += torch.bmm(gz_gate.unsqueeze(1), A_sel).squeeze(1)

            if "up" in targets:
                A_sel = stacked["A_up"][aids, e]
                B_sel = stacked["B_up"][aids, e]
                z_up = torch.bmm(A_sel, tokens.unsqueeze(2)).squeeze(2)
                gz_up = torch.bmm(B_sel.transpose(1, 2),
                                  grad_up.unsqueeze(2)).squeeze(2) * s
                dA_per = gz_up.unsqueeze(2) * tokens.unsqueeze(1)
                dB_per = grad_up.unsqueeze(2) * z_up.unsqueeze(1) * s
                aids_a = aids.view(-1, 1, 1).expand_as(dA_per)
                aids_b = aids.view(-1, 1, 1).expand_as(dB_per)
                grad_A_all = torch.zeros(K, RANK, H, device=tokens.device, dtype=dtype)
                grad_B_all = torch.zeros(K, MOE_INT, RANK, device=tokens.device, dtype=dtype)
                grad_A_all.scatter_add_(0, aids_a, dA_per)
                grad_B_all.scatter_add_(0, aids_b, dB_per)
                for k in range(K):
                    per_adapter_grads[k]["A_up"][e] = grad_A_all[k]
                    per_adapter_grads[k]["B_up"][e] = grad_B_all[k]
                grad_x2[idx] += torch.bmm(gz_up.unsqueeze(1), A_sel).squeeze(1)

    return grad_x2, per_adapter_grads


# ══════════════════════════════════════════════════════════════════════════
# Triton kernels for fused multi-adapter MoE LoRA
# ══════════════════════════════════════════════════════════════════════════

@triton.jit
def _fused_lora_fwd_kernel(
    x_ptr, a_ptr, b_ptr, out_ptr, aids_ptr,
    M, K_dim, N,
    scale,
    stride_xm, stride_xk,
    stride_a0, stride_a1, stride_a2,
    stride_b0, stride_b1, stride_b2,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, RANK: tl.constexpr,
    NUM_ADAPTERS: tl.constexpr,
):
    """Fused LoRA shrink+expand: out[m,n] += scale * B[aid,n,:] @ A[aid,:,:] @ x[m,:]

    One kernel launch replaces 2*K gathers + 2*K bmms.
    Intermediate z[RANK] stays in registers — never hits HBM.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < N

    aids = tl.load(aids_ptr + m_offs, mask=m_mask, other=0)

    # Shrink: z[m, r] = sum_k A[aid[m], r, k] * x[m, k], grouped by adapter
    z = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)
    for ki in tl.static_range(NUM_ADAPTERS):
        k_mask = (aids == ki)
        a_base = a_ptr + ki * stride_a0
        for k_start in range(0, K_dim, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            kk_mask = k_offs < K_dim
            x_tile = tl.load(
                x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                mask=(m_mask & k_mask)[:, None] & kk_mask[None, :], other=0.0)
            a_tile = tl.load(
                a_base + tl.arange(0, RANK)[:, None] * stride_a1
                + k_offs[None, :] * stride_a2,
                mask=kk_mask[None, :], other=0.0)
            z += tl.dot(x_tile.to(tl.float32), tl.trans(a_tile).to(tl.float32))

    # Expand: out[m, n] += scale * B[aid[m], n, :] @ z[m, :], grouped by adapter
    result = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ki in tl.static_range(NUM_ADAPTERS):
        k_mask = (aids == ki)
        b_tile = tl.load(
            b_ptr + ki * stride_b0
            + n_offs[:, None] * stride_b1
            + tl.arange(0, RANK)[None, :] * stride_b2,
            mask=n_mask[:, None], other=0.0).to(tl.float32)
        z_k = z * k_mask[:, None].to(tl.float32)
        result += tl.dot(z_k, tl.trans(b_tile))
    result *= scale

    existing = tl.load(
        out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on,
        mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(
        out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on,
        existing + result.to(existing.dtype),
        mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _fused_grad_scatter_kernel(
    a_ptr, b_ptr, aids_ptr, out_ptr,
    M, I_dim, J_dim,
    stride_am, stride_ai,
    stride_bm, stride_bj,
    stride_ok, stride_oi, stride_oj,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Fused outer-product scatter: out[k,i,j] = sum_{t:aid[t]==k} a[t,i]*b[t,j]

    Avoids materializing the full [M, I, J] intermediate tensor.
    The outer product is accumulated in registers per (i_tile, j_tile, adapter).
    """
    pid_j = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_k = tl.program_id(2)
    i_offs = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j_offs = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    i_mask = i_offs < I_dim
    j_mask = j_offs < J_dim
    acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask_local = m_offs < M
        aids = tl.load(aids_ptr + m_offs, mask=m_mask_local, other=-1)
        km = (aids == pid_k) & m_mask_local

        a_tile = tl.load(
            a_ptr + m_offs[:, None] * stride_am + i_offs[None, :] * stride_ai,
            mask=km[:, None] & i_mask[None, :], other=0.0)
        b_tile = tl.load(
            b_ptr + m_offs[:, None] * stride_bm + j_offs[None, :] * stride_bj,
            mask=km[:, None] & j_mask[None, :], other=0.0)
        acc += tl.dot(tl.trans(a_tile).to(tl.float32), b_tile.to(tl.float32))

    tl.store(
        out_ptr + pid_k * stride_ok + i_offs[:, None] * stride_oi
        + j_offs[None, :] * stride_oj,
        acc.to(out_ptr.dtype.element_ty),
        mask=i_mask[:, None] & j_mask[None, :])


# ── Python wrappers ──────────────────────────────────────────────────────

def fused_lora_fwd(x, stacked_a, stacked_b, aids, scale, num_adapters):
    """Fused LoRA forward: out = scale * B[aid,:,:] @ A[aid,:,:] @ x per token.

    stacked_a: [num_adapters, RANK, K_dim]
    stacked_b: [num_adapters, N, RANK]
    """
    M, K_dim = x.shape
    N = stacked_b.shape[1]
    out = torch.zeros(M, N, device=x.device, dtype=x.dtype)
    BLOCK_M = max(16, triton.next_power_of_2(M))
    _fused_lora_fwd_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, 64))](
        x, stacked_a, stacked_b, out, aids,
        M, K_dim, N, scale,
        x.stride(0), x.stride(1),
        stacked_a.stride(0), stacked_a.stride(1), stacked_a.stride(2),
        stacked_b.stride(0), stacked_b.stride(1), stacked_b.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=64, BLOCK_K=128,
        RANK=stacked_a.shape[1], NUM_ADAPTERS=num_adapters)
    return out


def fused_grad_scatter(a, b, aids, K, I_dim, J_dim):
    """Fused outer-product scatter: out[k,i,j] = sum_{t:aid==k} a[t,i]*b[t,j]"""
    M = a.shape[0]
    out = torch.zeros(K, I_dim, J_dim, device=a.device, dtype=a.dtype)
    BLOCK_I = min(triton.next_power_of_2(I_dim), 64)
    BLOCK_J = min(triton.next_power_of_2(J_dim), 128)
    BLOCK_M = max(16, triton.next_power_of_2(M))
    _fused_grad_scatter_kernel[(
        triton.cdiv(J_dim, BLOCK_J),
        triton.cdiv(I_dim, BLOCK_I),
        K,
    )](
        a, b, aids, out,
        M, I_dim, J_dim,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, BLOCK_M=BLOCK_M)
    return out


# ── Fused MoE forward/backward ──────────────────────────────────────────

def moe_forward_fused(x2_cat, W, adapters_moe, adapter_ids, cfg):
    """MoE forward using Triton-fused LoRA kernels."""
    K = len(adapters_moe)
    N_tok, Hl = x2_cat.shape
    dev, dtype = x2_cat.device, x2_cat.dtype
    s = SCALING
    targets = cfg.moe_targets
    n_lora = cfg.n_lora_experts
    has_moe_lora = cfg.mode in ("both", "moe_only")

    stacked = {}
    if has_moe_lora and adapters_moe:
        for key in adapters_moe[0]:
            stacked[key] = torch.stack([m[key] for m in adapters_moe])

    router_logits = x2_cat.detach() @ W["gate_weight"].T
    scores = torch.softmax(router_logits.float(), dim=-1).to(dtype)
    topk_scores, topk_ids = torch.topk(scores, TOP_K, dim=-1)
    topk_scores = topk_scores / (topk_scores.sum(-1, keepdim=True) + 1e-9)

    expert_map = _precompute_expert_assignment(
        topk_ids, topk_scores, adapter_ids, N_EXPERTS, dtype)

    w13 = W["w13"]
    w2  = W["w2"]
    output = torch.zeros(N_tok, Hl, device=dev, dtype=dtype)
    moe_cache = {"expert_map": expert_map, "experts": {}}

    for e, emap in expert_map.items():
        idx     = emap["indices"]
        score_e = emap["scores"]
        aids    = emap["aids"]
        tokens  = x2_cat[idx]

        gate_up = tokens @ w13[e].T

        if has_moe_lora and e < n_lora:
            if "gate" in targets:
                delta = fused_lora_fwd(
                    tokens, stacked["A_gate"][:, e], stacked["B_gate"][:, e],
                    aids, s, K)
                gate_up[:, :MOE_INT] += delta
            if "up" in targets:
                delta = fused_lora_fwd(
                    tokens, stacked["A_up"][:, e], stacked["B_up"][:, e],
                    aids, s, K)
                gate_up[:, MOE_INT:] += delta

        gate = gate_up[:, :MOE_INT]
        up   = gate_up[:, MOE_INT:]
        act  = F.silu(gate) * up

        out_e = act @ w2[e].T
        if has_moe_lora and e < n_lora and "down" in targets:
            delta = fused_lora_fwd(
                act, stacked["A_down"][:, e], stacked["B_down"][:, e],
                aids, s, K)
            out_e = out_e + delta

        output[idx] += out_e * score_e
        moe_cache["experts"][e] = {
            "tokens": tokens.detach(), "gate_up": gate_up.detach(),
            "act": act.detach(), "idx": idx, "score_e": score_e.detach(),
            "aids": aids,
        }

    return output, moe_cache


def moe_backward_fused(grad_hidden, W, adapters_moe, moe_cache, cfg):
    """MoE backward using Triton-fused gradient scatter kernel."""
    K = len(adapters_moe)
    dtype = grad_hidden.dtype
    s = SCALING
    targets = cfg.moe_targets
    n_lora = cfg.n_lora_experts
    has_moe_lora = cfg.mode in ("both", "moe_only")

    w13 = W["w13"]
    w2  = W["w2"]
    grad_x2 = torch.zeros_like(grad_hidden)

    per_adapter_grads = []
    for k in range(K):
        g = {}
        if has_moe_lora:
            for key in adapters_moe[k]:
                g[key] = torch.zeros_like(adapters_moe[k][key])
        per_adapter_grads.append(g)

    stacked = {}
    if has_moe_lora and adapters_moe:
        for key in adapters_moe[0]:
            stacked[key] = torch.stack([m[key] for m in adapters_moe])

    for e, ecache in moe_cache["experts"].items():
        tokens  = ecache["tokens"]
        gate_up = ecache["gate_up"]
        act     = ecache["act"]
        idx     = ecache["idx"]
        score_e = ecache["score_e"]
        aids    = ecache["aids"]

        gate = gate_up[:, :MOE_INT]
        up   = gate_up[:, MOE_INT:]
        grad_out_e = grad_hidden[idx] * score_e
        grad_act = grad_out_e @ w2[e]

        if has_moe_lora and e < n_lora and "down" in targets:
            A_sel = stacked["A_down"][:, e]              # [K, r, MOE_INT]
            B_sel = stacked["B_down"][:, e]              # [K, H, r]
            # z_down and gz_down via fused forward kernel (reuse for grad path)
            z_down = fused_lora_fwd(act, A_sel, torch.eye(
                RANK, device=act.device, dtype=dtype).unsqueeze(0).expand(K, -1, -1),
                aids, 1.0, K)
            # Simpler: compute z/gz via bmm (these are small, rank-sized)
            A_tok = A_sel[aids]                           # [n, r, MOE_INT]
            B_tok = B_sel[aids]                           # [n, H, r]
            z_down = torch.bmm(A_tok, act.unsqueeze(2)).squeeze(2)
            gz_down = torch.bmm(B_tok.transpose(1, 2),
                                grad_out_e.unsqueeze(2)).squeeze(2) * s

            # Fused gradient scatter (no intermediate materialization)
            grad_A = fused_grad_scatter(gz_down, act, aids, K, RANK, MOE_INT)
            grad_B = fused_grad_scatter(grad_out_e, z_down * s, aids, K, H, RANK)
            for k in range(K):
                per_adapter_grads[k]["A_down"][e] = grad_A[k]
                per_adapter_grads[k]["B_down"][e] = grad_B[k]

            grad_act = grad_act + torch.bmm(
                gz_down.unsqueeze(1), A_tok).squeeze(1)

        sig = torch.sigmoid(gate)
        silu_deriv = sig * (1.0 + gate * (1.0 - sig))
        grad_gate = grad_act * silu_deriv * up
        grad_up   = grad_act * F.silu(gate)
        grad_gate_up = torch.cat([grad_gate, grad_up], dim=-1)
        grad_x2[idx] += grad_gate_up @ w13[e]

        if has_moe_lora and e < n_lora:
            if "gate" in targets:
                A_tok = stacked["A_gate"][aids, e]
                B_tok = stacked["B_gate"][aids, e]
                z_gate = torch.bmm(A_tok, tokens.unsqueeze(2)).squeeze(2)
                gz_gate = torch.bmm(B_tok.transpose(1, 2),
                                    grad_gate.unsqueeze(2)).squeeze(2) * s
                grad_A = fused_grad_scatter(gz_gate, tokens, aids, K, RANK, H)
                grad_B = fused_grad_scatter(grad_gate, z_gate * s, aids, K, MOE_INT, RANK)
                for k in range(K):
                    per_adapter_grads[k]["A_gate"][e] = grad_A[k]
                    per_adapter_grads[k]["B_gate"][e] = grad_B[k]
                grad_x2[idx] += torch.bmm(
                    gz_gate.unsqueeze(1), A_tok).squeeze(1)

            if "up" in targets:
                A_tok = stacked["A_up"][aids, e]
                B_tok = stacked["B_up"][aids, e]
                z_up = torch.bmm(A_tok, tokens.unsqueeze(2)).squeeze(2)
                gz_up = torch.bmm(B_tok.transpose(1, 2),
                                  grad_up.unsqueeze(2)).squeeze(2) * s
                grad_A = fused_grad_scatter(gz_up, tokens, aids, K, RANK, H)
                grad_B = fused_grad_scatter(grad_up, z_up * s, aids, K, MOE_INT, RANK)
                for k in range(K):
                    per_adapter_grads[k]["A_up"][e] = grad_A[k]
                    per_adapter_grads[k]["B_up"][e] = grad_B[k]
                grad_x2[idx] += torch.bmm(
                    gz_up.unsqueeze(1), A_tok).squeeze(1)

    return grad_x2, per_adapter_grads


# ══════════════════════════════════════════════════════════════════════════
# Multi-adapter attention helpers
# ══════════════════════════════════════════════════════════════════════════

def attention_forward(tokens: torch.Tensor, W: dict, L: dict,
                      rope_freqs: torch.Tensor, cfg):
    """Attention forward for one adapter's batch. Returns (x2, attn_cache)."""
    s = SCALING
    has_attn_lora = cfg.mode in ("both", "attn_only")

    hidden   = W["embed"][tokens]
    residual = hidden.detach().clone()
    res_a    = (hidden + residual).detach()
    x_norm   = rms_norm(res_a, W["w_ln_in"])
    xd       = x_norm.detach()

    q_base = xd @ W["W_q"].T
    k_base = xd @ W["W_k"].T
    v_base = xd @ W["W_v"].T

    if has_attn_lora:
        q = q_base + (xd @ L["A_q"].T) @ L["B_q"].T * s
        v = v_base + (xd @ L["A_v"].T) @ L["B_v"].T * s
    else:
        q, v = q_base, v_base
    k = k_base
    q_pre_qknorm = q.detach().clone()

    Tl = tokens.shape[0]
    q_normed = rms_norm(q.view(Tl, N_HEADS, HEAD_DIM), W["w_qnorm"]).view(Tl, Q_SZ)
    k_normed = rms_norm(k.view(Tl, N_KV,    HEAD_DIM), W["w_knorm"]).view(Tl, KV_SZ)
    q_rot = rope_fwd(q_normed.view(Tl, N_HEADS, HEAD_DIM), rope_freqs).view(Tl, Q_SZ)
    k_rot = rope_fwd(k_normed.view(Tl, N_KV,    HEAD_DIM), rope_freqs).view(Tl, KV_SZ)
    sdpa_out = causal_sdpa(q_rot, k_rot, v)

    so     = sdpa_out.detach()
    o_base = so @ W["W_o"].T
    if has_attn_lora:
        o_local = o_base + (so @ L["A_o"].T) @ L["B_o"].T * s
    else:
        o_local = o_base

    res_b = o_local + residual
    x2    = rms_norm(res_b, W["w_ln_post"])

    cache = {"x_norm": x_norm, "q_pre_qknorm": q_pre_qknorm,
             "k_rot": k_rot, "q_rot": q_rot, "v": v,
             "sdpa_out": sdpa_out, "res_b": res_b}
    return x2, cache


def attention_backward(grad_x2: torch.Tensor, W: dict, L: dict,
                       attn_cache: dict, rope_freqs: torch.Tensor,
                       cfg) -> dict:
    """Attention backward for one adapter. Returns attn_grads dict."""
    s = SCALING
    has_attn_lora = cfg.mode in ("both", "attn_only")
    xn  = attn_cache["x_norm"]
    qpn = attn_cache["q_pre_qknorm"]
    kr  = attn_cache["k_rot"]
    qr  = attn_cache["q_rot"]
    v_  = attn_cache["v"]
    so  = attn_cache["sdpa_out"].detach()
    rb  = attn_cache["res_b"]
    Tl  = xn.shape[0]

    grad_res_b, _ = rms_norm_bwd(grad_x2, rb, W["w_ln_post"])
    grad_o_local  = grad_res_b

    attn_grads = {}
    if has_attn_lora:
        z_o  = so.float() @ L["A_o"].T.float()
        gz_o = grad_o_local.float() @ L["B_o"].float() * s
        attn_grads["A_o"] = (gz_o.T @ so.float()).to(L["A_o"].dtype)
        attn_grads["B_o"] = (grad_o_local.float().T @ z_o * s).to(L["B_o"].dtype)
        grad_sdpa = (grad_o_local.float() @ W["W_o"].float()
                     + gz_o @ L["A_o"].float()).to(qr.dtype)
    else:
        grad_sdpa = (grad_o_local.float() @ W["W_o"].float()).to(qr.dtype)

    grad_q_rot, grad_v = sdpa_bwd(qr, kr, v_, grad_sdpa)
    grad_q_normed = rope_bwd(
        grad_q_rot.view(Tl, N_HEADS, HEAD_DIM), rope_freqs).view(Tl, Q_SZ)
    grad_q = qknorm_bwd(qpn, grad_q_normed, W["w_qnorm"])

    if has_attn_lora:
        xd = xn.detach().float()
        z_q  = xd @ L["A_q"].T.float()
        gz_q = grad_q.float() @ L["B_q"].float() * s
        attn_grads["A_q"] = (gz_q.T @ xd).to(L["A_q"].dtype)
        attn_grads["B_q"] = (grad_q.float().T @ z_q * s).to(L["B_q"].dtype)
        z_v  = xd @ L["A_v"].T.float()
        gz_v = grad_v.float() @ L["B_v"].float() * s
        attn_grads["A_v"] = (gz_v.T @ xd).to(L["A_v"].dtype)
        attn_grads["B_v"] = (grad_v.float().T @ z_v * s).to(L["B_v"].dtype)

    return attn_grads


# ══════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════

def summarize_timing(name: str, samples_ms: list[float]) -> str:
    if not samples_ms:
        return f"  {name:<12}  (no data)"
    mean = statistics.mean(samples_ms)
    p50  = statistics.median(samples_ms)
    sd   = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    p99  = sorted(samples_ms)[max(0, int(len(samples_ms) * 0.99) - 1)]
    return f"  {name:<12}  mean={mean:8.3f}ms  p50={p50:8.3f}ms  p99={p99:8.3f}ms  sd={sd:7.3f}ms"


# ══════════════════════════════════════════════════════════════════════════
# Multi-adapter training loop
# ══════════════════════════════════════════════════════════════════════════

def run_multi_adapter(cfg, W, adapters_attn, adapters_moe, rope_freqs, seqs):
    """Train K adapters per step.  Compares naive vs batched (Punica-style)."""
    K       = cfg.num_adapters
    dev     = cfg.device
    dtype   = cfg.torch_dtype
    strategy = cfg.multi_adapter
    dev_idx = int(dev.split(":")[-1]) if ":" in dev else 0

    all_params = []
    for k in range(K):
        all_params.extend(adapters_attn[k].values())
        all_params.extend(adapters_moe[k].values())
    optimizer = torch.optim.AdamW(all_params, lr=cfg.lr, weight_decay=0.01)

    n_attn = sum(sum(p.numel() for p in adapters_attn[k].values()) for k in range(K))
    n_moe  = sum(sum(p.numel() for p in adapters_moe[k].values()) for k in range(K))

    print(f"\n=== Multi-adapter: K={K}  strategy={strategy} ===")
    print(f"  total attn LoRA: {n_attn:,}  ({n_attn // K:,} × {K})")
    print(f"  total MoE  LoRA: {n_moe:,}  ({n_moe // K:,} × {K})")
    lora_bytes = sum(p.numel() * p.element_size() for p in all_params)
    print(f"  total LoRA memory: {lora_bytes / 1e6:.2f} MB")

    torch.cuda.reset_peak_memory_stats(dev_idx)
    total_steps = cfg.warmup + cfg.steps
    fwd_ms, bwd_ms, opt_ms, step_ms = [], [], [], []

    print(f"\n{'step':>6}  {'loss':>8}  {'fwd_ms':>8}  {'bwd_ms':>8}  {'opt_ms':>8}")
    print("-" * 52)

    for step in range(total_steps):
        # Each adapter gets a different sequence
        batch_per_adapter = [seqs[(step * K + k) % seqs.shape[0]] for k in range(K)]

        ef  = torch.cuda.Event(enable_timing=True)
        ef2 = torch.cuda.Event(enable_timing=True)
        eb  = torch.cuda.Event(enable_timing=True)
        eb2 = torch.cuda.Event(enable_timing=True)
        eo  = torch.cuda.Event(enable_timing=True)
        eo2 = torch.cuda.Event(enable_timing=True)

        if strategy == "naive":
            # ── Naive: loop over K adapters sequentially ──────────
            ef.record()
            caches = []
            losses = []
            for k in range(K):
                h, cache = forward_one_layer(batch_per_adapter[k], W,
                                             adapters_attn[k], adapters_moe[k],
                                             rope_freqs, cfg)
                grad_h, loss_k = lm_head_grad(h, W["W_lm"], batch_per_adapter[k])
                cache["grad_h"] = grad_h
                caches.append(cache)
                losses.append(loss_k)
            ef2.record()

            eb.record()
            all_attn_grads = []
            all_moe_grads = []
            for k in range(K):
                ag, mg = backward_one_layer(caches[k]["grad_h"], W,
                                            adapters_attn[k], adapters_moe[k],
                                            caches[k], rope_freqs, cfg)
                all_attn_grads.append(ag)
                all_moe_grads.append(mg)
            eb2.record()

        else:
            # ── Batched / Fused: expert-major ─────────────────────
            moe_fwd_fn = moe_forward_fused if strategy == "fused" else moe_forward_batched
            moe_bwd_fn = moe_backward_fused if strategy == "fused" else moe_backward_batched

            ef.record()
            x2_list = []
            attn_caches = []
            for k in range(K):
                x2_k, ac_k = attention_forward(batch_per_adapter[k], W,
                                               adapters_attn[k], rope_freqs, cfg)
                x2_list.append(x2_k)
                attn_caches.append(ac_k)

            x2_cat = torch.cat(x2_list, dim=0)
            adapter_ids = torch.arange(K, device=dev).repeat_interleave(T)
            moe_out_cat, moe_cache = moe_fwd_fn(
                x2_cat, W, adapters_moe, adapter_ids, cfg)

            losses = []
            grad_moe_list = []
            for k in range(K):
                sl = slice(k * T, (k + 1) * T)
                h_normed_k = rms_norm(moe_out_cat[sl], W["w_ln_final"])
                grad_h_k, loss_k = lm_head_grad(h_normed_k, W["W_lm"],
                                                 batch_per_adapter[k])
                grad_ho_k, _ = rms_norm_bwd(grad_h_k, moe_out_cat[sl],
                                            W["w_ln_final"])
                grad_moe_list.append(grad_ho_k)
                losses.append(loss_k)
            grad_moe_cat = torch.cat(grad_moe_list, dim=0)
            ef2.record()

            eb.record()
            grad_x2_cat, per_adapter_moe_grads = moe_bwd_fn(
                grad_moe_cat, W, adapters_moe, moe_cache, cfg)

            all_attn_grads = []
            all_moe_grads = per_adapter_moe_grads
            for k in range(K):
                sl = slice(k * T, (k + 1) * T)
                ag_k = attention_backward(grad_x2_cat[sl], W,
                                          adapters_attn[k], attn_caches[k],
                                          rope_freqs, cfg)
                all_attn_grads.append(ag_k)
            eb2.record()

        # ── Optimizer step (same for both strategies) ──────────
        eo.record()
        optimizer.zero_grad(set_to_none=True)
        for k in range(K):
            for name, param in adapters_attn[k].items():
                if name in all_attn_grads[k]:
                    param.grad = all_attn_grads[k][name]
            for name, param in adapters_moe[k].items():
                if name in all_moe_grads[k]:
                    param.grad = all_moe_grads[k][name]
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        eo2.record()
        torch.cuda.synchronize()

        f = ef.elapsed_time(ef2)
        b = eb.elapsed_time(eb2)
        o = eo.elapsed_time(eo2)
        loss_avg = sum(losses) / K

        if step >= cfg.warmup:
            fwd_ms.append(f)
            bwd_ms.append(b)
            opt_ms.append(o)
            step_ms.append(f + b + o)

        if step % 20 == 0 or step == total_steps - 1:
            tag = "W" if step < cfg.warmup else " "
            print(f"{step:>5}{tag} {loss_avg:>8.4f}  {f:>8.2f}  {b:>8.2f}  {o:>8.2f}")

    peak_mem = torch.cuda.max_memory_allocated(dev_idx)
    print("\n" + "=" * 60)
    print(f"=== Multi-adapter timing ({strategy}, K={K}, {cfg.steps} steps) ===")
    print(summarize_timing("forward", fwd_ms))
    print(summarize_timing("backward", bwd_ms))
    print(summarize_timing("optimizer", opt_ms))
    print(summarize_timing("total", step_ms))
    if step_ms:
        mean_step = statistics.mean(step_ms)
        tok_per_s = (T * K) / (mean_step / 1000.0)
        print(f"\n  throughput: {tok_per_s:.1f} tok/s  ({T}×{K} tokens / {mean_step:.2f} ms)")
    print(f"\n  peak GPU: {peak_mem / 1e9:.3f} GB")


# ══════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════

def main():
    cfg = parse_args()
    apply_cfg(cfg)
    torch.manual_seed(42)
    dev   = cfg.device
    dtype = cfg.torch_dtype

    W = init_weights(dev, dtype)
    rope_freqs = build_rope_freqs(T, HEAD_DIM, device=dev)
    N_SEQS = max(8, cfg.num_adapters * 2)
    seqs   = torch.randint(0, VOCAB, (N_SEQS, T), device=dev)

    print(f"=== qwen3_moe_lora_microbench — Qwen3-30B-A3B single layer ===")
    print(f"  mode={cfg.mode}  dtype={cfg.dtype}  device={dev}")
    print(f"  H={H}  N_HEADS={N_HEADS}  N_KV={N_KV}  HEAD_DIM={HEAD_DIM}")
    print(f"  num_experts={N_EXPERTS}  top_k={TOP_K}  moe_intermediate={MOE_INT}")
    print(f"  rank={RANK}  scaling={SCALING}")
    print(f"  seq_len={T}  vocab={VOCAB}  lr={cfg.lr}")
    print(f"  moe_lora_targets={cfg.moe_lora_targets}  n_lora_experts={cfg.n_lora_experts}/{N_EXPERTS}")
    base_bytes = sum(v.numel() * v.element_size() for v in W.values())
    print(f"  base weights: {base_bytes / 1e9:.3f} GB")

    if cfg.num_adapters > 1:
        # ── Multi-adapter path ────────────────────────────────────
        adapters_attn = []
        adapters_moe  = []
        for k in range(cfg.num_adapters):
            torch.manual_seed(42 + k)
            a = init_attn_lora(dev, dtype) if cfg.mode in ("both", "attn_only") else {}
            m = init_moe_lora(dev, dtype, cfg) if cfg.mode in ("both", "moe_only") else {}
            adapters_attn.append(a)
            adapters_moe.append(m)
        run_multi_adapter(cfg, W, adapters_attn, adapters_moe, rope_freqs, seqs)
        return

    # ── Single-adapter path (original) ────────────────────────────
    L_attn = init_attn_lora(dev, dtype) if cfg.mode in ("both", "attn_only") else {}
    L_moe  = init_moe_lora(dev, dtype, cfg) if cfg.mode in ("both", "moe_only") else {}

    all_params = list(L_attn.values()) + list(L_moe.values())
    if not all_params:
        print("ERROR: no trainable params (check --mode)")
        return
    optimizer  = torch.optim.AdamW(all_params, lr=cfg.lr, weight_decay=0.01)

    n_attn = sum(p.numel() for p in L_attn.values())
    n_moe  = sum(p.numel() for p in L_moe.values())
    print(f"  num_adapters=1")
    print(f"  attn LoRA params: {n_attn:,}")
    print(f"  MoE  LoRA params: {n_moe:,}")
    print(f"  total trainable:  {n_attn + n_moe:,}")
    print(f"  steps={cfg.steps}  warmup={cfg.warmup}")

    lora_bytes = sum(p.numel() * p.element_size() for p in all_params)
    print(f"  LoRA params:   {lora_bytes / 1e6:.2f} MB")

    dev_idx = int(dev.split(":")[-1]) if ":" in dev else 0
    torch.cuda.reset_peak_memory_stats(dev_idx)

    fwd_ms, bwd_ms, opt_ms, step_ms = [], [], [], []
    total_steps = cfg.warmup + cfg.steps

    print(f"\n{'step':>6}  {'loss':>8}  {'fwd_ms':>8}  {'bwd_ms':>8}  {'opt_ms':>8}")
    print("-" * 52)

    for step in range(total_steps):
        tokens = seqs[step % N_SEQS]

        ef = torch.cuda.Event(enable_timing=True)
        ef2 = torch.cuda.Event(enable_timing=True)
        eb = torch.cuda.Event(enable_timing=True)
        eb2 = torch.cuda.Event(enable_timing=True)
        eo = torch.cuda.Event(enable_timing=True)
        eo2 = torch.cuda.Event(enable_timing=True)

        ef.record()
        h_normed, cache = forward_one_layer(tokens, W, L_attn, L_moe,
                                            rope_freqs, cfg)
        grad_h_normed, loss = lm_head_grad(h_normed, W["W_lm"], tokens)
        ef2.record()

        eb.record()
        attn_grads, moe_grads = backward_one_layer(
            grad_h_normed, W, L_attn, L_moe, cache, rope_freqs, cfg)
        eb2.record()

        eo.record()
        optimizer.zero_grad(set_to_none=True)
        for name, param in L_attn.items():
            if name in attn_grads:
                param.grad = attn_grads[name]
        for name, param in L_moe.items():
            if name in moe_grads:
                param.grad = moe_grads[name]
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        eo2.record()

        torch.cuda.synchronize()

        f = ef.elapsed_time(ef2)
        b = eb.elapsed_time(eb2)
        o = eo.elapsed_time(eo2)

        if step >= cfg.warmup:
            fwd_ms.append(f)
            bwd_ms.append(b)
            opt_ms.append(o)
            step_ms.append(f + b + o)

        if step % 20 == 0 or step == total_steps - 1:
            tag = "W" if step < cfg.warmup else " "
            print(f"{step:>5}{tag} {loss:>8.4f}  {f:>8.2f}  {b:>8.2f}  {o:>8.2f}")

    peak_mem = torch.cuda.max_memory_allocated(dev_idx)
    print("\n" + "=" * 60)
    print(f"=== Timing summary ({cfg.steps} timed steps) ===")
    print(summarize_timing("forward", fwd_ms))
    print(summarize_timing("backward", bwd_ms))
    print(summarize_timing("optimizer", opt_ms))
    print(summarize_timing("total", step_ms))

    if step_ms:
        mean_step = statistics.mean(step_ms)
        tok_per_s = T / (mean_step / 1000.0)
        print(f"\n  throughput: {tok_per_s:.1f} tok/s  ({T} tokens / {mean_step:.2f} ms)")

    print(f"\n=== Parameters ===")
    print(f"  attn LoRA:  {n_attn:>12,}")
    print(f"  MoE  LoRA:  {n_moe:>12,}")
    print(f"  total:      {n_attn + n_moe:>12,}")

    print(f"\n=== Memory ===")
    print(f"  base weights:  {base_bytes / 1e9:.3f} GB")
    print(f"  LoRA params:   {lora_bytes / 1e6:.2f} MB")
    print(f"  peak GPU:      {peak_mem / 1e9:.3f} GB")

    if cache and "moe_cache" in cache:
        n_active = len(cache["moe_cache"]["experts"])
        avg_tok = statistics.mean(
            ec["tokens"].shape[0]
            for ec in cache["moe_cache"]["experts"].values()
        ) if n_active > 0 else 0
        print(f"\n=== MoE routing (last step) ===")
        print(f"  active experts: {n_active}/{N_EXPERTS}")
        print(f"  avg tokens/expert: {avg_tok:.1f}")
        print(f"  expected: {T}×{TOP_K}/{N_EXPERTS} = {T * TOP_K / N_EXPERTS:.1f}")


if __name__ == "__main__":
    main()
