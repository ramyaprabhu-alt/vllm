"""
bt_lora_layer_bench_real.py — Same manual forward+backward as bt_lora_layer_bench.py
but loaded with REAL Qwen3-30B-A3B layer-0 weights and the real LoRA adapter.

No vLLM, no NCCL, no loss.backward().

Expected result: CE loss decreases from its initial value (~2-4, since the base model
is pre-trained) as LoRA overfits to the fixed training sequences.

Run:
    .venv/bin/python bt_lora_layer_bench_real.py
"""

from __future__ import annotations
import json
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_DIR  = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
ADAPTER    = ("/mnt/nfs/home/ramya/slora-plus/S-LoRA/test/qwen3/adapters"
              "/qwen3-toy-lora/adapter_model.safetensors")
SHARD_1    = f"{MODEL_DIR}/model-00001-of-00016.safetensors"
SHARD_16   = f"{MODEL_DIR}/model-00016-of-00016.safetensors"

# ── Qwen3-30B-A3B dimensions ────────────────────────────────────────────────
H        = 2048
N_HEADS  = 32
N_KV     = 4
HEAD_DIM = 128
Q_SZ     = N_HEADS * HEAD_DIM    # 4096
KV_SZ    = N_KV    * HEAD_DIM    # 512
RANK     = 16
SCALING  = 1.0
VOCAB    = 151936

T        = 128    # sequence length
STEPS    = 300
LR       = 2e-4
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════
# Load weights — only the keys we need (avoids pulling 600 MB of MoE experts)
# ══════════════════════════════════════════════════════════════════════════

def load_weights() -> dict:
    """Load layer-0 attention + norm weights, embedding, lm_head."""
    dev = DEVICE
    print("Loading model weights (layer 0 attention + norm + embed + lm_head)...")

    W = {}

    # ── Shard 1: layer-0 attention, norms, embedding ──────────────────────
    needed_s1 = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.embed_tokens.weight",
    ]
    with safe_open(SHARD_1, framework="pt", device=dev) as f:
        for key in needed_s1:
            W[key] = f.get_tensor(key).float()

    # ── Shard 16: lm_head, final norm ────────────────────────────────────
    needed_s16 = ["lm_head.weight", "model.norm.weight"]
    with safe_open(SHARD_16, framework="pt", device=dev) as f:
        for key in needed_s16:
            W[key] = f.get_tensor(key).float()

    print(f"  q_proj: {W['model.layers.0.self_attn.q_proj.weight'].shape}")
    print(f"  k_proj: {W['model.layers.0.self_attn.k_proj.weight'].shape}")
    print(f"  v_proj: {W['model.layers.0.self_attn.v_proj.weight'].shape}")
    print(f"  o_proj: {W['model.layers.0.self_attn.o_proj.weight'].shape}")
    print(f"  embed:  {W['model.embed_tokens.weight'].shape}")
    print(f"  lm_head:{W['lm_head.weight'].shape}")
    print("  done.\n")

    return {
        "W_q":        W["model.layers.0.self_attn.q_proj.weight"],
        "W_k":        W["model.layers.0.self_attn.k_proj.weight"],
        "W_v":        W["model.layers.0.self_attn.v_proj.weight"],
        "W_o":        W["model.layers.0.self_attn.o_proj.weight"],
        "w_qnorm":    W["model.layers.0.self_attn.q_norm.weight"],
        "w_knorm":    W["model.layers.0.self_attn.k_norm.weight"],
        "w_ln_in":    W["model.layers.0.input_layernorm.weight"],
        "w_ln_post":  W["model.layers.0.post_attention_layernorm.weight"],
        "w_ln_final": W["model.norm.weight"],
        "embed":      W["model.embed_tokens.weight"],
        "W_lm":       W["lm_head.weight"],
    }


def load_lora() -> dict:
    """Load LoRA adapter weights for layer 0, TP=1 (unshard)."""
    print(f"Loading LoRA adapter from {ADAPTER} ...")
    raw = load_file(ADAPTER, device="cpu")

    # The adapter has 48 layers × 4 projs × 2 (A+B) = 384 keys.
    # Key format: base_model.model.model.layers.{i}.self_attn.{proj}.lora_{A/B}.weight
    # For TP=1 we use the full (unsharded) weights.
    def get(proj, ab):
        key = f"base_model.model.model.layers.0.self_attn.{proj}.lora_{ab}.weight"
        return raw[key].float().to(DEVICE).requires_grad_(True)

    L = {
        "A_q": get("q_proj", "A"),   "B_q": get("q_proj", "B"),
        "A_v": get("v_proj", "A"),   "B_v": get("v_proj", "B"),
        "A_o": get("o_proj", "A"),   "B_o": get("o_proj", "B"),
    }
    print(f"  A_q: {L['A_q'].shape}  B_q: {L['B_q'].shape}")
    print(f"  A_v: {L['A_v'].shape}  B_v: {L['B_v'].shape}")
    print(f"  A_o: {L['A_o'].shape}  B_o: {L['B_o'].shape}")
    print("  done.\n")
    return L


# ══════════════════════════════════════════════════════════════════════════
# All primitive ops — identical to bt_lora_layer_bench.py
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
    T, nh, hd = x.shape
    xc = torch.view_as_complex(x.float().reshape(T, nh, hd // 2, 2))
    return torch.view_as_real(xc * fc[:T].unsqueeze(1)).flatten(3).to(x.dtype)


def rope_bwd(grad: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
    T, nh, hd = grad.shape
    gc = torch.view_as_complex(grad.float().reshape(T, nh, hd // 2, 2))
    return torch.view_as_real(gc * fc[:T].conj().unsqueeze(1)).flatten(3).to(grad.dtype)


def causal_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    T = q.shape[0]
    q3 = q.view(T, N_HEADS, HEAD_DIM).transpose(0, 1)
    k3 = k.view(T, N_KV,    HEAD_DIM).transpose(0, 1)
    v3 = v.view(T, N_KV,    HEAD_DIM).transpose(0, 1)
    if N_KV < N_HEADS:
        rep = N_HEADS // N_KV
        k3  = k3.repeat_interleave(rep, 0)
        v3  = v3.repeat_interleave(rep, 0)
    return F.scaled_dot_product_attention(q3, k3, v3, is_causal=True) \
             .transpose(0, 1).reshape(T, Q_SZ)


def sdpa_bwd(q_rot, k_rot, v, grad_out):
    T = q_rot.shape[0]
    with torch.enable_grad():
        ql = q_rot.detach().requires_grad_(True)
        vl = v.detach().requires_grad_(True)
        q3 = ql.view(T, N_HEADS, HEAD_DIM).transpose(0, 1)
        k3 = k_rot.detach().view(T, N_KV, HEAD_DIM).transpose(0, 1)
        v3 = vl.view(T, N_KV,    HEAD_DIM).transpose(0, 1)
        if N_KV < N_HEADS:
            rep = N_HEADS // N_KV
            k3  = k3.repeat_interleave(rep, 0)
            v3  = v3.repeat_interleave(rep, 0)
        out = F.scaled_dot_product_attention(q3, k3, v3, is_causal=True) \
                .transpose(0, 1).reshape(T, Q_SZ)
        out.backward(grad_out.to(out.dtype))
    return ql.grad, vl.grad


def qknorm_bwd(q_pre, grad_q_normed, w_qnorm):
    T = q_pre.shape[0]
    dx, _ = rms_norm_bwd(grad_q_normed.view(T * N_HEADS, HEAD_DIM),
                          q_pre.view(T * N_HEADS, HEAD_DIM), w_qnorm)
    return dx.view(T, Q_SZ)


def lm_head_grad(h_normed, W_lm, tokens):
    """Full-vocab CE loss (no TP sharding here — no OOB issue)."""
    logits     = h_normed.float() @ W_lm.T          # [T, VOCAB]
    sh_logits  = logits[:-1]                          # [T-1, VOCAB]
    sh_labels  = tokens[1:].long()                    # [T-1]

    loss_scalar = F.cross_entropy(sh_logits, sh_labels).item()

    probs   = torch.softmax(sh_logits, dim=-1)
    n_valid = sh_labels.shape[0]
    probs[torch.arange(n_valid), sh_labels] -= 1.0
    probs  /= n_valid

    grad_h             = torch.zeros_like(h_normed)
    grad_h[:-1]        = (probs.to(h_normed.dtype) @ W_lm)
    return grad_h, loss_scalar


# ══════════════════════════════════════════════════════════════════════════
# Forward / Backward — identical logic to bt_lora_layer_bench.py
# ══════════════════════════════════════════════════════════════════════════

def forward_one_layer(tokens, W, L, rope_freqs):
    s = SCALING

    hidden   = W["embed"][tokens].float().detach()
    residual = hidden.clone()

    res_a  = (hidden + residual).detach()
    x_norm = rms_norm(res_a, W["w_ln_in"])

    xd     = x_norm.detach()
    q_base = xd @ W["W_q"].T
    k_base = xd @ W["W_k"].T
    v_base = xd @ W["W_v"].T

    q = q_base + (xd @ L["A_q"].T) @ L["B_q"].T * s
    v = v_base + (xd @ L["A_v"].T) @ L["B_v"].T * s
    k = k_base

    q_pre_qknorm = q.detach().clone()

    q_normed = rms_norm(q.view(T, N_HEADS, HEAD_DIM), W["w_qnorm"]).view(T, Q_SZ)
    k_normed = rms_norm(k.view(T, N_KV,    HEAD_DIM), W["w_knorm"]).view(T, KV_SZ)

    q_rot = rope_fwd(q_normed.view(T, N_HEADS, HEAD_DIM), rope_freqs).view(T, Q_SZ)
    k_rot = rope_fwd(k_normed.view(T, N_KV,    HEAD_DIM), rope_freqs).view(T, KV_SZ)

    sdpa_out = causal_sdpa(q_rot, k_rot, v)

    so      = sdpa_out.detach()
    o_base  = so @ W["W_o"].T
    o_local = o_base + (so @ L["A_o"].T) @ L["B_o"].T * s

    res_b      = o_local + residual
    x2         = rms_norm(res_b, W["w_ln_post"])
    hidden_out = x2                              # MoE = identity

    h_normed = rms_norm(hidden_out, W["w_ln_final"])

    cache = dict(x_norm=x_norm, res_a=res_a,
                 q_pre_qknorm=q_pre_qknorm,
                 q_normed=q_normed, k_rot=k_rot, q_rot=q_rot, v=v,
                 sdpa_out=sdpa_out, o_local=o_local,
                 res_b=res_b, hidden_out=hidden_out, residual=residual)
    return h_normed, cache


def backward_one_layer(grad_h_normed, W, L, cache, rope_freqs):
    s   = SCALING
    xn  = cache["x_norm"]
    qpn = cache["q_pre_qknorm"]
    kr  = cache["k_rot"]
    qr  = cache["q_rot"]
    v_  = cache["v"]
    so  = cache["sdpa_out"].detach()
    rb  = cache["res_b"]
    ho  = cache["hidden_out"]

    grad_ho, _    = rms_norm_bwd(grad_h_normed, ho,  W["w_ln_final"])
    grad_res_b, _ = rms_norm_bwd(grad_ho,       rb,  W["w_ln_post"])
    grad_o_local  = grad_res_b

    z_o   = so.float()           @ L["A_o"].T.float()
    gz_o  = grad_o_local.float() @ L["B_o"].float() * s
    dA_o  = (gz_o.T @ so.float()).to(L["A_o"].dtype)
    dB_o  = (grad_o_local.float().T @ z_o * s).to(L["B_o"].dtype)

    grad_sdpa = (grad_o_local.float() @ W["W_o"].float()
                 + gz_o @ L["A_o"].float()).to(qr.dtype)

    grad_q_rot, grad_v = sdpa_bwd(qr, kr, v_, grad_sdpa)

    grad_q_normed = rope_bwd(grad_q_rot.view(T, N_HEADS, HEAD_DIM),
                              rope_freqs).view(T, Q_SZ)
    grad_q        = qknorm_bwd(qpn, grad_q_normed, W["w_qnorm"])

    xd   = xn.detach().float()
    z_q  = xd @ L["A_q"].T.float()
    gz_q = grad_q.float() @ L["B_q"].float() * s
    dA_q = (gz_q.T @ xd).to(L["A_q"].dtype)
    dB_q = (grad_q.float().T @ z_q * s).to(L["B_q"].dtype)

    z_v  = xd @ L["A_v"].T.float()
    gz_v = grad_v.float() @ L["B_v"].float() * s
    dA_v = (gz_v.T @ xd).to(L["A_v"].dtype)
    dB_v = (grad_v.float().T @ z_v * s).to(L["B_v"].dtype)

    return {"A_q": dA_q, "B_q": dB_q,
            "A_v": dA_v, "B_v": dB_v,
            "A_o": dA_o, "B_o": dB_o}


# ══════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    dev = DEVICE

    W          = load_weights()
    L          = load_lora()
    params     = list(L.values())
    optimizer  = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    rope_freqs = build_rope_freqs(T, HEAD_DIM, device=dev)

    # Fixed training sequences — 8 random token IDs from the real vocabulary
    N_SEQS = 8
    seqs   = torch.randint(0, VOCAB, (N_SEQS, T), device=dev)

    n_params = sum(p.numel() for p in params)
    print(f"bt_lora_layer_bench_real — Qwen3-30B-A3B layer 0 (real weights)")
    print(f"  H={H}  N_HEADS={N_HEADS}  N_KV={N_KV}  VOCAB={VOCAB}")
    print(f"  LoRA rank={RANK}  trainable params={n_params:,}")
    print(f"  device={dev}  steps={STEPS}  lr={LR}\n")
    print(f"{'step':>6}  {'loss':>8}  {'note'}")
    print("-" * 40)

    for step in range(STEPS):
        tokens = seqs[step % N_SEQS]

        h_normed, cache = forward_one_layer(tokens, W, L, rope_freqs)
        grad_h_normed, loss = lm_head_grad(h_normed, W["W_lm"], tokens)
        grads = backward_one_layer(grad_h_normed, W, L, cache, rope_freqs)

        optimizer.zero_grad(set_to_none=True)
        for name, param in L.items():
            param.grad = grads[name]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        if step % 10 == 0:
            note = "← start" if step == 0 else ""
            print(f"{step:>6}  {loss:>8.4f}  {note}")

    print("-" * 40)
    print("Done. Loss should have decreased — real pre-trained weights means")
    print("initial loss is ~2-4 (better than random) and LoRA overfits downward.")


if __name__ == "__main__":
    main()
