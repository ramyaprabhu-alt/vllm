"""
bench_subops.py — Micro-benchmark for BubbleTea C+D backward sub-op kernels.

Measures GPU execution time for the dominant operations in each sub-op type
using synthetic BF16 tensors matching Qwen3-30B-A3B dimensions.  No model
loading required — only the matmul shapes matter for memory-bandwidth timing.

Sub-op types benchmarked:
  passthrough_chunk  — backward through N experts' MoE FFN (dominant cost)
  lora_bwd           — LoRA A/B backward (q/v/o projections)
  optimizer_step     — AdamW on LoRA parameters

Usage:
    python bench_subops.py [--t-ft 128] [--chunks 8] [--warmup 20] [--iters 200]
"""

import argparse
import time
import torch
import torch.nn.functional as F

# Qwen3-30B-A3B dimensions (TP=2 per-rank values)
H           = 2048   # hidden size
E_INT       = 1408   # expert intermediate (w1/w3 output dim each)
N_EXPERTS   = 128    # total experts (EP=2: 64 per GPU)
N_LOCAL_EXP = 64     # experts per EP rank
N_HEADS_LOC = 16     # Q heads per TP rank (32 total / TP=2)
N_KV_LOC    = 4      # KV heads per TP rank (not split in GQA)
HEAD_DIM    = 128
Q_SZ        = N_HEADS_LOC * HEAD_DIM   # 2048
KV_SZ       = N_KV_LOC   * HEAD_DIM   # 512
LORA_RANK   = 8
LORA_ALPHA  = 16.0

def time_ops(fns: list, warmup: int, iters: int, label: str) -> dict:
    """Run fns() warmup+iters times, time with CUDA events, return stats."""
    # warmup
    for _ in range(warmup):
        for fn in fns:
            fn()
    torch.cuda.synchronize()

    e_start = torch.cuda.Event(enable_timing=True)
    e_end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        e_start.record()
        for fn in fns:
            fn()
        e_end.record()
        torch.cuda.synchronize()
        times.append(e_start.elapsed_time(e_end))

    times.sort()
    n = len(times)
    return {
        "label":  label,
        "p50_ms": times[n // 2],
        "p95_ms": times[int(n * 0.95)],
        "p99_ms": times[int(n * 0.99)],
        "mean_ms": sum(times) / n,
        "min_ms":  times[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-ft",   type=int, default=128)
    parser.add_argument("--chunks", type=int, default=8,
                        help="Number of passthrough chunks (VLLM_FT_BWD_PASSTHROUGH_CHUNKS)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters",  type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    T   = args.t_ft
    N_C = args.chunks
    experts_per_chunk = N_LOCAL_EXP // N_C  # 8 with chunks=8

    print(f"BubbleTea sub-op micro-benchmark")
    print(f"  Qwen3-30B-A3B: H={H}, E_INT={E_INT}, N_local_experts={N_LOCAL_EXP}")
    print(f"  t_ft={T}, chunks={N_C} ({experts_per_chunk} experts/chunk)")
    print(f"  warmup={args.warmup}, iters={args.iters}")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print()

    # ── Pre-allocate tensors ──────────────────────────────────────────────
    # Expert weights per chunk (BF16, on GPU)
    # w13: [2*E_INT, H] — gate+up combined (F.linear(x, w13) = x @ w13.T = [T, 2*E_INT])
    # w2:  [H, E_INT]   — down proj        (F.linear(x, w2)  = x @ w2.T  = [T, H])
    w13 = [torch.randn(2 * E_INT, H, dtype=torch.bfloat16, device=device)
           for _ in range(experts_per_chunk)]
    w2  = [torch.randn(H, E_INT,  dtype=torch.bfloat16, device=device)
           for _ in range(experts_per_chunk)]

    # Activations / gradients
    tokens       = torch.randn(T, H,          dtype=torch.bfloat16, device=device)
    grad_gate_up = torch.randn(T, 2 * E_INT,  dtype=torch.bfloat16, device=device)
    grad_x_acc   = torch.zeros(T, H,          dtype=torch.bfloat16, device=device)

    # LoRA weights (q_proj example: H -> n_heads*head_dim = 4096, rank=8)
    Q_OUT = 4096   # q_proj output (32 heads × 128 dim)
    lora_A_q = torch.randn(LORA_RANK, H,     dtype=torch.bfloat16, device=device)
    lora_B_q = torch.randn(Q_OUT, LORA_RANK, dtype=torch.bfloat16, device=device)
    grad_lora_A = torch.zeros_like(lora_A_q)
    grad_lora_B = torch.zeros_like(lora_B_q)
    inp_q   = torch.randn(T, H,     dtype=torch.bfloat16, device=device)
    grad_q  = torch.randn(T, Q_OUT, dtype=torch.bfloat16, device=device)

    # AdamW state for LoRA params (~4.3M total, use representative subset)
    n_lora_params = LORA_RANK * H + Q_OUT * LORA_RANK  # one q_proj lora A+B
    param_flat = torch.randn(n_lora_params, dtype=torch.float32, device=device)
    grad_flat  = torch.randn(n_lora_params, dtype=torch.float32, device=device)
    m_flat     = torch.zeros(n_lora_params, dtype=torch.float32, device=device)
    v_flat     = torch.zeros(n_lora_params, dtype=torch.float32, device=device)

    results = []

    # ── 1. Passthrough chunk (one chunk = experts_per_chunk experts) ─────
    # Ops per expert: re-forward (gate_up), bwd through w2 input, bwd through w13 input
    def passthrough_chunk():
        for e in range(experts_per_chunk):
            gate_up = F.linear(tokens, w13[e])                  # [T,H]@[H,2E] → [T,2E], reads w13
            gate, up = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up                             # [T, E_INT]
            grad_x_acc.add_(F.linear(act, w2[e]))               # [T,E]@[E,H] → [T,H], reads w2
            grad_x_acc.add_(grad_gate_up @ w13[e])              # [T,2E]@[2E,H] → [T,H], reads w13

    r = time_ops([passthrough_chunk], args.warmup, args.iters,
                 f"passthrough_chunk ({experts_per_chunk} experts, "
                 f"{experts_per_chunk * (2*H*E_INT + H*E_INT) * 2 / 1e6:.0f}MB reads)")
    results.append(r)

    # ── 2. LoRA backward (one projection) ────────────────────────────────
    def lora_bwd():
        # dL/dA = B^T @ dL/dout @ x^T  -> not full backward, just the dominant matmuls
        grad_lora_B.add_(grad_q.t() @ F.linear(inp_q, lora_A_q))  # reads lora_A
        grad_lora_A.add_(lora_B_q.t() @ grad_q.t() @ inp_q)       # reads lora_B

    r = time_ops([lora_bwd], args.warmup, args.iters,
                 f"lora_bwd (q_proj, rank={LORA_RANK}, "
                 f"{(LORA_RANK*H + Q_OUT*LORA_RANK)*2/1e6:.2f}MB reads)")
    results.append(r)

    # ── 3. AdamW optimizer step on LoRA params ────────────────────────────
    lr, b1, b2, eps, wd = 1e-4, 0.9, 0.999, 1e-8, 0.01
    def optimizer_step():
        m_flat.mul_(b1).add_(grad_flat, alpha=1 - b1)
        v_flat.mul_(b2).addcmul_(grad_flat, grad_flat, value=1 - b2)
        param_flat.addcdiv_(m_flat, v_flat.sqrt().add_(eps), value=-lr)

    r = time_ops([optimizer_step], args.warmup, args.iters,
                 f"optimizer_step (AdamW, {n_lora_params/1e6:.2f}M params)")
    results.append(r)

    # ── 4. Full backward cycle: all chunks for one layer ─────────────────
    all_w13 = [torch.randn(2*E_INT, H, dtype=torch.bfloat16, device=device)
               for _ in range(N_LOCAL_EXP)]
    all_w2  = [torch.randn(H, E_INT,  dtype=torch.bfloat16, device=device)
               for _ in range(N_LOCAL_EXP)]

    def full_layer_passthrough():
        for e in range(N_LOCAL_EXP):
            gate_up = F.linear(tokens, all_w13[e])
            gate, up = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up
            grad_x_acc.add_(F.linear(act, all_w2[e]))
            grad_x_acc.add_(grad_gate_up @ all_w13[e])

    r = time_ops([full_layer_passthrough], args.warmup, args.iters,
                 f"full_layer_passthrough (all {N_LOCAL_EXP} experts, "
                 f"{N_LOCAL_EXP*(2*H*E_INT + H*E_INT)*2/1e6:.0f}MB reads)")
    results.append(r)

    # ── 5. O-projection backward (base weight + LoRA) ────────────────────
    # Forward: o_local = sdpa_out @ o_w.T  (sdpa_out=[T,Q_SZ], o_w=[H,Q_SZ])
    # Backward: grad_sdpa = grad_o @ o_w  reads o_w [H, Q_SZ] = 8MB
    o_w     = torch.randn(H, Q_SZ,         dtype=torch.bfloat16, device=device)
    A_o     = torch.randn(LORA_RANK, Q_SZ, dtype=torch.bfloat16, device=device)
    B_o     = torch.randn(H, LORA_RANK,    dtype=torch.bfloat16, device=device)
    grad_o  = torch.randn(T, H,            dtype=torch.bfloat16, device=device)
    sdpa_out = torch.randn(T, Q_SZ,        dtype=torch.bfloat16, device=device)

    def o_proj_bwd():
        grad_sdpa = grad_o @ o_w                        # [T,H]@[H,Q_SZ] → reads 8MB
        grad_z    = grad_o @ B_o                        # [T,H]@[H,r] → reads 32KB
        grad_sdpa = grad_sdpa + grad_z @ A_o            # [T,r]@[r,Q_SZ] → reads 64KB

    r = time_ops([o_proj_bwd], args.warmup, args.iters,
                 f"o_proj_bwd (base+LoRA, {H*Q_SZ*2/1e6:.0f}MB o_w reads)")
    results.append(r)

    # ── 6. QKV forward recompute (needed to recover activations) ─────────
    # bt_lora_trainer re-runs QKV projections inside _attn_qkv_recompute
    # qkv_w: [Q_SZ+KV_SZ+KV_SZ, H] per rank = [3072, 2048] = 12MB
    qkv_w   = torch.randn(Q_SZ + 2*KV_SZ, H, dtype=torch.bfloat16, device=device)
    x_norm  = torch.randn(T, H,            dtype=torch.bfloat16, device=device)
    A_q     = torch.randn(LORA_RANK, H,    dtype=torch.bfloat16, device=device)
    B_q     = torch.randn(Q_SZ, LORA_RANK, dtype=torch.bfloat16, device=device)
    A_v     = torch.randn(LORA_RANK, H,    dtype=torch.bfloat16, device=device)
    B_v     = torch.randn(KV_SZ, LORA_RANK,dtype=torch.bfloat16, device=device)

    def qkv_recompute():
        q = F.linear(x_norm, qkv_w[:Q_SZ])             # reads Q_SZ*H*2 = 8MB
        k = F.linear(x_norm, qkv_w[Q_SZ:Q_SZ+KV_SZ])  # reads KV_SZ*H*2 = 2MB
        v = F.linear(x_norm, qkv_w[Q_SZ+KV_SZ:])      # reads KV_SZ*H*2 = 2MB
        q = q + (x_norm @ A_q.T) @ B_q.T               # LoRA delta
        v = v + (x_norm @ A_v.T) @ B_v.T               # LoRA delta

    r = time_ops([qkv_recompute], args.warmup, args.iters,
                 f"qkv_recompute (base+LoRA, {(Q_SZ+2*KV_SZ)*H*2/1e6:.0f}MB reads)")
    results.append(r)

    # ── 7. SDPA backward (FlashAttn, T=128 causal) ────────────────────────
    # Use N_HEADS_LOC for all — GQA head-count difference doesn't affect timing
    q_leaf = torch.randn(1, N_HEADS_LOC, T, HEAD_DIM,
                         dtype=torch.bfloat16, device=device, requires_grad=True)
    k_det  = torch.randn(1, N_HEADS_LOC, T, HEAD_DIM,
                         dtype=torch.bfloat16, device=device)
    v_leaf = torch.randn(1, N_HEADS_LOC, T, HEAD_DIM,
                         dtype=torch.bfloat16, device=device, requires_grad=True)
    grad_sdpa_bwd = torch.randn(1, N_HEADS_LOC, T, HEAD_DIM,
                                dtype=torch.bfloat16, device=device)
    rep = N_HEADS_LOC // N_KV_LOC  # keep for full_attn_bwd

    def sdpa_bwd():
        out = F.scaled_dot_product_attention(q_leaf, k_det, v_leaf, is_causal=True)
        out.backward(grad_sdpa_bwd)
        q_leaf.grad = None
        v_leaf.grad = None

    r = time_ops([sdpa_bwd], args.warmup, args.iters,
                 f"sdpa_bwd (FlashAttn causal, T={T}, {N_HEADS_LOC}h×{HEAD_DIM}d)")
    results.append(r)

    # ── 8. QKV backward through base weights (grad_x_norm contribution) ──
    # grad_x += grad_q @ qkv_w[:Q_SZ]  reads Q_SZ*H*2 = 8MB
    # grad_x += grad_v @ qkv_w[KV:]    reads KV_SZ*H*2 = 2MB
    grad_q_pn   = torch.randn(T, Q_SZ, dtype=torch.bfloat16, device=device)
    grad_v_bwd  = torch.randn(T, KV_SZ, dtype=torch.bfloat16, device=device)
    grad_x_attn = torch.zeros(T, H,    dtype=torch.bfloat16, device=device)

    def qkv_bwd():
        grad_x_attn.add_(grad_q_pn  @ qkv_w[:Q_SZ])       # reads 8MB
        grad_x_attn.add_(grad_v_bwd @ qkv_w[Q_SZ+KV_SZ:]) # reads 2MB
        # LoRA contributions
        grad_x_attn.add_((grad_q_pn @ B_q) @ A_q)          # reads 32KB + 64KB
        grad_x_attn.add_((grad_v_bwd @ B_v) @ A_v)         # reads 16KB + 32KB

    r = time_ops([qkv_bwd], args.warmup, args.iters,
                 f"qkv_bwd (base+LoRA, {(Q_SZ+KV_SZ)*H*2/1e6:.0f}MB reads)")
    results.append(r)

    # ── 9. Full attention backward sub-op (all of the above combined) ────
    def full_attn_bwd():
        # O backward
        gs = grad_o @ o_w
        grad_x_attn.add_(gs + (grad_o @ B_o) @ A_o)
        # QKV recompute
        q = F.linear(x_norm, qkv_w[:Q_SZ]) + (x_norm @ A_q.T) @ B_q.T
        v = F.linear(x_norm, qkv_w[Q_SZ+KV_SZ:]) + (x_norm @ A_v.T) @ B_v.T
        # SDPA backward
        q3 = q.view(T, N_HEADS_LOC, HEAD_DIM).transpose(0,1).unsqueeze(0).detach().requires_grad_(True)
        v3 = v.view(T, N_KV_LOC,   HEAD_DIM).transpose(0,1).unsqueeze(0).detach().requires_grad_(True)
        v3e = v3.repeat_interleave(rep, dim=1)   # [1, NH, T, HD]
        out = F.scaled_dot_product_attention(q3, k_det, v3e, is_causal=True)
        out.backward(grad_sdpa_bwd)
        # QKV backward
        gq = q3.grad.squeeze(0).transpose(0,1).reshape(T, Q_SZ)
        grad_x_attn.add_(gq @ qkv_w[:Q_SZ])
        grad_x_attn.add_(grad_v_bwd @ qkv_w[Q_SZ+KV_SZ:])

    r = time_ops([full_attn_bwd], args.warmup, args.iters,
                 "full_attn_bwd (o_proj + qkv_recompute + sdpa_bwd + qkv_bwd)")
    results.append(r)

    # ── Print results ─────────────────────────────────────────────────────
    print(f"{'Sub-op':<60} {'p50':>8} {'mean':>8} {'p95':>8} {'p99':>8}")
    print("-" * 92)
    for r in results:
        print(f"{r['label']:<60} {r['p50_ms']:>7.3f}ms {r['mean_ms']:>7.3f}ms "
              f"{r['p95_ms']:>7.3f}ms {r['p99_ms']:>7.3f}ms")

    print()
    print(f"Allreduce bubble reference (from serving run): p50=0.28ms (rank0), 0.33ms (rank1)")
    chunk_p50 = results[0]["p50_ms"]
    print(f"Passthrough chunk p50: {chunk_p50:.3f}ms  →  fits in bubble: {'YES' if chunk_p50 < 0.28 else 'NO (spills)'}")
    print(f"fills_per_trigger=2 uses: {2 * chunk_p50:.3f}ms of {0.28:.3f}ms bubble = {2*chunk_p50/0.28*100:.0f}%")
    print(f"Max fills per bubble:      {int(0.28 / chunk_p50)}")


if __name__ == "__main__":
    main()
