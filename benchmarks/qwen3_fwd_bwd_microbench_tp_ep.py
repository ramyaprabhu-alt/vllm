"""
qwen3_fwd_bwd_microbench_tp_ep.py — Distributed TP variant of
qwen3_fwd_bwd_microbench.py: forward vs backward pass timing for
Qwen3-30B-A3B with LoRA adapters (q/k/v/o_proj, rank=16, alpha=16),
sharded across 2 GPUs using tp_plan="auto" (transformers'
Qwen3MoeConfig.base_model_tp_plan):

  - TP for the attention path — colwise q/k/v_proj, rowwise o_proj,
    replicated-with-grad-allreduce q_norm/k_norm.
  - TP for the MoE path — packed_colwise gate_up_proj, rowwise down_proj,
    moe_tp_experts wrapper (all_reduce_forward on expert output, so each
    rank processes ALL 128 experts with half-width weight shards).
  - lm_head is replicated (not sharded) under the auto plan.

Note on tp_plan: passing a custom dict with "layers.*" patterns fails to
shard attention because transformers matches those paths against the full
model name (which starts with "model.layers.*"), but does match top-level
keys like "lm_head".  The resulting hybrid — lm_head sharded + attention
un-hooked — breaks the backward pass with NaN gradients.  tp_plan="auto"
uses the config's own plan, which is consistent and NaN-free.

Both ranks must see *identical* tokens (colwise/rowwise pairs assume
replicated input), so the batch is broadcast from rank 0.

LoRA wraps q/k/v/o_proj exactly as in the single-GPU benchmark, but the
adapter matrices mirror their base layer's sharding — see TPLoRALinear.

Run (2 GPUs):
    torchrun --nproc_per_node=2 qwen3_fwd_bwd_microbench_tp_ep.py [--t-ft N] [--reps N] [--warmup N]
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM

MODEL_DIR = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
VOCAB     = 151936

# LoRA config — matches qwen3-toy-lora/adapter_config.json
# (r=16, lora_alpha=16, target_modules=[q_proj, k_proj, v_proj, o_proj])
RANK    = 16
ALPHA   = 16.0
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

# How each LoRA target's base layer is sharded under the plan below — drives
# how TPLoRALinear shapes/replicates its adapter matrices.
TARGET_STYLE = {"q_proj": "colwise", "k_proj": "colwise", "v_proj": "colwise", "o_proj": "rowwise"}

# tp_plan="auto" uses Qwen3MoeConfig.base_model_tp_plan, which correctly
# applies ColwiseParallel/RowwiseParallel forward hooks to attention and
# MoeTensorParalellExperts hooks to MoE — all consistent in forward and
# backward.  A custom dict whose "layers.*" keys don't match the full-model
# path "model.layers.*" skips those hooks while still matching "lm_head",
# producing a broken mixed-hook model that generates NaN gradients.


def _kaiming_uniform_bound(fan_in: int) -> float:
    """Reproduces the bound nn.init.kaiming_uniform_(t, a=sqrt(5)) derives from
    t's fan_in. Used to seed a row-sharded lora_A at the *full* matrix's scale
    (see TPLoRALinear) instead of the narrower local-shard scale fan_in would
    otherwise pick up — keeping init statistics comparable to the 1-GPU run."""
    gain = nn.init.calculate_gain("leaky_relu", math.sqrt(5))
    return gain * math.sqrt(3.0 / fan_in)


class _SumAllReduce(torch.autograd.Function):
    """All-reduce-sum with a *correct* backward — plain dist.all_reduce has no
    autograd kernel (PyTorch warns "may lead to silently incorrect behavior"
    and, worse, in practice corrupts the whole upstream graph: this is what
    produced the NaNs in the first version of this file). Forward combines
    each rank's partial delta into the replicated full delta; backward is the
    identity, because that replicated delta is consumed identically by every
    rank afterwards, so the incoming grad each rank already has *is* its
    share — no further communication needed (mirrors transformers'
    all_reduce_forward / Megatron's "g")."""

    @staticmethod
    def forward(ctx, x, group):
        if dist.get_world_size(group) > 1:
            x = x.contiguous()
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def _all_reduce_sum(x: torch.Tensor, group) -> torch.Tensor:
    return _SumAllReduce.apply(x, group)


class TPLoRALinear(nn.Module):
    """LoRA wrapper for a frozen base nn.Linear that TP has already sharded
    (colwise: output sharded across ranks, rowwise: input sharded). Mirrors
    LoRALinear from qwen3_fwd_bwd_microbench.py, but keeps lora_A/lora_B
    sharded consistently with `base` so base(x) + delta stays equivalent to
    running the un-sharded adapter — and does the extra collectives that
    requires:

      colwise base (q/k/v_proj — output sharded, e.g. out=4096 -> 2048/rank):
        lora_B is column-sharded to match (local out_features), so each
        rank's local `(x @ A.T) @ B_local.T` is already the correct slice
        of the full delta — no communication in forward. lora_A is the
        *same full* [rank, in_features] matrix on every rank (broadcast
        from rank 0 at init); since it now contributes to the loss via two
        independent paths (one per rank), its gradient is a sum of partials
        and must be all-reduced — exactly transformers' trick for q_norm/
        k_norm ("replicated_with_grad_allreduce").

      rowwise base (o_proj — input sharded, e.g. in=4096 -> 2048/rank):
        lora_A is row-sharded to match (local in_features) so it can
        consume the local input shard; lora_B is the same full
        [out_features, rank] matrix on every rank. Each rank's local delta
        `x_local @ A_local.T @ B.T` is now only a *partial* sum (the true
        delta is `(x_0 @ A_0.T + x_1 @ A_1.T) @ B.T`, which distributes into
        a sum of per-rank local deltas), so it must be all-reduced — the
        same all-reduce RowwiseParallel performs on base(x)'s own partial
        output. Summing then adding == adding then summing, so
        `base(x) + all_reduce(delta_local)` is exactly the un-sharded
        `base(x) + delta`.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, style: str, tp_group):
        super().__init__()
        self.base     = base
        self.style    = style
        self.tp_group = tp_group
        device = next(base.parameters()).device
        dtype  = next(base.parameters()).dtype
        # transformers rewrites base.in_features/out_features to the *local*
        # shard sizes in place once it shards the layer (see e.g.
        # ColwiseParallel/RowwiseParallel.update_module_attributes), so these
        # already give us the right local shapes for lora_A / lora_B.
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features,  device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        self.scale  = alpha / rank

        if style == "colwise":
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            dist.broadcast(self.lora_A.data, src=0, group=tp_group)
            self.lora_A.tp_replicated = True
        elif style == "rowwise":
            full_in = base.in_features * dist.get_world_size(tp_group)
            bound = _kaiming_uniform_bound(full_in)
            nn.init.uniform_(self.lora_A, -bound, bound)
            self.lora_B.tp_replicated = True
        else:
            raise ValueError(f"unsupported TP style for LoRA target: {style!r}")

    def forward(self, x):
        delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        if self.style == "rowwise":
            delta = _all_reduce_sum(delta, self.tp_group)
        return self.base(x) + delta


def add_lora(model, rank: int, alpha: float, targets: tuple[str, ...], tp_group) -> list[nn.Parameter]:
    """Replace target attention projections with TP-aware LoRA wrappers in-place."""
    for layer in model.model.layers:
        for name in targets:
            base = getattr(layer.self_attn, name)
            setattr(layer.self_attn, name,
                    TPLoRALinear(base, rank, alpha, TARGET_STYLE[name], tp_group))
    # Global freeze — matches qwen3_fwd_bwd_microbench.py / standalone_finetune.py.
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    return [p for p in model.parameters() if p.requires_grad]


# ── Reporting (rank 0 only) ──────────────────────────────────────────────────

def summarize(name: str, samples_ms: list[float], n_tokens: int) -> None:
    mean = statistics.mean(samples_ms)
    sd   = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    p50  = statistics.median(samples_ms)
    p99  = sorted(samples_ms)[max(0, int(len(samples_ms) * 0.99) - 1)]
    print(f"  {name:<10}  mean={mean:8.2f}ms  stdev={sd:7.2f}ms  "
          f"p50={p50:8.2f}ms  p99={p99:8.2f}ms  "
          f"({n_tokens / mean * 1000:9.1f} tok/s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t-ft",   type=int, default=128, help="tokens per step (matches production t_ft)")
    ap.add_argument("--reps",   type=int, default=20,  help="timed repetitions")
    ap.add_argument("--warmup", type=int, default=5,   help="warmup steps (untimed)")
    args = ap.parse_args()

    if "RANK" not in os.environ:
        raise RuntimeError(
            "qwen3_fwd_bwd_microbench_tp_ep.py needs a distributed launch:\n"
            "    torchrun --nproc_per_node=2 qwen3_fwd_bwd_microbench_tp_ep.py [...]"
        )
    dist_rank  = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    def log(msg: str) -> None:
        if dist_rank == 0:
            print(msg)

    torch.manual_seed(0)

    log(f"Loading Qwen3-30B-A3B (bf16) sharded TP over {os.environ['WORLD_SIZE']} GPUs ...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, tp_plan="auto")
    model.train()
    tp_group = model._device_mesh.get_group()
    torch.cuda.synchronize()
    log(f"  loaded in {time.perf_counter() - t0:.1f}s — "
        f"{torch.cuda.memory_allocated(local_rank) / 1e9:.1f} GB allocated on rank {dist_rank} (cuda:{local_rank})")

    lora_params = add_lora(model, RANK, ALPHA, TARGETS, tp_group)
    n_lora = sum(p.numel() for p in lora_params)
    replicated = [p for p in lora_params if getattr(p, "tp_replicated", False)]
    log(f"LoRA: rank={RANK}  alpha={ALPHA}  targets={TARGETS}  scale={ALPHA / RANK}")
    log(f"  trainable params (this rank's local shards + replicas): {n_lora:,} "
        f"({len(lora_params)} tensors, {len(replicated)} replicated w/ grad-allreduce)")

    optimizer = torch.optim.AdamW(lora_params, lr=2e-4, weight_decay=0.01)

    T = args.t_ft
    # Random token IDs from the real vocabulary, fixed across all steps (same
    # content-independent-timing approach as the 1-GPU benchmark) — and
    # broadcast from rank 0 so every rank routes *identical* tokens, which the
    # EP router and the colwise/rowwise-paired collectives both assume.
    batch = torch.randint(0, VOCAB, (1, T + 1), device=device)
    dist.broadcast(batch, src=0, group=tp_group)
    input_ids = batch[:, :-1]
    labels    = batch[:, 1:]

    n_steps = args.warmup + args.reps
    fwd_ms, bwd_ms, step_ms = [], [], []

    log(f"\nRunning {args.warmup} warmup + {args.reps} timed steps "
        f"(t_ft={T} tokens/step, TP over {os.environ['WORLD_SIZE']} GPUs)...")
    dist.barrier()
    for i in range(n_steps):
        optimizer.zero_grad(set_to_none=True)

        if dist_rank == 0:
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end   = torch.cuda.Event(enable_timing=True)
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end   = torch.cuda.Event(enable_timing=True)
            fwd_start.record()

        out = model(input_ids=input_ids, labels=labels)

        if dist_rank == 0:
            fwd_end.record()

        if i == 0:
            torch.cuda.synchronize()
            log(f"  [diag] after fwd (step 1): "
                f"{torch.cuda.memory_allocated(local_rank) / 1e9:.2f} GB allocated, "
                f"{torch.cuda.max_memory_allocated(local_rank) / 1e9:.2f} GB peak (rank {dist_rank})")

        if dist_rank == 0:
            bwd_start.record()

        out.loss.backward()

        if i < 2:
            for n, p in model.named_parameters():
                if p.requires_grad and ("layers.0.self_attn" in n):
                    g = p.grad
                    print(f"  [DBG r{dist_rank}] {n:55s} shape={tuple(p.shape)} "
                          f"|p|={p.float().norm().item():.4e} "
                          f"|g|={'None' if g is None else f'{g.float().norm().item():.4e}'} "
                          f"g_nan={'-' if g is None else bool(torch.isnan(g).any().item())} "
                          f"g_inf={'-' if g is None else bool(torch.isinf(g).any().item())}",
                          flush=True)

        if dist_rank == 0:
            bwd_end.record()

        # Replicated LoRA params (colwise lora_A, rowwise lora_B) only
        # accumulate the gradient contribution from *this* rank's path —
        # sum (not average) the partials to recover the true full-matrix
        # gradient, mirroring transformers' ReplicatedWithGradAllReduce.
        for p in replicated:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=tp_group)

        optimizer.step()
        torch.cuda.synchronize()

        f_ms = b_ms = 0.0
        if dist_rank == 0:
            f_ms = fwd_start.elapsed_time(fwd_end)
            b_ms = bwd_start.elapsed_time(bwd_end)
            if i >= args.warmup:
                fwd_ms.append(f_ms)
                bwd_ms.append(b_ms)
                step_ms.append(f_ms + b_ms)
            tag = "warmup" if i < args.warmup else "timed "
            print(f"  [{tag}] step {i + 1:>3}/{n_steps}  "
                  f"loss={out.loss.item():.4f}  fwd={f_ms:8.2f}ms  bwd={b_ms:8.2f}ms")

    if dist_rank == 0:
        print(f"\n=== Qwen3-30B-A3B forward/backward microbenchmark "
              f"(TP x{os.environ['WORLD_SIZE']}, t_ft={T}, {args.reps} timed reps) ===")
        summarize("forward",  fwd_ms,  T)
        summarize("backward", bwd_ms,  T)
        summarize("fwd+bwd",  step_ms, T)
        print(f"\n  backward / forward ratio: "
              f"{statistics.mean(bwd_ms) / statistics.mean(fwd_ms):.2f}x")
        print(f"  peak GPU {local_rank} memory: {torch.cuda.max_memory_allocated(local_rank) / 1e9:.1f} GB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
