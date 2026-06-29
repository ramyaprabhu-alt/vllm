"""
qwen3_fwd_bwd_microbench.py — Microbenchmark: forward vs backward pass timing
for Qwen3-30B-A3B with LoRA adapters (q/k/v/o_proj, rank=16, alpha=16 —
matching qwen3-toy-lora's adapter_config.json and bt_lora_trainer._LORA_ALPHA).

Loads the full model (bf16, ~61GB) onto a single A100-80GB GPU — the same
single-GPU setup standalone_finetune.py uses (61.1GB fits comfortably in 85GB,
per Sessions_3_6_2026.md). Single GPU also avoids cross-device transfers that
would otherwise blur the forward/backward timing split.

Each step times the forward pass (model(...) → loss) and the backward pass
(loss.backward()) separately with paired CUDA events, after a warmup period
(early steps pay one-time CUDA context / kernel JIT / cache-fill costs that
don't represent steady-state per-step cost).

Run:
    .venv/bin/python qwen3_fwd_bwd_microbench.py [--t-ft N] [--reps N] [--warmup N]
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

MODEL_DIR = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
# cuda:0 is occupied by another process (nvidia-smi showed ~975MB used, 99% util),
# which pushed this benchmark's full-graph backward over the 79GB budget into OOM.
# cuda:1 is idle — use it instead.
DEVICE      = "cuda:1"
DEVICE_IDX  = 1
VOCAB     = 151936

# LoRA config — matches qwen3-toy-lora/adapter_config.json
# (r=16, lora_alpha=16, target_modules=[q_proj, k_proj, v_proj, o_proj])
# and bt_lora_trainer._LORA_ALPHA — scale = alpha / rank = 1.0.
RANK    = 16
ALPHA   = 16.0
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    """Wraps a frozen base nn.Linear with a trainable low-rank A/B delta."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        device = next(base.parameters()).device
        dtype  = next(base.parameters()).dtype
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features,  device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def add_lora(model, rank: int, alpha: float, targets: tuple[str, ...]) -> list[nn.Parameter]:
    """Replace target attention projections with LoRA-wrapped versions in-place."""
    for layer in model.model.layers:
        for name in targets:
            base = getattr(layer.self_attn, name)
            setattr(layer.self_attn, name, LoRALinear(base, rank, alpha))
    # Global freeze — matches standalone_finetune.py. Wrapping q/k/v/o_proj
    # alone does NOT freeze the rest of the ~30B-param base (lm_head,
    # embed_tokens, MoE experts/routers, norms): those default to
    # requires_grad=True from from_pretrained, so backward() was computing
    # and storing gradients for ~3B actively-routed params per step — the
    # exact ~17GB overhead that turned the 61GB model into an OOM during
    # backward (confirmed via the post-forward memory diagnostic: forward
    # alone only used 61.24GB, backward pushed it to 78.74GB).
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    params = [p for p in model.parameters() if p.requires_grad]
    return params


# ── Reporting ───────────────────────────────────────────────────────────────

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

    torch.manual_seed(0)

    print(f"Loading Qwen3-30B-A3B (bf16) onto {DEVICE} ...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).to(DEVICE)
    model.train()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s — "
          f"{torch.cuda.memory_allocated(DEVICE_IDX) / 1e9:.1f} GB allocated on GPU {DEVICE_IDX}")

    lora_params = add_lora(model, RANK, ALPHA, TARGETS)
    n_lora = sum(p.numel() for p in lora_params)
    print(f"LoRA: rank={RANK}  alpha={ALPHA}  targets={TARGETS}  scale={ALPHA / RANK}")
    print(f"  trainable params: {n_lora:,} ({len(lora_params)} tensors)")

    optimizer = torch.optim.AdamW(lora_params, lr=2e-4, weight_decay=0.01)

    T = args.t_ft
    # Random token IDs from the real vocabulary, fixed across all steps —
    # same content-independent-timing approach as bt_lora_layer_bench_real.py.
    # Reusing one batch also keeps MoE routing/expert-load identical step to
    # step, so fwd/bwd timings are directly comparable across reps.
    batch     = torch.randint(0, VOCAB, (1, T + 1), device=DEVICE)
    input_ids = batch[:, :-1]
    labels    = batch[:, 1:]

    n_steps = args.warmup + args.reps
    fwd_ms, bwd_ms, step_ms = [], [], []

    print(f"\nRunning {args.warmup} warmup + {args.reps} timed steps "
          f"(t_ft={T} tokens/step)...")
    for i in range(n_steps):
        optimizer.zero_grad(set_to_none=True)

        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end   = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end   = torch.cuda.Event(enable_timing=True)

        fwd_start.record()
        out = model(input_ids=input_ids, labels=labels)
        fwd_end.record()

        if i == 0:
            torch.cuda.synchronize()
            print(f"  [diag] after fwd (step 1): "
                  f"{torch.cuda.memory_allocated(DEVICE_IDX) / 1e9:.2f} GB allocated, "
                  f"{torch.cuda.max_memory_allocated(DEVICE_IDX) / 1e9:.2f} GB peak")

        bwd_start.record()
        out.loss.backward()
        bwd_end.record()

        optimizer.step()
        torch.cuda.synchronize()

        f_ms = fwd_start.elapsed_time(fwd_end)
        b_ms = bwd_start.elapsed_time(bwd_end)
        if i >= args.warmup:
            fwd_ms.append(f_ms)
            bwd_ms.append(b_ms)
            step_ms.append(f_ms + b_ms)

        tag = "warmup" if i < args.warmup else "timed "
        print(f"  [{tag}] step {i + 1:>3}/{n_steps}  "
              f"loss={out.loss.item():.4f}  fwd={f_ms:8.2f}ms  bwd={b_ms:8.2f}ms")

    print(f"\n=== Qwen3-30B-A3B forward/backward microbenchmark "
          f"(t_ft={T}, {args.reps} timed reps) ===")
    summarize("forward",  fwd_ms,  T)
    summarize("backward", bwd_ms,  T)
    summarize("fwd+bwd",  step_ms, T)
    print(f"\n  backward / forward ratio: "
          f"{statistics.mean(bwd_ms) / statistics.mean(fwd_ms):.2f}x")
    print(f"  peak GPU {DEVICE_IDX} memory: {torch.cuda.max_memory_allocated(DEVICE_IDX) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
