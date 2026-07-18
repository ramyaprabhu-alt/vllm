#!/usr/bin/env python3
"""make_toy_lora.py — generate an *untrained* (init-only) LoRA adapter for a
BubbleTea target model, shaped correctly for that model's attention config.

No adapter-generation script survived on disk for either the existing
qwen3-toy-lora or qwen15-moe-toy-lora adapters used by BubbleTea's
scripts/bench_arxiv.sh / scripts/run_qwen15_moe_a27b.sh. This reproduces them
(and any future model's adapter) from a plain PEFT LoraConfig -- the resulting
adapter is randomly initialized, not fine-tuned; BubbleTeaLoRATrainer trains it
for real once the server is running.

Default hyperparameters (r=16, lora_alpha=16, dropout=0.05, bias=none,
target_modules=q/k/v/o_proj) match the existing qwen15-moe-toy-lora adapter's
adapter_config.json exactly, so re-running this for Qwen1.5-MoE-A2.7B
reproduces the same shapes.

Requires `peft`/`transformers`/`torch` in the active venv (not installed in
vLLM's .venv by default -- BubbleTea only *loads* the resulting
adapter_model.safetensors via safetensors.torch.load_file, it never imports
peft, so this script is meant to be run once, standalone, e.g. from a small
side venv: `pip install -U transformers accelerate peft`).

Usage:
    python scripts/make_toy_lora.py \\
        --model /mnt/nfs/home/ramya/models/Qwen/Qwen1.5-MoE-A2.7B \\
        --out /mnt/nfs/home/ramya/scratch/qwen15-moe-toy-lora
"""

import argparse
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_lora_shapes(model) -> None:
    print("\n=== LoRA parameter shapes (only these should be trainable) ===")
    found = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            print(f"{name}: {tuple(param.shape)}  dtype={param.dtype}")
            found += 1
    if found == 0:
        print("WARNING: No LoRA parameters found. Check target_modules / PEFT config.")
    print("============================================================\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Path or HF id of the base model")
    ap.add_argument("--out", required=True, help="Output directory for the adapter")
    ap.add_argument("--r", type=int, default=16, help="LoRA rank")
    ap.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    ap.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    ap.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Module names LoRA is applied to (attention projections)",
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to from_pretrained (needed for some MoE configs)",
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        trust_remote_code=args.trust_remote_code,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()
    print_lora_shapes(model)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved *untrained* LoRA adapter to: {args.out}")


if __name__ == "__main__":
    main()
