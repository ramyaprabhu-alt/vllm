"""
standalone_finetune.py — Standalone LoRA finetuning throughput ceiling.

Loads Qwen3-30B-A3B across 2 GPUs, adds LoRA adapters matching
qwen3-toy-lora (rank=8, q/k/v_proj, 48 layers), trains on ShareGPT
at t_ft=128 tokens/step, and reports samples/sec throughput.

Used to establish the training throughput ceiling for comparing against
LLMStation co-serving (0.10 samp/s) and BubbleTea co-serving (~0.004 samp/s).
"""

import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
SHAREGPT  = "/mnt/nfs/home/ramya/scratch/ShareGPT_V3_unfiltered_cleaned_split.json"
T_FT      = 128
RANK      = 8
ALPHA     = 16.0
WARMUP    = 5
STEPS     = 50


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        device = next(base.parameters()).device
        dtype  = next(base.parameters()).dtype
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def add_lora(model, rank=8, alpha=16.0):
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj"):
            base = getattr(layer.self_attn, name)
            setattr(layer.self_attn, name, LoRALinear(base, rank, alpha))
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n


def load_chunks(tokenizer, n):
    data = json.loads(Path(SHAREGPT).read_text())
    chunks = []
    for conv in data:
        text = " ".join(t["value"] for t in conv.get("conversations", []))
        ids  = tokenizer.encode(text, add_special_tokens=False)
        for i in range(0, len(ids) - T_FT, T_FT):
            chunks.append(ids[i : i + T_FT + 1])
            if len(chunks) >= n:
                return chunks
    return chunks


def main():
    print(f"Loading model from {MODEL_DIR} across 2 GPUs...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    model.train()
    add_lora(model, RANK, ALPHA)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    print(f"LoRA params: {sum(p.numel() for p in lora_params):,} ({len(lora_params)} tensors)")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU {i}: {alloc:.1f} GB allocated after model load")

    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

    print(f"Loading {WARMUP + STEPS} training chunks (t_ft={T_FT})...")
    chunks = load_chunks(tokenizer, WARMUP + STEPS)
    first_device = torch.device("cuda:0")

    def step(chunk):
        ids = torch.tensor(chunk, dtype=torch.long, device=first_device)
        out = model(input_ids=ids[:-1].unsqueeze(0), labels=ids[1:].unsqueeze(0))
        out.loss.backward()
        return out.loss.item()

    print(f"Warmup ({WARMUP} steps)...")
    for i in range(WARMUP):
        optimizer.zero_grad()
        step(chunks[i])
        optimizer.step()
    torch.cuda.synchronize()

    print(f"Timing {STEPS} steps...")
    t0 = time.perf_counter()
    losses = []
    for i in range(STEPS):
        optimizer.zero_grad()
        loss = step(chunks[WARMUP + i])
        optimizer.step()
        losses.append(loss)
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  step {i+1}/{STEPS}  loss={loss:.4f}  ({elapsed:.1f}s so far)")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sps = STEPS / elapsed
    print(f"\n=== Standalone LoRA finetuning (1 GPU, t_ft={T_FT}) ===")
    print(f"  Steps:        {STEPS}")
    print(f"  Elapsed:      {elapsed:.1f}s")
    print(f"  Samples/sec:  {sps:.4f}  (1 sample = {T_FT} tokens, fwd+bwd+optim)")
    print(f"  Tokens/sec:   {sps * T_FT:.1f}")
    print(f"  Mean loss:    {sum(losses)/len(losses):.4f}")
    print(f"\nFor comparison (co-serving, same 50-prompt arxiv trace at 0.2 qps):")
    print(f"  LLMStation:   0.1000 samp/s  ({sps/0.1000:.1f}x slower than ceiling)")
    print(f"  BubbleTea:    0.0035 samp/s  ({sps/0.0035:.1f}x slower than ceiling)")


if __name__ == "__main__":
    main()
