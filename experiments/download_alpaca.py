"""
Pre-download and cache yahma/alpaca-cleaned to ~/scratch so the BubbleTea
LoRA trainer doesn't fetch it from the network inside the vLLM worker.

Run once before the integration test (or before the benchmark):
    /mnt/nfs/home/ramya/vllm/.venv/bin/python download_alpaca.py

The download is ~24 MB and takes ~30 seconds on a typical connection.
After this script completes, subsequent calls to
    datasets.load_dataset("yahma/alpaca-cleaned", cache_dir="~/scratch")
load from local Arrow files in under a second.
"""

import sys
from pathlib import Path

CACHE_DIR = Path.home() / "scratch"

print(f"Cache dir: {CACHE_DIR}")
print("Downloading yahma/alpaca-cleaned ...")

import datasets

ds = datasets.load_dataset("yahma/alpaca-cleaned", cache_dir=str(CACHE_DIR))
# After this runs, the server can be launched with HF_DATASETS_OFFLINE=1 so
# bt_lora_trainer._make_data_iter() never stalls on a hub check.

train = ds["train"]
print(f"Downloaded {len(train)} training examples.")
print(f"Columns: {train.column_names}")
print(f"Cached at: {CACHE_DIR}/yahma___alpaca-cleaned")

# Quick sanity check
sample = train[0]
print(f"\nSample instruction: {sample['instruction'][:80]}")
print("\nDone. Dataset is cached and ready for bt_lora_trainer.")
