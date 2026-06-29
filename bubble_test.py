#!/usr/bin/env python3
"""
Send a long no-think request to the vLLM server and report TTFT + bubble profile.
Usage:
    python bubble_test.py [--tokens N] [--port PORT]
"""
import argparse
import glob
import json
import os
import statistics
import time

import requests
from transformers import AutoTokenizer

MODEL = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
BUBBLE_OUT = "/tmp/vllm_bubble_profile.json"   # legacy, unused
RANK_FILES = ["/tmp/vllm_bubble_rank0.json", "/tmp/vllm_bubble_rank1.json"]


def build_prompt(tokenizer, target_tokens: int) -> str:
    """Repeat a passage until we hit target_tokens."""
    chunk = (
        "The study of distributed systems encompasses the design and analysis "
        "of algorithms for networks of computers. Fault tolerance, consistency, "
        "and availability are the three pillars that every distributed system "
        "must balance according to the CAP theorem. Modern systems like "
        "Kubernetes, Kafka, and Cassandra each make explicit trade-offs among "
        "these properties. Expert parallelism in large language models partitions "
        "the MoE expert layers across GPU ranks, reducing per-device memory at "
        "the cost of all-to-all communication latency during the forward pass. "
    )
    # Repeat until we exceed target
    text = chunk
    while len(tokenizer.encode(text)) < target_tokens:
        text = text + chunk
    # Trim to exact length
    ids = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def send_request(prompt: str, port: int) -> tuple[float, str]:
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 32,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=300)
    ttft = time.perf_counter() - t0
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return ttft, content


def load_last_prefill(path: str) -> list[dict] | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        recs = json.load(f)
    if not recs:
        return None
    last_id = max(r["prefill_id"] for r in recs)
    return [r for r in recs if r["prefill_id"] == last_id]


def compare_ranks(rank_files: list[str], target_tokens: int):
    datasets = []
    for path in rank_files:
        label = os.path.basename(path).replace("vllm_bubble_rank","").replace(".json","")
        recs = load_last_prefill(path)
        if recs is None:
            print(f"  [rank {label}: empty]")
        else:
            datasets.append((label, recs))

    if not datasets:
        print("  [no rank profiles found]")
        return

    print(f"\n=== Per-rank EP timing (last prefill, ~{target_tokens} tokens) ===")
    print(f"  Config: TP=2+EP, allgather_reducescatter, 48 MoE layers\n")

    rank_ffn_totals = []
    for rank_id, recs in datasets:  # rank_id is now a label string
        n_tokens = recs[0]["n_tokens"]
        ffn   = [r["ffn_ms"]       for r in recs]
        ar    = [r["allreduce_ms"] for r in recs]
        total = [r["total_ms"]     for r in recs]
        s_ffn, s_ar = sum(ffn), sum(ar)
        rank_ffn_totals.append(s_ffn)

        def _p(vals):
            s = sorted(vals)
            n = len(s)
            return (f"mean={statistics.mean(s):.1f}  "
                    f"p25={s[n//4]:.1f}  p50={s[n//2]:.1f}  "
                    f"p75={s[3*n//4]:.1f}  p95={s[int(.95*n)]:.1f}  max={s[-1]:.1f} ms")

        print(f"  ── Rank {rank_id} (GPU {rank_id}, {n_tokens} tokens, {len(recs)} layers) ──")
        print(f"    FFN:        {_p(ffn)}")
        print(f"    allreduce:  {_p(ar)}")
        print(f"    total/layer:{_p(total)}")
        print(f"    cumulative: FFN={s_ffn:.1f}ms  allreduce={s_ar:.1f}ms  total={s_ffn+s_ar:.1f}ms\n")

    if len(rank_ffn_totals) == 2:
        imbalance = abs(rank_ffn_totals[0] - rank_ffn_totals[1]) / max(rank_ffn_totals) * 100
        print(f"  Load imbalance: {imbalance:.1f}%  "
              f"(rank0={rank_ffn_totals[0]:.1f}ms vs rank1={rank_ffn_totals[1]:.1f}ms FFN total)")
        print(f"  TP all_reduce blocks at max(rank0, rank1) — slower rank sets the pace.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=18000)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Loading tokenizer from {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print(f"Building {args.tokens}-token prompt...")
    prompt = build_prompt(tokenizer, args.tokens)
    actual_tokens = len(tokenizer.encode(prompt))
    print(f"  Actual token count: {actual_tokens}")

    for f in glob.glob("/tmp/vllm_bubble_rank*.json"):
        os.remove(f)

    print(f"\nSending request to port {args.port} (thinking disabled)...")
    ttft, reply = send_request(prompt, args.port)
    print(f"  TTFT: {ttft*1000:.0f}ms")
    print(f"  Reply: {reply[:120]!r}")

    rank_files = sorted(glob.glob("/tmp/vllm_bubble_rank*.json"))
    print(f"  Found rank files: {rank_files}")
    compare_ranks(rank_files, args.tokens)


if __name__ == "__main__":
    main()
