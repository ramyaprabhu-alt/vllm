#!/usr/bin/env python3
"""
bubble_sched_live_test.py — measure TTFT overhead of the bubble scheduler.

The scheduler drains backward sub-ops purely through prefill bubbles on both
ranks.  This test quantifies the cost: how much extra TTFT does arming the
scheduler add vs a clean baseline?

ARCHITECTURE NOTE
-----------------
vLLM runs each TP rank in a separate worker process.  submit_backward_job()
lives in moe_runner.py and must be called INSIDE the worker process.  This
test cannot directly call it from outside.

Instead it uses VLLM_BUBBLE_SCHED_DEMO=1 (set in the server launch), which
makes each worker auto-submit a perpetual stream of demo backward jobs
immediately after the first qualifying prefill.  The test then measures TTFT
with and without this flag.

To run:
    # Baseline (no scheduler):
    python bubble_sched_live_test.py --mode baseline --t-in 16384 --reps 20

    # With scheduler:
    VLLM_BUBBLE_SCHED_DEMO=1 bash run_qwen3_30b_a3b.sh --enforce-eager \
        --no-enable-chunked-prefill --max-num-batched-tokens 32768 \
        --no-enable-prefix-caching
    python bubble_sched_live_test.py --mode sched --t-in 16384 --reps 20

    # Side-by-side (requires both servers on different ports):
    python bubble_sched_live_test.py --mode compare \
        --baseline-port 8000 --sched-port 8001 --t-in 16384 --reps 20
"""
from __future__ import annotations

import argparse
import statistics
import threading
import time

import requests
from transformers import AutoTokenizer

MODEL = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"

# ── Prompt builder ────────────────────────────────────────────────────────────

_DIVERSE_TEXT = (
    "The mitochondria generates ATP through oxidative phosphorylation. "
    "def merge_sort(arr): return arr if len(arr)<=1 else merge(merge_sort(arr[:len(arr)//2]),merge_sort(arr[len(arr)//2:])). "
    "SELECT name, COUNT(*) FROM employees GROUP BY department HAVING COUNT(*)>5. "
    "The Riemann hypothesis states all non-trivial zeros of zeta(s) have real part 1/2. "
    "import torch; x = torch.randn(128, 2048, device='cuda', dtype=torch.bfloat16). "
    "Paris is the capital of France and home to the Eiffel Tower built in 1889. "
)


def build_prompt(tokenizer, target_tokens: int, prompt_type: str = "diverse") -> str:
    if prompt_type == "repeated":
        tok_id = tokenizer.encode("the ", add_special_tokens=False)[:1]
        return tokenizer.decode(tok_id * target_tokens)

    # Diverse: pad _DIVERSE_TEXT to target length
    base_ids = tokenizer.encode(_DIVERSE_TEXT, add_special_tokens=False)
    pad_id   = tokenizer.encode("the ", add_special_tokens=False)[:1]
    needed   = max(0, target_tokens - len(base_ids))
    ids      = base_ids + pad_id * needed
    return tokenizer.decode(ids[:target_tokens])


# ── Request sender ────────────────────────────────────────────────────────────

def send_one(prompt: str, port: int) -> float | None:
    """Return TTFT in ms, or None on failure."""
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return (time.perf_counter() - t0) * 1e3
    except Exception as e:
        print(f"    request error: {e}")
        return None


def warmup(port: int, tokenizer, t_in: int, n: int = 3):
    print(f"  Warming up (port {port}, {n} requests)...")
    prompt = build_prompt(tokenizer, t_in)
    for _ in range(n):
        send_one(prompt + f" [{time.time_ns()}]", port)


def measure_ttft(port: int, tokenizer, t_in: int, reps: int,
                 prompt_type: str = "diverse") -> list[float]:
    results = []
    for i in range(reps):
        prompt = build_prompt(tokenizer, t_in, prompt_type)
        prompt += f" [sched-test-{i}-{time.time_ns()}]"
        ms = send_one(prompt, port)
        if ms is not None:
            results.append(ms)
            print(f"    rep {i:2d}: {ms:.1f} ms")
    return results


# ── Statistics ────────────────────────────────────────────────────────────────

def report(label: str, ttfts: list[float]):
    if not ttfts:
        print(f"  {label}: no data")
        return
    s = sorted(ttfts)
    n = len(s)
    print(f"\n  {label}  (n={n})")
    print(f"    mean={statistics.mean(s):.1f}ms  "
          f"p50={s[n//2]:.1f}ms  "
          f"p95={s[int(0.95*n)]:.1f}ms  "
          f"min={s[0]:.1f}ms  max={s[-1]:.1f}ms")


def compare(baseline: list[float], sched: list[float]):
    if not baseline or not sched:
        return
    overhead_mean = statistics.mean(sched) - statistics.mean(baseline)
    overhead_p50  = sorted(sched)[len(sched)//2] - sorted(baseline)[len(baseline)//2]
    print(f"\n  Scheduler overhead vs baseline:")
    print(f"    mean: {overhead_mean:+.1f} ms")
    print(f"    p50:  {overhead_p50:+.1f} ms")
    pct = overhead_mean / statistics.mean(baseline) * 100
    print(f"    relative (mean): {pct:+.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["baseline", "sched", "compare"],
                        default="sched")
    parser.add_argument("--port",          type=int, default=8000)
    parser.add_argument("--baseline-port", type=int, default=8000)
    parser.add_argument("--sched-port",    type=int, default=8001)
    parser.add_argument("--t-in",          type=int, default=16384)
    parser.add_argument("--reps",          type=int, default=20)
    parser.add_argument("--prompt-type",   choices=["diverse", "repeated"],
                        default="diverse")
    parser.add_argument("--warmup-reps",   type=int, default=3)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print(f"Bubble scheduler live test")
    print(f"  T_in={args.t_in}  reps={args.reps}  prompt={args.prompt_type}\n")

    if args.mode == "baseline":
        warmup(args.port, tok, args.t_in, args.warmup_reps)
        print(f"\nBaseline TTFT (port {args.port}):")
        ttfts = measure_ttft(args.port, tok, args.t_in, args.reps, args.prompt_type)
        report("baseline", ttfts)

    elif args.mode == "sched":
        print("NOTE: server must be running with VLLM_BUBBLE_SCHED_DEMO=1")
        print("      (set in run_qwen3_30b_a3b.sh or the launch command)\n")
        warmup(args.port, tok, args.t_in, args.warmup_reps)
        print(f"\nScheduler TTFT (port {args.port}):")
        ttfts = measure_ttft(args.port, tok, args.t_in, args.reps, args.prompt_type)
        report("sched", ttfts)

    elif args.mode == "compare":
        print(f"Baseline port: {args.baseline_port}   Sched port: {args.sched_port}\n")
        warmup(args.baseline_port, tok, args.t_in, args.warmup_reps)
        warmup(args.sched_port,    tok, args.t_in, args.warmup_reps)

        print(f"\nBaseline TTFT (port {args.baseline_port}):")
        base_ttfts = measure_ttft(args.baseline_port, tok, args.t_in,
                                  args.reps, args.prompt_type)

        print(f"\nScheduler TTFT (port {args.sched_port}):")
        sched_ttfts = measure_ttft(args.sched_port, tok, args.t_in,
                                   args.reps, args.prompt_type)

        report("baseline", base_ttfts)
        report("sched",    sched_ttfts)
        compare(base_ttfts, sched_ttfts)


if __name__ == "__main__":
    main()
