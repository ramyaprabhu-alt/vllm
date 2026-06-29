#!/usr/bin/env python3
"""
Find bubble opportunities in vLLM EP forward pass (no EPLB).

Two bubble types:
  1. Rank-imbalance bubble: rank 1 waits at all_reduce while rank 0 finishes FFN
     = r1_allreduce_ms - r0_allreduce_ms per layer  (rank 1 perspective)
  2. Inter-MoE-layer gap: time between one layer's all_reduce exit and next layer's
     FFN entry = covers attention, norms, linear projections
     = next_layer.t_entry - prev_layer.t_exit  (wall-clock, per rank)

Usage:
    python find_bubbles.py [--tokens T_IN] [--bs BS] [--port PORT]
"""
import argparse
import glob
import json
import os
import statistics
import threading
import time

import requests
from transformers import AutoTokenizer

MODEL = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"


def build_prompt(tokenizer, n_tokens):
    # Repeat a single common word — same token ID repeated → maximum routing
    # imbalance since every token has the same hidden state and routes identically.
    single = "the "
    ids = tokenizer.encode(single, add_special_tokens=False)[:1]  # one token
    repeated_ids = ids * n_tokens
    return tokenizer.decode(repeated_ids, skip_special_tokens=True)


def send_batch(prompt, bs, port):
    results = [None] * bs
    def _req(i):
        resp = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={"model": MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 4, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False}},
            timeout=300)
        resp.raise_for_status()
        results[i] = True
    threads = [threading.Thread(target=_req, args=(i,)) for i in range(bs)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    return time.perf_counter() - t0


def get_max_pids():
    ids = {}
    for path in glob.glob("/tmp/vllm_bubble_rank*.json"):
        label = os.path.basename(path).replace("vllm_bubble_rank","").replace(".json","")
        try:
            recs = json.load(open(path))
            ids[label] = max(r["prefill_id"] for r in recs) if recs else 0
        except: ids[label] = 0
    return ids


def wait_new(old, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        new = get_max_pids()
        if old and all(new.get(k,0) > old.get(k,0) for k in old):
            return new
        if not old and new:
            return new
        time.sleep(0.05)
    return None


def load_prefill(path, pid):
    try:
        recs = json.load(open(path))
        return sorted([r for r in recs if r["prefill_id"] == pid],
                      key=lambda r: r["layer"])
    except: return []


def analyze(recs_r0, recs_r1, t_in, bs):
    n = min(len(recs_r0), len(recs_r1))
    if n == 0:
        print("No data"); return

    # ── Bubble type 1: rank-imbalance wait ───────────────────────────────────
    # Rank 1 arrives at all_reduce barrier first (FFN < rank 0).
    # Its all_reduce_ms includes: wait_for_r0 + NVLink_transfer.
    # Rank 0's all_reduce_ms is just the NVLink_transfer (arrives last).
    # So: wait = r1_ar - r0_ar  per layer  (positive when r1 faster)
    imbalance_wait = []
    for l in range(n):
        wait = recs_r1[l]["allreduce_ms"] - recs_r0[l]["allreduce_ms"]
        imbalance_wait.append(max(0.0, wait))

    # ── Bubble type 2: inter-MoE gap (attention+norm window) ─────────────────
    # gap[i] = entry_time[i+1] - exit_time[i]  (per rank, wall-clock ms)
    def inter_gaps(recs):
        gaps = []
        for i in range(len(recs) - 1):
            if "t_entry" in recs[i+1] and "t_exit" in recs[i]:
                gap_ms = (recs[i+1]["t_entry"] - recs[i]["t_exit"]) * 1000
                gaps.append(max(0.0, gap_ms))
        return gaps

    gaps_r0 = inter_gaps(recs_r0)
    gaps_r1 = inter_gaps(recs_r1)

    def stats(vals, label):
        if not vals: return
        s = sorted(vals)
        n = len(s)
        print(f"  {label:<38s}  n={n:3d}  "
              f"min={s[0]:6.2f}  p25={s[n//4]:6.2f}  p50={s[n//2]:6.2f}  "
              f"p75={s[3*n//4]:6.2f}  p95={s[int(.95*n)]:6.2f}  max={s[-1]:6.2f}  "
              f"sum={sum(vals):7.1f}  ms")

    tok = recs_r0[0]["n_tokens"] if recs_r0 else "?"
    print(f"\n{'='*90}")
    print(f"T_in={t_in}  BS={bs}  actual_tokens={tok}  MoE_layers={n}")
    print(f"{'='*90}\n")

    print("── Bubble type 1: rank-imbalance wait (rank 1 idle at all_reduce barrier) ──")
    stats(imbalance_wait, "rank1 wait/layer (ms)")
    nonzero = [w for w in imbalance_wait if w > 0.5]
    print(f"  layers with >0.5ms wait: {len(nonzero)}/{n}  "
          f"total exploitable: {sum(imbalance_wait):.1f}ms")

    print(f"\n── Bubble type 2: inter-MoE gap (attention + norms + projections) ──")
    stats(gaps_r0, "rank 0 inter-layer gap (ms)")
    stats(gaps_r1, "rank 1 inter-layer gap (ms)")
    if gaps_r0:
        print(f"  rank 0 total inter-MoE time: {sum(gaps_r0):.1f}ms  "
              f"(covers all attention layers)")

    # ── Summary ──────────────────────────────────────────────────────────────
    ffn_r0  = sum(r["ffn_ms"] for r in recs_r0[:n])
    ffn_r1  = sum(r["ffn_ms"] for r in recs_r1[:n])
    ar_r0   = sum(r["allreduce_ms"] for r in recs_r0[:n])
    ar_r1   = sum(r["allreduce_ms"] for r in recs_r1[:n])
    gap_tot = sum(gaps_r0) if gaps_r0 else 0
    total_bubble1 = sum(imbalance_wait)

    print(f"\n── Summary ──")
    print(f"  Rank 0: FFN={ffn_r0:.1f}ms  allreduce={ar_r0:.1f}ms")
    print(f"  Rank 1: FFN={ffn_r1:.1f}ms  allreduce={ar_r1:.1f}ms")
    print(f"  Imbalance gap (total rank1 wait): {total_bubble1:.1f}ms  "
          f"({100*total_bubble1/(ffn_r0+ar_r0+gap_tot):.1f}% of prefill)")
    print(f"  Inter-MoE gaps (attention window): {gap_tot:.1f}ms  "
          f"({100*gap_tot/(ffn_r0+ar_r0+gap_tot):.1f}% of prefill)")
    print(f"  FFN+allreduce: {ffn_r0+ar_r0:.1f}ms  "
          f"({100*(ffn_r0+ar_r0)/(ffn_r0+ar_r0+gap_tot):.1f}% of prefill)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192, 16384, 24576, 32000])
    parser.add_argument("--bs",     type=int, default=1)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    for t_in in args.tokens:
        prompt = build_prompt(tok, t_in)
        unique = f" [bubble-find-{t_in}-{args.bs}-{time.time_ns()}]"
        prompt = prompt + unique

        old = get_max_pids()
        ttft = send_batch(prompt, args.bs, args.port)
        new = wait_new(old, timeout=20)
        if new is None:
            print(f"T_in={t_in}: timeout"); continue

        time.sleep(0.2)
        r0 = load_prefill("/tmp/vllm_bubble_rank0.json", new.get("0", 0))
        r1 = load_prefill("/tmp/vllm_bubble_rank1.json", new.get("1", 0))

        if not r0 or not r1:
            print(f"T_in={t_in}: missing rank data"); continue

        analyze(r0, r1, t_in, args.bs)
        print(f"  TTFT: {ttft*1000:.0f}ms\n")


if __name__ == "__main__":
    main()
