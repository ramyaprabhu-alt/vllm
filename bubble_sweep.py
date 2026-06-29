#!/usr/bin/env python3
"""
Sweep context lengths and batch sizes, measuring per-rank EP bubble timing.
Tracks prefill IDs to reliably capture each config's forward pass.

Usage:
    python bubble_sweep.py [--port PORT] [--reps N]
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

# (T_in, BS) — total_tokens = T_in * BS, must fit in max_num_batched_tokens=32768
CONFIGS = [
    (256,   1), (256,   4), (256,  16), (256,  64),
    (512,   1), (512,   4), (512,  16), (512,  32),
    (1024,  1), (1024,  4), (1024,  8), (1024, 16),
    (2048,  1), (2048,  4), (2048,  8),
    (4096,  1), (4096,  4), (4096,  8),
    (8192,  1), (8192,  2), (8192,  4),
    (16384, 1), (16384, 2),
]


def build_prompt(tokenizer, n_tokens: int) -> str:
    chunk = (
        "The study of distributed systems encompasses design and analysis "
        "of algorithms for networks of computers. Expert parallelism in large "
        "language models partitions MoE expert layers across GPU ranks. "
    )
    text = chunk
    while len(tokenizer.encode(text)) < n_tokens:
        text += chunk
    ids = tokenizer.encode(text)[:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def send_one(prompt: str, port: int, results: list, idx: int):
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        results[idx] = time.perf_counter() - t0
    except Exception as e:
        results[idx] = None
        print(f"      req {idx} failed: {e}")


def get_max_prefill_ids() -> dict[str, int]:
    """Return current max prefill_id per rank file."""
    ids = {}
    for path in glob.glob("/tmp/vllm_bubble_rank*.json"):
        label = os.path.basename(path).replace("vllm_bubble_rank","").replace(".json","")
        try:
            with open(path) as f:
                recs = json.load(f)
            ids[label] = max(r["prefill_id"] for r in recs) if recs else 0
        except Exception:
            ids[label] = 0
    return ids


def wait_for_new_prefill(old_ids: dict[str, int], timeout: float = 10.0) -> dict[str, int] | None:
    """Poll until every known rank has a prefill_id > old_ids."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        new_ids = get_max_prefill_ids()
        # Need all ranks that existed before to have incremented
        if old_ids and all(new_ids.get(k, 0) > old_ids.get(k, 0) for k in old_ids):
            return new_ids
        # Also handle first-ever request (no old files)
        if not old_ids and new_ids:
            return new_ids
        time.sleep(0.05)
    return None


def get_prefill(rank_file: str, prefill_id: int) -> list[dict]:
    try:
        with open(rank_file) as f:
            recs = json.load(f)
        return [r for r in recs if r["prefill_id"] == prefill_id]
    except Exception:
        return []


def stats_for(recs: list[dict]) -> dict:
    if not recs:
        return {}
    ffn = [r["ffn_ms"] for r in recs]
    ar  = [r["allreduce_ms"] for r in recs]
    s = sorted(ffn)
    n = len(s)
    return {
        "n_tokens": recs[0]["n_tokens"],
        "n_layers": len(recs),
        "ffn_mean": statistics.mean(ffn),
        "ffn_sum":  sum(ffn),
        "ffn_p50":  s[n//2],
        "ffn_p95":  s[int(.95*n)],
        "ffn_max":  s[-1],
        "ar_mean":  statistics.mean(ar),
        "ar_sum":   sum(ar),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    prompt_cache: dict[int, str] = {}
    for t in sorted(set(t for t, _ in CONFIGS)):
        p = build_prompt(tok, t)
        actual = len(tok.encode(p))
        prompt_cache[t] = p
        print(f"  T_in={t:6d} → {actual} actual tokens")

    # Warm up prefill_id tracker (server may have done memory profiling)
    time.sleep(0.5)
    baseline_ids = get_max_prefill_ids()
    print(f"\nBaseline prefill_ids: {baseline_ids}")

    hdr = (f"\n{'T_in':>6} {'BS':>4} {'total':>6} {'actual_tok':>10} "
           f"{'lat_ms':>7} "
           f"{'r0_ffn_sum':>11} {'r0_ffn_p50':>10} {'r0_ar_mean':>10} "
           f"{'r1_ffn_sum':>11} {'r1_ffn_p50':>10} {'r1_ar_mean':>10} "
           f"{'imbal%':>7} {'ar%':>5}")
    print(hdr)
    print("-" * len(hdr))

    all_results = []

    for (t_in, bs) in CONFIGS:
        prompt = prompt_cache[t_in]
        rep_data = []

        for rep in range(args.reps):
            old_ids = get_max_prefill_ids()

            # Unique suffix per rep to bust prefix cache
            unique = f" [sweep-rep-{t_in}-{bs}-{rep}-{time.time_ns()}]"
            rep_prompt = prompt + unique

            results = [None] * bs
            threads = [threading.Thread(target=send_one, args=(rep_prompt, args.port, results, i))
                       for i in range(bs)]
            t0 = time.perf_counter()
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            lat = max((r for r in results if r is not None), default=None)
            if lat is None:
                continue

            # Wait for profiling to flush (all 48 layers done)
            new_ids = wait_for_new_prefill(old_ids, timeout=15.0)
            if new_ids is None:
                print(f"  [{t_in}×{bs} rep{rep}] timeout waiting for profile")
                continue

            # Collect per-rank stats for this specific prefill
            rank_stats = {}
            for path in sorted(glob.glob("/tmp/vllm_bubble_rank*.json")):
                label = os.path.basename(path).replace("vllm_bubble_rank","").replace(".json","")
                pid = new_ids.get(label, 0)
                recs = get_prefill(path, pid)
                if recs:
                    rank_stats[label] = stats_for(recs)

            if len(rank_stats) < 2:
                print(f"  [{t_in}×{bs} rep{rep}] only {len(rank_stats)} ranks profiled")
                continue

            rep_data.append({"lat_ms": lat * 1000, "ranks": rank_stats})

        if not rep_data:
            print(f"  {t_in:>6} {bs:>4}  [all reps failed]")
            continue

        def ravg(rank, field):
            vals = [d["ranks"][rank][field] for d in rep_data if rank in d["ranks"]]
            return statistics.mean(vals) if vals else 0

        r0 = rep_data[0]["ranks"].get("0", {})
        r1 = rep_data[0]["ranks"].get("1", {})
        if not r0 or not r1:
            print(f"  [{t_in}×{bs}] missing rank(s): {list(rep_data[0]['ranks'].keys())}")
            continue

        actual_tok = ravg("0", "n_tokens")
        lat   = statistics.mean(d["lat_ms"] for d in rep_data)
        r0_fs = ravg("0", "ffn_sum");  r0_fp50 = ravg("0", "ffn_p50"); r0_ar = ravg("0", "ar_mean")
        r1_fs = ravg("1", "ffn_sum");  r1_fp50 = ravg("1", "ffn_p50"); r1_ar = ravg("1", "ar_mean")

        imbal  = abs(r0_fs - r1_fs) / max(r0_fs, r1_fs) * 100
        ar_frac = (r0_ar * 48) / (r0_fs + r0_ar * 48) * 100

        print(f"{t_in:>6} {bs:>4} {t_in*bs:>6} {actual_tok:>10.0f} "
              f"{lat:>7.0f} "
              f"{r0_fs:>10.1f}ms {r0_fp50:>9.2f}ms {r0_ar:>9.2f}ms "
              f"{r1_fs:>10.1f}ms {r1_fp50:>9.2f}ms {r1_ar:>9.2f}ms "
              f"{imbal:>6.1f}% {ar_frac:>4.1f}%")

        all_results.append({
            "t_in": t_in, "bs": bs, "total_tok": t_in * bs,
            "actual_tok": actual_tok, "lat_ms": lat,
            "r0_ffn_sum": r0_fs, "r0_ffn_p50": r0_fp50, "r0_ar_mean": r0_ar,
            "r1_ffn_sum": r1_fs, "r1_ffn_p50": r1_fp50, "r1_ar_mean": r1_ar,
            "imbalance_pct": imbal, "ar_frac_pct": ar_frac,
        })

    out = "/tmp/bubble_sweep_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} configs to {out}")


if __name__ == "__main__":
    main()
