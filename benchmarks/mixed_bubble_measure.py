#!/usr/bin/env python3
"""
Measure rank-imbalance bubble sizes with prompts that generate bubbles on
BOTH GPUs, not just rank 1.

Repeated-token prompts ("the the the...") concentrate routing on whichever
experts happen to process that token, keeping rank 0 busy and rank 1 idle
every layer.  Diverse prompts spread routing across both ranks' expert
subsets, so the fast/slow role can swap per layer: some layers bubble on
rank 0, some on rank 1.

Per-layer bubble attribution:
  r1_allreduce_ms > r0_allreduce_ms  →  rank 1 entered barrier first
                                         rank 1 had the bubble that layer
  r0_allreduce_ms > r1_allreduce_ms  →  rank 0 entered barrier first
                                         rank 0 had the bubble that layer

Usage:
    # Server must be running with VLLM_BUBBLE_PROFILE=1
    python mixed_bubble_measure.py [--port 8000] [--tokens 8192] [--reps 3]
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


# ── Prompt library ────────────────────────────────────────────────────────────
# Each entry: (label, template)
# template is a string; {PAD} is replaced with repeated filler to hit T_in.
PROMPT_TEMPLATES = [
    (
        "repeated_the",
        # Baseline: one token repeated — all tokens route identically,
        # routing imbalance is maximised, bubble always on same rank.
        "{PAD_THE}",
    ),
    (
        "english_wiki",
        # Diverse English prose — varied vocabulary activates different experts
        # across layers, more likely to split bubbles across both ranks.
        "The mitochondria is a membrane-bound organelle found in the cytoplasm "
        "of eukaryotic cells. It generates most of the cell's supply of "
        "adenosine triphosphate, used as a source of chemical energy. "
        "Beyond supplying cellular energy, mitochondria are involved in other "
        "tasks, such as signaling, cellular differentiation, and cell death, "
        "as well as maintaining control of the cell cycle and cell growth. "
        "The organelle was first discovered by Richard Altmann in 1894, who "
        "called them bioblasts. The name mitochondrion was coined by Carl "
        "Benda in 1898. Mitochondria have two membranes, an outer membrane "
        "and an inner membrane. The inner membrane has large numbers of "
        "invaginations called cristae. "
        "{PAD_WIKI}",
    ),
    (
        "python_code",
        # Code tokens (keywords, punctuation, identifiers) have distinct
        # distributions from natural language, which may activate different
        # expert subsets.
        "def quicksort(arr):\n"
        "    if len(arr) <= 1:\n"
        "        return arr\n"
        "    pivot = arr[len(arr) // 2]\n"
        "    left = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n"
        "    right = [x for x in arr if x > pivot]\n"
        "    return quicksort(left) + middle + quicksort(right)\n\n"
        "class BinaryTree:\n"
        "    def __init__(self, value):\n"
        "        self.value = value\n"
        "        self.left = None\n"
        "        self.right = None\n\n"
        "    def insert(self, value):\n"
        "        if value < self.value:\n"
        "            if self.left is None:\n"
        "                self.left = BinaryTree(value)\n"
        "            else:\n"
        "                self.left.insert(value)\n"
        "        else:\n"
        "            if self.right is None:\n"
        "                self.right = BinaryTree(value)\n"
        "            else:\n"
        "                self.right.insert(value)\n"
        "{PAD_CODE}",
    ),
    (
        "mixed_domains",
        # Alternating sentence types — the shift in semantic domain between
        # sentences forces routing to different expert regions mid-sequence.
        "Paris is the capital of France. "
        "def factorial(n): return 1 if n==0 else n*factorial(n-1). "
        "The square root of 144 is 12. "
        "Photosynthesis converts light energy into chemical energy. "
        "SELECT * FROM users WHERE age > 30 ORDER BY name; "
        "The speed of light is approximately 299,792,458 metres per second. "
        "import numpy as np; x = np.linspace(0, 2*np.pi, 100). "
        "In 1969, humans first landed on the Moon during the Apollo 11 mission. "
        "The Fibonacci sequence begins 0, 1, 1, 2, 3, 5, 8, 13, 21, 34. "
        "Water molecules consist of two hydrogen atoms bonded to one oxygen atom. "
        "{PAD_MIX}",
    ),
]


def build_prompt(tokenizer, template_label: str, template: str,
                 target_tokens: int) -> str:
    """Expand template to approximately target_tokens tokens."""
    filler_word = "the "
    filler_id = tokenizer.encode(filler_word, add_special_tokens=False)[:1]

    # Tokenize the fixed part to see how many tokens it takes
    fixed = template
    for pad_key in ["{PAD_THE}", "{PAD_WIKI}", "{PAD_CODE}", "{PAD_MIX}"]:
        fixed = fixed.replace(pad_key, "")
    fixed_tokens = len(tokenizer.encode(fixed, add_special_tokens=False))

    pad_needed = max(0, target_tokens - fixed_tokens)
    pad_text = tokenizer.decode(filler_id * pad_needed, skip_special_tokens=True)

    prompt = template
    for pad_key in ["{PAD_THE}", "{PAD_WIKI}", "{PAD_CODE}", "{PAD_MIX}"]:
        prompt = prompt.replace(pad_key, pad_text)

    return prompt


# ── Server interaction ────────────────────────────────────────────────────────

def send_request(prompt: str, port: int) -> bool:
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
            timeout=300,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"    request failed: {e}")
        return False


def get_max_pids():
    ids = {}
    for path in glob.glob("/tmp/vllm_bubble_rank*.json"):
        label = os.path.basename(path).replace("vllm_bubble_rank", "").replace(".json", "")
        try:
            recs = json.load(open(path))
            ids[label] = max(r["prefill_id"] for r in recs) if recs else 0
        except Exception:
            ids[label] = 0
    return ids


def wait_new_prefill(old_pids: dict, timeout: float = 30.0) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        new = get_max_pids()
        if old_pids and all(new.get(k, 0) > old_pids.get(k, 0) for k in old_pids):
            return new
        if not old_pids and new:
            return new
        time.sleep(0.05)
    return None


def load_prefill(rank: int, pid: int) -> list[dict]:
    path = f"/tmp/vllm_bubble_rank{rank}.json"
    try:
        recs = json.load(open(path))
        return sorted(
            [r for r in recs if r["prefill_id"] == pid],
            key=lambda r: r["layer"],
        )
    except Exception:
        return []


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_both_ranks(recs_r0: list[dict], recs_r1: list[dict],
                       label: str, t_in: int) -> dict:
    """Per-layer bubble attribution for both ranks."""
    n = min(len(recs_r0), len(recs_r1))
    if n == 0:
        return {}

    r0_bubbles = []   # bubble size when rank 0 is the fast (idle) rank
    r1_bubbles = []   # bubble size when rank 1 is the fast (idle) rank
    r0_layers  = []   # layer indices where rank 0 had the bubble
    r1_layers  = []   # layer indices where rank 1 had the bubble

    for i in range(n):
        diff = recs_r1[i]["allreduce_ms"] - recs_r0[i]["allreduce_ms"]
        if diff > 0.0:
            # Rank 1 spent more time in the all_reduce → it entered first
            # → rank 1 was the fast rank → rank 1 had the bubble
            r1_bubbles.append(diff)
            r1_layers.append(i + 1)
            r0_bubbles.append(0.0)
        elif diff < 0.0:
            # Rank 0 entered the all_reduce first → rank 0 had the bubble
            r0_bubbles.append(-diff)
            r0_layers.append(i + 1)
            r1_bubbles.append(0.0)
        else:
            r0_bubbles.append(0.0)
            r1_bubbles.append(0.0)

    r0_nonzero = [b for b in r0_bubbles if b > 0.1]
    r1_nonzero = [b for b in r1_bubbles if b > 0.1]

    def fmt(vals):
        if not vals:
            return "no exploitable layers"
        s = sorted(vals)
        return (f"n={len(s):2d}  p50={statistics.median(s):.2f}ms  "
                f"max={s[-1]:.2f}ms  total={sum(s):.1f}ms")

    print(f"\n  [{label}]  T_in={t_in}  MoE layers={n}")
    print(f"    Rank 0 bubble (rank 0 fast):  {fmt(r0_nonzero)}")
    if r0_layers:
        print(f"      layers: {r0_layers[:20]}{'...' if len(r0_layers)>20 else ''}")
    print(f"    Rank 1 bubble (rank 1 fast):  {fmt(r1_nonzero)}")
    if r1_layers:
        print(f"      layers: {r1_layers[:20]}{'...' if len(r1_layers)>20 else ''}")
    print(f"    Distribution: {len(r0_nonzero)} layers on rank 0, "
          f"{len(r1_nonzero)} layers on rank 1, "
          f"{n - len(r0_nonzero) - len(r1_nonzero)} balanced")
    total_r0 = sum(r0_nonzero)
    total_r1 = sum(r1_nonzero)
    total = total_r0 + total_r1
    if total > 0:
        print(f"    Total bubble: {total:.1f}ms  "
              f"(rank 0: {total_r0:.1f}ms = {100*total_r0/total:.0f}%,  "
              f"rank 1: {total_r1:.1f}ms = {100*total_r1/total:.0f}%)")

    return {
        "label": label,
        "t_in": t_in,
        "n_layers": n,
        "r0_n": len(r0_nonzero),
        "r1_n": len(r1_nonzero),
        "r0_total": total_r0,
        "r1_total": total_r1,
        "r0_p50": statistics.median(r0_nonzero) if r0_nonzero else 0.0,
        "r1_p50": statistics.median(r1_nonzero) if r1_nonzero else 0.0,
        "r0_max": max(r0_nonzero) if r0_nonzero else 0.0,
        "r1_max": max(r1_nonzero) if r1_nonzero else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--reps",   type=int, default=3,
                        help="Repetitions per (prompt_type, T_in) combination")
    parser.add_argument("--prompts", nargs="+",
                        default=["repeated_the", "english_wiki", "python_code", "mixed_domains"],
                        help="Prompt types to test (subset of PROMPT_TEMPLATES labels)")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    selected = {label: tmpl for label, tmpl in PROMPT_TEMPLATES
                if label in args.prompts}
    if not selected:
        print(f"No matching prompt types. Available: {[l for l,_ in PROMPT_TEMPLATES]}")
        return

    all_results = []

    for t_in in args.tokens:
        print(f"\n{'='*80}")
        print(f"T_in = {t_in} tokens")
        print(f"{'='*80}")

        for label, template in selected.items():
            rep_results = []
            for rep in range(args.reps):
                prompt = build_prompt(tok, label, template, t_in)
                # Cache-busting suffix so vLLM does a fresh prefill each rep
                prompt += f" [mbm-{label}-{t_in}-{rep}-{time.time_ns()}]"

                old_pids = get_max_pids()
                t0 = time.perf_counter()
                ok = send_request(prompt, args.port)
                ttft_ms = (time.perf_counter() - t0) * 1e3
                if not ok:
                    continue

                new_pids = wait_new_prefill(old_pids, timeout=30)
                if new_pids is None:
                    print(f"    [{label}] rep {rep}: timeout waiting for profile data")
                    continue
                time.sleep(0.1)

                r0 = load_prefill(0, new_pids.get("0", 0))
                r1 = load_prefill(1, new_pids.get("1", 0))
                if not r0 or not r1:
                    print(f"    [{label}] rep {rep}: missing rank data")
                    continue

                result = analyze_both_ranks(r0, r1, f"{label} rep{rep}", t_in)
                if result:
                    result["ttft_ms"] = ttft_ms
                    rep_results.append(result)

            # Aggregate across reps
            if len(rep_results) > 1:
                print(f"\n  ── {label} aggregate ({len(rep_results)} reps) ──")
                r0_totals = [r["r0_total"] for r in rep_results]
                r1_totals = [r["r1_total"] for r in rep_results]
                r0_ns     = [r["r0_n"]     for r in rep_results]
                r1_ns     = [r["r1_n"]     for r in rep_results]
                print(f"    Rank 0 layers: {statistics.mean(r0_ns):.1f} ± {statistics.stdev(r0_ns):.1f}  "
                      f"total bubble: {statistics.mean(r0_totals):.1f} ± {statistics.stdev(r0_totals):.1f} ms")
                print(f"    Rank 1 layers: {statistics.mean(r1_ns):.1f} ± {statistics.stdev(r1_ns):.1f}  "
                      f"total bubble: {statistics.mean(r1_totals):.1f} ± {statistics.stdev(r1_totals):.1f} ms")

            all_results.extend(rep_results)

    # ── Summary table ─────────────────────────────────────────────────────────
    if not all_results:
        return

    print(f"\n\n{'='*90}")
    print("SUMMARY — bubble distribution across ranks per prompt type and T_in")
    print(f"{'='*90}")
    print(f"  {'prompt_type':<20} {'T_in':>6}  "
          f"{'r0_layers':>9} {'r0_total':>9} {'r0_p50':>7}  "
          f"{'r1_layers':>9} {'r1_total':>9} {'r1_p50':>7}")
    print(f"  {'-'*88}")

    seen = set()
    for r in all_results:
        key = (r["label"].rsplit(" rep", 1)[0], r["t_in"])
        if key in seen:
            continue
        seen.add(key)
        label_base = key[0]
        t = key[1]
        matching = [x for x in all_results
                    if x["label"].startswith(label_base) and x["t_in"] == t]
        r0_n = statistics.mean(x["r0_n"] for x in matching)
        r1_n = statistics.mean(x["r1_n"] for x in matching)
        r0_t = statistics.mean(x["r0_total"] for x in matching)
        r1_t = statistics.mean(x["r1_total"] for x in matching)
        r0_p = statistics.mean(x["r0_p50"] for x in matching)
        r1_p = statistics.mean(x["r1_p50"] for x in matching)
        print(f"  {label_base:<20} {t:>6}  "
              f"{r0_n:>9.1f} {r0_t:>9.1f} {r0_p:>7.2f}  "
              f"{r1_n:>9.1f} {r1_t:>9.1f} {r1_p:>7.2f}")


if __name__ == "__main__":
    main()
