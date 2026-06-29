#!/usr/bin/env python3
"""
ft_fwd_placement_exp.py — measure TTFT and TPOT overhead of three FT forward
pass placement strategies in vLLM.

Three conditions (plus baseline):
  A  baseline  No FT forward at all.
  B  prefill   Sub-ops run synchronously on main stream per MoE layer during
               prefill.  Full FT fwd latency added to TTFT.
  C  decode    Sub-ops dispatched on a low-priority stream during decode steps.
               TPOT overhead; minimal TTFT impact.
  D  bubble    Sub-ops scheduled into per-layer TP all_reduce bubbles.
               Near-zero overhead if bubbles cover all sub-ops.

Usage
-----
Each condition requires a server running with the matching VLLM_FT_FWD_MODE:

  # Condition A — baseline:
  VLLM_FT_FWD_MODE=off python -m vllm.entrypoints.openai.api_server \\
      --model /mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B \\
      --tensor-parallel-size 2 --enforce-eager &

  # Condition B — prefill:
  VLLM_FT_FWD_MODE=prefill VLLM_FT_FWD_DEMO=1 VLLM_FT_FWD_FILLS=9 \\
      python -m vllm.entrypoints.openai.api_server ... &

  # Condition C — decode:
  VLLM_FT_FWD_MODE=decode VLLM_FT_FWD_DEMO=1 VLLM_FT_FWD_FILLS=9 \\
      python -m vllm.entrypoints.openai.api_server ... &

  # Condition D — bubble:
  VLLM_FT_FWD_MODE=bubble VLLM_FT_FWD_DEMO=1 VLLM_FT_FWD_FILLS=1 \\
      python -m vllm.entrypoints.openai.api_server ... &

Then measure each condition:
  python ft_fwd_placement_exp.py --condition A --port 8000
  python ft_fwd_placement_exp.py --condition B --port 8001
  ...
  python ft_fwd_placement_exp.py --condition C --port 8002 --measure-tpot

Or run all on one server sequentially (restart between conditions):
  python ft_fwd_placement_exp.py --condition A --port 8000 --save results_A.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

MODEL = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
FWD_COMPLETIONS_LOG = "/tmp/vllm_fwd_completions.log"
COMBINED_COMPLETIONS_LOG = "/tmp/vllm_combined_completions.log"
T_FT = 128  # tokens per FT forward pass (must match VLLM_FT_FWD_DEMO build)

# ── Prompt builder ────────────────────────────────────────────────────────────

_DIVERSE_TEXT = (
    "The mitochondria generates ATP through oxidative phosphorylation. "
    "def merge_sort(arr): return arr if len(arr)<=1 else merge(merge_sort(arr[:len(arr)//2]),merge_sort(arr[len(arr)//2:])). "
    "SELECT name, COUNT(*) FROM employees GROUP BY department HAVING COUNT(*)>5. "
    "The Riemann hypothesis states all non-trivial zeros of zeta(s) have real part 1/2. "
    "import torch; x = torch.randn(128, 2048, device='cuda', dtype=torch.bfloat16). "
    "Paris is the capital of France and home to the Eiffel Tower built in 1889. "
)


def build_prompt(tokenizer, target_tokens: int) -> str:
    base_ids = tokenizer.encode(_DIVERSE_TEXT, add_special_tokens=False)
    pad_id   = tokenizer.encode("the ", add_special_tokens=False)[:1]
    needed   = max(0, target_tokens - len(base_ids))
    ids      = base_ids + pad_id * needed
    return tokenizer.decode(ids[:target_tokens])


# ── Request senders ───────────────────────────────────────────────────────────

def send_ttft(prompt: str, port: int) -> float | None:
    """Non-streaming request; total response time ≈ TTFT for max_tokens=4."""
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


def send_tpot(prompt: str, port: int, max_tokens: int = 100) -> tuple[float, float] | None:
    """Streaming request; returns (ttft_ms, mean_tpot_ms).

    Uses SSE streaming to timestamp first and subsequent tokens.
    Returns None on failure.
    """
    t0 = time.perf_counter()
    ttft_ms = None
    token_times = []
    try:
        with requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    now = (time.perf_counter() - t0) * 1e3
                    if ttft_ms is None:
                        ttft_ms = now
                    else:
                        token_times.append(now)
        if ttft_ms is None:
            return None
        mean_tpot = statistics.mean(token_times) / len(token_times) if len(token_times) > 1 else 0.0
        # mean_tpot: average gap between consecutive tokens
        if len(token_times) >= 2:
            gaps = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
            mean_tpot = statistics.mean(gaps)
        return ttft_ms, mean_tpot
    except Exception as e:
        print(f"    streaming error: {e}")
        return None


# ── Measurement loops ─────────────────────────────────────────────────────────

def measure_ttft(port: int, tokenizer, t_in: int, reps: int) -> list[float]:
    results = []
    for i in range(reps):
        prompt = build_prompt(tokenizer, t_in) + f" [fwd-exp-ttft-{i}-{time.time_ns()}]"
        ms = send_ttft(prompt, port)
        if ms is not None:
            results.append(ms)
            print(f"    TTFT rep {i:2d}: {ms:.1f} ms")
    return results


def measure_tpot(port: int, tokenizer, t_in: int, reps: int,
                 max_tokens: int = 100) -> tuple[list[float], list[float]]:
    """Returns (ttft_list, tpot_list)."""
    ttfts, tpots = [], []
    for i in range(reps):
        prompt = build_prompt(tokenizer, t_in) + f" [fwd-exp-tpot-{i}-{time.time_ns()}]"
        res = send_tpot(prompt, port, max_tokens)
        if res is not None:
            ttft_ms, tpot_ms = res
            ttfts.append(ttft_ms)
            tpots.append(tpot_ms)
            print(f"    TPOT rep {i:2d}: TTFT={ttft_ms:.1f}ms  TPOT={tpot_ms:.2f}ms/tok")
    return ttfts, tpots


def warmup(port: int, tokenizer, t_in: int, n: int = 3):
    print(f"  Warming up (port {port}, {n} requests, t_in={t_in})...")
    prompt = build_prompt(tokenizer, t_in)
    for _ in range(n):
        send_ttft(prompt + f" [warmup-{time.time_ns()}]", port)


# ── Statistics ────────────────────────────────────────────────────────────────

def pct(s: list[float], p: float) -> float:
    idx = min(int(p * len(s)), len(s) - 1)
    return sorted(s)[idx]


def report_metric(label: str, vals: list[float], unit: str = "ms"):
    if not vals:
        print(f"  {label}: no data")
        return
    s = sorted(vals)
    n = len(s)
    print(f"  {label:30s}  mean={statistics.mean(s):.1f}{unit}  "
          f"p50={s[n//2]:.1f}{unit}  p95={s[int(0.95*n)]:.1f}{unit}  "
          f"p99={s[min(int(0.99*n), n-1)]:.1f}{unit}  "
          f"min={s[0]:.1f}{unit}  max={s[-1]:.1f}{unit}")


def clear_fwd_log():
    try:
        Path(FWD_COMPLETIONS_LOG).unlink(missing_ok=True)
    except Exception:
        pass


def clear_combined_log():
    try:
        Path(COMBINED_COMPLETIONS_LOG).unlink(missing_ok=True)
    except Exception:
        pass


def read_fwd_throughput(wall_start: float, wall_end: float, t_ft: int = T_FT) -> dict:
    """Parse the completion log written by _fwd_demo_resubmit in each worker.

    Each line is a Unix timestamp (float) written when one forward pass
    (all 432 sub-ops) finishes.  Two worker ranks each write their own line,
    so we deduplicate by rounding to the nearest 10 ms.

    Returns a dict with n_passes, duration_s, passes_per_sec, tokens_per_sec,
    and the raw timestamps list.
    """
    try:
        raw = Path(FWD_COMPLETIONS_LOG).read_text().strip().splitlines()
    except FileNotFoundError:
        return {"error": "log not found — server may not have VLLM_FT_FWD_DEMO=1"}

    timestamps = []
    for line in raw:
        try:
            t = float(line.strip())
            if wall_start <= t <= wall_end:
                timestamps.append(t)
        except ValueError:
            pass

    if not timestamps:
        return {"n_passes_raw": 0, "n_passes_dedup": 0, "duration_s": wall_end - wall_start,
                "passes_per_sec": 0.0, "tokens_per_sec": 0.0, "timestamps": []}

    # Deduplicate: both TP ranks write a completion; they finish within a few ms
    # of each other.  Keep only one entry per 50 ms window.
    timestamps_sorted = sorted(timestamps)
    deduped = [timestamps_sorted[0]]
    for t in timestamps_sorted[1:]:
        if t - deduped[-1] > 0.05:  # 50 ms gap = different pass
            deduped.append(t)

    duration_s = wall_end - wall_start
    passes_per_sec = len(deduped) / duration_s if duration_s > 0 else 0.0
    tokens_per_sec = passes_per_sec * t_ft

    inter_pass = []
    for i in range(1, len(deduped)):
        inter_pass.append(deduped[i] - deduped[i - 1])

    return {
        "n_passes_raw":    len(timestamps),
        "n_passes_dedup":  len(deduped),
        "duration_s":      duration_s,
        "passes_per_sec":  passes_per_sec,
        "tokens_per_sec":  tokens_per_sec,
        "mean_inter_pass_s": statistics.mean(inter_pass) if inter_pass else None,
        "timestamps":      deduped,
    }


def read_combined_throughput(wall_start: float, wall_end: float, t_ft: int = T_FT) -> dict:
    """Parse the combined completion log written by _combined_bwd_done / _combined_dd_done.

    Each line is a Unix timestamp written when a full fwd+bwd training step
    completes.  Two TP ranks each write, so deduplicate by 50 ms window.
    """
    try:
        raw = Path(COMBINED_COMPLETIONS_LOG).read_text().strip().splitlines()
    except FileNotFoundError:
        return {"error": "log not found — server may not have VLLM_FT_COMBINED_MODE set"}

    timestamps = []
    for line in raw:
        try:
            t = float(line.strip())
            if wall_start <= t <= wall_end:
                timestamps.append(t)
        except ValueError:
            pass

    if not timestamps:
        return {"n_steps_raw": 0, "n_steps_dedup": 0, "duration_s": wall_end - wall_start,
                "steps_per_sec": 0.0, "tokens_per_sec": 0.0, "timestamps": []}

    timestamps_sorted = sorted(timestamps)
    deduped = [timestamps_sorted[0]]
    for t in timestamps_sorted[1:]:
        if t - deduped[-1] > 0.05:
            deduped.append(t)

    duration_s = wall_end - wall_start
    steps_per_sec = len(deduped) / duration_s if duration_s > 0 else 0.0
    inter_step = [deduped[i] - deduped[i-1] for i in range(1, len(deduped))]

    return {
        "n_steps_raw":    len(timestamps),
        "n_steps_dedup":  len(deduped),
        "duration_s":     duration_s,
        "steps_per_sec":  steps_per_sec,
        "tokens_per_sec": steps_per_sec * t_ft,
        "mean_inter_step_s": statistics.mean(inter_step) if inter_step else None,
        "timestamps":     deduped,
    }


def report_combined_throughput(condition: str, thr: dict):
    if "error" in thr:
        print(f"  FT training-step throughput: {thr['error']}")
        return
    n   = thr["n_steps_dedup"]
    dur = thr["duration_s"]
    sps = thr["steps_per_sec"]
    tps = thr["tokens_per_sec"]
    ist = thr["mean_inter_step_s"]
    print(f"  [{condition}] FT train-step (fwd+bwd)  "
          f"{n} steps in {dur:.1f}s  →  {sps:.3f} steps/sec  "
          f"({tps:.1f} FT-tokens/sec)")
    if ist is not None:
        print(f"  [{condition}] FT mean inter-step       {ist*1000:.0f} ms")


def report_fwd_throughput(condition: str, thr: dict):
    if "error" in thr:
        print(f"  FT-fwd throughput: {thr['error']}")
        return
    n = thr["n_passes_dedup"]
    dur = thr["duration_s"]
    pps = thr["passes_per_sec"]
    tps = thr["tokens_per_sec"]
    ip  = thr["mean_inter_pass_s"]
    print(f"  [{condition}] FT-fwd passes          "
          f"{n} passes in {dur:.1f}s  →  {pps:.3f} passes/sec  "
          f"({tps:.1f} FT-tokens/sec)")
    if ip is not None:
        print(f"  [{condition}] FT-fwd mean inter-pass  {ip*1000:.0f} ms")


def overhead_vs(label: str, vals: list[float], baseline: list[float]):
    if not vals or not baseline:
        return
    oh_mean = statistics.mean(vals) - statistics.mean(baseline)
    oh_p50  = pct(vals, 0.5) - pct(baseline, 0.5)
    pct_rel = oh_mean / statistics.mean(baseline) * 100
    print(f"  {label:30s}  overhead  mean={oh_mean:+.1f}ms  p50={oh_p50:+.1f}ms  ({pct_rel:+.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--condition", choices=["A", "B", "C", "D", "BD", "CD", "DD"],
                        required=True,
                        help="Which placement strategy to measure. "
                             "A/B/C/D = single-pass modes. "
                             "BD/CD/DD = combined fwd+bwd training step.")
    parser.add_argument("--port",         type=int,   default=8000)
    parser.add_argument("--t-in",         type=int,   default=8192,
                        help="Prompt length in tokens for TTFT measurement")
    parser.add_argument("--t-in-tpot",    type=int,   default=128,
                        help="Prompt length for TPOT measurement (short = more decode)")
    parser.add_argument("--reps",         type=int,   default=20,
                        help="Number of requests per metric")
    parser.add_argument("--max-tokens",   type=int,   default=100,
                        help="Output tokens for TPOT measurement")
    parser.add_argument("--measure-tpot", action="store_true",
                        help="Also run the TPOT measurement pass")
    parser.add_argument("--warmup-reps",  type=int,   default=3)
    parser.add_argument("--baseline-json", type=str,  default=None,
                        help="Path to saved condition-A JSON for overhead comparison")
    parser.add_argument("--save",         type=str,   default=None,
                        help="Save results to this JSON file")
    args = parser.parse_args()

    CONDITION_LABELS = {
        "A":  "A  baseline  (no FT, VLLM_FT_FWD_MODE=off)",
        "B":  "B  prefill   (fwd sync on main stream, VLLM_FT_FWD_MODE=prefill)",
        "C":  "C  decode    (fwd async low-pri stream, VLLM_FT_FWD_MODE=decode)",
        "D":  "D  bubble    (fwd in TP bubbles, VLLM_FT_FWD_MODE=bubble)",
        "BD": "BD B+D        (fwd sync prefill + bwd in bubbles, VLLM_FT_COMBINED_MODE=B+D)",
        "CD": "CD C+D        (fwd decode stream + bwd in bubbles, VLLM_FT_COMBINED_MODE=C+D)",
        "DD": "DD D+D        (fwd+bwd concatenated in bubbles, VLLM_FT_COMBINED_MODE=D+D)",
    }
    # Server env var required for each condition
    COMBINED_CONDITIONS = {"BD", "CD", "DD"}
    MODE_NEEDED = {
        "A": "VLLM_FT_FWD_MODE=off",
        "B": "VLLM_FT_FWD_MODE=prefill  VLLM_FT_FWD_DEMO=1",
        "C": "VLLM_FT_FWD_MODE=decode   VLLM_FT_FWD_DEMO=1",
        "D": "VLLM_FT_FWD_MODE=bubble   VLLM_FT_FWD_DEMO=1",
        "BD": "VLLM_FT_COMBINED_MODE=B+D",
        "CD": "VLLM_FT_COMBINED_MODE=C+D",
        "DD": "VLLM_FT_COMBINED_MODE=D+D",
    }

    print(f"\nFT Forward Placement Experiment")
    print(f"{'='*60}")
    print(f"  Condition : {CONDITION_LABELS[args.condition]}")
    print(f"  Port      : {args.port}")
    print(f"  T_in TTFT : {args.t_in} tokens")
    print(f"  Reps      : {args.reps}")
    if args.measure_tpot:
        print(f"  T_in TPOT : {args.t_in_tpot} tokens, max_out={args.max_tokens}")
    print()
    print(f"  NOTE: Server must be running with {MODE_NEEDED[args.condition]}")
    if args.condition in COMBINED_CONDITIONS:
        print(f"        (combined mode: full fwd+bwd training step pairs)")
    print()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # Clear the relevant completion log so we only count passes from this run.
    if args.condition in ("B", "C", "D"):
        clear_fwd_log()
    if args.condition in COMBINED_CONDITIONS:
        clear_combined_log()

    warmup(args.port, tok, args.t_in, args.warmup_reps)

    # ── TTFT measurement ──────────────────────────────────────────────────────
    print(f"\nTTFT measurement (t_in={args.t_in}, max_tokens=4):")
    wall_start = time.time()
    ttfts = measure_ttft(args.port, tok, args.t_in, args.reps)
    wall_end_ttft = time.time()
    print()
    report_metric(f"[{args.condition}] TTFT", ttfts)

    # ── TPOT measurement (optional) ───────────────────────────────────────────
    tpots_ttft, tpots = [], []
    if args.measure_tpot:
        print(f"\nTPOT measurement (t_in={args.t_in_tpot}, max_tokens={args.max_tokens}):")
        tpots_ttft, tpots = measure_tpot(
            args.port, tok, args.t_in_tpot, args.reps, args.max_tokens)
        print()
        report_metric(f"[{args.condition}] TPOT-run TTFT", tpots_ttft)
        report_metric(f"[{args.condition}] TPOT",          tpots, unit="ms/tok")

    wall_end = time.time()

    # ── FT throughput ─────────────────────────────────────────────────────────
    if args.condition in ("B", "C", "D"):
        print()
        thr_ttft = read_fwd_throughput(wall_start, wall_end_ttft)
        thr_all  = read_fwd_throughput(wall_start, wall_end)
        report_fwd_throughput(args.condition + " TTFT-window", thr_ttft)
        if args.measure_tpot:
            report_fwd_throughput(args.condition + " full-window ", thr_all)
    if args.condition in COMBINED_CONDITIONS:
        print()
        cthr_ttft = read_combined_throughput(wall_start, wall_end_ttft)
        cthr_all  = read_combined_throughput(wall_start, wall_end)
        report_combined_throughput(args.condition + " TTFT-window", cthr_ttft)
        if args.measure_tpot:
            report_combined_throughput(args.condition + " full-window ", cthr_all)

    # ── Overhead vs baseline ──────────────────────────────────────────────────
    if args.baseline_json:
        try:
            base = json.loads(Path(args.baseline_json).read_text())
            print()
            print("Overhead vs baseline (A):")
            overhead_vs(f"[{args.condition}] TTFT overhead", ttfts, base.get("ttft", []))
            if args.measure_tpot and base.get("tpot"):
                overhead_vs(f"[{args.condition}] TPOT overhead", tpots, base.get("tpot", []))
        except Exception as e:
            print(f"  (could not load baseline: {e})")

    # ── Save results ──────────────────────────────────────────────────────────
    if args.save:
        out = {
            "condition": args.condition,
            "mode": MODE_NEEDED[args.condition],
            "t_in": args.t_in,
            "reps": args.reps,
            "ttft": ttfts,
            "wall_start": wall_start,
            "wall_end": wall_end,
        }
        if args.condition in ("B", "C", "D"):
            out["fwd_throughput_ttft_window"] = thr_ttft
            if args.measure_tpot:
                out["fwd_throughput_full_window"] = thr_all
        if args.condition in COMBINED_CONDITIONS:
            out["combined_throughput_ttft_window"] = cthr_ttft
            if args.measure_tpot:
                out["combined_throughput_full_window"] = cthr_all
        if args.measure_tpot:
            out["tpot"] = tpots
            out["tpot_ttft"] = tpots_ttft
            out["t_in_tpot"] = args.t_in_tpot
        Path(args.save).write_text(json.dumps(out, indent=2))
        print(f"\n  Results saved to {args.save}")

    print()


if __name__ == "__main__":
    main()
