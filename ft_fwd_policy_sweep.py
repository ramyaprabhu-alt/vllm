#!/usr/bin/env python3
"""
ft_fwd_policy_sweep.py — FT forward placement policy grid benchmark.

Sweeps T_in × decode_BS for conditions A / B / C / D, then emits a
per-cell policy recommendation and a summary CSV.

Conditions
----------
  A  baseline  no FT forward (VLLM_FT_FWD_MODE=off)
  B  prefill   FT fwd sync on main stream during prefill, FILLS=9
                 → 1 complete FT pass per inference request
  C  decode    FT fwd on low-priority stream during decode, FILLS=9
                 → pass accumulates over decode steps; completes within
                   a single request at max_tokens=100 (900 sub-ops > 432)
  D  bubble    FT fwd into TP all_reduce rank-imbalance gaps, FILLS=1
                 → 48 sub-ops per prefill; 1 pass per ~9 prefills (low
                   overhead, lower throughput)

FT throughput
-------------
/tmp/vllm_fwd_completions.log is written by each TP rank as a Unix
timestamp (float, one per line) when a full forward pass completes.
Both ranks write within a few ms of each other → deduplicate at 50 ms.
Throughput is computed over the measurement window for each (T_in, BS).
For D at small BS where < 1 pass completes per window, we fall back to
condition-level throughput (accumulated over the full T_in sweep).

Usage
-----
    # Full grid (runs all 4 conditions, ~90 min total)
    python ft_fwd_policy_sweep.py

    # Resume / run a single condition against an existing condition-A baseline
    python ft_fwd_policy_sweep.py --conditions B,C,D --skip-existing

    # Smaller grid for a quick check
    python ft_fwd_policy_sweep.py --t-ins 4096,16384 --batch-sizes 1,8 --n-reps 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

MODEL     = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
VENV_BIN  = "/mnt/nfs/home/ramya/vllm/.venv/bin"
PORT      = 8000
BASE_URL  = f"http://localhost:{PORT}"
COMPL_LOG = "/tmp/vllm_fwd_completions.log"

T_FT         = 128   # FT tokens per forward pass (VLLM_FT_FWD_DEMO default)
MAX_TOKENS   = 100   # decode steps per inference request
N_WARMUP     = 5     # warmup requests before each condition's grid

DEFAULT_T_INS      = [1024, 4096, 8192, 16384]
DEFAULT_BATCH_SIZES = [1, 4, 8, 16]
DEFAULT_N_REPS     = 5

# env vars + FILLS setting per condition
COND_ENV = {
    "A": {"VLLM_FT_FWD_MODE": "off"},
    "B": {"VLLM_FT_FWD_MODE": "prefill", "VLLM_FT_FWD_DEMO": "1",
          "VLLM_FT_FWD_FILLS": "9"},
    "C": {"VLLM_FT_FWD_MODE": "decode",  "VLLM_FT_FWD_DEMO": "1",
          "VLLM_FT_FWD_FILLS": "9"},
    "D": {"VLLM_FT_FWD_MODE": "bubble",  "VLLM_FT_FWD_DEMO": "1",
          "VLLM_FT_FWD_FILLS": "1"},
}

SERVER_FLAGS = [
    "--tensor-parallel-size", "2",
    "--enable-expert-parallel",
    "--enable-ep-weight-filter",
    "--all2all-backend", "allgather_reducescatter",
    "--moe-backend", "triton",
    "--dtype", "bfloat16",
    "--max-model-len", "32768",
    "--gpu-memory-utilization", "0.92",
    "--max-num-seqs", "256",
    "--max-num-batched-tokens", "32768",
    "--trust-remote-code",
    "--host", "0.0.0.0",
    "--port", str(PORT),
    "--enforce-eager",
    "--no-enable-chunked-prefill",
    "--no-enable-prefix-caching",
]

# ── Prompt builder ────────────────────────────────────────────────────────────

_DIVERSE = (
    "The mitochondria generates ATP through oxidative phosphorylation. "
    "def merge_sort(arr): return arr if len(arr)<=1 else merge(merge_sort(arr[:len(arr)//2]),merge_sort(arr[len(arr)//2:])). "
    "SELECT name, COUNT(*) FROM employees GROUP BY department HAVING COUNT(*)>5. "
    "The Riemann hypothesis states all non-trivial zeros of zeta(s) have real part 1/2. "
    "import torch; x = torch.randn(128, 2048, device='cuda', dtype=torch.bfloat16). "
    "Paris is the capital of France and home to the Eiffel Tower built in 1889. "
)


def build_prompt(tokenizer, target: int, salt: str = "") -> str:
    base = tokenizer.encode(_DIVERSE, add_special_tokens=False)
    pad  = tokenizer.encode("the ", add_special_tokens=False)[:1]
    ids  = base + pad * max(0, target - len(base))
    # Decode to exactly target tokens, then append salt to bust prefix cache.
    # Salt adds a few extra tokens but that's fine — T_in is approximate.
    prompt = tokenizer.decode(ids[:target])
    if salt:
        prompt += f" [{salt}]"
    return prompt


# ── Server management ─────────────────────────────────────────────────────────

def start_server(cond: str, log_path: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["PATH"] = f"{VENV_BIN}:{env.get('PATH', '')}"
    env.update(COND_ENV[cond])
    Path(COMPL_LOG).unlink(missing_ok=True)
    cmd = [f"{VENV_BIN}/vllm", "serve", MODEL] + SERVER_FLAGS
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf,
                                preexec_fn=os.setsid)
    print(f"[server] condition={cond} pid={proc.pid} log={log_path}")
    return proc


def wait_ready(timeout: int = 300) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server did not become ready in time")


def kill_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=40)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    print(f"[server] pid={proc.pid} stopped")
    time.sleep(6)   # CUDA context teardown


# ── Request sender ────────────────────────────────────────────────────────────

def send_streaming(prompt: str) -> dict | None:
    """
    Returns {"ttft_ms": float, "tpot_ms": float, "n_tokens": int} or None.
    TTFT = time to first content token.
    TPOT = mean inter-token latency over tokens 2..N.
    """
    t0 = time.perf_counter()
    ttft_ms = None
    tok_times: list[float] = []
    try:
        with requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model":        MODEL,
                "messages":     [{"role": "user", "content": prompt}],
                "max_tokens":   MAX_TOKENS,
                "temperature":  0,
                "stream":       True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            stream=True,
            timeout=600,
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode() if isinstance(raw, bytes) else raw
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                if choices[0].get("delta", {}).get("content"):
                    now = (time.perf_counter() - t0) * 1e3
                    if ttft_ms is None:
                        ttft_ms = now
                    tok_times.append(now)
    except Exception as e:
        print(f"  [request error] {e}")
        return None

    if ttft_ms is None or len(tok_times) < 2:
        return None

    itls = [tok_times[i] - tok_times[i - 1] for i in range(1, len(tok_times))]
    return {
        "ttft_ms":  ttft_ms,
        "tpot_ms":  statistics.mean(itls),
        "n_tokens": len(tok_times),
    }


# ── FT throughput (from completion log) ──────────────────────────────────────

def read_throughput(wall_start: float, wall_end: float) -> dict:
    """
    Counts deduped FT forward-pass completions in [wall_start, wall_end].
    Dedup: both TP ranks write a timestamp within a few ms → keep one per 50ms.
    Returns passes_dedup, duration_s, toks_per_sec (= passes × T_FT / dur).
    """
    try:
        raw = Path(COMPL_LOG).read_text().strip().splitlines()
    except FileNotFoundError:
        return {"passes": 0, "duration_s": wall_end - wall_start, "toks_per_sec": 0.0}

    ts_in_window = []
    for line in raw:
        try:
            t = float(line.strip())
            if wall_start <= t <= wall_end:
                ts_in_window.append(t)
        except ValueError:
            pass

    if not ts_in_window:
        return {"passes": 0, "duration_s": wall_end - wall_start, "toks_per_sec": 0.0}

    ts_in_window.sort()
    deduped = [ts_in_window[0]]
    for t in ts_in_window[1:]:
        if t - deduped[-1] > 0.05:
            deduped.append(t)

    dur = wall_end - wall_start
    passes = len(deduped)
    return {
        "passes":     passes,
        "duration_s": dur,
        "toks_per_sec": passes * T_FT / dur if dur > 0 else 0.0,
    }


# ── Grid measurement ──────────────────────────────────────────────────────────

def run_grid(
    tokenizer,
    t_ins: list[int],
    batch_sizes: list[int],
    n_reps: int,
) -> list[dict]:
    """
    Sweep (T_in, BS) grid. Returns list of per-(T_in, BS) aggregate dicts.
    Each (T_in, BS) runs n_reps batches of BS concurrent requests.
    """
    rows = []

    for t_in in t_ins:
        print(f"\n  T_in={t_in}")

        # Condition-level throughput fallback: track entire T_in block
        cond_wall_start = time.time()

        for bs in batch_sizes:
            ttfts_all: list[float] = []
            tpots_all: list[float] = []

            # Per-(T_in, BS) throughput window
            bs_wall_start = time.time()

            for rep in range(n_reps):
                salt = f"{t_in}-{bs}-{rep}-{time.time_ns()}"
                prompts = [build_prompt(tokenizer, t_in, salt=f"{salt}-{i}")
                           for i in range(bs)]

                with ThreadPoolExecutor(max_workers=bs) as ex:
                    futures = [ex.submit(send_streaming, p) for p in prompts]
                    for fut in as_completed(futures):
                        res = fut.result()
                        if res is not None:
                            ttfts_all.append(res["ttft_ms"])
                            tpots_all.append(res["tpot_ms"])

                if ttfts_all:
                    print(f"    bs={bs:2d} rep={rep}  "
                          f"ttft={statistics.mean(ttfts_all[-bs:]):7.1f}ms  "
                          f"tpot={statistics.mean(tpots_all[-bs:]):6.2f}ms")

            bs_wall_end = time.time()
            thr = read_throughput(bs_wall_start, bs_wall_end)

            if ttfts_all:
                rows.append({
                    "t_in":          t_in,
                    "bs":            bs,
                    "ttft_mean":     statistics.mean(ttfts_all),
                    "ttft_p50":      sorted(ttfts_all)[len(ttfts_all) // 2],
                    "tpot_mean":     statistics.mean(tpots_all),
                    "tpot_p50":      sorted(tpots_all)[len(tpots_all) // 2],
                    "n_samples":     len(ttfts_all),
                    "ft_passes":     thr["passes"],
                    "ft_duration_s": thr["duration_s"],
                    "ft_toks_per_s": thr["toks_per_sec"],
                    # fallback flag — set below if needed
                    "ft_tput_from_fallback": False,
                })

        # Fallback: if any (T_in, BS) row for this T_in had 0 passes, use
        # the T_in-block aggregate throughput instead.
        cond_wall_end = time.time()
        cond_thr = read_throughput(cond_wall_start, cond_wall_end)
        if cond_thr["passes"] > 0:
            n_bs = len(batch_sizes)
            fallback_toks_per_s = cond_thr["toks_per_sec"] / n_bs  # avg over BS cells
            for row in rows:
                if row["t_in"] == t_in and row["ft_passes"] == 0:
                    row["ft_toks_per_s"] = fallback_toks_per_s
                    row["ft_tput_from_fallback"] = True

    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-dir",    default="fwd_policy_results")
    ap.add_argument("--t-ins",       default=",".join(map(str, DEFAULT_T_INS)))
    ap.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    ap.add_argument("--n-reps",      type=int, default=DEFAULT_N_REPS)
    ap.add_argument("--conditions",  default="A,B,C,D")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a condition if its JSON already exists in save-dir")
    args = ap.parse_args()

    t_ins       = [int(x) for x in args.t_ins.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    conditions  = args.conditions.split(",")
    save_dir    = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("[init] loading tokenizer …")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    # Verify prompt lengths
    for t in t_ins:
        actual = len(tok.encode(build_prompt(tok, t), add_special_tokens=False))
        print(f"  T_in={t:5d}  actual={actual}")

    all_results: dict[str, list[dict]] = {}
    server_proc = None

    def cleanup():
        if server_proc is not None:
            kill_server(server_proc)

    import atexit
    atexit.register(cleanup)

    for cond in conditions:
        out_path = save_dir / f"results_{cond}.json"
        if args.skip_existing and out_path.exists():
            print(f"\n[{cond}] skipping (found {out_path})")
            all_results[cond] = json.loads(out_path.read_text())
            continue

        print(f"\n{'='*60}\n[{cond}] starting server …")
        log_path = str(save_dir / f"server_{cond}.log")
        server_proc = start_server(cond, log_path)

        try:
            wait_ready()
            print(f"[{cond}] server ready — warming up ({N_WARMUP} req at T_in={t_ins[0]}) …")
            for i in range(N_WARMUP):
                send_streaming(build_prompt(tok, t_ins[0], salt=f"warmup-{i}"))
            print(f"[{cond}] warmup done — starting grid …")

            rows = run_grid(tok, t_ins, batch_sizes, args.n_reps)
            all_results[cond] = rows
            out_path.write_text(json.dumps(rows, indent=2))
            print(f"[{cond}] saved → {out_path}")

        finally:
            kill_server(server_proc)
            server_proc = None

    # ── Analysis ──────────────────────────────────────────────────────────────

    if "A" not in all_results:
        print("\n[analysis] condition A required for overhead — done.")
        return

    def by_cell(rows: list[dict]) -> dict[tuple, dict]:
        return {(r["t_in"], r["bs"]): r for r in rows}

    a_cells = by_cell(all_results["A"])

    # Build summary CSV
    csv_path = save_dir / "summary.csv"
    fieldnames = [
        "condition", "t_in", "bs",
        "ttft_mean_ms", "tpot_mean_ms",
        "ttft_ovhd_ms", "tpot_ovhd_ms", "total_ovhd_ms",
        "ft_toks_per_s", "ft_tput_from_fallback",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cond in conditions:
            for row in all_results.get(cond, []):
                key = (row["t_in"], row["bs"])
                a = a_cells.get(key, {})
                ttft_ovhd = row["ttft_mean"] - a.get("ttft_mean", row["ttft_mean"])
                tpot_ovhd = row["tpot_mean"] - a.get("tpot_mean", row["tpot_mean"])
                total_ovhd = ttft_ovhd + tpot_ovhd * MAX_TOKENS
                w.writerow({
                    "condition":           cond,
                    "t_in":                row["t_in"],
                    "bs":                  row["bs"],
                    "ttft_mean_ms":        f"{row['ttft_mean']:.1f}",
                    "tpot_mean_ms":        f"{row['tpot_mean']:.2f}",
                    "ttft_ovhd_ms":        f"{ttft_ovhd:.1f}",
                    "tpot_ovhd_ms":        f"{tpot_ovhd:.2f}",
                    "total_ovhd_ms":       f"{total_ovhd:.1f}",
                    "ft_toks_per_s":       f"{row['ft_toks_per_s']:.1f}",
                    "ft_tput_from_fallback": row["ft_tput_from_fallback"],
                })
    print(f"\nSaved {csv_path}")

    # ── Recommendation table ──────────────────────────────────────────────────

    active_conds = [c for c in conditions if c != "A"]
    if not active_conds:
        return

    print(f"\n{'='*78}")
    print(f"RECOMMENDATION TABLE  "
          f"(best policy = min total_ovhd = TTFT_ovhd + TPOT_ovhd × {MAX_TOKENS})")
    print(f"\n{'T_in':>7}  {'BS':>4}  {'best':>4}  "
          f"{'total+':>9}  {'TTFT+':>8}  {'TPOT+':>7}  {'FT tok/s':>9}"
          f"  (runners-up)")
    print("-" * 78)

    rec_path = save_dir / "recommendations.txt"
    rec_lines = [
        "T_in,BS,best_policy,total_ovhd_ms,ttft_ovhd_ms,tpot_ovhd_ms,ft_toks_per_s\n"
    ]

    for t_in in t_ins:
        for bs in batch_sizes:
            key = (t_in, bs)
            a = a_cells.get(key)
            if a is None:
                continue

            candidates = []
            for cond in active_conds:
                cell = by_cell(all_results.get(cond, [])).get(key)
                if cell is None:
                    continue
                ttft_ovhd = cell["ttft_mean"] - a["ttft_mean"]
                tpot_ovhd = cell["tpot_mean"] - a["tpot_mean"]
                total_ovhd = ttft_ovhd + tpot_ovhd * MAX_TOKENS
                candidates.append({
                    "cond":       cond,
                    "ttft_ovhd":  ttft_ovhd,
                    "tpot_ovhd":  tpot_ovhd,
                    "total_ovhd": total_ovhd,
                    "ft_tput":    cell["ft_toks_per_s"],
                    "fallback":   cell["ft_tput_from_fallback"],
                })

            if not candidates:
                continue

            candidates.sort(key=lambda c: c["total_ovhd"])
            best = candidates[0]
            runners = ", ".join(
                f"{c['cond']}({c['total_ovhd']:+.0f}ms)"
                for c in candidates[1:]
            )
            ft_str = f"{best['ft_tput']:7.1f}{'*' if best['fallback'] else ' '}"
            print(f"{t_in:7d}  {bs:4d}  {best['cond']:>4}  "
                  f"{best['total_ovhd']:+9.1f}ms  "
                  f"{best['ttft_ovhd']:+8.1f}ms  "
                  f"{best['tpot_ovhd']:+7.2f}ms  "
                  f"{ft_str}  ({runners})")
            rec_lines.append(
                f"{t_in},{bs},{best['cond']},{best['total_ovhd']:.1f},"
                f"{best['ttft_ovhd']:.1f},{best['tpot_ovhd']:.2f},{best['ft_tput']:.1f}\n"
            )

    rec_path.write_text("".join(rec_lines))
    print(f"\n* = FT throughput estimated from T_in-block fallback (< 1 pass per (T_in,BS) window)")
    print(f"Saved {rec_path}")


if __name__ == "__main__":
    main()
