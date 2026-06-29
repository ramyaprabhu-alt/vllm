#!/usr/bin/env python3
"""
Sweep VLLM_FT_COMBINED_T_FT and measure TPOT overhead in C+D mode.

For each t_ft value, starts a BubbleTea server in C+D mode and runs
vllm bench serve with synthetic T_in=8192 prompts so the KV cache is
large enough to reproduce the HBM bandwidth contention seen in the
arxiv benchmark.

Usage:
    python tpot_tft_sweep.py
    python tpot_tft_sweep.py --t-ft-values 8,16,32,64,128
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

BT_VENV   = "/mnt/nfs/home/ramya/vllm/.venv"
MODEL_DIR = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
PORT      = 8000
RESULT_DIR = Path("/mnt/nfs/home/ramya/vllm/tpot_tft_results")

SERVER_FLAGS = [
    "--tensor-parallel-size", "2",
    "--enable-expert-parallel",
    "--enable-ep-weight-filter",
    "--all2all-backend", "allgather_reducescatter",
    "--moe-backend", "triton",
    "--dtype", "bfloat16",
    "--max-model-len=16384",
    "--gpu-memory-utilization=0.9",
    "--max-num-seqs=256",
    "--enforce-eager",
    "--no-enable-chunked-prefill",
    "--no-enable-prefix-caching",
    "--max-num-batched-tokens=16384",
    "--trust-remote-code",
    "--host", "0.0.0.0",
    "--port", str(PORT),
]

def _wait_server(timeout=600):
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=3)
            return True
        except Exception:
            time.sleep(5)
    return False

def _kill_server():
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{PORT}"], text=True).strip()
        for pid in out.splitlines():
            try: os.kill(int(pid), signal.SIGTERM)
            except Exception: pass
    except Exception:
        pass
    time.sleep(6)
    for pat in [r"vllm\.worker", r"vllm serve", r"VLLM::Worker_TP"]:
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    subprocess.run(["pkill", "-9", "-x", "VLLM::Worker_TP"], capture_output=True)
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True).strip().splitlines()
            if all(int(x) < 5120 for x in out if x.strip().isdigit()):
                break
        except Exception:
            break
        time.sleep(4)

def _parse_tpot(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    out = {}
    for key, pat in [
        ("mean_tpot_ms", r"Mean TPOT \(ms\):\s+([\d.]+)"),
        ("p50_tpot_ms",  r"Median TPOT \(ms\):\s+([\d.]+)"),
        ("p99_tpot_ms",  r"P99 TPOT \(ms\):\s+([\d.]+)"),
        ("mean_ttft_ms", r"Mean TTFT \(ms\):\s+([\d.]+)"),
        ("successful",   r"Successful requests:\s+(\d+)"),
    ]:
        m = re.search(pat, text)
        if m: out[key] = float(m.group(1))
    return out

def run_condition(label: str, t_ft: int | None, num_prompts: int, request_rate: float,
                  combined_mode: str = "C+D") -> dict:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    log_dir  = RESULT_DIR / label.replace(" ", "_")
    log_dir.mkdir(parents=True, exist_ok=True)
    srv_log  = log_dir / "server.log"
    bench_log= log_dir / "bench.log"

    env = os.environ.copy()
    env["PATH"]                    = f"{BT_VENV}/bin:{env.get('PATH','')}"
    env["CUDA_VISIBLE_DEVICES"]    = "0,1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if t_ft is not None:
        env["VLLM_FT_COMBINED_MODE"]   = combined_mode
        env["VLLM_FT_COMBINED_T_FT"]   = str(t_ft)
        Path("/tmp/vllm_combined_completions.log").unlink(missing_ok=True)

    subprocess.run(["pkill", "-9", "-f", "VLLM::Worker_TP"], capture_output=True)
    subprocess.run(["pkill", "-9", "-x", "VLLM::Worker_TP"], capture_output=True)
    time.sleep(2)

    srv_cmd = [f"{BT_VENV}/bin/vllm", "serve", MODEL_DIR] + SERVER_FLAGS
    print(f"  Starting server (t_ft={t_ft})...")
    with open(srv_log, "w") as f:
        proc = subprocess.Popen(srv_cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    if not _wait_server():
        print("  ERROR: server timed out")
        proc.terminate()
        return {"error": "timeout"}

    bench_cmd = [
        f"{BT_VENV}/bin/vllm", "bench", "serve",
        "--backend", "vllm",
        "--model", MODEL_DIR,
        "--dataset-name", "random",
        "--input-len", "8192",
        "--output-len", "100",
        "--num-prompts", str(num_prompts),
        "--request-rate", str(request_rate),
        "--port", str(PORT),
    ]
    print(f"  Benchmarking ({num_prompts} prompts @ {request_rate} req/s)...")
    with open(bench_log, "w") as f:
        subprocess.run(bench_cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    _kill_server()

    metrics = _parse_tpot(bench_log)
    print(f"  mean TPOT={metrics.get('mean_tpot_ms','?')} ms  "
          f"p99 TPOT={metrics.get('p99_tpot_ms','?')} ms  "
          f"mean TTFT={metrics.get('mean_ttft_ms','?')} ms")

    result = {"label": label, "t_ft": t_ft, **metrics}
    (log_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-ft-values", default="8,16,32,64,128",
                        help="Comma-separated t_ft values to sweep (default: 8,16,32,64,128)")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--request-rate", type=float, default=1.0)
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the vLLM inference-only baseline run")
    args = parser.parse_args()

    t_ft_values = [int(x) for x in args.t_ft_values.split(",")]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    if not args.no_baseline:
        r = run_condition("baseline (no FT)", t_ft=None,
                          num_prompts=args.num_prompts, request_rate=args.request_rate)
        results.append(r)

    for t_ft in t_ft_values:
        r = run_condition(f"C+D t_ft={t_ft}", t_ft=t_ft,
                          num_prompts=args.num_prompts, request_rate=args.request_rate,
                          combined_mode="C+D")
        results.append(r)

    for t_ft in t_ft_values:
        r = run_condition(f"C+D_batch t_ft={t_ft}", t_ft=t_ft,
                          num_prompts=args.num_prompts, request_rate=args.request_rate,
                          combined_mode="C+D_batch")
        results.append(r)

    # Summary table
    baseline_tpot = next((r["mean_tpot_ms"] for r in results if r.get("t_ft") is None), None)

    print(f"\n{'='*65}")
    print(f"  {'Condition':<22} {'mean TPOT':>10} {'p99 TPOT':>10} {'overhead':>10} {'mean TTFT':>11}")
    print(f"{'='*65}")
    for r in results:
        ovhd = ""
        if baseline_tpot and r.get("mean_tpot_ms") and r.get("t_ft") is not None:
            delta = r["mean_tpot_ms"] - baseline_tpot
            ovhd = f"{delta:>+.1f} ms"
        print(f"  {r['label']:<22} "
              f"{r.get('mean_tpot_ms', '—'):>10.1f} "
              f"{r.get('p99_tpot_ms',  '—'):>10.1f} "
              f"{ovhd:>10} "
              f"{r.get('mean_ttft_ms', '—'):>11.1f}")
    print(f"{'='*65}")

    (RESULT_DIR / "sweep_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults in {RESULT_DIR}/")


if __name__ == "__main__":
    main()
