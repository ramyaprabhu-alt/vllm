#!/usr/bin/env python3
"""
bubble_qps_sweep.py — sweep vLLM over datasets × QPS, capturing bubble sizes + TTFT + TBT.

For each (dataset, qps) combination:
  - Launches vLLM with VLLM_BUBBLE_PROFILE=1
  - Runs vllm bench serve for 500 requests
  - Collects per-layer allreduce (bubble) timings from /tmp/vllm_bubble_rank*.json
  - Collects TTFT and TBT (TPOT) from bench output
  - Writes a summary table to results/summary.txt

Usage:
    python bubble_qps_sweep.py [--result-dir PATH]
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

VENV          = "/mnt/nfs/home/ramya/vllm/.venv"
MODEL         = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
PORT          = 8000
NUM_PROMPTS   = 500
OUTPUT_LEN    = 256   # fixed output length for arxiv; sharegpt uses its own

DATASETS = [
    {
        "name":    "arxiv",
        "flags":   [
            "--dataset-name", "custom",
            "--dataset-path", "/mnt/nfs/home/ramya/scratch/arxiv_bench_500.jsonl",
            "--custom-output-len", str(OUTPUT_LEN),
            "--disable-shuffle",
        ],
    },
    {
        "name":    "sharegpt",
        "flags":   [
            "--dataset-name", "sharegpt",
            "--dataset-path", "/mnt/nfs/home/ramya/scratch/ShareGPT_V3_unfiltered_cleaned_split.json",
        ],
    },
]

QPS_VALUES = [0.5, 1.0, 2.0, 4.0]

RESULT_DIR = Path("/mnt/nfs/home/ramya/vllm/compare_results/bubble_qps_sweep")

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def wait_server(port: int, timeout: int = 600) -> bool:
    import urllib.request, urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(5)
    return False


def _pids_on_port(port: int) -> list[int]:
    """Return PIDs listening on port by parsing /proc/net/tcp (no lsof needed)."""
    import socket
    hex_port = f"{port:04X}"
    pids = []
    # Build inode → pid map from /proc/*/fd
    inode_to_pid: dict[str, int] = {}
    try:
        for pid_dir in Path("/proc").iterdir():
            if not pid_dir.name.isdigit():
                continue
            fd_dir = pid_dir / "fd"
            try:
                for fd in fd_dir.iterdir():
                    target = fd.resolve()
                    if "socket:[" in str(target):
                        inode = str(target).split("[")[1].rstrip("]")
                        inode_to_pid[inode] = int(pid_dir.name)
            except Exception:
                pass
    except Exception:
        pass
    # Find inodes for our port in /proc/net/tcp and tcp6
    for tcp_file in ["/proc/net/tcp", "/proc/net/tcp6"]:
        try:
            for line in Path(tcp_file).read_text().splitlines()[1:]:
                parts = line.split()
                if len(parts) < 10:
                    continue
                local = parts[1]  # "00000000:1F40" or ipv6 variant
                if local.endswith(f":{hex_port}") and parts[3] == "0A":  # 0A = LISTEN
                    inode = parts[9]
                    if inode in inode_to_pid:
                        pids.append(inode_to_pid[inode])
        except Exception:
            pass
    return list(set(pids))


def _port_in_use(port: int) -> bool:
    import socket as _socket
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex(("127.0.0.1", port))
        s.close()
        return result == 0
    except Exception:
        return False


def kill_server(port: int) -> None:
    log("  Stopping server...")
    # Kill any PIDs holding the port
    for pid in _pids_on_port(port):
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    for pattern in ["vllm serve", "vllm.worker", "VLLM::Worker_TP"]:
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)
    subprocess.run(["pkill", "-9", "-x", "VLLM::Worker_TP"], capture_output=True)
    # Wait for GPU memory to free
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True).strip().splitlines()
            if all(int(x) < 5120 for x in out if x.strip().isdigit()):
                log("  GPU memory freed.")
                break
        except Exception:
            break
        time.sleep(5)
    # Wait for port to clear, force-kill any straggler
    deadline = time.time() + 30
    while time.time() < deadline:
        if not _port_in_use(port):
            break
        for pid in _pids_on_port(port):
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(1)
    time.sleep(3)


def parse_bench_output(text: str) -> dict:
    patterns = {
        "successful_requests": r"Successful requests:\s+(\d+)",
        "mean_ttft_ms":        r"Mean TTFT \(ms\):\s+([\d.]+)",
        "p50_ttft_ms":         r"P50 TTFT \(ms\):\s+([\d.]+)",
        "p99_ttft_ms":         r"P99 TTFT \(ms\):\s+([\d.]+)",
        "mean_tpot_ms":        r"Mean TPOT \(ms\):\s+([\d.]+)",
        "p50_tpot_ms":         r"P50 TPOT \(ms\):\s+([\d.]+)",
        "p99_tpot_ms":         r"P99 TPOT \(ms\):\s+([\d.]+)",
        "mean_itl_ms":         r"Mean ITL \(ms\):\s+([\d.]+)",
        "p99_itl_ms":          r"P99 ITL \(ms\):\s+([\d.]+)",
        "req_throughput":      r"Request throughput \(req/s\):\s+([\d.]+)",
    }
    result = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        result[key] = float(m.group(1)) if m else None
    return result


def parse_bubble_json(path: Path) -> dict:
    """Return summary stats over all allreduce_ms values in the bubble log."""
    if not path.exists():
        return {}
    records = json.loads(path.read_text())
    if not records:
        return {}
    vals = sorted(r["allreduce_ms"] for r in records)
    n = len(vals)
    total = sum(vals)
    # per-prefill total bubble (sum of allreduce_ms across all 48 layers per batch)
    from collections import defaultdict
    per_prefill: dict = defaultdict(float)
    per_prefill_ntok: dict = {}
    for r in records:
        pid = r["prefill_id"]
        per_prefill[pid] += r["allreduce_ms"]
        per_prefill_ntok[pid] = r.get("n_tokens", 0)
    pf_totals = sorted(per_prefill.values())
    pf_ntoks  = [per_prefill_ntok[pid] for pid in sorted(per_prefill)]

    def pct(lst, p):
        idx = max(0, min(len(lst) - 1, int(len(lst) * p / 100)))
        return lst[idx]

    return {
        "n_layer_records":    n,
        "n_prefills":         len(per_prefill),
        "mean_allreduce_ms":  round(total / n, 3),
        "p50_allreduce_ms":   round(pct(vals, 50), 3),
        "p95_allreduce_ms":   round(pct(vals, 95), 3),
        "p99_allreduce_ms":   round(pct(vals, 99), 3),
        "mean_bubble_per_prefill_ms": round(sum(pf_totals) / len(pf_totals), 1),
        "p50_bubble_per_prefill_ms":  round(pct(pf_totals, 50), 1),
        "p95_bubble_per_prefill_ms":  round(pct(pf_totals, 95), 1),
        "mean_n_tokens":      round(sum(pf_ntoks) / len(pf_ntoks)) if pf_ntoks else None,
        "p50_n_tokens":       round(pct(sorted(pf_ntoks), 50)) if pf_ntoks else None,
    }


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_one(dataset: dict, qps: float, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    server_log = run_dir / "server.log"
    bench_log  = run_dir / "bench.log"

    env = {**os.environ, "VLLM_BUBBLE_PROFILE": "1",
           "PATH": f"{VENV}/bin:{os.environ.get('PATH', '')}"}

    server_cmd = [
        f"{VENV}/bin/vllm", "serve", MODEL,
        "--tensor-parallel-size", "2",
        "--enable-expert-parallel",
        "--enable-ep-weight-filter",
        "--all2all-backend", "allgather_reducescatter",
        "--moe-backend", "triton",
        "--dtype", "bfloat16",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.85",
        "--max-num-seqs", "256",
        "--enforce-eager",
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--trust-remote-code",
        "--host", "0.0.0.0", "--port", str(PORT),
    ]

    # Remove stale bubble logs
    for rank in range(2):
        Path(f"/tmp/vllm_bubble_rank{rank}.json").unlink(missing_ok=True)

    log(f"  Launching server → {server_log.name}")
    with open(server_log, "w") as f:
        proc = subprocess.Popen(server_cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    log(f"  Server PID: {proc.pid}")

    log("  Waiting for server (up to 600s)...")
    if not wait_server(PORT):
        log("  ERROR: server did not become ready — skipping.")
        proc.terminate()
        kill_server(PORT)
        return {"error": "server_timeout"}

    bench_cmd = [
        f"{VENV}/bin/vllm", "bench", "serve",
        "--num-prompts", str(NUM_PROMPTS),
        "--request-rate", str(qps),
        "--host", "localhost", "--port", str(PORT),
    ] + dataset["flags"]

    log(f"  Benchmarking {NUM_PROMPTS} prompts @ {qps} req/s...")
    with open(bench_log, "w") as f:
        subprocess.run(bench_cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    bench_text = bench_log.read_text(errors="replace")
    inference  = parse_bench_output(bench_text)

    # Copy bubble logs before killing the server — /tmp/ is ephemeral
    bubble = {}
    for rank in range(2):
        src = Path(f"/tmp/vllm_bubble_rank{rank}.json")
        dst = run_dir / f"bubble_rank{rank}.json"
        if src.exists():
            shutil.copy(src, dst)
            bubble[f"rank{rank}"] = parse_bubble_json(dst)
        else:
            log(f"  WARNING: no bubble log for rank {rank}")

    kill_server(PORT)

    result = {
        "dataset": dataset["name"],
        "qps":     qps,
        "inference": inference,
        "bubble":  bubble,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def print_summary(results: list[dict]) -> str:
    lines = []
    lines.append("")
    lines.append("=" * 110)
    lines.append(f"{'Dataset':<10} {'QPS':>5}  {'Reqs':>5}  "
                 f"{'TTFT_mean':>10} {'TTFT_p50':>9} {'TTFT_p99':>9}  "
                 f"{'TBT_mean':>9} {'TBT_p50':>8} {'TBT_p99':>8}  "
                 f"{'Bubble_p50/layer':>16} {'Bubble_p95/layer':>16} {'Bub/prefill_p50':>15} "
                 f"{'Tok_p50':>8}")
    lines.append("─" * 110)
    lines.append(f"{'':10} {'':5}  {'':5}  "
                 f"{'(ms)':>10} {'(ms)':>9} {'(ms)':>9}  "
                 f"{'(ms)':>9} {'(ms)':>8} {'(ms)':>8}  "
                 f"{'(ms)':>16} {'(ms)':>16} {'(ms)':>15} "
                 f"{'(tokens)':>8}")
    lines.append("─" * 110)

    for r in results:
        if "error" in r:
            lines.append(f"{r['dataset']:<10} {r['qps']:>5.1f}  ERROR: {r['error']}")
            continue
        inf  = r.get("inference", {})
        bub  = r.get("bubble", {}).get("rank1", {})   # rank1 = light EP rank (larger bubbles)

        def f(v, fmt=".1f"):
            return f"{v:{fmt}}" if v is not None else "—"

        lines.append(
            f"{r['dataset']:<10} {r['qps']:>5.1f}  "
            f"{f(inf.get('successful_requests'), '.0f'):>5}  "
            f"{f(inf.get('mean_ttft_ms')):>10} "
            f"{f(inf.get('p50_ttft_ms')):>9} "
            f"{f(inf.get('p99_ttft_ms')):>9}  "
            f"{f(inf.get('mean_tpot_ms')):>9} "
            f"{f(inf.get('p50_tpot_ms')):>8} "
            f"{f(inf.get('p99_tpot_ms')):>8}  "
            f"{f(bub.get('p50_allreduce_ms'), '.3f'):>16} "
            f"{f(bub.get('p95_allreduce_ms'), '.3f'):>16} "
            f"{f(bub.get('p50_bubble_per_prefill_ms'), '.1f'):>15} "
            f"{f(bub.get('p50_n_tokens'), '.0f'):>8}"
        )
    lines.append("=" * 110)
    lines.append("")
    lines.append("  Bubble columns use rank 1 (light EP rank — consistently larger bubbles).")
    lines.append("  TBT = TPOT (mean inter-token latency averaged over decode phase).")
    lines.append("")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=str(RESULT_DIR))
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    log_path = result_dir / "sweep.log"
    log(f"Results → {result_dir}")
    log(f"Datasets: {[d['name'] for d in DATASETS]}")
    log(f"QPS values: {QPS_VALUES}")
    log(f"Prompts per run: {NUM_PROMPTS}")
    log("")

    results = []
    total = len(DATASETS) * len(QPS_VALUES)
    idx = 0
    for dataset in DATASETS:
        for qps in QPS_VALUES:
            idx += 1
            tag = f"{dataset['name']}_qps{qps}"
            run_dir = result_dir / tag
            log(f"[{idx}/{total}] {dataset['name']} @ {qps} qps → {run_dir.name}")
            if (run_dir / "result.json").exists():
                log("  Already done — loading cached result.")
                result = json.loads((run_dir / "result.json").read_text())
            else:
                result = run_one(dataset, qps, run_dir)
            results.append(result)
            with open(log_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            log("")

    summary = print_summary(results)
    print(summary)
    (result_dir / "summary.txt").write_text(summary)
    log(f"Summary written to {result_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
