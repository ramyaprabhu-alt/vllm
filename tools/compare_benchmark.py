#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
compare_benchmark.py — vLLM vs LLMStation vs Bubble Tea co-serving benchmark.

Runs any subset of three systems end-to-end on Qwen3-30B-A3B (2×A100), using
the same ShareGPT inference workload for each, and prints a side-by-side
comparison of inference latency and training throughput.

What each system does
---------------------
vLLM (inference-only baseline):
    Stock vLLM serve with no training.  Establishes the inference latency floor.

LLMStation:
    Spawns training workers (real LoRA weight updates) inside the vLLM
    process via --enable-lms.  Workers run under CUDA MPS with tasklet-
    based preemption (--lms-forward/backward-tasklets/wait).

Bubble Tea:
    Runs FT forward sub-ops on a secondary CUDA stream during decode steps
    (mode C) and FT backward sub-ops inside rank-imbalance idle windows at
    the TP all_reduce barrier during prefill (mode D).  No MPS, no GPU
    time-slice; backward uses synthetic Qwen3-30B-A3B-shaped tensors to
    measure interference without a full training pipeline.

Usage
-----
    # All three systems sequentially (recommended):
    python compare_benchmark.py

    # Subset:
    python compare_benchmark.py --system vllm
    python compare_benchmark.py --system llmstation
    python compare_benchmark.py --system bubble_tea
    python compare_benchmark.py --system vllm,llmstation
    python compare_benchmark.py --system vllm,bubble_tea

    # Adjust workload:
    python compare_benchmark.py --request-rate 2 --num-prompts 400

Results
-------
    compare_results/vllm.json
    compare_results/llmstation.json
    compare_results/bubble_tea.json
    Printed comparison table on stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import regex as re

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_DIR = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
SHAREGPT = Path.home() / "scratch/ShareGPT_V3_unfiltered_cleaned_split.json"

LMS_VENV = "/mnt/nfs/home/ramya/llmstation/.venv-lms"
LMS_BENCH_DIR = "/mnt/nfs/home/ramya/llmstation/python"
LMS_BENCH = f"{LMS_BENCH_DIR}/benchmark_serving.py"
LMS_LORA_MODULES = "/mnt/nfs/home/ramya/llmstation/python/lora_modules.txt"
LMS_PYTORCH_LIB = "/mnt/nfs/home/ramya/llmstation-builds/pytorch-v2.4.0-lms/torch/lib"
LMS_NCCL_SO = (
    "/mnt/nfs/home/ramya/llmstation-builds/pytorch-v2.4.0-lms"
    "/build/nccl/lib/libnccl.so.2.20.5"
)

BT_VENV = "/mnt/nfs/home/ramya/vllm/.venv-bubble"
BT_BENCH_DIR = "/mnt/nfs/home/ramya/vllm/benchmarks"
BT_BENCH = f"{BT_BENCH_DIR}/benchmark_serving.py"
BT_COMPLETIONS = Path("/tmp/vllm_combined_completions.log")  # written by C+D mode
BT_FWD_BWD_LOG = Path("/tmp/vllm_bt_fwd_bwd.log")  # per-pass log

# LoRA adapter for BubbleTea real training (set VLLM_FT_LORA_PATH in env)
BT_LORA_ADAPTER = (
    "/mnt/nfs/home/ramya/slora-plus/S-LoRA/test/qwen3/adapters/qwen3-toy-lora"
)

PORT = 8000


# ── Helpers ────────────────────────────────────────────────────────────────────


def _lora_path(key: str = "Qwen/Qwen3-30B-A3B-LoRA") -> str:
    with open(LMS_LORA_MODULES) as f:
        for line in f:
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip()
    raise FileNotFoundError(
        f"LoRA key '{key}' not in {LMS_LORA_MODULES}. "
        "Run: python llmstation/python/qwen3_lora_create.py"
        " --output-dir ~/scratch/qwen3-toy-lora"
    )


def _wait_server(port: int, timeout: int = 600) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
            return True
        except Exception:
            time.sleep(5)
    return False


def _warmup_server(
    port: int, env: dict, args, log, wait_for_training: bool = False
) -> None:
    """Send warmup requests to prime Triton JIT and optionally wait for training.

    Sends 20 short completions so both systems have the same warm-up state before
    timing begins.  For BubbleTea with real training, also polls until the first
    optimizer step completes (visible in BT_COMPLETIONS) so the benchmark measures
    steady-state, not training start-up latency.
    """
    import concurrent.futures as _cf
    import json as _json
    import urllib.error
    import urllib.request

    log("  Warming up (20 short requests)...")
    payload = _json.dumps(
        {
            "model": MODEL_DIR,
            "prompt": "Briefly summarise recent advances in large language models.",
            "max_tokens": 32,
            "temperature": 0.0,
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    n_ok = 0
    for _ in range(20):
        try:
            req = urllib.request.Request(
                f"http://localhost:{port}/v1/completions",
                data=payload,
                headers=headers,
                method="POST",
            )
            urllib.request.urlopen(req, timeout=60)
            n_ok += 1
        except Exception:
            pass
    log(f"  Warmup done ({n_ok}/20 requests succeeded).")

    if wait_for_training and env.get("VLLM_FT_LORA_PATH", ""):
        # Root cause of the old 120s/300s timeouts, and of sending more of
        # the SAME short request during the wait, and of switching to a long
        # prompt sent one-at-a-time: bubble-gated training sub-ops need (a) a
        # prefill batch with n_tok >= 512 (moe_runner.py's _sched_min_tokens
        # -- the ~12-token short prompt never clears this) AND (b) genuine
        # request concurrency. Checked every real run in this project's
        # history, including the slowest ever run (0.4qps,
        # qwen15moe_arxiv_200_0.4qps_bubble_cycle3_0710): its first training
        # step fired at "Running: 3 reqs", never at "Running: 1" -- Poisson
        # arrivals mean even nominally slow traffic still overlaps sometimes,
        # which strictly sequential one-at-a-time polling (every earlier
        # attempt here) never does. Fix: fire several long-prompt requests
        # CONCURRENTLY per cycle (not one blocking call at a time) so the
        # server actually sees overlapping in-flight requests during the
        # wait, matching every real run's condition instead of an
        # artificially serialized one.
        long_prompt = (
            "Briefly summarise recent advances in large language models. " * 120
        )
        long_payload = _json.dumps(
            {
                "model": MODEL_DIR,
                "prompt": long_prompt,
                "max_tokens": 1,
                "temperature": 0.0,
            }
        ).encode()
        concurrency = 4

        def _fire_one() -> None:
            try:
                req = urllib.request.Request(
                    f"http://localhost:{port}/v1/completions",
                    data=long_payload,
                    headers=headers,
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=60)
            except Exception:
                pass

        # 30s, not 300s: 2026-07-16 established across 6 back-to-back trials
        # (idle wait, sequential short/long-prompt polling, 4-way concurrent
        # polling) that no synthetic warmup traffic tried so far ever
        # triggers the first optimizer step -- training only ever started
        # once the real benchmark's own (varied-content) traffic began. A
        # long wait here just burns GPU time with no observed benefit; kept
        # short rather than removed in case a future warmup scheme does work.
        training_wait_s = 30
        log(
            f"  Waiting for first BubbleTea optimizer step (up to "
            f"{training_wait_s}s, polling with {concurrency} concurrent "
            "long-prompt requests per cycle so the server sees overlapping "
            "in-flight requests, not one at a time)..."
        )
        deadline = time.time() + training_wait_s
        prev = BT_COMPLETIONS.stat().st_size if BT_COMPLETIONS.exists() else -1
        with _cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
            while time.time() < deadline:
                if BT_COMPLETIONS.exists():
                    sz = BT_COMPLETIONS.stat().st_size
                    if sz > prev:
                        log("  Training started — first optimizer step logged.")
                        break
                futures = [pool.submit(_fire_one) for _ in range(concurrency)]
                _cf.wait(futures, timeout=60)
            else:
                log(
                    f"  WARNING: training did not start within "
                    f"{training_wait_s}s — check server log."
                )


def _port_is_free(port: int) -> bool:
    """Check whether *port* is free to bind, without depending on lsof/ss
    (neither is installed on this host — a prior lsof-based check here
    silently no-op'd via a broad `except Exception`, masking real port
    contention and causing the LLMStation "Address already in use" race).

    A plain `connect()` probe is NOT enough: a socket left in TIME_WAIT after
    the previous server closes its listener refuses new connections (so a
    connect-based probe reports "free") but still blocks a fresh `bind()` on
    that port for up to ~60s. Scan /proc/net/tcp{,6} directly instead — any
    entry for the port (LISTEN, TIME_WAIT, whatever) means bind() can fail.
    """
    target = f"{port:04X}"
    for proc_net in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(proc_net).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) > 1 and fields[1].rsplit(":", 1)[-1].upper() == target:
                return False
    return True


def _pids_on_port(port: int) -> list[int]:
    """Return PIDs holding an open socket on *port*, parsed from /proc — a
    dependency-free stand-in for `lsof -ti :port`."""
    target = f"{port:04X}"
    inodes: set[str] = set()
    for proc_net in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(proc_net).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10:
                continue
            if fields[1].rsplit(":", 1)[-1].upper() == target:
                inodes.add(fields[9])
    if not inodes:
        return []
    pids = []
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            fds = list((pid_dir / "fd").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(fd)
            except OSError:
                continue
            if link.startswith("socket:[") and link[8:-1] in inodes:
                pids.append(int(pid_dir.name))
                break
    return pids


def _kill_server(port: int, log) -> None:
    log(f"  Stopping server on port {port}...")
    for pid in _pids_on_port(port):
        with contextlib.suppress(Exception):
            os.kill(pid, signal.SIGTERM)
    time.sleep(6)
    # Kill worker processes via cmdline patterns and via comm name.
    # Workers rename themselves (prctl PR_SET_NAME) to "VLLM::Worker_TP*".
    # Linux truncates comm to 15 chars → actual comm is "VLLM::Worker_TP".
    # pkill -x matches comm (15 chars), pkill -f matches full cmdline.
    for pattern in [
        r"vllm\.worker",
        r"vllm serve",
        r"ray::",
        r"multiprocessing.spawn import spawn_main",
        r"VLLM::Worker_TP",
    ]:
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)
    # Also match by truncated comm in case cmdline was cleared
    subprocess.run(["pkill", "-9", "-x", "VLLM::Worker_TP"], capture_output=True)
    # Poll until GPU memory is freed (both GPUs < 5 GiB)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            out = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                .strip()
                .splitlines()
            )
            if all(int(x) < 5120 for x in out if x.strip().isdigit()):
                log("  GPU memory freed.")
                break
        except Exception:
            break
        time.sleep(5)
    else:
        log("  WARNING: GPU memory may not be fully freed — proceeding anyway.")
    # Wait for port to be released. A closed listening socket can sit in
    # TIME_WAIT for up to ~60s (2*MSL) during which nothing owns it (so
    # SIGKILL has nothing to target) but a fresh bind() still fails — so the
    # budget here must exceed that, not just cover process-teardown lag.
    deadline = time.time() + 75
    while time.time() < deadline:
        if _port_is_free(port):
            break
        # If something still owns the socket (not just TIME_WAIT), force-kill it.
        for pid in _pids_on_port(port):
            with contextlib.suppress(Exception):
                os.kill(pid, signal.SIGKILL)
        time.sleep(2)
    else:
        log(f"  WARNING: port {port} may still be in TIME_WAIT — proceeding anyway.")
    time.sleep(2)  # brief grace period for OS to fully release the socket


def _parse_bench_log(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    patterns = {
        "successful_requests": r"Successful requests:\s+(\d+)",
        "duration_s": r"Benchmark duration \(s\):\s+([\d.]+)",
        "req_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
        "out_tok_s": r"Output token throughput \(tok/s\):\s+([\d.]+)",
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
        "p50_ttft_ms": r"Median TTFT \(ms\):\s+([\d.]+)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([\d.]+)",
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
        "p50_tpot_ms": r"Median TPOT \(ms\):\s+([\d.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([\d.]+)",
        "p99_itl_ms": r"P99 ITL \(ms\):\s+([\d.]+)",
    }
    result = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            result[key] = float(m.group(1))
    return result


def _parse_lms_log(lms_log: Path, start_line: int, end_line: int) -> dict:
    """Compute weighted-average PEFT throughput from lms.log."""
    pat_ft = re.compile(r"finetune:\s*([\d.]+)")
    pat_dur = re.compile(r"Duration\(s\):\s*([\d.]+)")
    values, durations = [], []
    if not lms_log.exists():
        return {
            "peft_samples_s": None,
            "training_steps": 0,
            "note": "lms.log not found — training metrics unavailable",
        }
    lines = lms_log.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if i < start_line + 2 or i > end_line - 2:
            continue
        m1, m2 = pat_ft.search(line), pat_dur.search(line)
        if m1:
            values.append(float(m1.group(1)))
        if m2:
            durations.append(float(m2.group(1)))
    if not values or not durations:
        return {"peft_samples_s": None, "training_steps": 0}
    avg = sum(x * y for x, y in zip(values, durations)) / sum(durations)
    return {"peft_samples_s": round(avg, 4), "training_steps": len(values)}


def _count_timestamps_in_window(path: Path, t_start: float, t_end: float) -> int:
    """Count Unix-timestamp lines in *path* that fall within [t_start, t_end]."""
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(errors="replace").splitlines():
        try:
            ts = float(line.strip())
            if t_start <= ts <= t_end:
                n += 1
        except ValueError:
            pass
    return n


def _parse_bt_completions(
    path: Path, t_start: float, t_end: float, t_ft: int = 128
) -> dict:
    """Count C+D training steps that completed during the benchmark window."""
    n = _count_timestamps_in_window(path, t_start, t_end)
    if n == 0 and not path.exists():
        return {
            "training_steps": 0,
            "peft_samples_s": None,
            "note": "no completions log — VLLM_FT_COMBINED_MODE may not be set",
        }
    duration = t_end - t_start
    result = {
        "training_steps": n,
        "peft_samples_s": round(n / duration, 4) if (n > 0 and duration > 0) else None,
        "note": f"each step = one C+D fwd+bwd pass over t_ft={t_ft} tokens",
    }
    # Count individual fwd+bwd passes from the per-pass log
    fwd_bwd = _count_timestamps_in_window(BT_FWD_BWD_LOG, t_start, t_end)
    if fwd_bwd > 0:
        result["fwd_bwd_passes"] = fwd_bwd
        result["fwd_bwd_tok_s"] = round(fwd_bwd * t_ft / duration, 2)
    return result


# ── Dataset flag builder ───────────────────────────────────────────────────────


def _bench_cmd(args) -> list[str]:
    """Return the benchmark command to run against the launched server.

    If --trace-file is set, replay real BurstGPT arrival timing/token lengths
    via burstgpt_trace_replay.py instead of `vllm bench serve`.
    """
    if args.trace_file:
        return [
            f"{BT_VENV}/bin/python",
            f"{BT_BENCH_DIR}/burstgpt_trace_replay.py",
            "--model",
            MODEL_DIR,
            "--backend",
            "vllm",
            "--trace-file",
            str(args.trace_file),
            "--port",
            str(PORT),
        ]
    return (
        [
            f"{BT_VENV}/bin/vllm",
            "bench",
            "serve",
            "--backend",
            "vllm",
            "--model",
            MODEL_DIR,
        ]
        + _bench_dataset_flags(args)
        + [
            "--request-rate",
            str(args.request_rate),
            "--num-prompts",
            str(args.num_prompts),
            "--port",
            str(PORT),
        ]
    )


def _bench_desc(args) -> str:
    if args.trace_file:
        return f"BurstGPT trace replay ({args.trace_file.name})"
    return f"{args.num_prompts} prompts @ {args.request_rate} req/s"


def _bench_dataset_flags(args) -> list[str]:
    """Return the dataset flags for vllm bench serve based on args."""
    if args.dataset_name == "custom":
        return [
            "--dataset-name",
            "custom",
            "--dataset-path",
            str(args.dataset_path),
            "--custom-output-len",
            str(args.custom_output_len),
            "--disable-shuffle",  # already ordered longest-first in the JSONL
        ]
    else:
        flags = [
            "--dataset-name",
            args.dataset_name,
            "--dataset-path",
            str(args.dataset_path),
        ]
        if args.dataset_name == "sharegpt" and args.sharegpt_output_len:
            flags += ["--sharegpt-output-len", str(args.sharegpt_output_len)]
        return flags


# ── vLLM (inference-only baseline) ────────────────────────────────────────────


def run_vllm(args, result_dir: Path, log) -> dict:
    log("\n" + "=" * 60)
    log("SYSTEM: vLLM (inference-only baseline)")
    log("=" * 60)

    bench_log = result_dir / "vllm_bench.log"
    server_log = result_dir / "vllm_server.log"

    env = os.environ.copy()
    env["PATH"] = f"{BT_VENV}/bin:{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    _cu13_lib = f"{BT_VENV}/lib/python3.12/site-packages/nvidia/cu13/lib"
    env["LD_LIBRARY_PATH"] = f"{_cu13_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    subprocess.run(["pkill", "-9", "-f", "VLLM::Worker_TP"], capture_output=True)
    subprocess.run(["pkill", "-9", "-x", "VLLM::Worker_TP"], capture_output=True)
    time.sleep(2)

    server_cmd = [
        f"{BT_VENV}/bin/vllm",
        "serve",
        MODEL_DIR,
        "--tensor-parallel-size",
        "2",
        "--enable-expert-parallel",
        "--enable-ep-weight-filter",
        "--all2all-backend",
        "allgather_reducescatter",
        "--moe-backend",
        "triton",
        "--dtype",
        "bfloat16",
        f"--max-model-len={args.max_model_len}",
        f"--gpu-memory-utilization={args.gpu_mem_util}",
        "--max-num-seqs",
        "256",
        "--enforce-eager",
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
        f"--max-num-batched-tokens={args.max_model_len}",
        "--trust-remote-code",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
    ]
    log("  Launching vLLM server...")
    with open(server_log, "w") as srv_f:
        server_proc = subprocess.Popen(
            server_cmd, env=env, stdout=srv_f, stderr=subprocess.STDOUT
        )
    log(f"  Server PID: {server_proc.pid}")

    log("  Waiting for server to be ready (up to 600s)...")
    if not _wait_server(PORT):
        log("  ERROR: server did not become ready in time.")
        server_proc.terminate()
        return {"error": "server timeout"}

    _warmup_server(PORT, env, args, log, wait_for_training=False)

    bench_cmd = _bench_cmd(args)
    log(f"\n  Running benchmark ({_bench_desc(args)})...")
    with open(bench_log, "w") as bf:
        subprocess.run(bench_cmd, env=env, stdout=bf, stderr=subprocess.STDOUT)

    _kill_server(PORT, log)

    inference = _parse_bench_log(bench_log)
    result = {
        "system": "vllm",
        "config": {
            "max_model_len": args.max_model_len,
            "gpu_mem_util": args.gpu_mem_util,
            "ft": False,
            "request_rate": args.request_rate,
            "num_prompts": args.num_prompts,
            "trace_file": str(args.trace_file) if args.trace_file else None,
        },
        "inference": inference,
        "training": {
            "training_steps": 0,
            "peft_samples_s": None,
            "note": "inference-only baseline",
        },
    }
    out = result_dir / "vllm.json"
    out.write_text(json.dumps(result, indent=2))
    log(f"\n  Results written to {out}")
    return result


# ── LLMStation ─────────────────────────────────────────────────────────────────


def run_llmstation(args, result_dir: Path, log) -> dict:
    log("\n" + "=" * 60)
    log("SYSTEM: LLMStation")
    log("=" * 60)

    result_dir = result_dir.resolve()  # must be absolute: server runs with cwd=/tmp
    lora_path = _lora_path()
    lms_log = result_dir / "lms.log"
    bench_log = result_dir / "llmstation_bench.log"
    mps_dir = result_dir / "nvidia-mps"
    mps_log = result_dir / "nvidia-log"

    # LMS env
    env = os.environ.copy()
    # Override the expandable_segments allocator config inherited from the
    # caller's shell (bench_arxiv.sh exports it for the BubbleTea/vLLM CUDA13
    # build). LLMStation's older multiproc_gpu_executor shares tensors across
    # TP worker processes via CUDA IPC, which PyTorch refuses to do for
    # expandable-segment allocations ("Tensors allocated with
    # expandable_segments:True cannot be shared between processes").
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    env["PATH"] = f"{LMS_VENV}/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{LMS_PYTORCH_LIB}:{env.get('LD_LIBRARY_PATH', '')}"
    env["VLLM_NCCL_SO_PATH"] = LMS_NCCL_SO
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["CUDA_MPS_PIPE_DIRECTORY"] = str(mps_dir)
    env["CUDA_MPS_LOG_DIRECTORY"] = str(mps_log)
    # Do NOT set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE here: per vllm's own
    # arg_utils.py ("...for training workers only"), this cap is meant to
    # apply solely to the LMS finetune subprocess, which sets it on its own
    # os.environ from the --lms-mps-percent CLI flag (already passed below)
    # right before that subprocess initializes CUDA. Setting it in the
    # top-level env here instead leaked the cap onto the *inference* engine
    # and TP workers too — with --lms-mps-percent 20 that meant inference
    # itself was starved to 20% of the GPU's SMs (observed: ~129s mean TTFT).

    # Start MPS — quit any stale daemon first so a leftover instance from a
    # prior failed run doesn't cause "already running" on the new start.
    mps_dir.mkdir(parents=True, exist_ok=True)
    mps_log.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["sh", "-c", "echo quit | nvidia-cuda-mps-control"],
        env=env,
        capture_output=True,  # ignore errors if nothing was running
    )
    time.sleep(1)
    pct = args.lms_mps_percent
    log(f"  Starting CUDA MPS daemon (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={pct})...")
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)
    time.sleep(2)

    # Launch server
    vllm_bin = f"{LMS_VENV}/bin/vllm"
    lms_log.unlink(missing_ok=True)
    lms_server_log = result_dir / "llmstation_server.log"
    server_cmd = [
        vllm_bin,
        "serve",
        MODEL_DIR,
        "-tp=2",
        "--disable-async-output-proc",
        "--disable-log-requests",
        f"--max-model-len={args.max_model_len}",
        "--enforce-eager",
        f"--gpu-memory-utilization={args.gpu_mem_util}",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
    ]
    if args.ft:
        server_cmd += [
            "--enable-lora",
            "--max-loras",
            "4",
            "--max-lora-rank=8",
            "--enable-lms",
            # LLMStation's finetune() loads yahma/alpaca-cleaned via
            # datasets.load_dataset(cache_dir=self.load_config.download_dir).
            # download_dir defaults to None, i.e. ~/.cache/huggingface, which
            # doesn't have this dataset cached — and HF_DATASETS_OFFLINE/
            # TRANSFORMERS_OFFLINE inherited from bench_arxiv.sh's shell env
            # then block it from fetching it, so the finetune worker crashes
            # at startup (0 training steps, no error surfaced to the caller).
            # Point it at the same shared cache BubbleTea already populated.
            "--download-dir=/mnt/nfs/home/ramya/scratch",
            f"--lms-output={result_dir.resolve()}",
            f"--lms-forward-tasklets={args.lms_fwd_tasklets}",
            f"--lms-forward-wait={args.lms_fwd_wait}",
            f"--lms-backward-tasklets={args.lms_bwd_tasklets}",
            f"--lms-backward-wait={args.lms_bwd_wait}",
            f"--lms-mps-percent={args.lms_mps_percent}",
            "--lora-modules",
            f"chat-lora-0={lora_path}",
            f"chat-lora-1={lora_path}",
            f"chat-lora-2={lora_path}",
            f"chat-lora-3={lora_path}",
        ]
    log(f"  Launching LLMStation server (PID logged to {lms_server_log})...")
    with open(lms_server_log, "w") as srv_f:
        server_proc = subprocess.Popen(
            server_cmd,
            env=env,
            stdout=srv_f,
            stderr=subprocess.STDOUT,
            cwd="/tmp",  # prevent local vllm/ dir from shadowing LMS venv's package
        )
    log(f"  Server PID: {server_proc.pid}")

    log("  Waiting for server to be ready (up to 600s)...")
    if not _wait_server(PORT):
        log("  ERROR: server did not become ready in time.")
        server_proc.terminate()
        # Full teardown (not just terminate()) — a timed-out LMS server can
        # still hold the port/GPU memory, which previously caused the next
        # system in the sweep to hit "Address already in use" on bind.
        _kill_server(PORT, log)
        subprocess.run(
            ["sh", "-c", "echo quit | nvidia-cuda-mps-control"],
            env=env,
            capture_output=True,
        )
        return {"error": "server timeout"}

    _warmup_server(PORT, env, args, log, wait_for_training=False)

    start_line = sum(1 for _ in open(lms_log)) if lms_log.exists() else 0  # noqa: SIM115

    # Benchmark — use a clean BT env (no LMS LD_LIBRARY_PATH) so that the
    # BT_VENV Python doesn't crash on torch import due to LMS lib conflicts.
    bench_env = os.environ.copy()
    bench_env["PATH"] = f"{BT_VENV}/bin:{bench_env.get('PATH', '')}"
    bench_env["CUDA_VISIBLE_DEVICES"] = "0,1"
    bench_cmd = _bench_cmd(args)
    log(f"\n  Running benchmark ({_bench_desc(args)})...")
    with open(bench_log, "w") as bf:
        subprocess.run(bench_cmd, env=bench_env, stdout=bf, stderr=subprocess.STDOUT)

    end_line = sum(1 for _ in open(lms_log)) if lms_log.exists() else 0  # noqa: SIM115

    # Tear down
    _kill_server(PORT, log)
    log("  Stopping CUDA MPS...")
    subprocess.run(
        ["sh", "-c", "echo quit | nvidia-cuda-mps-control"],
        env=env,
        capture_output=True,
    )

    # Parse results
    inference = _parse_bench_log(bench_log)
    if args.ft:
        training = _parse_lms_log(lms_log, start_line, end_line)
    else:
        training = {
            "training_steps": 0,
            "peft_samples_s": None,
            "note": "inference-only baseline",
        }

    result = {
        "system": "llmstation",
        "config": {
            "max_model_len": args.max_model_len,
            "gpu_mem_util": args.gpu_mem_util,
            "ft": args.ft,
            "lms_mps_percent": args.lms_mps_percent,
            "lms_fwd_tasklets": args.lms_fwd_tasklets,
            "lms_fwd_wait_ms": args.lms_fwd_wait * 1000,
            "lms_bwd_tasklets": args.lms_bwd_tasklets,
            "lms_bwd_wait_ms": args.lms_bwd_wait * 1000,
            "request_rate": args.request_rate,
            "num_prompts": args.num_prompts,
            "trace_file": str(args.trace_file) if args.trace_file else None,
        },
        "inference": inference,
        "training": training,
    }
    out = result_dir / "llmstation.json"
    out.write_text(json.dumps(result, indent=2))
    log(f"\n  Results written to {out}")
    return result


# ── Bubble tea ─────────────────────────────────────────────────────────────────


def run_bubble_tea(args, result_dir: Path, log) -> dict:
    mode_label = "C+D" if args.ft else "inference-only"
    if args.ft and args.sched_mode == "green_ctx":
        mode_label += f" bwd=green_ctx={args.green_ctx_sms}SM"
    if args.ft and args.fwd_green_ctx:
        mode_label += f" fwd=green_ctx={args.green_ctx_sms}SM"
    log("\n" + "=" * 60)
    log(f"SYSTEM: Bubble tea ({mode_label})")
    log("=" * 60)

    bench_log = result_dir / "bubble_tea_bench.log"
    server_log = result_dir / "bubble_tea_server.log"

    env = os.environ.copy()
    env["PATH"] = f"{BT_VENV}/bin:{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.tp_size))
    # .venv-bubble ships CUDA 13 runtime; make it available to the worker processes.
    _cu13_lib = f"{BT_VENV}/lib/python3.12/site-packages/nvidia/cu13/lib"
    env["LD_LIBRARY_PATH"] = f"{_cu13_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    if args.ft:
        env["VLLM_FT_COMBINED_MODE"] = args.ft_mode
        # Enable real LoRA training: trainer initialises inside the worker at
        # model-load time (see qwen3_moe.py:_maybe_init_bt_trainer). Respect an
        # already-set VLLM_FT_LORA_PATH (e.g. bench_arxiv.sh's LORA= override
        # for a non-Qwen3 model) instead of unconditionally forcing the Qwen3
        # toy adapter -- silently loading the wrong-shaped adapter here caused
        # every Qwen1.5-MoE-A2.7B benchmark to crash on the first LoRA forward
        # (q_proj.lora_B loaded as Qwen3's [4096,16] instead of [2048,16]).
        env["VLLM_FT_LORA_PATH"] = os.environ.get("VLLM_FT_LORA_PATH", BT_LORA_ADAPTER)
        env["VLLM_FT_TOKENIZER_PATH"] = MODEL_DIR
        env["VLLM_FT_CACHE_DIR"] = "/mnt/nfs/home/ramya/scratch"
        env["VLLM_FT_COMBINED_T_FT"] = str(args.t_ft)
        if args.ft_mode == "C+D_dynamic":
            env["VLLM_FT_COMBINED_T_FT_MAX"] = str(args.t_ft_max)
            env["VLLM_FT_WINDOW_OVERHEAD_FRAC"] = str(args.window_overhead_frac)
            env["VLLM_FT_WINDOW_PROBE_CYCLE"] = str(args.window_probe_cycle)
        if args.trigger_rank is not None:
            env["VLLM_FT_COMBINED_TRIGGER_RANK"] = str(args.trigger_rank)
        # Prevent bt_lora_trainer from contacting the HF hub for dataset
        # freshness checks — a stale unauthenticated request stalls 60+s and
        # hangs the vLLM worker during model load.  Run download_alpaca.py
        # once to populate the cache before benchmarking.
        env["HF_DATASETS_OFFLINE"] = "1"  # datasets: use cache, no hub check
        env["TRANSFORMERS_OFFLINE"] = "1"  # transformers: no hub check on tokenizer
        env["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
        env["VLLM_FT_SCHED_MODE"] = args.sched_mode
        env["VLLM_FT_BWD_MODE"] = args.bwd_mode
        if args.bwd_mode in ("slo_budget", "slo_budget_both"):
            env["VLLM_FT_SLOD_MS"] = str(args.slo_d_ms)
        if args.fwd_green_ctx:
            env["VLLM_FT_FWD_GREEN_CTX"] = "1"
        if args.sched_mode == "green_ctx" or args.fwd_green_ctx:
            env["VLLM_FT_GREEN_CTX_SMS"] = str(args.green_ctx_sms)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # On a cold Triton cache, fused_moe_kernel JIT for large prefill shapes
    # can exceed the default 300s timeout.  600s covers first-run compilation;
    # subsequent runs hit the disk cache and complete in <1ms.
    env["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "600"

    # Kill any stale workers before starting (comm is truncated to 15 chars)
    subprocess.run(["pkill", "-9", "-f", "VLLM::Worker_TP"], capture_output=True)
    subprocess.run(["pkill", "-9", "-x", "VLLM::Worker_TP"], capture_output=True)
    time.sleep(2)

    if args.ft:
        BT_COMPLETIONS.unlink(missing_ok=True)
        BT_FWD_BWD_LOG.unlink(missing_ok=True)

    server_cmd = [
        f"{BT_VENV}/bin/vllm",
        "serve",
        MODEL_DIR,
        "--tensor-parallel-size",
        str(args.tp_size),
        "--enable-expert-parallel",
        "--enable-ep-weight-filter",
        "--all2all-backend",
        "allgather_reducescatter",
        "--moe-backend",
        "triton",  # A100: no FlashInfer cubin on SM80
        "--dtype",
        "bfloat16",
        f"--max-model-len={args.max_model_len}",
        f"--gpu-memory-utilization={args.gpu_mem_util}",
        "--max-num-seqs",
        "256",
        "--enforce-eager",
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
        f"--max-num-batched-tokens={args.max_model_len}",
        "--trust-remote-code",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
    ]
    if args.enable_eplb:
        # Same config as scripts/run_qwen3_30b_a3b.sh: 8 redundant expert
        # slots per GPU (64 base + 8), async non-blocking rebalance.
        eplb_config = (
            '{"num_redundant_experts": 8, "window_size": 1000, '
            '"step_interval": 3000, "use_async": true, '
            '"log_balancedness": true, "log_balancedness_interval": 100}'
        )
        server_cmd += ["--enable-eplb", "--eplb-config", eplb_config]
    # Note: --enable-lora is intentionally omitted. The real LoRA trainer
    # (bt_lora_trainer.py) loads adapter weights directly from the safetensors
    # file and does not use vLLM's LoRA serving infrastructure. Adding
    # --enable-lora + --enable-ep-weight-filter together crashes the vLLM v1
    # dummy-profiling run with an IndexError.
    log(f"  Launching bubble tea server ({mode_label})...")
    with open(server_log, "w") as srv_f:
        server_proc = subprocess.Popen(
            server_cmd, env=env, stdout=srv_f, stderr=subprocess.STDOUT
        )
    log(f"  Server PID: {server_proc.pid}")

    log("  Waiting for server to be ready (up to 600s)...")
    if not _wait_server(PORT):
        log("  ERROR: server did not become ready in time.")
        server_proc.terminate()
        return {"error": "server timeout"}

    # ── Warmup: send requests to prime Triton JIT and (if ft) training data ──
    _warmup_server(PORT, env, args, log, wait_for_training=args.ft)

    # Benchmark
    bench_cmd = _bench_cmd(args)
    log(f"\n  Running benchmark ({_bench_desc(args)})...")
    t_bench_start = time.time()
    with open(bench_log, "w") as bf:
        subprocess.run(bench_cmd, env=env, stdout=bf, stderr=subprocess.STDOUT)
    t_bench_end = time.time()

    # Tear down
    _kill_server(PORT, log)

    # Parse results
    inference = _parse_bench_log(bench_log)
    lora_path = env.get("VLLM_FT_LORA_PATH", "")
    if args.ft:
        # If real training was active, parse training stats from the trainer's
        # completion log (written by bt_lora_trainer via the BT_COMPLETIONS path).
        # The file contains one Unix timestamp per completed training step.
        # Fall back to the synthetic-mode completions parser if no steps logged.
        # C+D_dynamic fuses a per-step SLO-solved window <= t_ft_max, not a
        # fixed t_ft -- t_ft_max is only an upper bound here, so fwd_bwd_tok_s
        # (which assumes every pass moved exactly t_ft tokens) overstates
        # dynamic-mode throughput; flagged in the note rather than silently
        # reported as if it were exact, since per-pass window sizes aren't
        # aggregated into this log today.
        _report_t_ft = args.t_ft_max if args.ft_mode == "C+D_dynamic" else args.t_ft
        training = _parse_bt_completions(
            BT_COMPLETIONS, t_bench_start, t_bench_end, t_ft=_report_t_ft
        )
        if args.ft_mode == "C+D_dynamic" and "note" in training:
            training["note"] += (
                f" (t_ft={_report_t_ft} is a max, not fixed -- see [window] "
                "server log for the actual per-step solved window)"
            )
        # Annotate whether real or synthetic training ran
        if lora_path and training.get("training_steps", 0) > 0:
            training["note"] = (
                "real LoRA training (q/k/v/o rank=16, alpaca-cleaned, "
                f"t_ft={_report_t_ft}"
                + (" (max)" if args.ft_mode == "C+D_dynamic" else "")
                + f", adapter={Path(lora_path).name})"
            )
    else:
        training = {
            "training_steps": 0,
            "peft_samples_s": None,
            "note": "inference-only baseline",
        }

    config = {
        "max_model_len": args.max_model_len,
        "gpu_mem_util": args.gpu_mem_util,
        "ft": args.ft,
        "request_rate": args.request_rate,
        "num_prompts": args.num_prompts,
        "trace_file": str(args.trace_file) if args.trace_file else None,
        "real_training": bool(lora_path),
        "enable_eplb": args.enable_eplb,
        "tp_size": args.tp_size,
    }
    if args.ft:
        config["combined_mode"] = args.ft_mode
        config["t_ft"] = args.t_ft
        config["sched_mode"] = args.sched_mode
        config["fwd_green_ctx"] = args.fwd_green_ctx
        if args.sched_mode == "green_ctx" or args.fwd_green_ctx:
            config["green_ctx_sms"] = args.green_ctx_sms
        # Recorded explicitly so result JSONs are self-documenting about which
        # arm actually ran (an earlier compare_results/armB_*/ folder was
        # mislabeled -- it had bwd_mode implicitly "bubble", i.e. arm C, with
        # nothing in the JSON to catch that after the fact).
        config["bwd_mode"] = args.bwd_mode
        if args.bwd_mode in ("slo_budget", "slo_budget_both"):
            config["slo_d_ms"] = args.slo_d_ms
        if args.ft_mode == "C+D_dynamic":
            config["t_ft_max"] = args.t_ft_max
            config["window_overhead_frac"] = args.window_overhead_frac
            config["window_probe_cycle"] = args.window_probe_cycle

    result = {
        "system": "bubble_tea",
        "config": config,
        "inference": inference,
        "training": training,
    }
    out = result_dir / "bubble_tea.json"
    out.write_text(json.dumps(result, indent=2))
    log(f"\n  Results written to {out}")
    return result


# ── Comparison table ───────────────────────────────────────────────────────────


def _fmt(val, fmt=".1f", suffix=""):
    if val is None:
        return "—"
    return f"{val:{fmt}}{suffix}"


def print_comparison(results: dict) -> None:
    """results: dict keyed by system name (vllm / llmstation / bubble_tea)."""
    vr = results.get("vllm", {})
    lms = results.get("llmstation", {})
    bt = results.get("bubble_tea", {})

    vi = vr.get("inference", {})
    vt = vr.get("training", {})
    li = lms.get("inference", {})
    lt = lms.get("training", {})
    bi = bt.get("inference", {})
    btt = bt.get("training", {})

    COL = 15
    print()
    print("=" * (38 + 3 * COL + 2))
    print(f"{'Metric':<38}{'vLLM':>{COL}}{'LLMStation':>{COL}}{'Bubble Tea':>{COL}}")
    print("=" * (38 + 3 * COL + 2))

    def row(label, vval, lval, bval):
        if vval is None and lval is None and bval is None and label.startswith("──"):
            print(f"\n{label}")
            return
        print(f"  {label:<36}{_fmt(vval):>{COL}}{_fmt(lval):>{COL}}{_fmt(bval):>{COL}}")

    row("── Inference ──────────────────────", None, None, None)
    row(
        "Successful requests",
        vi.get("successful_requests"),
        li.get("successful_requests"),
        bi.get("successful_requests"),
    )
    row(
        "Benchmark duration (s)",
        vi.get("duration_s"),
        li.get("duration_s"),
        bi.get("duration_s"),
    )
    row(
        "Request throughput (r/s)",
        vi.get("req_throughput"),
        li.get("req_throughput"),
        bi.get("req_throughput"),
    )
    row(
        "Output tok tput (tok/s)",
        vi.get("out_tok_s"),
        li.get("out_tok_s"),
        bi.get("out_tok_s"),
    )
    row(
        "Mean TTFT (ms)",
        vi.get("mean_ttft_ms"),
        li.get("mean_ttft_ms"),
        bi.get("mean_ttft_ms"),
    )
    row(
        "P50  TTFT (ms)",
        vi.get("p50_ttft_ms"),
        li.get("p50_ttft_ms"),
        bi.get("p50_ttft_ms"),
    )
    row(
        "P99  TTFT (ms)",
        vi.get("p99_ttft_ms"),
        li.get("p99_ttft_ms"),
        bi.get("p99_ttft_ms"),
    )
    row(
        "Mean TPOT (ms)",
        vi.get("mean_tpot_ms"),
        li.get("mean_tpot_ms"),
        bi.get("mean_tpot_ms"),
    )
    row(
        "P50  TPOT (ms)",
        vi.get("p50_tpot_ms"),
        li.get("p50_tpot_ms"),
        bi.get("p50_tpot_ms"),
    )
    row(
        "P99  TPOT (ms)",
        vi.get("p99_tpot_ms"),
        li.get("p99_tpot_ms"),
        bi.get("p99_tpot_ms"),
    )
    row(
        "P99  ITL  (ms)",
        vi.get("p99_itl_ms"),
        li.get("p99_itl_ms"),
        bi.get("p99_itl_ms"),
    )
    row("── Training ────────────────────────", None, None, None)
    row(
        "Training steps",
        vt.get("training_steps"),
        lt.get("training_steps"),
        btt.get("training_steps"),
    )
    row(
        "PEFT throughput (samp/s)",
        vt.get("peft_samples_s"),
        lt.get("peft_samples_s"),
        btt.get("peft_samples_s"),
    )
    row("Fwd+bwd passes", None, None, btt.get("fwd_bwd_passes"))
    row("Fwd+bwd tok/s", None, None, btt.get("fwd_bwd_tok_s"))

    # Delta rows vs vLLM baseline (if present)
    print()
    if vi.get("p99_ttft_ms"):
        base = vi["p99_ttft_ms"]
        parts = []
        if li.get("p99_ttft_ms"):
            parts.append(f"lms {li['p99_ttft_ms'] - base:>+.1f} ms")
        if bi.get("p99_ttft_ms"):
            parts.append(f"bt {bi['p99_ttft_ms'] - base:>+.1f} ms")
        if parts:
            print(f"  P99 TTFT overhead vs vLLM:    {',  '.join(parts)}")
    if lt.get("peft_samples_s") and btt.get("peft_samples_s"):
        ratio = btt["peft_samples_s"] / lt["peft_samples_s"]
        print(f"  PEFT tput ratio (bt / lms):   {ratio:.2f}×")

    print()
    print("  Note: Bubble Tea training uses synthetic Qwen3-30B-A3B-shaped tensors")
    print("  (t_ft=128); LLMStation uses real ShareGPT fine-tuning data.")
    print("=" * (38 + 3 * COL + 2))


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    global MODEL_DIR
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--system",
        default="all",
        help=(
            "Comma-separated list of systems to run, or 'all'. "
            "Choices: vllm, llmstation, bubble_tea  (default: all)"
        ),
    )
    parser.add_argument(
        "--model",
        default=MODEL_DIR,
        help=f"Path to the base model to serve (default: {MODEL_DIR})",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=1.0,
        help="Inference request rate in req/s (default: 1)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=200,
        help="Number of inference prompts (default: 200)",
    )
    parser.add_argument(
        "--dataset-name",
        default="sharegpt",
        choices=["sharegpt", "custom"],
        help="Dataset format (default: sharegpt)",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=SHAREGPT,
        help="Path to dataset file (ShareGPT JSON or custom JSONL)",
    )
    parser.add_argument(
        "--custom-output-len",
        type=int,
        default=256,
        help="Fixed output token count for custom dataset (default: 256)",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Override output len for sharegpt dataset",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=None,
        help="Path to a BurstGPT trace-window JSON (see "
        "compare_results/burstgpt_traces/). When set, replays the "
        "trace's real request arrival timing and token lengths via "
        "burstgpt_trace_replay.py instead of `vllm bench serve` "
        "with --dataset-name/--request-rate/--num-prompts.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(__file__).parent / "compare_results",
        help="Directory to write result JSON files",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max model length for both systems (default: 4096)",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.5,
        help="GPU memory utilization for both systems (default: 0.5)",
    )
    # LLMStation tasklet config
    parser.add_argument(
        "--lms-mps-percent",
        type=int,
        default=10,
        help="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for LLMStation training workers "
        "(default: 10). Prior runs used 100 which gave training workers "
        "all SMs and caused catastrophic inference latency.",
    )
    parser.add_argument("--lms-fwd-tasklets", type=int, default=8)
    parser.add_argument(
        "--lms-fwd-wait",
        type=float,
        default=0.005,
        help="LMS forward wait in seconds (default: 0.005 = 5ms)",
    )
    parser.add_argument("--lms-bwd-tasklets", type=int, default=8)
    parser.add_argument(
        "--lms-bwd-wait",
        type=float,
        default=0.005,
        help="LMS backward wait in seconds (default: 0.005 = 5ms)",
    )
    parser.add_argument(
        "--t-ft",
        type=int,
        default=128,
        help="Training tokens per C+D cycle (default: 128)",
    )
    parser.add_argument(
        "--ft-mode",
        default="C+D",
        choices=["C+D", "D+D", "C+D_batch", "B+D", "C+D_dynamic"],
        help="BubbleTea training mode (default: C+D). C+D_dynamic is C+D_batch's "
        "MoE-FFN token fusion with a per-step SLO-solved window instead of a "
        "fixed --t-ft size -- see --t-ft-max/--window-overhead-frac/"
        "--window-probe-cycle.",
    )
    parser.add_argument(
        "--t-ft-max",
        type=int,
        default=512,
        help="Max FT tokens the trainer produces per round for --ft-mode "
        "C+D_dynamic (VLLM_FT_COMBINED_T_FT_MAX; default 512). The solved "
        "per-step window is clamped to this; --t-ft is ignored in this mode.",
    )
    parser.add_argument(
        "--window-overhead-frac",
        type=float,
        default=0.15,
        help="Relative per-bucket SLO ceiling for --ft-mode C+D_dynamic's "
        "window solve: don't let a step's latency exceed (1+frac)*l0 for its "
        "own n_tok bucket (VLLM_FT_WINDOW_OVERHEAD_FRAC; default 0.15).",
    )
    parser.add_argument(
        "--window-probe-cycle",
        type=int,
        default=10,
        help="Probe cadence (in steps, per n_tok bucket) for --ft-mode "
        "C+D_dynamic's window solver's l0/l1 liveness probes "
        "(VLLM_FT_WINDOW_PROBE_CYCLE; default 10 -- unvalidated starting "
        "guess, needs its own calibration pass, see session notes on "
        "VLLM_FT_EP_STARVE_PROBE_CYCLE's cycle=10->3 retune).",
    )
    parser.add_argument(
        "--trigger-rank",
        type=int,
        default=None,
        help="Only this TP rank dispatches training sub-ops "
        "(default: all ranks). Use 1 for light EP rank only.",
    )
    parser.add_argument(
        "--sched-mode",
        default="bubble",
        choices=["bubble", "green_ctx"],
        help="Backward scheduler: bubble (default) or "
        "green_ctx (naive baseline, runs continuously)",
    )
    parser.add_argument(
        "--fwd-green-ctx",
        action="store_true",
        default=False,
        help="Run forward scheduler on a green context stream "
        "with dedicated SMs (default: off)",
    )
    parser.add_argument(
        "--green-ctx-sms",
        type=int,
        default=8,
        help="SMs dedicated to training in green_ctx mode (default: 8)",
    )
    parser.add_argument(
        "--bwd-mode",
        default="bubble",
        choices=["bubble", "decode", "both", "slo_budget", "slo_budget_both"],
        help="Backward sub-op trigger (VLLM_FT_BWD_MODE): bubble (default, "
        "EP-bubble gated), decode (fixed per-layer count during decode, "
        "LLMStation-style), both (bubble + fixed-count decode), slo_budget "
        "(Arm B: LLMStation's actual Eq. 1 SLO-budget rule, solved per "
        "decode iteration instead of a fixed count), or slo_budget_both "
        "(Arm B+C: slo_budget's decode dispatch + bubble-gated prefill "
        "dispatch, both active -- see moe_runner.py)",
    )
    parser.add_argument(
        "--slo-d-ms",
        type=float,
        default=80.0,
        help="Decode-iteration SLO in ms for --bwd-mode slo_budget's Eq. 1 "
        "solve (VLLM_FT_SLOD_MS; default 80, matching the LLMStation "
        "paper's larger-model default)",
    )
    parser.add_argument(
        "--ft",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable fine-tuning co-serving (default: on). "
        "Use --no-ft for inference-only baseline.",
    )
    parser.add_argument(
        "--enable-eplb",
        action="store_true",
        default=False,
        help="Enable expert-parallel load balancing on the bubble_tea server "
        "(--enable-eplb + --eplb-config, same 8-redundant-expert async "
        "config as scripts/run_qwen3_30b_a3b.sh). Default: off, matching "
        "every prior sweep in this file's history.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=2,
        help="Tensor-parallel (and expert-parallel) size for the bubble_tea "
        "server, and number of GPUs (0..tp_size-1) exposed via "
        "CUDA_VISIBLE_DEVICES. Default: 2, matching every prior sweep in "
        "this file's history. Untested above 2 as of 2026-07-16.",
    )
    args = parser.parse_args()
    MODEL_DIR = args.model

    # Validate
    if args.trace_file is not None:
        if not args.trace_file.exists():
            print(f"ERROR: trace file not found: {args.trace_file}", file=sys.stderr)
            sys.exit(1)
    elif not args.dataset_path.exists():
        print(f"ERROR: dataset not found: {args.dataset_path}", file=sys.stderr)
        sys.exit(1)

    # Normalise --system into a set
    VALID = {"vllm", "llmstation", "bubble_tea"}
    if args.system.lower() == "all":
        systems = {"vllm", "llmstation", "bubble_tea"}
    else:
        systems = set(s.strip() for s in args.system.split(","))
        unknown = systems - VALID
        if unknown:
            print(
                f"ERROR: unknown system(s): {unknown}. Valid: {VALID}", file=sys.stderr
            )
            sys.exit(1)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    log = print

    results = {}

    if "vllm" in systems:
        results["vllm"] = run_vllm(args, args.result_dir, log)

    if "llmstation" in systems:
        results["llmstation"] = run_llmstation(args, args.result_dir, log)

    if "bubble_tea" in systems:
        results["bubble_tea"] = run_bubble_tea(args, args.result_dir, log)

    # Merge in any results that already exist on disk from prior runs so that
    # partial re-runs (e.g. --system llmstation,bubble_tea) still produce the
    # full 3-way comparison table.
    for sys_name in ("vllm", "llmstation", "bubble_tea"):
        if sys_name not in results:
            existing = args.result_dir / f"{sys_name}.json"
            if existing.exists():
                with open(existing) as _f:
                    results[sys_name] = json.load(_f)
                log(f"Loaded existing {sys_name} results from {existing}")

    if len(results) > 1:
        print_comparison(results)
    elif results:
        name, r = next(iter(results.items()))
        print(f"\n{name} results:")
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
