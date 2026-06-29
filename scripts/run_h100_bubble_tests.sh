#!/bin/bash
# H100 bubble scheduler test suite — all phases sequential.
# Results tee'd to run_h100_bubble_tests.log alongside this script.
#
# Phase 1: no-EPLB bubble measurement (mixed_bubble_measure.py)
# Phase 2: EPLB bubble measurement
# Phase 3: TTFT baseline (bubble_sched_live_test.py --mode baseline)
# Phase 4: TTFT with scheduler demo (trigger_rank=1, fills=1)
# Phase 5: detailed per-layer bubble profile (find_bubbles.py, no-EPLB)

set -euo pipefail

VENV="/mnt/nfs/home/ramya/vllm/.venv"
PYTHON="$VENV/bin/python"
VLLM_BIN="$VENV/bin/vllm"
export PATH="$VENV/bin:$PATH"
MODEL="/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
SCRIPTS_DIR="/mnt/nfs/home/ramya/vllm"
PORT=8000
LOG="$SCRIPTS_DIR/run_h100_bubble_tests.log"

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

wait_server_ready() {
    local port=$1
    log "  waiting for server on port $port ..."
    for i in $(seq 1 120); do
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            log "  server ready after ${i}s"
            return 0
        fi
        sleep 5
    done
    log "  ERROR: server not ready after 600s"
    return 1
}

kill_server() {
    local port=$1
    local pid
    pid=$(lsof -ti ":$port" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        log "  killing server pid $pid on port $port"
        kill "$pid" 2>/dev/null || true
        sleep 8
        kill -9 "$pid" 2>/dev/null || true
    fi
    # Kill any stray vllm workers and TP processes
    pkill -9 -f "vllm.worker\|RayWorker\|vllm serve" 2>/dev/null || true
    sleep 5
    # Wait until both GPUs show <5 GiB used (model weights freed)
    log "  waiting for GPU memory to free..."
    local waited=0
    while true; do
        local used0 used1
        used0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0 2>/dev/null | tr -d ' ')
        used1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 2>/dev/null | tr -d ' ')
        # memory.used is in MiB; 5120 MiB = 5 GiB
        if [ "${used0:-99999}" -lt 5120 ] && [ "${used1:-99999}" -lt 5120 ]; then
            log "  GPU memory freed (GPU0: ${used0}MiB, GPU1: ${used1}MiB)"
            break
        fi
        log "  still waiting... GPU0: ${used0}MiB, GPU1: ${used1}MiB (waited ${waited}s)"
        sleep 5
        waited=$((waited + 5))
        if [ "$waited" -ge 120 ]; then
            log "  WARNING: GPU memory not fully freed after 120s, proceeding anyway"
            break
        fi
    done
}

# Common inference flags for profiling (deterministic, no prefix-cache reuse)
COMMON_FLAGS=(
    --tensor-parallel-size 2
    --enable-expert-parallel
    --enable-ep-weight-filter
    --all2all-backend allgather_reducescatter
    --moe-backend auto
    --enable-flashinfer-autotune
    --dtype bfloat16
    --max-model-len 32768
    --gpu-memory-utilization 0.92
    --max-num-seqs 256
    --enforce-eager
    --no-enable-chunked-prefill
    --no-enable-prefix-caching
    --max-num-batched-tokens 32768
    --trust-remote-code
    --host 0.0.0.0
    --port "$PORT"
)

EPLB_FLAGS=(
    --enable-eplb
    --eplb-config '{"num_redundant_experts":8,"window_size":1000,"step_interval":3000,"use_async":true}'
)

# ── Init ──────────────────────────────────────────────────────────────────────

: > "$LOG"
log "=== H100 Bubble Scheduler Test Suite ==="
log "Hardware: 2x H100-SXM4-80GB (SM90)"
log "Model:    Qwen3-30B-A3B"
log ""

rm -f /tmp/vllm_bubble_rank*.json

# ── Phase 1: no-EPLB bubble measurement ───────────────────────────────────────

log "=== PHASE 1: Bubble measurement — no-EPLB ==="

VLLM_BUBBLE_PROFILE=1 "$VLLM_BIN" serve "$MODEL" \
    "${COMMON_FLAGS[@]}" >> "$LOG" 2>&1 &
SERVER_PID=$!
log "  server PID $SERVER_PID"
wait_server_ready "$PORT"

log "  running mixed_bubble_measure.py (no-EPLB) ..."
"$PYTHON" "$SCRIPTS_DIR/mixed_bubble_measure.py" \
    --port "$PORT" \
    --tokens 4096 8192 16384 \
    --reps 3 2>&1 | tee -a "$LOG"

log "  running find_bubbles.py (no-EPLB, per-layer detail) ..."
"$PYTHON" "$SCRIPTS_DIR/find_bubbles.py" \
    --port "$PORT" \
    --tokens 4096 8192 16384 \
    --bs 1 2>&1 | tee -a "$LOG"

kill_server "$PORT"
log "Phase 1 complete."
log ""

# ── Phase 2: EPLB bubble measurement ─────────────────────────────────────────

log "=== PHASE 2: Bubble measurement — EPLB ==="

rm -f /tmp/vllm_bubble_rank*.json

VLLM_BUBBLE_PROFILE=1 "$VLLM_BIN" serve "$MODEL" \
    "${COMMON_FLAGS[@]}" "${EPLB_FLAGS[@]}" >> "$LOG" 2>&1 &
SERVER_PID=$!
log "  server PID $SERVER_PID"
wait_server_ready "$PORT"

log "  running mixed_bubble_measure.py (EPLB) ..."
"$PYTHON" "$SCRIPTS_DIR/mixed_bubble_measure.py" \
    --port "$PORT" \
    --tokens 4096 8192 16384 \
    --reps 3 2>&1 | tee -a "$LOG"

kill_server "$PORT"
log "Phase 2 complete."
log ""

# ── Phase 3: TTFT baseline ────────────────────────────────────────────────────

log "=== PHASE 3: TTFT baseline (no scheduler) ==="

"$VLLM_BIN" serve "$MODEL" \
    "${COMMON_FLAGS[@]}" >> "$LOG" 2>&1 &
SERVER_PID=$!
log "  server PID $SERVER_PID"
wait_server_ready "$PORT"

log "  running bubble_sched_live_test.py --mode baseline (T_in=16384, 20 reps) ..."
"$PYTHON" "$SCRIPTS_DIR/bubble_sched_live_test.py" \
    --mode baseline \
    --port "$PORT" \
    --t-in 16384 \
    --reps 20 \
    --warmup-reps 5 2>&1 | tee -a "$LOG"

kill_server "$PORT"
log "Phase 3 complete."
log ""

# ── Phase 4: TTFT with scheduler (trigger_rank=1, fills=1) ────────────────────

log "=== PHASE 4: TTFT scheduler (trigger_rank=1, fills=1) ==="

VLLM_BUBBLE_SCHED_DEMO=1 VLLM_BUBBLE_SCHED_RANK=0 VLLM_BUBBLE_SCHED_FILLS=1 \
    "$VLLM_BIN" serve "$MODEL" \
    "${COMMON_FLAGS[@]}" >> "$LOG" 2>&1 &
SERVER_PID=$!
log "  server PID $SERVER_PID"
wait_server_ready "$PORT"

log "  running bubble_sched_live_test.py --mode sched (T_in=16384, 20 reps) ..."
"$PYTHON" "$SCRIPTS_DIR/bubble_sched_live_test.py" \
    --mode sched \
    --port "$PORT" \
    --t-in 16384 \
    --reps 20 \
    --warmup-reps 5 2>&1 | tee -a "$LOG"

kill_server "$PORT"
log "Phase 4 complete."
log ""

log "=== ALL PHASES COMPLETE ==="
log "Full log: $LOG"
