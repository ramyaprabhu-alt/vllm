#!/bin/bash
# Re-measure bubbles with fixed profiling (single synchronize() per prefill).
set -euo pipefail

VENV="/mnt/nfs/home/ramya/vllm/.venv"
PYTHON="$VENV/bin/python"
VLLM_BIN="$VENV/bin/vllm"
export PATH="$VENV/bin:$PATH"
MODEL="/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
SCRIPTS_DIR="/mnt/nfs/home/ramya/vllm"
PORT=8000
LOG="$SCRIPTS_DIR/run_h100_bubble_fixed.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

wait_server_ready() {
    log "  waiting for server on port $1 ..."
    for i in $(seq 1 120); do
        curl -sf "http://localhost:$1/health" > /dev/null 2>&1 && log "  server ready after $((i*5))s" && return 0
        sleep 5
    done
    log "  ERROR: server not ready after 600s"; return 1
}

kill_server() {
    local pid; pid=$(lsof -ti ":$PORT" 2>/dev/null || true)
    [ -n "$pid" ] && { log "  killing $pid"; kill "$pid" 2>/dev/null || true; sleep 8; kill -9 "$pid" 2>/dev/null || true; }
    pkill -9 -f "vllm.worker\|vllm serve\|VLLM::Worker" 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | xargs -r kill -9 2>/dev/null || true
    sleep 5
    log "  waiting for GPU memory to free..."
    local w=0
    while true; do
        local u0 u1
        u0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0 2>/dev/null | tr -d ' ')
        u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 2>/dev/null | tr -d ' ')
        [ "${u0:-99999}" -lt 5120 ] && [ "${u1:-99999}" -lt 5120 ] && log "  freed (GPU0:${u0}MiB GPU1:${u1}MiB)" && break
        log "  GPU0:${u0}MiB GPU1:${u1}MiB (${w}s)"; sleep 5; w=$((w+5))
        [ "$w" -ge 120 ] && log "  WARNING: giving up after 120s" && break
    done
}

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

: > "$LOG"
rm -f /tmp/vllm_bubble_rank*.json
log "=== H100 Bubble Measurement (fixed profiling: 1 sync/prefill) ==="
log ""

# ── Phase 1: no-EPLB ─────────────────────────────────────────────────────────
log "=== PHASE 1: no-EPLB ==="

VLLM_BUBBLE_PROFILE=1 "$VLLM_BIN" serve "$MODEL" \
    "${COMMON_FLAGS[@]}" >> "$LOG" 2>&1 &
SERVER_PID=$!
log "  server PID $SERVER_PID"
wait_server_ready "$PORT"

log "  running mixed_bubble_measure.py ..."
"$PYTHON" "$SCRIPTS_DIR/mixed_bubble_measure.py" \
    --port "$PORT" --tokens 4096 8192 16384 --reps 3 2>&1 | tee -a "$LOG"

log "  running find_bubbles.py (per-layer detail) ..."
"$PYTHON" "$SCRIPTS_DIR/find_bubbles.py" \
    --port "$PORT" --tokens 4096 8192 16384 --bs 1 2>&1 | tee -a "$LOG"

kill_server
log "Phase 1 complete."
log ""

# ── Phase 2: EPLB ─────────────────────────────────────────────────────────────
log "=== PHASE 2: EPLB ==="
rm -f /tmp/vllm_bubble_rank*.json

VLLM_BUBBLE_PROFILE=1 "$VLLM_BIN" serve "$MODEL" \
    "${COMMON_FLAGS[@]}" \
    --enable-eplb \
    --eplb-config '{"num_redundant_experts":8,"window_size":1000,"step_interval":3000,"use_async":true}' \
    >> "$LOG" 2>&1 &
SERVER_PID=$!
log "  server PID $SERVER_PID"
wait_server_ready "$PORT"

log "  running mixed_bubble_measure.py ..."
"$PYTHON" "$SCRIPTS_DIR/mixed_bubble_measure.py" \
    --port "$PORT" --tokens 4096 8192 16384 --reps 3 2>&1 | tee -a "$LOG"

kill_server
log "Phase 2 complete."
log ""
log "=== DONE ==="
