#!/bin/bash
set -euo pipefail
VENV="/mnt/nfs/home/ramya/vllm/.venv"
PYTHON="$VENV/bin/python"
VLLM_BIN="$VENV/bin/vllm"
export PATH="$VENV/bin:$PATH"
MODEL="/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
PORT=8000
LOG="/mnt/nfs/home/ramya/vllm/run_ttft_comparison.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

wait_ready() {
    for i in $(seq 1 120); do
        curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && log "  ready after $((i*5))s" && return 0
        sleep 5
    done; return 1
}

kill_server() {
    lsof -ti ":$PORT" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    pkill -9 -f "vllm.worker\|vllm serve\|VLLM::Worker" 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | xargs -r kill -9 2>/dev/null || true
    sleep 5
    local w=0
    while true; do
        u0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0 2>/dev/null | tr -d ' ')
        u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 2>/dev/null | tr -d ' ')
        [ "${u0:-99999}" -lt 5120 ] && [ "${u1:-99999}" -lt 5120 ] && log "  GPU freed" && break
        sleep 5; w=$((w+5)); [ "$w" -ge 120 ] && break
    done
}

COMMON=(
    --tensor-parallel-size 2 --enable-expert-parallel --enable-ep-weight-filter
    --all2all-backend allgather_reducescatter --moe-backend auto
    --enable-flashinfer-autotune --dtype bfloat16
    --max-model-len 32768 --gpu-memory-utilization 0.92 --max-num-seqs 256
    --enforce-eager --no-enable-chunked-prefill --no-enable-prefix-caching
    --max-num-batched-tokens 32768 --trust-remote-code --host 0.0.0.0 --port "$PORT"
)

: > "$LOG"
log "=== TTFT comparison: no-EPLB vs EPLB ==="

# ── no-EPLB ───────────────────────────────────────────────────────────────────
log "=== no-EPLB ==="
"$VLLM_BIN" serve "$MODEL" "${COMMON[@]}" >> "$LOG" 2>&1 &
wait_ready
for t_in in 4096 8192 16384; do
    log "  T_in=$t_in"
    "$PYTHON" /mnt/nfs/home/ramya/vllm/bubble_sched_live_test.py \
        --mode baseline --port "$PORT" --t-in "$t_in" --reps 20 --warmup-reps 5 \
        2>&1 | tee -a "$LOG"
done
kill_server
log "no-EPLB done."

# ── EPLB ──────────────────────────────────────────────────────────────────────
log "=== EPLB ==="
"$VLLM_BIN" serve "$MODEL" "${COMMON[@]}" \
    --enable-eplb \
    --eplb-config '{"num_redundant_experts":8,"window_size":1000,"step_interval":3000,"use_async":true}' \
    >> "$LOG" 2>&1 &
wait_ready
for t_in in 4096 8192 16384; do
    log "  T_in=$t_in"
    "$PYTHON" /mnt/nfs/home/ramya/vllm/bubble_sched_live_test.py \
        --mode baseline --port "$PORT" --t-in "$t_in" --reps 20 --warmup-reps 5 \
        2>&1 | tee -a "$LOG"
done
kill_server
log "EPLB done."
log "=== ALL DONE ==="
