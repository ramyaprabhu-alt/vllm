#!/bin/bash
# Group A LLMStation recovery: re-run all 9 LLMStation configs
# that failed due to port-binding race in the main sweep.

set -euo pipefail

PYTHON=".venv/bin/python"
LOG="compare_results/group_a_lms_recovery.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

wait_port_free() {
    local max_wait=60
    local waited=0
    while ss -tlnp 2>/dev/null | grep -q ":8000 "; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -ge $max_wait ]; then
            log "  WARNING: port 8000 still in use after ${max_wait}s"
            return 1
        fi
    done
    return 0
}

INPUT_LEN=2048
MAX_MODEL_LEN=4096
GPU_MEM=0.5
T_FT=512

DISTS=(poisson gamma burst_lull)
RATES=(0.4 1.0 4.0)

total=9
done_count=0

for dist in "${DISTS[@]}"; do
  for rate in "${RATES[@]}"; do
    done_count=$((done_count + 1))
    trace="compare_results/synthetic_traces/synth_${INPUT_LEN}_out256_${dist}_${rate}rps.json"
    result_dir="compare_results/synth_${INPUT_LEN}_${dist}_${rate}rps"

    # Remove stale LLMStation logs from failed run
    rm -f "$result_dir"/llmstation_bench.log "$result_dir"/llmstation_server.log
    rm -f "$result_dir"/llmstation.json

    log "[$done_count/$total] Waiting for port 8000..."
    wait_port_free

    log "[$done_count/$total] START llmstation / $dist / ${rate}rps"

    if $PYTHON compare_benchmark.py \
      --system llmstation \
      --trace-file "$trace" \
      --max-model-len $MAX_MODEL_LEN --gpu-mem-util $GPU_MEM \
      --t-ft $T_FT \
      --result-dir "$result_dir" >> "$LOG" 2>&1; then
      log "[$done_count/$total] DONE  llmstation / $dist / ${rate}rps"
    else
      log "[$done_count/$total] FAIL  llmstation / $dist / ${rate}rps (exit=$?)"
    fi

    # Extra cooldown between runs
    sleep 10
  done
done

log "=== LLMStation recovery complete ($done_count/$total) ==="
