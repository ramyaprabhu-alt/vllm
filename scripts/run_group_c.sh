#!/bin/bash
# Group C: 8192 input, max-model-len=16384, gpu-mem-util=0.7
# 9 configs × 3 systems = 27 runs

set -euo pipefail

PYTHON=".venv/bin/python"
LOG="compare_results/group_c_progress.log"
mkdir -p compare_results

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

INPUT_LEN=8192
MAX_MODEL_LEN=16384
GPU_MEM=0.7
T_FT=512

DISTS=(poisson gamma burst_lull)
RATES=(0.4 1.0 4.0)
SYSTEMS=(vllm llmstation bubble_tea)

total=27
done_count=0

for dist in "${DISTS[@]}"; do
  for rate in "${RATES[@]}"; do
    trace="compare_results/synthetic_traces/synth_${INPUT_LEN}_out256_${dist}_${rate}rps.json"
    result_dir="compare_results/synth_${INPUT_LEN}_${dist}_${rate}rps"

    for sys in "${SYSTEMS[@]}"; do
      done_count=$((done_count + 1))

      log "[$done_count/$total] Waiting for port 8000..."
      wait_port_free

      log "[$done_count/$total] START $sys / $dist / ${rate}rps"

      if $PYTHON compare_benchmark.py \
        --system "$sys" \
        --trace-file "$trace" \
        --max-model-len $MAX_MODEL_LEN --gpu-mem-util $GPU_MEM \
        --t-ft $T_FT \
        --result-dir "$result_dir" >> "$LOG" 2>&1; then
        log "[$done_count/$total] DONE  $sys / $dist / ${rate}rps"
      else
        log "[$done_count/$total] FAIL  $sys / $dist / ${rate}rps (exit=$?)"
      fi

      # Cooldown between runs
      sleep 10
    done
  done
done

log "=== Group C complete ($done_count/$total runs) ==="
