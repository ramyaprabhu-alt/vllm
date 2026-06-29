#!/bin/bash
# Group A: 2048 input, max-model-len=4096, gpu-mem-util=0.5
# 9 configs × 3 systems = 27 runs (minus 1 smoke test already done)

set -euo pipefail

PYTHON=".venv/bin/python"
LOG="compare_results/group_a_progress.log"
mkdir -p compare_results

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

INPUT_LEN=2048
MAX_MODEL_LEN=4096
GPU_MEM=0.5
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
      # Skip the smoke test we already ran
      if [[ "$dist" == "poisson" && "$rate" == "1.0" && "$sys" == "vllm" ]]; then
        done_count=$((done_count + 1))
        log "[$done_count/$total] SKIP $sys / $dist / ${rate}rps (smoke test already done)"
        continue
      fi

      done_count=$((done_count + 1))
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
    done
  done
done

log "=== Group A complete ($done_count/$total runs) ==="
