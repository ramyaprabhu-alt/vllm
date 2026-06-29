#!/bin/bash
# Synthetic sweep: 8192 input tokens, all distributions and rates
# LMS-first ordering to avoid port-binding race
# Uses --lms-mps-percent=10 (new default) for fair comparison
set -e

cd /mnt/nfs/home/ramya/vllm
PYTHON=".venv/bin/python"
TRACES="compare_results/synthetic_traces"

DISTS=("poisson" "gamma" "burst_lull")
RATES=("0.4" "1.0" "4.0")

# Group C server config (8192 input tokens)
MAX_MODEL_LEN=16384
GPU_MEM_UTIL=0.7

echo "=== Synthetic sweep: 8192 input, mps-percent=10 ==="
echo "  max-model-len=$MAX_MODEL_LEN  gpu-mem-util=$GPU_MEM_UTIL"
echo "  Started: $(date)"
echo ""

for dist in "${DISTS[@]}"; do
    for rate in "${RATES[@]}"; do
        TRACE="$TRACES/synth_8192_out256_${dist}_${rate}rps.json"
        RESULT_DIR="compare_results/synth_8192_${dist}_${rate}rps"

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Config: 8192/${dist}/${rate}rps"
        echo "  Trace:  $TRACE"
        echo "  Output: $RESULT_DIR"
        echo "  Time:   $(date)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Run LLMStation first (port-binding race fix)
        echo ">>> LLMStation (mps-percent=10)..."
        $PYTHON compare_benchmark.py \
            --system llmstation \
            --trace-file "$TRACE" \
            --result-dir "$RESULT_DIR" \
            --max-model-len $MAX_MODEL_LEN \
            --gpu-mem-util $GPU_MEM_UTIL \
            --lms-mps-percent 10 \
            2>&1 | tee -a "${RESULT_DIR}/sweep.log"

        echo ""
        sleep 15  # cooldown between systems

        # Run vLLM
        echo ">>> vLLM..."
        $PYTHON compare_benchmark.py \
            --system vllm \
            --trace-file "$TRACE" \
            --result-dir "$RESULT_DIR" \
            --max-model-len $MAX_MODEL_LEN \
            --gpu-mem-util $GPU_MEM_UTIL \
            2>&1 | tee -a "${RESULT_DIR}/sweep.log"

        echo ""
        sleep 15

        # Run Bubble Tea
        echo ">>> Bubble Tea..."
        $PYTHON compare_benchmark.py \
            --system bubble_tea \
            --trace-file "$TRACE" \
            --result-dir "$RESULT_DIR" \
            --max-model-len $MAX_MODEL_LEN \
            --gpu-mem-util $GPU_MEM_UTIL \
            --t-ft 512 \
            2>&1 | tee -a "${RESULT_DIR}/sweep.log"

        echo ""
        echo "  Done: ${dist}/${rate}rps at $(date)"
        sleep 15
    done
done

echo ""
echo "=== Sweep complete: $(date) ==="
