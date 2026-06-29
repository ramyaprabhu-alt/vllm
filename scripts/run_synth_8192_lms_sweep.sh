#!/bin/bash
# LLMStation sweep: 8192 input, mps=10, mem=0.55
# Run LMS-first for all 9 configs, then vLLM+BT after
set -e
cd /mnt/nfs/home/ramya/vllm

DISTS=("poisson" "gamma" "burst_lull")
RATES=("0.4" "1.0" "4.0")
TRACES="compare_results/synthetic_traces"

echo "=== 8192 sweep: LMS mps=10 mem=0.55, then vLLM+BT mem=0.7 ==="
echo "  Started: $(date)"

for dist in "${DISTS[@]}"; do
    for rate in "${RATES[@]}"; do
        TRACE="$TRACES/synth_8192_out256_${dist}_${rate}rps.json"
        RESULT="compare_results/synth_8192_${dist}_${rate}rps"
        echo ""
        echo "━━━ ${dist}/${rate}rps ━━━ $(date) ━━━"

        echo ">>> LLMStation"
        .venv/bin/python compare_benchmark.py \
            --system llmstation \
            --trace-file "$TRACE" \
            --result-dir "$RESULT" \
            --max-model-len 16384 \
            --gpu-mem-util 0.55 \
            --lms-mps-percent 10
        sleep 15

        echo ">>> vLLM"
        .venv/bin/python compare_benchmark.py \
            --system vllm \
            --trace-file "$TRACE" \
            --result-dir "$RESULT" \
            --max-model-len 16384 \
            --gpu-mem-util 0.7
        sleep 15

        echo ">>> Bubble Tea"
        .venv/bin/python compare_benchmark.py \
            --system bubble_tea \
            --trace-file "$TRACE" \
            --result-dir "$RESULT" \
            --max-model-len 16384 \
            --gpu-mem-util 0.7 \
            --t-ft 512
        sleep 15
    done
done

echo ""
echo "=== Sweep complete: $(date) ==="
