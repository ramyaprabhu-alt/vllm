#!/bin/bash
# Green context vs BubbleTea vs vLLM comparison
# arxiv dataset, 200 prompts, 0.4 and 4.0 qps
set -e

PYTHON="/mnt/nfs/home/ramya/vllm/.venv/bin/python"
BENCH="/mnt/nfs/home/ramya/vllm/compare_benchmark.py"
ARXIV="/mnt/nfs/home/ramya/scratch/arxiv_bench_500.jsonl"

COMMON_ARGS="--dataset-name custom --dataset-path $ARXIV --num-prompts 200 \
  --max-model-len 16384 --gpu-mem-util 0.85 --ft-mode C+D_batch --t-ft 512 \
  --custom-output-len 256"

echo "=========================================="
echo "Starting green_ctx vs bubble vs vLLM sweep"
echo "$(date)"
echo "=========================================="

# --- 0.4 qps ---

echo ""
echo ">>> [1/6] vLLM baseline @ 0.4 qps"
$PYTHON $BENCH --system vllm --request-rate 0.4 \
  --result-dir compare_results/greenctx_vs_bubble_0.4qps \
  $COMMON_ARGS 2>&1 | tee compare_results/greenctx_vs_bubble_0.4qps/run.log

echo ""
echo ">>> [2/6] BubbleTea (bubble sched) @ 0.4 qps"
$PYTHON $BENCH --system bubble_tea --request-rate 0.4 \
  --sched-mode bubble \
  --result-dir compare_results/greenctx_vs_bubble_0.4qps \
  $COMMON_ARGS 2>&1 | tee -a compare_results/greenctx_vs_bubble_0.4qps/run.log

echo ""
echo ">>> [3/6] BubbleTea (green_ctx 8SM) @ 0.4 qps"
$PYTHON $BENCH --system bubble_tea --request-rate 0.4 \
  --sched-mode green_ctx --green-ctx-sms 8 \
  --result-dir compare_results/greenctx_0.4qps \
  $COMMON_ARGS 2>&1 | tee compare_results/greenctx_0.4qps/run.log

# --- 4.0 qps ---

echo ""
echo ">>> [4/6] vLLM baseline @ 4.0 qps"
$PYTHON $BENCH --system vllm --request-rate 4.0 \
  --result-dir compare_results/greenctx_vs_bubble_4.0qps \
  $COMMON_ARGS 2>&1 | tee compare_results/greenctx_vs_bubble_4.0qps/run.log

echo ""
echo ">>> [5/6] BubbleTea (bubble sched) @ 4.0 qps"
$PYTHON $BENCH --system bubble_tea --request-rate 4.0 \
  --sched-mode bubble \
  --result-dir compare_results/greenctx_vs_bubble_4.0qps \
  $COMMON_ARGS 2>&1 | tee -a compare_results/greenctx_vs_bubble_4.0qps/run.log

echo ""
echo ">>> [6/6] BubbleTea (green_ctx 8SM) @ 4.0 qps"
$PYTHON $BENCH --system bubble_tea --request-rate 4.0 \
  --sched-mode green_ctx --green-ctx-sms 8 \
  --result-dir compare_results/greenctx_4.0qps \
  $COMMON_ARGS 2>&1 | tee compare_results/greenctx_4.0qps/run.log

echo ""
echo "=========================================="
echo "All runs complete: $(date)"
echo "=========================================="
