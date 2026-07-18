#!/usr/bin/env bash
# bench_cl.sh — fixed-context-length vLLM vs BubbleTea benchmark
#
# Same env setup / correctness flags as bench_arxiv.sh, but points at a
# pre-generated fixed-context-length trace instead of filtering arxiv data
# on every invocation. Generate the trace once with the snippet below, then
# reuse it across the whole QPS sweep.
#
# Trace generation (adjust CL/N/DST as needed):
#   MODEL=/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B \
#   SRC=/mnt/nfs/home/ramya/scratch/arxiv_bench_500.jsonl \
#   DST=/mnt/nfs/home/ramya/scratch/cl2048_200.jsonl \
#   TARGET_LEN=2048 N=200 \
#   .venv-bubble/bin/python - <<'PYEOF'
#   import json, os
#   from transformers import AutoTokenizer
#   tok = AutoTokenizer.from_pretrained(os.environ["MODEL"])
#   prompts = [json.loads(l)["prompt"] for l in open(os.environ["SRC"])]
#   target_len, n = int(os.environ["TARGET_LEN"]), int(os.environ["N"])
#   out, idx = [], 0
#   for i in range(n):
#       text, ids = "", []
#       while len(ids) < target_len:
#           text += ("\n\n" if text else "") + prompts[idx % len(prompts)]; idx += 1
#           ids = tok.encode(text, add_special_tokens=False)
#       out.append({"prompt": tok.decode(ids[:target_len])})
#   with open(os.environ["DST"], "w") as f:
#       for d in out: f.write(json.dumps(d) + "\n")
#   PYEOF
#
# Usage:
#   CL=2048 bash scripts/bench_cl.sh                       # defaults below
#   CL=2048 NUM_PROMPTS=200 QPS=2.0 bash scripts/bench_cl.sh
#   CL=2048 bash scripts/bench_cl.sh --system bubble_tea --bwd-mode slo_budget --slo-d-ms 80

set -euo pipefail
cd "$(dirname "$0")/.."  # run from repo root

# ── Config ─────────────────────────────────────────────────────────────────────
VENV="${VENV:-/mnt/nfs/home/ramya/vllm/.venv-bubble}"
MODEL="${MODEL:-/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B}"
LORA="${LORA:-/mnt/nfs/home/ramya/slora-plus/S-LoRA/test/qwen3/adapters/qwen3-toy-lora}"
CACHE_DIR="${CACHE_DIR:-/mnt/nfs/home/ramya/scratch}"
CL="${CL:-2048}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
QPS="${QPS:-4.0}"
MAX_LEN="${MAX_LEN:-16384}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
GPU_MEM="${GPU_MEM:-0.85}"
CL_TRACE="${CL_TRACE:-$CACHE_DIR/cl${CL}_${NUM_PROMPTS}.jsonl}"
RESULT_DIR="${RESULT_DIR:-compare_results/cl${CL}_${NUM_PROMPTS}_${QPS}qps_$(date +%m%d_%H%M)}"

# ── Derived ────────────────────────────────────────────────────────────────────
CUDA13_LIB="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"

if [ ! -f "$CL_TRACE" ]; then
    echo "ERROR: trace file not found: $CL_TRACE" >&2
    echo "Generate it first with the snippet in this script's header comment." >&2
    exit 1
fi
N_AVAIL=$( wc -l < "$CL_TRACE" )
if [ "$N_AVAIL" -lt "$NUM_PROMPTS" ]; then
    echo "WARNING: only $N_AVAIL prompts available in $CL_TRACE (requested $NUM_PROMPTS)."
    echo "         Reducing NUM_PROMPTS to $N_AVAIL."
    NUM_PROMPTS=$N_AVAIL
fi

# ── Environment ────────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="$CUDA13_LIB:${LD_LIBRARY_PATH:-}"
export VLLM_FT_LORA_PATH="$LORA"       # detected by compare_benchmark.py for real_training flag
export VLLM_FT_TP_CORRECT=1            # TP-correct gradients via TCPStore exchange
export VLLM_FT_ACCUM_STEPS=1           # optimizer fires every backward pass (1 sample/step for benchmarking)
export HF_DATASETS_OFFLINE=1           # use local alpaca cache, no hub check
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================"
echo "  BubbleTea vs vLLM fixed-context-length benchmark"
echo "  Model:       $MODEL"
echo "  Context len: $CL (fixed)  Prompts: $NUM_PROMPTS  QPS: $QPS"
echo "  Max len:     $MAX_LEN  Output: $OUTPUT_LEN"
echo "  Trace:       $CL_TRACE"
echo "  LoRA:        $LORA"
echo "  Result dir:  $RESULT_DIR"
echo "================================================================"

echo
"$VENV/bin/python" tools/compare_benchmark.py \
    --model           "$MODEL" \
    --dataset-name    custom \
    --dataset-path    "$CL_TRACE" \
    --num-prompts     "$NUM_PROMPTS" \
    --request-rate    "$QPS" \
    --max-model-len   "$MAX_LEN" \
    --gpu-mem-util    "$GPU_MEM" \
    --result-dir      "$RESULT_DIR" \
    --custom-output-len "$OUTPUT_LEN" \
    --ft \
    --t-ft 128 \
    "$@"

echo
echo "Results written to $RESULT_DIR/"
for f in "$RESULT_DIR"/*.json; do
    [ -f "$f" ] || continue
    echo
    echo "── $(basename "$f") ──"
    "$VENV/bin/python" - "$f" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
inf = d.get("inference", {})
tr  = d.get("training",  {})
cfg = d.get("config",    {})
print(f"  System:         {d.get('system')}")
print(f"  real_training:  {cfg.get('real_training')}")
print(f"  Requests OK:    {inf.get('successful_requests')}")
print(f"  Mean TTFT:      {inf.get('mean_ttft_ms'):.1f} ms")
print(f"  p50  TTFT:      {inf.get('p50_ttft_ms'):.1f} ms")
print(f"  p99  TTFT:      {inf.get('p99_ttft_ms'):.1f} ms")
print(f"  Mean TPOT:      {inf.get('mean_tpot_ms'):.1f} ms")
print(f"  p50  TPOT:      {inf.get('p50_tpot_ms'):.1f} ms")
print(f"  Training steps: {tr.get('training_steps')}")
print(f"  Note:           {tr.get('note')}")
PYEOF
done
