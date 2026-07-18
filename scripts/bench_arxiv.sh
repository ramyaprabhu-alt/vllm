#!/usr/bin/env bash
# bench_arxiv.sh — fair vLLM vs BubbleTea benchmark on arxiv dataset
#
# Fixes vs ad-hoc runs:
#   1. Prompt filtering  — drops prompts that exceed max_model_len - output_len,
#                          eliminating VLLMValidationError rejections.
#   2. Equal warmup      — both servers process 20 warmup requests before timing;
#                          prevents cold-Triton-JIT advantage for the second system.
#   3. Training gate     — waits for first BubbleTea optimizer step before starting
#                          the timed benchmark, so results reflect steady-state.
#   4. Correct env       — VLLM_FT_LORA_PATH exported at shell level so
#                          compare_benchmark.py reports real_training=True correctly.
#   5. CUDA 13 runtime   — .venv-bubble ships libcudart.so.13 needed by _C.abi3.so.
#
# Usage:
#   bash scripts/bench_arxiv.sh                      # defaults below
#   NUM_PROMPTS=200 QPS=2.0 bash scripts/bench_arxiv.sh
#   bash scripts/bench_arxiv.sh --system vllm        # baseline only
#   bash scripts/bench_arxiv.sh --system bubble_tea  # BubbleTea only
#   bash scripts/bench_arxiv.sh --no-ft              # inference-only BubbleTea

set -euo pipefail
cd "$(dirname "$0")/.."  # run from repo root

# ── Config ─────────────────────────────────────────────────────────────────────
VENV="${VENV:-/mnt/nfs/home/ramya/vllm/.venv-bubble}"
MODEL="${MODEL:-/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B}"
LORA="${LORA:-/mnt/nfs/home/ramya/slora-plus/S-LoRA/test/qwen3/adapters/qwen3-toy-lora}"
CACHE_DIR="${CACHE_DIR:-/mnt/nfs/home/ramya/scratch}"
ARXIV_RAW="${ARXIV_RAW:-$CACHE_DIR/arxiv_bench_500.jsonl}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
QPS="${QPS:-4.0}"
MAX_LEN="${MAX_LEN:-16384}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
GPU_MEM="${GPU_MEM:-0.85}"
RESULT_DIR="${RESULT_DIR:-compare_results/arxiv_${NUM_PROMPTS}_${QPS}qps_$(date +%m%d_%H%M)}"

# ── Derived ────────────────────────────────────────────────────────────────────
MAX_INPUT=$(( MAX_LEN - OUTPUT_LEN ))
ARXIV_FILTERED="$CACHE_DIR/arxiv_filtered_${MAX_INPUT}.jsonl"
CUDA13_LIB="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"

# ── Environment ────────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="$CUDA13_LIB:${LD_LIBRARY_PATH:-}"
export VLLM_FT_LORA_PATH="$LORA"       # detected by compare_benchmark.py for real_training flag
export VLLM_FT_TP_CORRECT=1            # TP-correct gradients via TCPStore exchange
export VLLM_FT_ACCUM_STEPS=1           # optimizer fires every backward pass (1 sample/step for benchmarking)
export HF_DATASETS_OFFLINE=1           # use local alpaca cache, no hub check
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================"
echo "  BubbleTea vs vLLM arxiv benchmark"
echo "  Model:       $MODEL"
echo "  Prompts:     $NUM_PROMPTS  QPS: $QPS"
echo "  Max len:     $MAX_LEN  Output: $OUTPUT_LEN  Max input: $MAX_INPUT"
echo "  LoRA:        $LORA"
echo "  Result dir:  $RESULT_DIR"
echo "================================================================"

# ── 1. Filter prompts ─────────────────────────────────────────────────────────
if [ ! -f "$ARXIV_FILTERED" ]; then
    echo
    echo "Filtering arxiv prompts to <$MAX_INPUT input tokens..."
    MAX_INPUT=$MAX_INPUT ARXIV_RAW=$ARXIV_RAW ARXIV_FILTERED=$ARXIV_FILTERED \
    MODEL=$MODEL "$VENV/bin/python" - <<'PYEOF'
import json, os, sys
from pathlib import Path
from transformers import AutoTokenizer

max_input    = int(os.environ["MAX_INPUT"])
src          = os.environ["ARXIV_RAW"]
dst          = os.environ["ARXIV_FILTERED"]
model_path   = os.environ["MODEL"]

print(f"  Loading tokenizer from {model_path}...", flush=True)
tok = AutoTokenizer.from_pretrained(model_path)

kept, skipped = [], 0
with open(src) as f:
    for line in f:
        d = json.loads(line)
        n = len(tok.encode(d["prompt"], add_special_tokens=False))
        if n <= max_input:
            kept.append(d)
        else:
            skipped += 1

Path(dst).write_text("".join(json.dumps(d) + "\n" for d in kept))
print(f"  Kept {len(kept)} prompts, dropped {skipped} (too long). → {dst}", flush=True)
PYEOF
else
    echo "Using cached filtered prompts: $ARXIV_FILTERED"
fi

# Count available prompts
N_AVAIL=$( wc -l < "$ARXIV_FILTERED" )
if [ "$N_AVAIL" -lt "$NUM_PROMPTS" ]; then
    echo "WARNING: only $N_AVAIL filtered prompts available (requested $NUM_PROMPTS)."
    echo "         Reducing NUM_PROMPTS to $N_AVAIL."
    NUM_PROMPTS=$N_AVAIL
fi

# ── 2. Run benchmark ──────────────────────────────────────────────────────────
echo
"$VENV/bin/python" tools/compare_benchmark.py \
    --model           "$MODEL" \
    --dataset-name    custom \
    --dataset-path    "$ARXIV_FILTERED" \
    --num-prompts     "$NUM_PROMPTS" \
    --request-rate    "$QPS" \
    --max-model-len   "$MAX_LEN" \
    --gpu-mem-util    "$GPU_MEM" \
    --result-dir      "$RESULT_DIR" \
    --custom-output-len "$OUTPUT_LEN" \
    --ft \
    --t-ft 128 \
    "$@"

# ── 3. Summary ────────────────────────────────────────────────────────────────
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
