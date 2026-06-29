#!/bin/bash
# Qwen3-30B-A3B serving: TP=2, EP=2, async EPLB
# Hardware: auto-detected (A100 SM80 or H100 SM90)
# Model: 30B total / 3B active, 128 experts, 8 active/token, 48 layers BF16

set -euo pipefail

VENV="/mnt/nfs/home/ramya/vllm/.venv"
export PATH="$VENV/bin:$PATH"
MODEL="/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

# EPLB: 8 redundant expert slots per GPU (64 base + 8), async non-blocking rebalance
EPLB_CONFIG='{
  "num_redundant_experts": 8,
  "window_size": 1000,
  "step_interval": 3000,
  "use_async": true,
  "log_balancedness": true,
  "log_balancedness_interval": 100
}'

# Detect GPU architecture
GPU_ARCH=$("$VENV/bin/python" -c "import torch; sm=torch.cuda.get_device_capability(); print(f'{sm[0]}{sm[1]}')" 2>/dev/null || echo "80")
if [ "$GPU_ARCH" = "90" ] || [ "$GPU_ARCH" = "90a" ]; then
    GPU_DESC="H100 SM90"
    MOE_BACKEND="auto"          # FlashInfer CUTLASS on SM90
    FLASHINFER_FLAG="--enable-flashinfer-autotune"
else
    GPU_DESC="A100 SM80"
    MOE_BACKEND="triton"        # FlashInfer cubin absent on SM80; Triton JIT
    FLASHINFER_FLAG=""
fi

echo "=== Qwen3-30B-A3B EP Deployment ==="
echo "Model:        $MODEL"
echo "GPU:          $GPU_DESC (detected sm_$GPU_ARCH)"
echo "TP:           2 (attention)"
echo "EP:           2 (MoE, 64 experts/GPU + 8 redundant, local weights only)"
echo "MoE backend:  $MOE_BACKEND"
echo "EPLB:         async, step_interval=3000, window=1000"
echo "Max context:  $MAX_MODEL_LEN tokens"
echo "Port:         $PORT"
echo ""

exec "$VENV/bin/vllm" serve "$MODEL" \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --enable-eplb \
    --eplb-config "$EPLB_CONFIG" \
    --all2all-backend allgather_reducescatter \
    --moe-backend "$MOE_BACKEND" \
    $FLASHINFER_FLAG \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs 256 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$PORT" \
    "$@"
