#!/bin/bash
# Qwen1.5-MoE-A2.7B serving: TP=2, EP=2
# Hardware: A100 SM80
# Model: 14.3B total / 2.7B active, 60 experts, 4 active/token, 24 layers BF16
# BubbleTea: set VLLM_FT_LORA_PATH to enable LoRA training in EP bubbles

set -euo pipefail

# BubbleTea backward-dispatch mode. Default "bubble" gates ALL training
# dispatch on _ep_is_light_rank() during prefill only (zero dispatch during
# decode) -- on Qwen1.5-MoE-A2.7B's top-4-of-60 routing the light/heavy rank
# flips unpredictably prefill-to-prefill (unlike Qwen3-30B-A3B's top-8-of-128,
# where one rank is consistently light), so a rank can go an entire run
# without a single backward dispatch. That stalls its _fwd_round forever and
# _sync_fwd_round() times out against a peer that will never advance --
# no rndz timeout value fixes that. "both" keeps the prefill-bubble dispatch
# AND adds unconditional decode-phase dispatch, so every rank always makes
# progress regardless of which one is "light" this prefill.
export VLLM_FT_BWD_MODE="${VLLM_FT_BWD_MODE:-both}"

VENV="/mnt/nfs/home/ramya/vllm/.venv"
export PATH="$VENV/bin:$PATH"
MODEL="/mnt/nfs/home/ramya/models/Qwen/Qwen1.5-MoE-A2.7B"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# Detect GPU architecture
GPU_ARCH=$("$VENV/bin/python" -c "import torch; sm=torch.cuda.get_device_capability(); print(f'{sm[0]}{sm[1]}')" 2>/dev/null || echo "80")
if [ "$GPU_ARCH" = "90" ] || [ "$GPU_ARCH" = "90a" ]; then
    GPU_DESC="H100 SM90"
    MOE_BACKEND="auto"
    FLASHINFER_FLAG="--enable-flashinfer-autotune"
else
    GPU_DESC="A100 SM80"
    MOE_BACKEND="triton"
    FLASHINFER_FLAG=""
fi

echo "=== Qwen1.5-MoE-A2.7B EP Deployment ==="
echo "Model:        $MODEL"
echo "GPU:          $GPU_DESC (detected sm_$GPU_ARCH)"
echo "TP:           2 (attention)"
echo "EP:           2 (MoE, 30 experts/GPU)"
echo "MoE backend:  $MOE_BACKEND"
echo "Max context:  $MAX_MODEL_LEN tokens"
echo "Port:         $PORT"
if [ -n "${VLLM_FT_LORA_PATH:-}" ]; then
    echo "BubbleTea:    LoRA training from $VLLM_FT_LORA_PATH"
    echo "BWD mode:     $VLLM_FT_BWD_MODE"
fi
echo ""

exec "$VENV/bin/vllm" serve "$MODEL" \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
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
