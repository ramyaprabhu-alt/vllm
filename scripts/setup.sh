#!/usr/bin/env bash
# setup.sh — bootstrap the BubbleTea vLLM environment
#
# Usage:
#   bash scripts/setup.sh
#
# Optional environment variables (set before running, or export them):
#   VENV_DIR    — venv directory name              (default: .venv)
#   MODEL_DIR   — path to Qwen3-30B-A3B weights   (default: ~/models/Qwen/Qwen3-30B-A3B)
#   CACHE_DIR   — HuggingFace datasets cache       (default: ~/scratch)
#   HF_TOKEN    — HuggingFace token (required for model download if not cached)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[ok]${NC}  $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
die()  { echo -e "${RED}[err]${NC}  $*" >&2; exit 1; }

echo "=== BubbleTea setup ==="
echo "Repo: $REPO_ROOT"
echo

# ── 1. Prerequisites ──────────────────────────────────────────────────────────
echo "--- Checking prerequisites ---"

# CUDA
if ! command -v nvcc &>/dev/null && ! nvidia-smi &>/dev/null; then
    die "CUDA not found. This project requires 2× A100 or 2× H100 GPUs."
fi
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
ok "Found $GPU_COUNT GPU(s): $GPU_NAME"
if [ "$GPU_COUNT" -lt 2 ]; then
    warn "BubbleTea needs 2 GPUs (TP=2 EP=2). Benchmarks may fail with $GPU_COUNT GPU."
fi

# Python 3.12+
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    ok "System python3: $PY_VER (used only to bootstrap uv — venv will use 3.12)"
fi

# uv
if ! command -v uv &>/dev/null; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the rest of this script
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
ok "uv: $(uv --version)"

# ── 2. Python virtual environment ─────────────────────────────────────────────
echo
echo "--- Setting up Python environment ---"

VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating $VENV_DIR (Python 3.12) ..."
    uv venv "$VENV_DIR" --python 3.12
    ok "Created $VENV_DIR"
else
    ok "$VENV_DIR already exists — skipping creation"
fi

# ── 3. Install vLLM + dependencies ───────────────────────────────────────────
echo
echo "--- Installing vLLM (Python-only, precompiled CUDA kernels) ---"
echo "This may take a few minutes on first run ..."

VLLM_USE_PRECOMPILED=1 uv pip install --python "$VENV_DIR/bin/python" -e . --torch-backend=auto
ok "vLLM installed"

# BubbleTea extra deps (safetensors is pulled by vLLM; datasets/transformers/requests listed explicitly)
echo "Installing BubbleTea dependencies ..."
uv pip install --python "$VENV_DIR/bin/python" "safetensors>=0.4" "datasets>=2.18" "transformers>=4.40" "requests>=2.31"
ok "BubbleTea dependencies installed"

# ── 4. Pre-commit hooks (optional but recommended) ───────────────────────────
echo
echo "--- Pre-commit hooks ---"
if [ -f "requirements/lint.txt" ]; then
    uv pip install --python "$VENV_DIR/bin/python" -r requirements/lint.txt
    "$VENV_DIR/bin/pre-commit" install
    ok "pre-commit hooks installed"
else
    warn "requirements/lint.txt not found — skipping pre-commit"
fi

# ── 5. Model path check ───────────────────────────────────────────────────────
echo
echo "--- Model paths ---"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen/Qwen3-30B-A3B}"
CACHE_DIR="${CACHE_DIR:-$HOME/scratch}"

if [ -d "$MODEL_DIR" ]; then
    ok "Qwen3-30B-A3B found at $MODEL_DIR"
else
    warn "Model not found at $MODEL_DIR"
    echo "  Download with:"
    echo "    export HF_TOKEN=<your_token>"
    echo "    $VENV_DIR/bin/python -c \\"
    echo "      \"from huggingface_hub import snapshot_download; \\"
    echo "       snapshot_download('Qwen/Qwen3-30B-A3B', local_dir='$MODEL_DIR')\""
fi

# ── 6. Environment variable summary ──────────────────────────────────────────
echo
echo "=== Setup complete ==="
echo
echo "Add these to your shell profile (~/.bashrc or ~/.zshrc):"
echo
echo "  export MODEL_DIR=\"$MODEL_DIR\""
echo "  export CACHE_DIR=\"$CACHE_DIR\""
echo "  export HF_TOKEN=\"<your_huggingface_token>\""
echo "  export HF_DATASETS_OFFLINE=1   # use local cache, avoid hub checks"
echo "  export TRANSFORMERS_OFFLINE=1"
echo
echo "Activate the venv:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "Quick-start — serve Qwen3-30B-A3B with BubbleTea:"
echo "  VENV=\"\$PWD/$VENV_DIR\" MODEL_DIR=\$MODEL_DIR bash scripts/run_qwen3_30b_a3b.sh"
echo
echo "Run tests:"
echo "  $VENV_DIR/bin/python -m pytest tests/bubbletea/ -v"
