# BubbleTea

BubbleTea co-schedules LoRA fine-tuning inside the idle time that arises during
vLLM inference on MoE models. With Expert Parallelism (EP), whichever GPU rank
finishes its expert FFN first sits idle at the NCCL all-reduce barrier waiting
for the slower rank. BubbleTea dispatches training sub-ops into these "bubbles"
on a low-priority CUDA stream, achieving near-zero TTFT/TPOT overhead for
continuous fine-tuning alongside live inference.

Target model: **Qwen3-30B-A3B** (TP=2, EP=2, 48 MoE layers, 128 experts).

Measured idle budget per prefill:

- 8k context → ~34 ms total across 48 layers (24 layers exploitable)
- 16k context → ~68 ms (25 layers)
- 32k context → ~137 ms (31 layers)

---

## Hardware requirements

- 2× NVIDIA A100 80 GB or 2× H100 80 GB
- CUDA 12.x
- ~200 GB NFS/local storage for model weights + dataset cache

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/ramyaprabhu-alt/vllm.git
cd vllm
git checkout bubbletea
```

### 2. Run the setup script

```bash
bash scripts/setup.sh
```

This will:

- Install `uv` if not present
- Create `.venv` with Python 3.12
- Install vLLM (precompiled CUDA kernels — no C++ build needed)
- Install BubbleTea dependencies (`safetensors`, `datasets`, `transformers`, `requests`)
- Install pre-commit hooks

Expected time: ~5 minutes on first run (large wheel downloads).

### 3. Set environment variables

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export MODEL_DIR="/path/to/Qwen3-30B-A3B"   # directory with model weights
export CACHE_DIR="/path/to/hf-cache"         # HuggingFace datasets cache
export HF_TOKEN="hf_..."                     # HuggingFace token
export HF_DATASETS_OFFLINE=1                 # use local cache; avoid hub checks
export TRANSFORMERS_OFFLINE=1
```

> **Never hardcode your HF token in source files.** The setup script and
> experiments read it from the environment.

### 4. Download the model (if not already cached)

```bash
source .venv/bin/activate
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-30B-A3B', local_dir='$MODEL_DIR')
"
```

### 5. Download training data

```bash
.venv/bin/python experiments/download_alpaca.py
```

---

## Serving

Start a vLLM server with the Qwen3-30B-A3B model (TP=2, EP=2, async EPLB):

```bash
MODEL_DIR=$MODEL_DIR bash scripts/run_qwen3_30b_a3b.sh
```

> **Note:** `run_qwen3_30b_a3b.sh` and other scripts in `scripts/` have `VENV`
> and `MODEL` hardcoded to the original NFS paths. Either edit the variables at
> the top of each script or override them:
>
> ```bash
> VENV="$PWD/.venv" MODEL_DIR=$MODEL_DIR bash scripts/run_qwen3_30b_a3b.sh
> ```

---

## Running BubbleTea modes

BubbleTea is controlled via environment variables passed to `vllm serve`.
The main knob is `VLLM_FT_COMBINED_MODE`.

| Mode | Description |
| ---- | ----------- |
| `off` (default) | Pure inference, no training |
| `B+D` | Forward sync on main stream during prefill; backward in EP bubbles |
| `C+D` | Forward async on secondary stream during decode; backward in EP bubbles |
| `D+D` | Forward + backward concatenated into one bubble-scheduler job (prefill only) |
| `C+D_batch` | FT attention on secondary stream; MoE batched inline with inference tokens |

### Demo mode (synthetic ops, no real training data needed)

```bash
VLLM_FT_COMBINED_MODE=C+D \
  MODEL_DIR=$MODEL_DIR bash scripts/run_qwen3_30b_a3b.sh
```

### Real LoRA training

```bash
VLLM_FT_COMBINED_MODE=C+D \
  VLLM_FT_LORA_PATH=/path/to/adapter_dir \
  MODEL_DIR=$MODEL_DIR bash scripts/run_qwen3_30b_a3b.sh
```

`VLLM_FT_LORA_PATH` must point to a directory containing
`adapter_model.safetensors`. The trainer initialises automatically on model
load.

### Useful env vars

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `VLLM_FT_COMBINED_MODE` | `off` | Training placement mode (see above) |
| `VLLM_FT_LORA_PATH` | — | Path to `adapter_model.safetensors`; enables real training |
| `VLLM_FT_COMBINED_T_FT` | `128` | Number of training tokens per step |
| `VLLM_FT_ACCUM_STEPS` | `4` | Gradient accumulation steps before optimizer.step() |
| `VLLM_BUBBLE_SCHED_DEMO` | `0` | `1` = run backward demo loop with synthetic ops |
| `VLLM_FT_TIMING` | `0` | `1` = record pause-decode stall times to `/tmp/bt_timing.log` |
| `VLLM_BUBBLE_PROFILE` | `0` | `1` = write per-layer bubble measurements to `/tmp/vllm_bubble_rank*.json` |

---

## Running experiments

All experiment scripts talk to a running vLLM server (default `localhost:8000`).
Start the server first, then in a second terminal:

```bash
# Measure TTFT baseline vs. BubbleTea C+D
.venv/bin/python experiments/bubble_sched_live_test.py --mode baseline
.venv/bin/python experiments/bubble_sched_live_test.py --mode C+D

# Sweep throughput vs. bubble utilisation across QPS settings
.venv/bin/python experiments/bubble_sweep.py

# Forward placement comparison (B+D vs C+D vs D+D)
.venv/bin/python experiments/ft_fwd_placement_exp.py
```

---

## Running tests

```bash
.venv/bin/python -m pytest tests/bubbletea/ -v
```

Individual test files:

| File | What it checks |
| ---- | -------------- |
| `test_bt_lora_unit.py` | LoRA weight loading, optimizer step, param shapes |
| `test_bt_lora_tp_correctness.py` | TP-replicated params produce identical gradients |
| `test_bt_lora_tp_correct_safety.py` | No cross-rank gradient contamination |
| `test_bt_lora_moe_bwd_ep_correct.py` | EP-correct MoE backward (each rank sees its expert shard) |
| `test_bt_lora_cdbatch_moe_delta.py` | C+D_batch MoE delta correctness |
| `test_bt_lora_integration.py` | End-to-end: server starts, training step fires, weights update |

---

## Benchmarks

Standalone microbenchmarks (no server needed):

```bash
# Sub-op costs (Q/K/V/O proj bwd, attn SDPA bwd, MoE bwd)
.venv/bin/python benchmarks/bench_subops.py

# Per-layer TTFT overhead of BubbleTea vs. baseline
.venv/bin/python benchmarks/qwen3_fwd_bwd_microbench.py

# TP-correct backward overhead
.venv/bin/python benchmarks/bench_tp_correct_overhead.py
```

---

## Repository layout

```text
bubbletea/            Core library (trainer, backward_subops, forward_subops)
vllm/
  model_executor/
    layers/fused_moe/runner/
      bubble_scheduler.py   VllmBubbleScheduler — EP-bubble gated dispatch
      moe_runner.py         Modified: bubble hooks, FT placement modes
    models/
      qwen3_moe.py          Modified: C+D_batch inline MoE, trainer init
benchmarks/           Microbenchmarks
experiments/          Sweeps, live tests, demos
tests/bubbletea/      Correctness tests
scripts/              Server launch scripts (run_*.sh) + setup.sh
tools/                Analysis and debug utilities
```
