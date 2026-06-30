# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
bt_lora_trainer.py — Real LoRA training for BubbleTea C+D mode.

Replaces synthetic random-tensor ops with actual gradient updates to LoRA
adapter weights, using real alpaca-cleaned training data.  The trainer runs
entirely in-process inside the vLLM worker, coordinated by the bubble
scheduler in moe_runner.py.

Design
------
* Only the LoRA A/B matrices are trainable; all base model weights are frozen
  (detached from the autograd graph).
* The training forward is a full 48-layer forward pass using standard
  F.scaled_dot_product_attention (no paged KV cache) on a t_ft-token training
  batch, followed by a CE loss against the next-token labels.
* MoE FFN layers are called under torch.no_grad so that gradients only flow
  through the attention LoRA deltas.
* After `accum_steps` training calls, AdamW updates are applied and LoRA
  weights are synced back to vLLM's LoRA adapter manager so inference sees
  the updated weights.
* The trainer is TP-aware: it applies the required all-reduce after the
  row-parallel O projection, and handles GQA k/v head replication.
"""

from __future__ import annotations

import datetime
import itertools
import logging
import os
import queue
import struct
import threading
import time
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# ── RMSNorm helpers (manual, no autograd) ────────────────────────────────────
#
# Both helpers scale x by 1/amax before squaring to prevent float32 overflow
# when the training forward (EP approximation, no TP all-reduce) produces
# bfloat16 activations with |x| > sqrt(float32_max) ≈ 1.84e19.  Squaring such
# values overflows to Inf, making rms = Inf and x/rms = Inf/Inf = NaN.
# Scaling preserves the formula: rms(x) = amax * rms(x/amax), x_hat = x/rms.


@torch.no_grad()
def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f = x.float()
    amax = x_f.abs().amax(-1, keepdim=True).clamp(min=1.0)
    rms = ((x_f / amax).pow(2).mean(-1, keepdim=True) + eps / amax.pow(2)).sqrt() * amax
    return (x_f / rms * w.float()).to(x.dtype)


@torch.no_grad()
def _rms_norm_bwd_x(
    grad: torch.Tensor, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """grad_x only (we don't train the norm weight)."""
    x_f = x.float()
    amax = x_f.abs().amax(-1, keepdim=True).clamp(min=1.0)
    rms = ((x_f / amax).pow(2).mean(-1, keepdim=True) + eps / amax.pow(2)).sqrt() * amax
    x_hat = x_f / rms
    dxh = grad.float() * w.float()
    dx = (dxh - x_hat * (dxh * x_hat).mean(-1, keepdim=True)) / rms
    return dx.to(x.dtype)


# ── Constants (Qwen3-30B-A3B architecture) ─────────────────────────────────
_HIDDEN = 2048
_N_HEADS = 32
_N_KV = 4
_HEAD_DIM = 128
_N_LAYERS = 48
_LORA_ALPHA = 16

# Set VLLM_FT_SYNC_LORA_PARAMS=1 to broadcast replicated LoRA params from the
# rank that ran the optimizer to the other rank after each training step.  This
# keeps lora_A (colwise Q/K/V) and lora_B (rowwise O) consistent across TP
# ranks and ensures _sync_to_vllm on the non-updating rank uses current weights.
# The sync fires on the main CUDA stream inside the model forward(), where both
# TP ranks are guaranteed to participate — no deadlock risk.
_SYNC_LORA_PARAMS: bool = os.environ.get("VLLM_FT_SYNC_LORA_PARAMS", "0") == "1"

# Set VLLM_FT_TP_CORRECT=1 to enable TP-correct attention-LoRA training:
#   - all-reduce the row-parallel O-projection LoRA output in
#     _attn_lora_forward (matches TPLoRALinear's rowwise path in
#     qwen3_fwd_bwd_microbench_tp_ep.py)
#   - all-reduce gradients of TP-replicated LoRA params (q/k/v_proj.lora_A,
#     o_proj.lora_B) before the optimizer step
# Default off — without it, training proceeds with the partial-sum
# approximations BubbleTea has always used (no extra NCCL on bwd_stream).
_TP_CORRECT: bool = os.environ.get("VLLM_FT_TP_CORRECT", "0") == "1"

# Timeout (ms) for VLLM_FT_TP_CORRECT's TCPStore-based exchange (see
# _tp_correct_exchange_sum). If a peer rank's value for the current round
# isn't published within this window, we fall back to the local (uncorrected)
# value instead of blocking, and never wait longer than this. Tune based on
# observed cross-rank skew.
_TP_CORRECT_TIMEOUT_MS: int = int(
    os.environ.get("VLLM_FT_TP_CORRECT_TIMEOUT_MS", "500")
)

# Wire format for _tp_correct_exchange_sum's per-round header: a single
# little-endian int64 round id, prepended to the raw float32 tensor bytes.
_TP_CORRECT_HDR = struct.Struct("<q")

# Max bytes per TCPStore key for the VLLM_FT_TP_CORRECT exchange. The store's
# libuv backend hard-rejects messages over 8 MB and kills the client socket
# (poisoning every later exchange in the process), so large payloads — e.g.
# the ~25 MB items-3-5 replicated-LoRA grad pack at 48 layers — are split
# into chunks under per-chunk keys. 4 MB keeps a comfortable margin.
_TP_CORRECT_MAX_CHUNK: int = 4 * 1024 * 1024

# Set VLLM_FT_NAN_DEBUG=1 to log per-layer NaN/magnitude probes in the
# training forward (build_fwd_subops). Diagnostic for the long-standing
# "NaN in last fwd" warning: logs the first layer where NaN appears in
# hidden/residual/attn_out, the number of affected token rows, and the
# residual magnitude growth across layers.
_NAN_DEBUG: bool = os.environ.get("VLLM_FT_NAN_DEBUG", "0") == "1"


class _TpCorrectFuture:
    """Future for an in-flight async TP-correct exchange
    (_tp_correct_exchange_sum_async). result() returns the cross-rank sum
    moved to the local tensor's device/dtype, or the local (partial) tensor
    on timeout/round-mismatch/failure — the same per-call fallback semantics
    as the synchronous exchange. The publish half runs synchronously on the
    caller, so a consumer falling back never delays the peer's collect.
    """

    # Backstop only: the I/O job is internally bounded by the per-chunk
    # store.wait timeouts; this just guarantees result() can never hang.
    _RESULT_CAP_S = 2.0

    __slots__ = ("_local", "_evt", "_total")

    def __init__(self, local: torch.Tensor):
        self._local = local
        self._evt = threading.Event()
        self._total: torch.Tensor | None = None

    @classmethod
    def preset(cls, local: torch.Tensor) -> _TpCorrectFuture:
        """Already-resolved-to-local future, used when the exchange is
        disabled/gated off so consumers have a single code path."""
        fut = cls(local)
        fut._evt.set()
        return fut

    def _set(self, total_cpu: torch.Tensor | None) -> None:
        self._total = total_cpu
        self._evt.set()

    def result(self) -> torch.Tensor:
        if not self._evt.wait(self._RESULT_CAP_S) or self._total is None:
            return self._local
        return self._total.to(self._local.device, dtype=self._local.dtype)


# LoRA tensors that are TP-replicated (identical full matrix on every rank):
# lora_A for colwise Q/K/V projections, lora_B for rowwise O projection.
_REPLICATED_LORA_KEYS = (
    "q_proj.lora_A",
    "k_proj.lora_A",
    "v_proj.lora_A",
    "o_proj.lora_B",
)

import torch.distributed as dist  # noqa: E402


class _SumAllReduce(torch.autograd.Function):
    """All-reduce-sum with correct backward (identity).

    Used for the row-parallel O projection in _attn_lora_forward when the
    autograd graph is active (_forward_loss / training_step path).  Forward
    sums partial outputs across TP ranks; backward passes the gradient
    unchanged to each rank (same pattern as transformers' all_reduce_forward /
    Megatron's 'g' operator, and as _SumAllReduce in
    qwen3_fwd_bwd_microbench_tp_ep.py).
    """

    @staticmethod
    def forward(ctx, x, group):
        if dist.get_world_size(group) > 1:
            x = x.contiguous()
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# ── BubbleTeaLoRATrainer ──────────────────────────────────────────────────


class BubbleTeaLoRATrainer:
    """
    Real LoRA trainer for BubbleTea.

    Parameters
    ----------
    model        : the vLLM Qwen3MoeForCausalLM instance (already loaded)
    adapter_path : path to adapter_model.safetensors
    tokenizer_path : model directory (used to load tokenizer)
    device       : local CUDA device index (0 or 1 in TP=2)
    cache_dir    : HF datasets / tokenizer cache
    accum_steps  : gradient accumulation steps before an optimizer update
    lr           : AdamW learning rate
    t_ft         : tokens per training step (must match VLLM_FT_COMBINED_T_FT)
    """

    def __init__(
        self,
        model,
        adapter_path: str,
        tokenizer_path: str,
        device: int,
        cache_dir: str = "/mnt/nfs/home/ramya/scratch",
        accum_steps: int = 4,
        lr: float = 2e-4,
        t_ft: int = 128,
        ep_rank: int = 0,
        ep_size: int = 1,
        ep_expert_start: int = 0,
        ep_num_local_experts: int = 64,
    ):
        self.device = device
        self.accum_steps = accum_steps
        self.t_ft = t_ft
        # Number of expert-group chunks to split each _passthrough sub-op into.
        # Each chunk handles (n_local_experts / n_chunks) experts and reads only
        # that fraction of the expert weight matrices — small enough to complete
        # within one EP bubble (~5ms) without causing queue buildup on bwd_stream.
        self._bwd_passthrough_chunks: int = int(
            os.environ.get("VLLM_FT_BWD_PASSTHROUGH_CHUNKS", "8")
        )
        self._step = 0  # gradient-accumulation steps so far
        self.completed_steps = 0  # optimizer steps applied
        self.total_loss = 0.0  # accumulated loss for logging
        self._lock = threading.Lock()

        # ── Base model layers (read-only references) ──────────────────────
        self._model = model
        inner = model.model  # Qwen3MoeModel
        self._layers = inner.layers  # nn.ModuleList[DecoderLayer]
        self._embed = inner.embed_tokens  # VocabParallelEmbedding
        self._norm = inner.norm  # RMSNorm
        self._lm_head = model.lm_head  # ParallelLMHead

        # TP info
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            get_tp_group,
        )

        self._tp_size = get_tensor_model_parallel_world_size()
        self._tp_rank = get_tensor_model_parallel_rank()
        self._tp_group = get_tp_group()

        # Dedicated NCCL process group for training TP communication.
        # Kept separate from the inference TP group (self._tp_group) so that
        # training all-reduces on bwd_stream don't fight with inference
        # all-reduces on the main stream — using the same NCCL communicator
        # from two streams causes non-deterministic NCCL deadlocks.
        # Must be created at a point where all TP ranks call dist.new_group
        # together; the dist.barrier() guarantees that.
        self._train_tp_pg = None
        if self._tp_size > 1:
            try:
                dist.barrier()  # synchronise before new_group (collective op)
                self._train_tp_pg = dist.new_group(
                    ranks=list(range(dist.get_world_size())),
                    backend="nccl",
                )
                log.debug(
                    "[BubbleTea] training TP process group created (tp_size=%d)",
                    self._tp_size,
                )
            except Exception as _e:
                log.warning(
                    "[BubbleTea] Training TP process group creation failed (%s). "
                    "Training will proceed without TP communication — gradients "
                    "will be incorrect in multi-GPU mode.",
                    _e,
                )

        # Store-based exchange used only by the VLLM_FT_TP_CORRECT production
        # all-reduces (see _tp_correct_exchange_sum). Reuses the default
        # ProcessGroup's rendezvous TCPStore -- a plain key/value store, not a
        # collective ProcessGroup, so each call's store.wait() has its own
        # independent timeout and a skipped/late round can never desync or
        # poison future calls (unlike a persistent NCCL/Gloo ProcessGroup,
        # whose internal sequence counters permanently desync after one
        # missed collective). Falls back to the local value on timeout or any
        # error, and never touches _train_tp_pg/NCCL or
        # sync_replicated_params's collectives.
        self._tp_correct_store = None
        if self._tp_size > 1:
            try:
                self._tp_correct_store = dist.distributed_c10d._get_default_store()
            except Exception as _e:
                log.warning(
                    "[BubbleTea] Could not obtain default store for "
                    "VLLM_FT_TP_CORRECT exchange (%s). VLLM_FT_TP_CORRECT "
                    "will be disabled.",
                    _e,
                )

        # Per-rank counter of forward training rounds started (incremented in
        # _fwd_init). Used as the "round id" for item 1's per-layer exchange
        # via _tp_correct_exchange_sum -- both ranks process the same number
        # of forward rounds (driven by the same synchronized inference
        # timeline), so this stays in sync across ranks even though
        # individual sub-ops fire asymmetrically.
        self._fwd_round = 0

        # Param sync: set True after each optimizer step; cleared by
        # sync_replicated_params() which runs on the main CUDA stream where
        # both TP ranks are guaranteed to participate together.
        self._do_param_sync = (
            _SYNC_LORA_PARAMS
            and (self._tp_size > 1)
            and (self._train_tp_pg is not None)
        )
        self._pending_param_sync = False

        # See _TP_CORRECT above. Gates the O-proj all-reduce in
        # _attn_lora_forward and the replicated-grad all-reduce in
        # optimizer_step().
        self._tp_correct = (
            _TP_CORRECT
            and (self._tp_size > 1)
            and (self._train_tp_pg is not None)
            and (self._tp_correct_store is not None)
        )

        # EP expert info for _training_moe_forward
        self._ep_rank = ep_rank
        self._ep_size = ep_size
        self._ep_expert_start = ep_expert_start
        self._ep_num_local_experts = ep_num_local_experts

        # Per-rank head counts — read from the first attention layer so the
        # trainer works for any MoE architecture (Qwen3: 32Q/4KV, Qwen1.5-MoE: 16Q/8KV).
        _first_attn = inner.layers[0].self_attn
        self._n_heads_local = getattr(
            _first_attn, "num_heads", _N_HEADS // self._tp_size
        )
        self._n_kv_local = getattr(
            _first_attn, "num_kv_heads", max(1, _N_KV // self._tp_size)
        )

        # ── LoRA parameters ───────────────────────────────────────────────
        self._lora: dict[int, dict[str, torch.Tensor]] = {}  # layer -> proj -> tensor
        self._load_lora(adapter_path)

        # ── Optimizer ─────────────────────────────────────────────────────
        all_params = [p for layer_d in self._lora.values() for p in layer_d.values()]
        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)

        # ── Training data (lazy background init) ──────────────────────────────
        # The background thread is NOT started here.  It is started the first
        # time build_fwd_subops()._real_fwd() is called, which happens only
        # during decode steps — never during the memory-profiling run.
        # Starting the thread during __init__ (called from load_weights) races
        # with the EngineCore's profiling RPC and causes a 60-second timeout.
        self._data_iter: object = None
        self._data_ready = threading.Event()
        self._data_thread_started: bool = False
        self._tokenizer_path = tokenizer_path
        self._cache_dir = cache_dir
        self._next_batch: dict | None = None  # pre-fetched

        # Non-reentrant lock that prevents concurrent _real_fwd() executions.
        # The pre-data-ready fwd/bwd cycle runs at inference speed (dozens/s).
        # Without this guard, _fwd_job_queue accumulates many items and
        # multiple VllmBubbleScheduler instances launch _real_fwd() in
        # parallel, causing concurrent calls to layer.mlp() on the same
        # module and GPU corruption.
        self._fwd_running = threading.Lock()

        # Cooldown: don't start a new training forward within
        # _FWD_COOLDOWN_S seconds of the last one completing.
        # A 48-layer training forward takes ~20s on the secondary stream.
        # Running concurrently with a very long inference prefill (7000+
        # tokens, also ~20-50s) saturates the GPU and pushes the prefill
        # past the EngineCore's RPC timeout (~60s).  The cooldown ensures
        # the GPU has breathing room between training cycles.
        self._fwd_last_done: float = 0.0
        self._FWD_COOLDOWN_S: float = 0.0

        # ── Forward / backward state shared between sub-ops ──────────────────
        # _fwd_layer_state: per-layer state threaded through the N forward
        #   sub-ops (hidden, residual, positions, labels, ok flag).
        #   Set by the init sub-op; consumed and updated by each layer sub-op;
        #   cleared by the final layer sub-op after storing results.
        # _fwd: final results written by the last forward sub-op and read by
        #   the first backward sub-op (last_attn_out, labels).
        self._fwd_layer_state: dict | None = None
        self._fwd: dict = {}

        import warnings as _w

        n_params = sum(p.numel() for p in all_params)
        _w.warn(
            f"[BubbleTea LoRA] Trainer ready – {n_params} trainable params, "
            f"device={device}, tp={self._tp_rank + 1}/{self._tp_size}, "
            f"accum={accum_steps}, t_ft={t_ft}",
            stacklevel=2,
        )

        # Triton warmup is triggered from Qwen3MoeForCausalLM.forward() on the
        # first profiling call (where the ForwardContext IS set), not here in
        # __init__ (where it is not yet set).  See qwen3_moe.py.
        self._warmup_done: bool = False

    # ── LoRA loading ──────────────────────────────────────────────────────

    def _load_lora(self, adapter_path: str) -> None:
        """Load safetensors LoRA weights as float32 nn.Parameters."""
        weights = load_file(adapter_path, device=f"cuda:{self.device}")
        for key, tensor in weights.items():
            # key: "base_model.model.model.layers.{i}.self_attn.{proj}.lora_X.weight"
            # idx:   [0]       [1]   [2]   [3]   [4]   [5]      [6]    [7]      [8]
            parts = key.split(".")
            try:
                layer_idx = int(parts[4])
                proj = parts[6]  # q_proj / k_proj / v_proj / o_proj
                ab = parts[7]  # lora_A / lora_B
            except (IndexError, ValueError):
                continue

            if layer_idx not in self._lora:
                self._lora[layer_idx] = {}

            # Shard lora_B for column-parallel projections (Q, K, V).
            # For row-parallel (O), lora_A is sharded along its output dim.
            param_key = f"{proj}.{ab}"
            t = tensor.clone().to(torch.float32)

            if ab == "lora_B" and proj in ("q_proj", "k_proj", "v_proj"):
                # Column-parallel: shard B along output (rows).
                chunk = t.shape[0] // self._tp_size
                t = t[self._tp_rank * chunk : (self._tp_rank + 1) * chunk].contiguous()
            elif ab == "lora_A" and proj == "o_proj":
                # Row-parallel: shard A along input (columns).
                chunk = t.shape[1] // self._tp_size
                t = t[
                    :, self._tp_rank * chunk : (self._tp_rank + 1) * chunk
                ].contiguous()

            self._lora[layer_idx][param_key] = t.requires_grad_(True)

    # ── Data pipeline ─────────────────────────────────────────────────────

    def _warmup_training_kernels(self) -> None:
        """Warm up AdamW CUDA kernels (~50ms).

        Training path now uses pure-PyTorch MoE (no Triton), so there are no
        Triton kernels to pre-compile.  This method is kept for the qwen3_moe.py
        hook but is effectively a no-op.
        """
        import warnings as _w

        _w.warn(
            "[BubbleTea LoRA] warmup: pure-PyTorch MoE — no Triton to compile",
            stacklevel=2,
        )
        try:
            dummy = torch.zeros(
                1,
                1,
                device=f"cuda:{self.device}",
                dtype=torch.float32,
                requires_grad=True,
            )
            dummy.sum().backward()
            self.optimizer.zero_grad(set_to_none=True)
        except Exception as exc:
            _w.warn(f"[BubbleTea LoRA] warmup error (non-fatal): {exc}", stacklevel=2)
        _w.warn("[BubbleTea LoRA] warmup complete", stacklevel=2)

    def _build_data_iter_bg(self, tokenizer_path: str, cache_dir: str) -> None:
        """Background thread: download + tokenise dataset, then signal ready."""
        try:
            self._data_iter = self._make_data_iter(tokenizer_path, cache_dir)
            import warnings as _w

            _w.warn("[BubbleTea LoRA] training data ready", stacklevel=1)
        except Exception as exc:
            import warnings as _w

            _w.warn(
                f"[BubbleTea LoRA] data iterator init FAILED: {exc}",
                stacklevel=1,
            )
            log.error("[BubbleTea LoRA] data iterator init failed: %s", exc)
        finally:
            self._data_ready.set()

    def _make_data_iter(self, tokenizer_path: str, cache_dir: str):
        """Build alpaca-cleaned data iterator.

        Uses a plain Python generator to tokenise on-the-fly, avoiding
        ds.map() which pickles the tokeniser closure — and fast tokenisers
        contain pybind11 objects that are not pickleable.
        """
        import datasets
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="left",
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        pad_id = tokenizer.pad_token_id or 0
        t_ft = self.t_ft

        def _fmt(s) -> str:
            if s.get("input", "").strip():
                return (
                    f"### Instruction:\n{s['instruction']}\n\n"
                    f"### Input:\n{s['input']}\n\n"
                    f"### Response:\n{s['output']}"
                )
            return (
                f"### Instruction:\n{s['instruction']}\n\n### Response:\n{s['output']}"
            )

        # HF_DATASETS_OFFLINE=1 must be set in the server environment so this
        # call uses the local cache without contacting the HuggingFace hub.
        # An unauthenticated hub check stalls for 60+ seconds and holds the
        # Python GIL in this background thread, hanging the vLLM worker's
        # main loop.  Run download_alpaca.py once to populate the cache.
        ds_train = datasets.load_dataset(
            "yahma/alpaca-cleaned",
            cache_dir=cache_dir,
            split="train",
        )

        def _gen():
            for sample in itertools.cycle(ds_train):
                enc = tokenizer(
                    _fmt(sample),
                    padding="max_length",
                    truncation=True,
                    max_length=t_ft,
                    return_tensors="pt",
                )
                ids = enc["input_ids"]  # [1, T]
                labels = ids.clone()
                labels[ids == pad_id] = -100  # mask padding
                yield {"input_ids": ids, "labels": labels}

        return _gen()

    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids [1, T], labels [1, T]) on the right GPU.

        Callers must check _data_ready.is_set() before calling — _real_fwd
        does this and skips early if the iterator isn't ready yet.
        """
        if self._data_iter is None:
            raise RuntimeError("[BubbleTea LoRA] data iterator not ready")
        batch = next(self._data_iter)
        ids = batch["input_ids"].to(f"cuda:{self.device}")
        labels = batch["labels"].to(f"cuda:{self.device}")
        return ids, labels

    # ── Single-shot training step (used by sub-ops) ───────────────────────

    def training_step(self) -> float:
        """
        Run one complete forward + backward pass over a real training batch.
        Gradients are accumulated.  Call optimizer_step() every accum_steps.
        Returns the scalar loss value.
        """
        with torch.inference_mode(False), torch.enable_grad():
            loss = self._forward_loss()
        loss.backward()
        loss_val = loss.item()
        self.total_loss += loss_val
        self._step += 1
        try:
            with open("/tmp/vllm_bt_fwd_bwd.log", "a") as _f:
                _f.write(f"{time.time():.6f}\n")
        except Exception:
            pass
        if self._step % self.accum_steps == 0:
            self.optimizer_step()
        return loss_val

    def optimizer_step(self) -> None:
        """Apply accumulated gradients, zero them, sync weights to vLLM.

        Order matters: the grad all-reduce must come BEFORE clipping. The old
        clip-first order scaled each rank's partial grads by a rank-local
        coefficient, so the exchanged sums were inconsistently scaled —
        breaking the identical-post-step-params invariant items 3-5 rely on.
        """
        all_params = [p for d in self._lora.values() for p in d.values()]
        self._all_reduce_replicated_grads()
        self._clip_grad_norm_tp(all_params)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.completed_steps += 1
        avg_loss = self.total_loss / max(self.accum_steps, 1)
        self.total_loss = 0.0
        log.debug("[BubbleTea LoRA] step %d  loss=%.4f", self.completed_steps, avg_loss)
        import warnings as _w

        _w.warn(
            f"[BubbleTea LoRA] training step {self.completed_steps}"
            f"  loss={avg_loss:.4f}  device={self.device}",
            stacklevel=1,
        )

        # Write completion timestamp to the same log that _parse_bt_completions reads
        try:
            with open("/tmp/vllm_combined_completions.log", "a") as _f:
                _f.write(f"{time.time():.6f}\n")
        except Exception:
            pass

        # Write (step, loss) to a dedicated metrics file for loss-curve monitoring.
        try:
            with open("/tmp/vllm_training_loss.log", "a") as _f:
                _f.write(f"{self.completed_steps},{avg_loss:.6f}\n")
        except Exception:
            pass

        # Sync updated LoRA weights back to vLLM's LoRA adapter tensors
        self._sync_to_vllm()

        # Signal sync_replicated_params() (called from the model forward on the
        # main CUDA stream) that replicated params need broadcasting to the other
        # TP rank.
        if self._do_param_sync:
            self._pending_param_sync = True

    @torch.no_grad()
    def _clip_grad_norm_tp(self, params, max_norm: float = 1.0) -> None:
        """clip_grad_norm_ with a TP-consistent norm.

        After the grad all-reduce, replicated params hold identical grads on
        all ranks, but sharded params (lora_B for q/k/v, lora_A for o) only
        hold this rank's shard — a rank-local norm would give each rank a
        different clip coefficient and de-sync the replicated params again.
        The true global norm = replicated contribution (counted once; it's
        identical everywhere) + the sharded contributions summed across
        ranks (one scalar exchange). Falls back to the local norm — the old
        per-rank behavior — if the exchange misses.
        """
        if not (self._tp_correct and self._tp_size > 1):
            torch.nn.utils.clip_grad_norm_(params, max_norm)
            return
        repl_ids = {id(p) for p in self._replicated_lora_params()}
        sq_repl = 0.0
        sq_shard = 0.0
        for p in params:
            if p.grad is None:
                continue
            s = float(p.grad.detach().float().pow(2).sum())
            if id(p) in repl_ids:
                sq_repl += s
            else:
                sq_shard += s
        sq = torch.tensor([sq_shard], dtype=torch.float32)
        sq = self._tp_correct_exchange_sum(sq, "clipsq", self._step)
        total_norm = (sq_repl + float(sq[0])) ** 0.5
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(scale)

    def _replicated_lora_params(self) -> list[torch.Tensor]:
        """TP-replicated LoRA params (q/k/v_proj.lora_A, o_proj.lora_B) across
        all layers, in a deterministic order shared by sync_replicated_params()
        and _all_reduce_replicated_grads()."""
        out: list[torch.Tensor] = []
        for i in sorted(self._lora.keys()):
            d = self._lora[i]
            for key in _REPLICATED_LORA_KEYS:
                p = d.get(key)
                if p is not None:
                    out.append(p)
        return out

    def _tp_correct_exchange_gather(
        self,
        t: torch.Tensor,
        tag: str,
        round_id: int,
    ) -> list[torch.Tensor] | None:
        """Publish `t` and collect every peer TP rank's tensor (same shape +
        dtype) via the default rendezvous TCPStore — the building block for
        the VLLM_FT_TP_CORRECT production exchanges
        (_tp_correct_exchange_sum, and _lm_head_grad's vocab-parallel-CE
        stats, whose per-rank values must be combined with a max/rescale
        rather than a plain sum).

        Wire format: `t`'s NATIVE dtype, serialized via a uint8 view (numpy
        has no bfloat16, so going through float32 — as an earlier version
        did — doubled every bf16 payload for nothing: a bf16 value is
        represented exactly in fp32, so summing bf16-wire values in fp32 is
        bit-identical). Payloads are split into <=_TP_CORRECT_MAX_CHUNK-byte
        chunks under per-chunk keys: TCPStore's libuv backend hard-rejects
        messages over 8 MB AND kills the client socket, which at production
        scale (e.g. the ~25 MB items-3-5 grad pack for 48 layers) would
        permanently poison every later exchange in the process.

        Each rank publishes under fixed per-(tag, rank, chunk) keys,
        overwritten every call (bounded storage — no unbounded growth over a
        long-running server), each chunk prefixed with `round_id` so a peer
        can tell whether the value it reads corresponds to *this* round.

        For each peer rank: wait up to VLLM_FT_TP_CORRECT_TIMEOUT_MS for
        each of that peer's chunks, then check its round_id. If a wait times
        out, or a chunk's round_id doesn't match ours (the peer hasn't
        reached this round yet, or has already moved past it — the bubble
        scheduler fires sub-ops asymmetrically across ranks), returns None:
        the caller falls back to its local value for this call, rather than
        incorrectly combining values from different rounds.
        _train_tp_pg/NCCL and sync_replicated_params are untouched either
        way.

        Returns the peers' tensors (cpu, `t`'s dtype, in rank order,
        excluding this rank), or None on timeout/round-mismatch/failure.
        """
        if self._tp_correct_store is None:
            return None
        try:
            cpu_t = t.detach().to("cpu").contiguous()
            n_chunks = self._tp_correct_publish(cpu_t, tag, round_id)
            return self._tp_correct_collect(cpu_t, tag, round_id, n_chunks)
        except Exception as e:
            log.warning(
                "[BubbleTea] TP-correct exchange timed out/failed "
                "(%s); using local value for this round.",
                e,
            )
            return None

    def _tp_correct_publish(self, cpu_t: torch.Tensor, tag: str, round_id: int) -> int:
        """Chunk-publish cpu_t under per-(tag, rank, chunk) keys; returns the
        chunk count (the peer derives the same count from its identically
        shaped tensor). May raise; callers handle fallback."""
        raw = bytes(cpu_t.view(torch.uint8).numpy())
        n_chunks = max(1, -(-len(raw) // _TP_CORRECT_MAX_CHUNK))
        hdr = _TP_CORRECT_HDR.pack(round_id)
        for j in range(n_chunks):
            chunk = raw[j * _TP_CORRECT_MAX_CHUNK : (j + 1) * _TP_CORRECT_MAX_CHUNK]
            self._tp_correct_store.set(
                f"tpcorrect/{tag}/{self._tp_rank}/{j}", hdr + chunk
            )
        return n_chunks

    def _tp_correct_collect(
        self,
        cpu_t: torch.Tensor,
        tag: str,
        round_id: int,
        n_chunks: int,
    ) -> list[torch.Tensor] | None:
        """Collect every peer's chunks for (tag, round_id); None on
        round-mismatch. May raise on store timeout; callers handle fallback."""
        store = self._tp_correct_store
        timeout = datetime.timedelta(milliseconds=_TP_CORRECT_TIMEOUT_MS)
        timeout_s = _TP_CORRECT_TIMEOUT_MS / 1000.0
        peers: list[torch.Tensor] = []
        for r in range(self._tp_size):
            if r == self._tp_rank:
                continue
            parts: list[bytes] = []
            for j in range(n_chunks):
                peer_key = f"tpcorrect/{tag}/{r}/{j}"
                store.wait([peer_key], timeout)
                buf = store.get(peer_key)
                (peer_round,) = _TP_CORRECT_HDR.unpack(buf[: _TP_CORRECT_HDR.size])
                if peer_round != round_id:
                    # Key exists but holds data from a different round: the peer
                    # may not have published this round yet (publish-before-collect
                    # race).  Poll get() until the round_id matches or timeout.
                    deadline = time.monotonic() + timeout_s
                    while time.monotonic() < deadline:
                        time.sleep(0.001)
                        buf = store.get(peer_key)
                        (peer_round,) = _TP_CORRECT_HDR.unpack(
                            buf[: _TP_CORRECT_HDR.size]
                        )
                        if peer_round == round_id:
                            break
                    else:
                        return None  # peer is on a genuinely different round
                parts.append(buf[_TP_CORRECT_HDR.size :])
            peers.append(
                torch.frombuffer(bytearray(b"".join(parts)), dtype=cpu_t.dtype).reshape(
                    cpu_t.shape
                )
            )
        return peers

    # ── Async exchange: publish on the caller, collect on an I/O thread ────

    def _xchg_submit(self, job) -> None:
        """Run `job` on the dedicated exchange I/O thread (lazily started; a
        single thread keeps store traffic serialized in publish order)."""
        q = getattr(self, "_xchg_queue", None)
        if q is None:
            q = self._xchg_queue = queue.SimpleQueue()
            self._xchg_thread = threading.Thread(
                target=self._xchg_worker, name="bt-tpcorrect-io", daemon=True
            )
            self._xchg_thread.start()
        q.put(job)

    def _xchg_worker(self) -> None:
        while True:
            job = self._xchg_queue.get()
            try:
                job()
            except Exception as e:  # job() sets its future; this is a backstop
                log.warning("[BubbleTea] TP-correct async job error: %s", e)

    def _tp_correct_exchange_sum_async(
        self,
        t: torch.Tensor,
        tag: str,
        round_id: int,
    ) -> _TpCorrectFuture:
        """Async all-reduce-sum: publish the local value NOW (synchronously,
        so the peer's collect is never delayed by our queue), run the peer
        wait/get/fp32-sum on the I/O thread, and return a future. Consumers
        call result() at the latest possible point — typically the next
        bubble sub-op — so the TCP latency overlaps inter-bubble time and
        gradient-independent GPU work instead of blocking the worker thread
        inside one sub-op. result() falls back to the local value on
        timeout/round-mismatch/failure (same semantics as the sync path).
        """
        if (
            not getattr(self, "_tp_correct", False)
            or self._tp_size <= 1
            or self._tp_correct_store is None
        ):
            return _TpCorrectFuture.preset(t)
        fut = _TpCorrectFuture(t)
        try:
            cpu_t = t.detach().to("cpu").contiguous()
            n_chunks = self._tp_correct_publish(cpu_t, tag, round_id)
        except Exception as e:
            log.warning(
                "[BubbleTea] TP-correct async publish failed (%s); "
                "using local value for this round.",
                e,
            )
            fut._set(None)
            return fut

        def _collect_job():
            try:
                peers = self._tp_correct_collect(cpu_t, tag, round_id, n_chunks)
                if not peers:
                    fut._set(None)
                    return
                total = cpu_t.to(torch.float32)
                total = total.clone() if total is cpu_t else total
                for peer_t in peers:
                    total += peer_t.to(torch.float32)
                fut._set(total)
            except Exception as e:
                log.warning(
                    "[BubbleTea] TP-correct async collect failed "
                    "(%s); using local value for this round.",
                    e,
                )
                fut._set(None)

        self._xchg_submit(_collect_job)
        return fut

    def _tp_correct_exchange_sum(
        self, t: torch.Tensor, tag: str, round_id: int
    ) -> torch.Tensor:
        """All-reduce-sum `t` across TP ranks via _tp_correct_exchange_gather
        (TCPStore-based, bounded timeout), accumulating in float32 and
        returning in `t`'s dtype. Production call sites: _attn_lora_forward's
        no-grad O-proj all-reduce, the _fwd_init embedding correction,
        _training_moe_forward's output, the moebwd/attnbwd backward
        corrections, _all_reduce_replicated_grads, and _lm_head_grad's
        grad_hidden partial-sum. On timeout/round-mismatch, falls back to
        returning `t` unchanged — the _tp_correct=False partial-sum
        approximation for this one call.
        """
        peers = self._tp_correct_exchange_gather(t, tag, round_id)
        if not peers:  # None (fallback) or empty (tp_size == 1)
            return t
        total = t.detach().to("cpu", dtype=torch.float32).contiguous().clone()
        for peer_t in peers:
            total += peer_t.to(torch.float32)
        return total.to(t.device, dtype=t.dtype)

    # Streamed per-layer replicated keys: lora_A for column-parallel Q/K/V,
    # lora_B for row-parallel O.  Published immediately after the O-LoRA
    # sub-op so the exchange runs behind the next layer's passthrough chunks.
    _STREAMED_LORA_KEYS = (
        "q_proj.lora_A",
        "k_proj.lora_A",
        "v_proj.lora_A",
        "o_proj.lora_B",
    )

    @torch.no_grad()
    def _publish_layer_grads(self, layer_idx: int) -> None:
        """Stream layer `layer_idx`'s replicated-LoRA grads (items 3-5) as
        one small async exchange right after its o-LoRA sub-op produces them.
        Called only on rounds that end an accumulation cycle, when the grads
        hold the full accumulated sums. Consumed by
        _all_reduce_replicated_grads at optimizer time (same per-call
        fallback semantics: an unresolved layer keeps its local partials).
        """
        d = self._lora.get(layer_idx)
        if not d:
            return
        params = [d[k] for k in self._STREAMED_LORA_KEYS if k in d]
        if not params:
            return
        flat = torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).view(-1).float()
                for p in params
            ]
        )
        fut = self._tp_correct_exchange_sum_async(flat, f"grad_{layer_idx}", self._step)
        futs = getattr(self, "_pending_grad_futs", None)
        if futs is None:
            futs = self._pending_grad_futs = {}
        futs[layer_idx] = (params, fut)

    @torch.no_grad()
    def _all_reduce_replicated_grads(self) -> None:
        """All-reduce (sum) gradients of TP-replicated LoRA params before the
        optimizer step.

        Mirrors qwen3_fwd_bwd_microbench_tp_ep.py's post-backward grad
        all-reduce for `replicated` params: each rank's local backward only
        accumulates its own partial contribution to these gradients (they
        feed into the loss via two independent per-rank paths), so summing
        recovers the true full-matrix gradient.

        Every replicated param is included in the packed buffer regardless of
        whether this rank produced a gradient for it this step (missing grads
        are treated as zero), so both ranks build an identically-shaped
        buffer. optimizer_step() is reached asymmetrically across ranks, so
        this goes through _tp_correct_exchange_sum (TCPStore-based, bounded
        timeout + local-value fallback on skew) rather than
        _train_tp_pg/NCCL.

        Once this runs, both ranks hold identical post-all-reduce gradients
        for these params; combined with identical AdamW state, the optimizer
        step below produces identical post-step values on both ranks —
        sync_replicated_params()'s broadcast becomes a no-op for these params
        (still safe to leave enabled).
        """
        if not self._tp_correct:
            return

        # Streamed path: the backward sweep already published per-layer grad
        # exchanges (_publish_layer_grads); just resolve the futures (their
        # collects ran behind the rest of the backward on the I/O thread).
        futs = getattr(self, "_pending_grad_futs", None)
        if futs:
            for params, fut in futs.values():
                flat = fut.result()
                offset = 0
                for p in params:
                    n = p.numel()
                    g = flat[offset : offset + n].view_as(p).to(p.dtype)
                    if p.grad is None:
                        p.grad = g
                    else:
                        p.grad.copy_(g)
                    offset += n
            futs.clear()
            return

        # Packed-sync fallback (autograd/training_step path and tests that
        # call this directly without a streamed backward sweep).
        params = self._replicated_lora_params()
        if not params:
            return
        flat = torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).view(-1).float()
                for p in params
            ]
        )
        flat = self._tp_correct_exchange_sum(flat, "grad", self._step)
        offset = 0
        for p in params:
            n = p.numel()
            g = flat[offset : offset + n].view_as(p).to(p.dtype)
            if p.grad is None:
                p.grad = g
            else:
                p.grad.copy_(g)
            offset += n

    # ── Forward-round rendezvous ─────────────────────────────────────────────

    def _sync_fwd_round(self) -> bool:
        """Rendezvous both TP ranks to the same _fwd_round before training.

        The two TP workers advance _fwd_round independently: whichever rank
        sees lighter expert traffic finishes each training cycle faster and
        pulls ahead.  When they diverge the TP-correct exchange keys don't
        match and every TCPStore collect times out, silently setting
        bwd_state["ok"]=False and blocking the optimizer step.

        Both ranks publish their current round to "bt_rndz/{rank}" and poll
        until the peer has caught up.  The lagging rank jumps to the faster
        rank's value so they converge in one iteration.  Gracefully falls back
        (returns True) when TP-correct is disabled or the store is unavailable.

        Returns True when both ranks are on the same round, False on timeout.
        Caller must release _fwd_running and set _fwd_layer_state=None on False.
        """
        if not (
            self._tp_correct
            and self._tp_size == 2
            and self._tp_correct_store is not None
        ):
            return True

        import time as _t
        from datetime import timedelta as _td

        my_key = f"bt_rndz/{self._tp_rank}"
        peer_key = f"bt_rndz/{1 - self._tp_rank}"

        try:
            self._tp_correct_store.set(my_key, str(self._fwd_round).encode())
        except Exception as _e:
            log.debug("[BubbleTea] rndz publish failed: %s", _e)
            return True  # store unavailable; proceed unsynchronised

        deadline = _t.time() + 2.0
        while _t.time() < deadline:
            try:
                # wait() returns immediately after the first round (key exists).
                self._tp_correct_store.wait([peer_key], _td(milliseconds=50))
                peer_round = int(self._tp_correct_store.get(peer_key))
            except Exception:
                _t.sleep(0.005)
                continue

            if peer_round == self._fwd_round:
                return True

            if peer_round > self._fwd_round:
                # Jump to the faster rank's round and republish.
                self._fwd_round = peer_round
                try:
                    self._tp_correct_store.set(my_key, str(self._fwd_round).encode())
                except Exception:
                    return True
                continue  # immediately re-check; peer may already be satisfied

            # Peer is behind; it will jump to our round on its next poll.
            _t.sleep(0.005)

        import warnings as _w

        _w.warn(
            f"[BubbleTea LoRA] rndz timed out after 2s"
            f" (rank={self._tp_rank} round={self._fwd_round})",
            stacklevel=1,
        )
        log.debug(
            "[BubbleTea] rndz timed out after 2s (rank=%d round=%d)",
            self._tp_rank,
            self._fwd_round,
        )
        return False

    # ── Replicated-param sync (main CUDA stream, both TP ranks) ─────────────

    @torch.no_grad()
    def sync_replicated_params(self) -> None:
        """Broadcast replicated LoRA params from the optimizer rank to the other.

        Called at the top of Qwen3MoeForCausalLM.forward() so it runs on the
        main CUDA stream with both TP ranks guaranteed to participate together —
        the only safe place for cross-rank NCCL given the bubble-scheduler
        architecture.

        Replicated params:
          - lora_A for Q/K/V projections (colwise TP: same matrix on all ranks)
          - lora_B for O projection       (rowwise TP: same matrix on all ranks)

        Uses _train_tp_pg (a dedicated NCCL communicator) so these collectives
        don't interleave with inference TP all-reduces on the same communicator.

        After syncing, calls _sync_to_vllm() on the rank that RECEIVED the
        params (didn't run the optimizer) so inference weights are updated too.
        """
        if not self._do_param_sync or self._train_tp_pg is None:
            return

        dev = torch.device(f"cuda:{self.device}")

        # Cheap coordination: one int32 all-reduce to check if any rank has
        # a pending sync.  Both ranks call this every forward() — it costs
        # ~1µs and avoids the large param tensor exchange on most steps.
        pending = torch.tensor(
            int(self._pending_param_sync), dtype=torch.int32, device=dev
        )
        dist.all_reduce(pending, op=dist.ReduceOp.MAX, group=self._train_tp_pg)
        if pending.item() == 0:
            return

        i_ran_optimizer = self._pending_param_sync

        # Collect replicated params in a deterministic order
        params = self._replicated_lora_params()

        if not params:
            self._pending_param_sync = False
            return

        # Pack into one float32 buffer → single all-reduce instead of N calls.
        # The optimizer rank contributes its updated params; the other contributes
        # zeros.  After SUM: both ranks hold the optimizer rank's values.
        if i_ran_optimizer:
            flat = torch.cat([p.data.view(-1).float() for p in params])
        else:
            flat = torch.zeros(
                sum(p.numel() for p in params), dtype=torch.float32, device=dev
            )

        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self._train_tp_pg)

        # Unpack back into each param tensor
        offset = 0
        for p in params:
            n = p.numel()
            p.data.copy_(flat[offset : offset + n].view_as(p.data).to(p.dtype))
            offset += n

        self._pending_param_sync = False

        # The rank that received fresh params also needs to merge them into the
        # inference weights.  (The optimizer rank already called _sync_to_vllm
        # inside optimizer_step(); calling it again here on the same rank would
        # apply an unnecessary double-delta.)
        if not i_ran_optimizer:
            self._sync_to_vllm()
            log.debug(
                "[BubbleTea] synced replicated LoRA params from optimizer rank "
                "(tp_rank=%d received)",
                self._tp_rank,
            )
        else:
            log.debug(
                "[BubbleTea] synced replicated LoRA params to peer (tp_rank=%d sent)",
                self._tp_rank,
            )

    # ── Manual backward helpers ───────────────────────────────────────────

    @torch.no_grad()
    def _lm_head_grad(
        self,
        hidden_last: torch.Tensor,  # [T, H]
        labels: torch.Tensor,  # [1, T] int64
        round_id: int | None = None,
        async_grad: bool = False,
    ) -> tuple:
        """CE loss and gradient seed for the manual backward sweep.

        Returns (grad_hidden, loss) — or, with async_grad=True,
        (grad_hidden_LOCAL_PARTIAL, loss, future_or_None): the cross-rank
        grad sum is published asynchronously and the caller resolves the
        future at its consume point (bwd_state["_pending_gh"]); future is
        None when no exchange was needed (TP off / fallback path).

        The LM head is vocab-sharded across TP ranks (ParallelLMHead), so
        logits_local covers only this rank's vocab shard. Both TP ranks run
        the backward sub-ops, each on its own bubble-scheduler timeline, so
        synchronous NCCL here would deadlock on skew (item 7 in
        Sessions_10_6_2026.md).

        With _tp_correct off (default): softmax/CE over the local shard only.
        Tokens whose labels fall outside the shard are ignored, and the
        softmax denominator covers just the local shard — a different
        (smaller-denominator) distribution than the true full-vocab softmax.

        With _tp_correct on: Megatron-style vocab-parallel CE via two
        TCPStore exchanges (same per-round timeout/fallback semantics as the
        item 1/2 corrections; never touches _train_tp_pg/NCCL):
          1. "lmh_stats": per-token [max_logit, sum_exp, label_logit] — the
             peers' stats recover the full-vocab softmax denominator
             (log-sum-exp combined across shards) and the true CE loss;
          2. "lmh_grad": dL/d(hidden) = probs_full_vocab @ lm_w sums over the
             whole vocab, so each rank's shard yields only a partial term —
             exchange-summed like item 1's o_total.
        If exchange 1 fails (timeout/round mismatch), this round falls back
        to the local-shard path above; if only exchange 2 fails, the returned
        gradient is this rank's partial term of the correct gradient (same
        class of approximation as item 1's fallback).

        round_id should be the _fwd_round of the forward that produced
        hidden_last (stamped into trainer._fwd["round"]); defaults to the
        current _fwd_round.
        """

        def _ret(g, loss, fut=None):
            return (g, loss, fut) if async_grad else (g, loss)

        lm_w = self._lm_head.weight.detach().float()  # [vocab_local, H]
        logits_local = hidden_last.float() @ lm_w.T  # [T, vocab_local]

        shift_logits = logits_local[:-1].contiguous()  # [T-1, vocab_local]
        shift_labels = labels.squeeze(0)[1:].long()  # [T-1]

        vocab_local = lm_w.shape[0]
        v_start = self._tp_rank * vocab_local

        # Global validity (same on every TP rank — identical labels per round)
        valid = shift_labels != -100  # [T-1]
        n_valid = valid.sum().item()
        if n_valid == 0:
            return _ret(None, 0.0)

        # Tokens whose label falls inside this rank's vocab shard.
        owned = (
            valid & (shift_labels >= v_start) & (shift_labels < v_start + vocab_local)
        )
        owned_idx = shift_labels[owned] - v_start  # local column index

        if self._tp_correct and self._tp_size > 1:
            rid = round_id if round_id is not None else self._fwd_round
            local_max = shift_logits.max(dim=-1).values  # [T-1]
            local_sumexp = torch.exp(shift_logits - local_max[:, None]).sum(-1)  # [T-1]
            label_logit = torch.zeros_like(local_max)
            label_logit[owned] = shift_logits[owned, owned_idx]
            stats = torch.stack([local_max, local_sumexp, label_logit])  # [3, T-1]
            peer_stats = self._tp_correct_exchange_gather(stats, "lmh_stats", rid)
            if peer_stats is not None:
                all_stats = [stats] + [p.to(stats.device) for p in peer_stats]
                global_max = torch.stack([s[0] for s in all_stats]).max(dim=0).values
                # Each rank's sum_exp is relative to its own local max —
                # rescale to the global max before summing shards.
                sum_exp = torch.stack(
                    [s[1] * torch.exp(s[0] - global_max) for s in all_stats]
                ).sum(0)
                # label_logit is nonzero on exactly the owning rank's shard.
                label_logit_full = torch.stack([s[2] for s in all_stats]).sum(0)
                log_z = global_max + torch.log(sum_exp)  # [T-1]
                loss_scalar = ((log_z - label_logit_full)[valid].sum() / n_valid).item()

                probs = torch.exp(
                    shift_logits - log_z[:, None]
                )  # full-vocab softmax, local cols
                probs[~valid] = 0.0
                probs[owned, owned_idx] -= 1.0
                probs /= n_valid
                # dL/d(hidden) sums over the whole vocab — this rank's shard
                # yields only a partial term, exchange-summed like item 1's
                # o_total. (Padded to [T, H] before the exchange so the async
                # future resolves to the final shape directly.)
                grad_local = torch.zeros_like(hidden_last)
                grad_local[:-1] = (probs @ lm_w).to(hidden_last.dtype)
                if async_grad:
                    fut = self._tp_correct_exchange_sum_async(
                        grad_local.contiguous(), "lmh_grad", rid
                    )
                    return grad_local, loss_scalar, fut
                grad_full = self._tp_correct_exchange_sum(
                    grad_local.contiguous(), "lmh_grad", rid
                )
                return grad_full, loss_scalar
            # stats exchange failed — fall through to the local-shard path.

        if not owned.any():
            return _ret(None, 0.0)
        n_owned = owned.sum().item()

        local_idx = shift_labels.clone()
        local_idx[~owned] = -100
        local_idx[owned] -= v_start

        loss_scalar = F.cross_entropy(shift_logits, local_idx, ignore_index=-100).item()

        probs = torch.softmax(shift_logits, dim=-1)  # [T-1, vocab_local]
        probs[~owned] = (
            0.0  # ignored / out-of-shard tokens: zero grad (CE ignore_index)
        )
        probs[owned, owned_idx] -= 1.0
        probs /= n_owned
        grad_shifted = (probs @ lm_w).to(hidden_last.dtype)  # [T-1, H]

        # Pad the last token position with zero (no next-token target for T-1)
        grad_hidden = torch.zeros_like(hidden_last)
        grad_hidden[:-1] = grad_shifted
        return _ret(grad_hidden, loss_scalar)

    @torch.no_grad()
    def _moe_backward_passthrough(
        self,
        x2: torch.Tensor,  # [T, H] — FFN input (output of post-attn LN)
        grad_hidden: torch.Tensor,  # [T, H] — dL/d(MoE output)
        layer_mlp,
        saved_gate_up: dict | None = None,  # {local_e: Tensor[n, 2*d_inter]} from fwd
        e_start: int | None = None,  # first local expert index (chunked mode)
        e_end: int | None = None,  # exclusive upper bound  (chunked mode)
        grad_x2_acc: torch.Tensor
        | None = None,  # accumulate into this tensor (chunked)
    ) -> torch.Tensor:
        """Passthrough gradient through the frozen MoE FFN.

        Re-runs the MoE forward to recover gate_up intermediates, then computes
        dL/d(x2) without computing any expert weight gradients.

        If saved_gate_up is provided (populated by _training_moe_forward during
        the C+D forward sub-op), the forward re-run of w13 is skipped per expert
        — eliminating ~40% of the HBM reads in the backward pass.

        For non-MoE layers (no .gate attr, e.g. mock in unit tests) the MoE
        is an identity, so the gradient passes straight through.
        """
        if not hasattr(layer_mlp, "gate"):
            return grad_hidden.clone()

        dtype = x2.dtype

        # ── Routing (frozen) — same as _training_moe_forward ──────────────
        gate_w = layer_mlp.gate.weight.detach()  # [128, H]
        router_logits = x2 @ gate_w.T  # [T, 128]
        scores = torch.softmax(router_logits.float(), dim=-1).to(dtype)
        topk_scores, topk_ids = torch.topk(
            scores, layer_mlp.experts.top_k, dim=-1
        )  # [T, k]
        topk_scores = topk_scores / (topk_scores.sum(-1, keepdim=True) + 1e-9)

        w13 = layer_mlp.experts.w13_weight.detach()  # [E_local, 2*d_inter, H]
        w2 = layer_mlp.experts.w2_weight.detach()  # [E_local, H, d_inter]
        d_inter = w2.shape[2]

        # Chunked mode: accumulate into provided tensor; full mode: fresh zeros.
        grad_x2 = grad_x2_acc if grad_x2_acc is not None else torch.zeros_like(x2)
        _e_start = e_start if e_start is not None else 0
        _e_end = e_end if e_end is not None else self._ep_num_local_experts

        for local_e in range(_e_start, _e_end):
            global_e = self._ep_expert_start + local_e
            token_mask = (topk_ids == global_e).any(-1)  # [T]
            if not token_mask.any():
                continue

            tokens = x2[token_mask]  # [n, H]
            if saved_gate_up is not None and local_e in saved_gate_up:
                gate_up = saved_gate_up[local_e]  # [n, 2*d_inter] — no w13 read
            else:
                gate_up = tokens @ w13[local_e].T  # [n, 2*d_inter] — fallback
            gate = gate_up[:, :d_inter]  # [n, d_inter]
            up = gate_up[:, d_inter:]  # [n, d_inter]

            # Routing weight for this expert (same computation as forward)
            exp_mask = topk_ids[token_mask] == global_e  # [n, k]
            score_e = (topk_scores[token_mask] * exp_mask.to(dtype)).sum(
                -1, keepdim=True
            )  # [n, 1]

            # ── Backward through: output += (silu(gate)*up @ w2.T) * score_e ──

            # down projection: out_e = act @ w2[e].T  →  grad_act = grad_out_e @ w2[e]
            grad_out_e = grad_hidden[token_mask] * score_e  # [n, H]
            grad_act = grad_out_e @ w2[local_e]  # [n, d_inter]  (w2: [H, d_inter])

            # SwiGLU: act = silu(gate) * up
            #   d(act)/d(gate) = silu_deriv(gate) * up
            #   d(act)/d(up)   = silu(gate)
            sig = torch.sigmoid(gate)  # [n, d_inter]
            silu_deriv = sig * (1.0 + gate * (1.0 - sig))  # [n, d_inter]
            grad_gate = grad_act * silu_deriv * up  # [n, d_inter]
            grad_up = grad_act * F.silu(gate)  # [n, d_inter]
            grad_gate_up = torch.cat([grad_gate, grad_up], dim=-1)  # [n, 2*d_inter]

            # gate+up proj: grad_tokens = grad_gate_up @ w13[e]
            grad_x2[token_mask] += grad_gate_up @ w13[local_e]  # [n, H]

        # grad_x2 holds only this rank's local experts' contributions. The
        # cross-rank exchange-sum (item 9) lives in the *last* passthrough
        # chunk op (_make_passthrough_chunk_ops), once all local-expert
        # chunks have accumulated — not here, since this function is called
        # once per chunk.

        return grad_x2

    # ── Forward pass ──────────────────────────────────────────────────────

    def _forward_loss(self) -> torch.Tensor:
        """
        Full 48-layer forward on a real training batch.
        Base model weights are detached; only LoRA deltas participate in autograd.
        Returns scalar CE loss.
        """
        input_ids, labels = self._get_batch()  # [1, T]
        T = input_ids.shape[1]
        dev = f"cuda:{self.device}"

        # Token embeddings via the model's VocabParallelEmbedding
        with torch.no_grad():
            hidden = self._embed(input_ids.squeeze(0))  # [T, H]

        positions = torch.arange(T, device=dev, dtype=torch.long)
        residual: torch.Tensor | None = None
        last_attn_out: torch.Tensor | None = None

        for i, layer in enumerate(self._layers):
            hidden, residual, x_norm, attn_out, x2, _ = self._layer_forward(
                i, layer, hidden, residual, positions
            )
            last_attn_out = attn_out

        # _forward_loss is the single-shot autograd fallback path (used by
        # training_step and TestGradientFlow unit tests only — not production).
        # last_attn_out still carries grad via the LoRA deltas in _attn_lora_forward
        # (hidden is fully detached by the no_grad FFN block).
        with torch.no_grad():
            lm_w = self._lm_head.weight.detach()  # [vocab_local, H]
        logits_local = last_attn_out @ lm_w.T  # [T, vocab_local] — HAS grad

        # No TP all-gather: each rank computes loss on its local logit shard.
        # This keeps the training path entirely NCCL-free.
        logits = logits_local

        # Shift for next-token prediction: predict token i+1 from token i
        shift_logits = logits[:-1].contiguous()  # [T-1, vocab]
        shift_labels = labels.squeeze(0)[1:].contiguous()  # [T-1]
        loss = F.cross_entropy(
            shift_logits.float(),
            shift_labels,
            ignore_index=-100,
        )
        return loss

    def _layer_forward(
        self,
        layer_idx: int,
        layer,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One transformer layer forward.  All operations run under no_grad —
        the full forward pass is used for manual backward, not autograd.

        Returns (hidden, residual, x_norm, attn_out, x2) where:
          x_norm   — output of input LayerNorm, input to QKV projections.
                     Needed by Q/V LoRA grad sub-ops: dL/dA = x_norm.T @ grad @ B.
          attn_out — output of SDPA, before O projection.
                     Needed by O LoRA grad sub-op: dL/dA_o = attn_out.T @ grad @ B_o.
          x2       — output of post-attention LayerNorm, input to FFN.
                     Needed by FFN passthrough backward to propagate grad_hidden
                     back through SwiGLU and RMSNorm to reach attn_out.
        """
        # ── Input LayerNorm ──────────────────────────────────────────────
        with torch.no_grad():
            if residual is None:
                residual = hidden
                x_norm = layer.input_layernorm(hidden)
            else:
                x_norm, residual = layer.input_layernorm(hidden, residual)

        # ── Attention with LoRA ──────────────────────────────────────────
        attn_out = self._attn_lora_forward(
            layer_idx, layer.self_attn, x_norm, positions
        )

        # ── Post-attention LayerNorm + residual ──────────────────────────
        # x2 is the FFN input — saved for the FFN passthrough backward.
        with torch.no_grad():
            x2, residual = layer.post_attention_layernorm(attn_out, residual)

        # ── MoE FFN (frozen) ─────────────────────────────────────────────
        gate_up_out: dict = {}
        with torch.no_grad():
            hidden = self._training_moe_forward(
                layer.mlp, x2, gate_up_out=gate_up_out, layer_idx=layer_idx
            )

        return hidden, residual, x_norm, attn_out, x2, gate_up_out

    def _attn_lora_forward(
        self,
        layer_idx: int,
        attn_layer,
        x: torch.Tensor,  # [T, H]
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention projection with LoRA deltas.
        Only lora_A and lora_B are in the autograd graph; base weights are detached.
        """
        T = x.shape[0]
        nh = self._n_heads_local
        nkv = self._n_kv_local
        hd = _HEAD_DIM
        scaling = _LORA_ALPHA / 16  # lora_alpha / lora_rank = 1.0

        # ── Retrieve base QKV weights (detached) ─────────────────────────
        # qkv_proj.weight layout (per rank): [q_local + k_local + v_local, H]
        q_sz = nh * hd
        kv_sz = nkv * hd
        with torch.no_grad():
            qkv_w = attn_layer.qkv_proj.weight.detach()  # [q+k+v, H] per rank
            q_base = F.linear(x.detach(), qkv_w[:q_sz])  # [T, q_sz]
            k_base = F.linear(x.detach(), qkv_w[q_sz : q_sz + kv_sz])  # [T, kv_sz]
            v_base = F.linear(x.detach(), qkv_w[q_sz + kv_sz :])  # [T, kv_sz]
            # _sync_to_vllm merges the LoRA delta into qkv_proj.weight after each
            # optimizer step.  Subtract the previously-merged delta so the training
            # forward uses the pure base-model output (the explicit q_delta/v_delta
            # below then re-adds the current LoRA contribution without double-counting).
            xf = x.detach().float()
            prev_q = getattr(self, f"_prev_delta_{layer_idx}_q_proj", None)
            if prev_q is not None:
                q_base = (q_base.float() - F.linear(xf, prev_q.float())).to(
                    q_base.dtype
                )
            prev_k = getattr(self, f"_prev_delta_{layer_idx}_k_proj", None)
            if prev_k is not None:
                k_base = (k_base.float() - F.linear(xf, prev_k.float())).to(
                    k_base.dtype
                )
            prev_v = getattr(self, f"_prev_delta_{layer_idx}_v_proj", None)
            if prev_v is not None:
                v_base = (v_base.float() - F.linear(xf, prev_v.float())).to(
                    v_base.dtype
                )

        # ── LoRA deltas for Q, K, and V ──────────────────────────────────
        lora_d = self._lora.get(layer_idx, {})
        A_q = lora_d.get("q_proj.lora_A")
        B_q = lora_d.get("q_proj.lora_B")
        A_k = lora_d.get("k_proj.lora_A")
        B_k = lora_d.get("k_proj.lora_B")
        A_v = lora_d.get("v_proj.lora_A")
        B_v = lora_d.get("v_proj.lora_B")

        # x participates in autograd only through the LoRA path
        q_delta = (
            (x @ A_q.T.to(x.dtype)) @ B_q.T.to(x.dtype) * scaling
            if (A_q is not None and B_q is not None)
            else 0
        )
        k_delta = (
            (x @ A_k.T.to(x.dtype)) @ B_k.T.to(x.dtype) * scaling
            if (A_k is not None and B_k is not None)
            else 0
        )
        v_delta = (
            (x @ A_v.T.to(x.dtype)) @ B_v.T.to(x.dtype) * scaling
            if (A_v is not None and B_v is not None)
            else 0
        )

        q = q_base.detach() + q_delta  # [T, q_sz]  — grad flows through delta
        k = k_base.detach() + k_delta  # [T, kv_sz] — grad flows through delta
        v = v_base.detach() + v_delta  # [T, kv_sz]

        # ── QK norm (per head) — present in Qwen3, absent in Qwen1.5-MoE ──
        if hasattr(attn_layer, "q_norm"):
            with torch.no_grad():
                q = attn_layer.q_norm(q.view(T, nh, hd)).view(T, q_sz)
                k = attn_layer.k_norm(k.view(T, nkv, hd)).view(T, kv_sz)

        # ── Rotary embeddings ─────────────────────────────────────────────
        with torch.no_grad():
            q_rot, k_rot = attn_layer.rotary_emb(positions, q, k)

        # ── Standard causal attention (not paged) ────────────────────────
        # Add a batch dimension [1, H, T, D] so PyTorch dispatches flash attention.
        # Without the batch dim (3D input), SDPA falls back to the naive math
        # implementation which computes QK in bfloat16.  After QK-norm the
        # attention scores can reach ~11 → exp(11) ≈ 80k > bfloat16 max (65504)
        # → Inf → NaN in softmax → NaN residual propagating through all 48 layers.
        # Flash attention computes softmax in float32 internally, avoiding overflow.
        q3 = q_rot.view(T, nh, hd).transpose(0, 1).unsqueeze(0)  # [1, nh,  T, hd]
        k3 = k_rot.view(T, nkv, hd).transpose(0, 1).unsqueeze(0)  # [1, nkv, T, hd]
        v3 = v.view(T, nkv, hd).transpose(0, 1).unsqueeze(0)  # [1, nkv, T, hd]

        # GQA: replicate K and V to match Q head count
        if nkv < nh:
            rep = nh // nkv
            k3 = k3.repeat_interleave(rep, dim=1)
            v3 = v3.repeat_interleave(rep, dim=1)

        attn_out = F.scaled_dot_product_attention(q3, k3, v3, is_causal=True).squeeze(
            0
        )  # [nh, T, hd]
        attn_out = attn_out.transpose(0, 1).reshape(T, nh * hd)  # [T, q_sz]

        # ── O projection with LoRA delta ──────────────────────────────────
        with torch.no_grad():
            o_w = attn_layer.o_proj.weight.detach()  # [H, q_sz_local] row-parallel
            o_base = F.linear(attn_out.detach(), o_w)  # [T, H]
            prev_o = getattr(self, f"_prev_delta_{layer_idx}_o_proj", None)
            if prev_o is not None:
                o_base = (
                    o_base.float() - F.linear(attn_out.float(), prev_o.float())
                ).to(o_base.dtype)

        A_o = lora_d.get("o_proj.lora_A")
        B_o = lora_d.get("o_proj.lora_B")
        # For row-parallel O: lora_A is sharded along input dim
        o_delta = (
            (attn_out @ A_o.T.to(attn_out.dtype)) @ B_o.T.to(attn_out.dtype) * scaling
            if (A_o is not None and B_o is not None)
            else 0
        )

        # Row-parallel O: in a standard synchronous training loop, each rank would
        # all-reduce here. In BubbleTea, the training forward fires asynchronously
        # in the bubble scheduler worker thread; NCCL from that thread races with
        # inference NCCL on the main thread (dual-group contention → deadlock).
        # By default the output is therefore a partial rank-local contribution;
        # the loss gradient will be partial but the server stays live.
        o_total = o_base.detach() + o_delta  # [T, H] — partial row-parallel sum
        if self._tp_correct:
            if torch.is_grad_enabled():
                # Autograd path (_forward_loss / training_step): the identity
                # backward of _SumAllReduce makes the rest of the backward
                # graph (computed from local shards) correct for the
                # all-reduced o_total.
                o_total = _SumAllReduce.apply(o_total, self._train_tp_pg)
            else:
                # Manual no-grad production path (build_fwd_subops). Uses a
                # TCPStore-based exchange with a bounded timeout +
                # local-value fallback (see _tp_correct_exchange_sum) --
                # never touches _train_tp_pg/NCCL, so a skewed/asymmetric
                # sub-op firing cannot poison sync_replicated_params's
                # collectives.
                o_total = self._tp_correct_exchange_sum(
                    o_total.contiguous(), f"o_{layer_idx}", self._fwd_round
                )
        return o_total

    # ── Pure-PyTorch MoE forward (training path) ─────────────────────────

    def _training_moe_forward(
        self,
        layer_mlp,
        hidden: torch.Tensor,
        gate_up_out: dict | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """Local-expert MoE forward (training path, non-C+D_batch).

        Avoids Triton JIT (no fused_moe_kernel). Each GPU computes the
        contribution from its own local experts only — `hidden` is replicated
        across ranks (same batch per round) and the routing is deterministic
        (replicated gate weight), so the full MoE output is exactly the sum
        of the per-rank local-expert partials.

        With _tp_correct off: returns the local partial directly (the
        original single-rank approximation). With _tp_correct on (and EP
        actually sharding experts): exchange-sums the partial across ranks
        via the TCPStore mechanism — the non-C+D_batch counterpart of item
        8's piggybacked inference all-to-all. Assumes the EP group spans the
        same ranks as the TP group (true for the 2-GPU deployment).

        Falls back to a direct call for non-MoE layers (e.g. mock/dense layers
        in unit tests that don't have a .gate attribute).
        """
        if not hasattr(layer_mlp, "gate"):
            return layer_mlp(hidden) if callable(layer_mlp) else hidden

        T, H = hidden.shape
        dev, dtype = hidden.device, hidden.dtype

        # Gate weight is replicated on all ranks — same routing on every GPU
        gate_w = layer_mlp.gate.weight.detach()  # [128, H]
        router_logits = hidden.detach() @ gate_w.T  # [T, 128]
        scores = torch.softmax(router_logits.float(), dim=-1).to(dtype)
        topk_scores, topk_ids = torch.topk(
            scores, layer_mlp.experts.top_k, dim=-1
        )  # [T, k]
        # norm_topk_prob=True: renormalise selected scores
        topk_scores = topk_scores / (topk_scores.sum(-1, keepdim=True) + 1e-9)

        w13 = layer_mlp.experts.w13_weight.detach()  # [E_local, 2*768, H]
        w2 = layer_mlp.experts.w2_weight.detach()  # [E_local, H, 768]
        d_inter = w2.shape[2]
        output = torch.zeros(T, H, device=dev, dtype=dtype)

        for local_e in range(self._ep_num_local_experts):
            global_e = self._ep_expert_start + local_e
            token_mask = (topk_ids == global_e).any(-1)  # [T] bool
            if not token_mask.any():
                continue
            tokens = hidden[token_mask]  # [n, H]
            gate_up = tokens @ w13[local_e].T  # [n, 2*768]
            if gate_up_out is not None:
                gate_up_out[local_e] = gate_up.detach()  # save for passthrough backward
            act = F.silu(gate_up[:, :d_inter]) * gate_up[:, d_inter:]
            out_e = act @ w2[local_e].T  # [n, H]
            # Routing weight for this expert per token
            exp_mask = topk_ids[token_mask] == global_e  # [n, k]
            score_e = (topk_scores[token_mask] * exp_mask.to(dtype)).sum(
                -1, keepdim=True
            )
            output[token_mask] += out_e * score_e

        # Shared expert path (Qwen1.5-MoE / Qwen2-MoE): all tokens pass through
        # a replicated MLP gated by a learned per-token scalar.  Access weights
        # directly to avoid NCCL from the bwd_stream worker thread.
        se = getattr(layer_mlp, "shared_expert", None)
        if se is not None:
            gu_w = (
                se.gate_up_proj.weight.detach()
            )  # [2*d_local, H] column-parallel shard
            d_w = se.down_proj.weight.detach()  # [H_out, d_local] row-parallel shard
            gate_up = hidden @ gu_w.T  # [T, 2*d_local]
            d_local = gate_up.shape[-1] // 2
            act = F.silu(gate_up[:, :d_local]) * gate_up[:, d_local:]  # SiluAndMul
            se_out = F.linear(act, d_w)  # [T, H_out] — partial; summed below
            if getattr(se, "expert_gate", None) is not None:
                sg_w = se.expert_gate.weight.detach()  # [1, H]
                se_gate = torch.sigmoid(hidden @ sg_w.T)  # [T, 1]
                se_out = se_gate * se_out
            output = output + se_out
            if gate_up_out is not None:
                gate_up_out["shared"] = gate_up.detach()

        # `output` holds only this rank's local experts' + shared expert's partial.
        # Exchange-sum across ranks to reconstruct the full MoE output (same
        # per-round timeout/fallback semantics as item 1's o_total).
        if self._tp_correct and self._ep_size > 1 and layer_idx is not None:
            output = self._tp_correct_exchange_sum(
                output.contiguous(), f"moefwd_{layer_idx}", self._fwd_round
            )

        return output

    # ── Sync LoRA weights back to vLLM's adapter manager ─────────────────

    def _sync_to_vllm(self) -> None:
        """
        Sync updated LoRA weights back to the inference model so that
        subsequent inference requests benefit from the trained weights.

        Strategy: merge the LoRA delta into the base QKV/O projection weights
        in-place (W_eff = W_base + lora_B @ lora_A * scale). We track the
        previous delta and subtract it first so updates don't compound.
        """
        try:
            scaling = _LORA_ALPHA / 16
            for layer_idx, proj_dict in self._lora.items():
                layer = self._layers[layer_idx]
                attn = layer.self_attn

                # qkv_proj.weight layout per rank: [q_local + k_local + v_local, H]
                q_sz = self._n_heads_local * _HEAD_DIM
                kv_sz = self._n_kv_local * _HEAD_DIM

                for proj_ab, param in proj_dict.items():
                    proj, ab = proj_ab.rsplit(".", 1)  # e.g. "q_proj", "lora_A"
                    if ab == "lora_A":
                        continue  # process once per projection when we see lora_B

                    A_key = proj + ".lora_A"
                    B_key = proj + ".lora_B"
                    if A_key not in proj_dict or B_key not in proj_dict:
                        continue
                    A = proj_dict[A_key].data.to(torch.bfloat16)  # [r, in_dim]
                    B = proj_dict[B_key].data.to(torch.bfloat16)  # [out_local, r]
                    delta = (B @ A) * scaling  # [out_local, in_dim]

                    prev_key = f"_prev_delta_{layer_idx}_{proj}"
                    prev = getattr(self, prev_key, None)

                    with torch.no_grad():
                        if proj == "q_proj":
                            w = attn.qkv_proj.weight[:q_sz]
                            if prev is not None:
                                w -= prev
                            w += delta
                        elif proj == "k_proj":
                            w = attn.qkv_proj.weight[q_sz : q_sz + kv_sz]
                            if prev is not None:
                                w -= prev
                            w += delta
                        elif proj == "v_proj":
                            w = attn.qkv_proj.weight[q_sz + kv_sz :]
                            if prev is not None:
                                w -= prev
                            w += delta
                        elif proj == "o_proj":
                            w = attn.o_proj.weight
                            if prev is not None:
                                w -= prev
                            w += delta

                    setattr(self, prev_key, delta.clone())

        except Exception as exc:
            log.debug("[BubbleTea LoRA] sync_to_vllm failed: %s", exc)

    # ── Sub-op builders (for bubble scheduler integration) ────────────────

    def build_fwd_subops(self) -> list:
        """
        Return 1 + N_layers sub-ops for the bubble scheduler.

        Sub-op 0 (init): checks readiness, loads a batch, embeds tokens.
        Sub-ops 1..N (per-layer): each runs one transformer layer's forward
            (_layer_forward = attention with LoRA deltas + frozen MoE FFN).

        Each sub-op is ~2-5 ms and fits in one EP all-to-all bubble during
        prefill.  State is threaded via trainer._fwd_layer_state.
        The lock (_fwd_running) is acquired in the init sub-op and released
        in the final layer sub-op (or immediately if any guard fails).
        """
        trainer = self
        N = len(trainer._layers)

        def _fwd_init():
            # ── Guards (no lock held yet) ──────────────────────────────────
            import time as _t

            # Start data thread on first call.
            if not trainer._data_thread_started:
                trainer._data_thread_started = True
                threading.Thread(
                    target=trainer._build_data_iter_bg,
                    args=(trainer._tokenizer_path, trainer._cache_dir),
                    daemon=True,
                    name="bt-lora-data-init",
                ).start()

            if not trainer._data_ready.is_set():
                if not getattr(trainer, "_fwd_data_wait_warned", False):
                    trainer._fwd_data_wait_warned = True
                    import warnings as _w

                    _w.warn(
                        f"[BubbleTea LoRA] fwd_init: waiting for training data "
                        f"(device={trainer.device})",
                        stacklevel=1,
                    )
                trainer._fwd_layer_state = None
                return

            if _t.time() - trainer._fwd_last_done < trainer._FWD_COOLDOWN_S:
                trainer._fwd_layer_state = None
                return

            # Prevent concurrent forwards (lock stays held until last layer).
            if not trainer._fwd_running.acquire(blocking=False):
                log.debug(
                    "[BubbleTea] fwd_init: lock busy (rank=%d round=%d)",
                    trainer._tp_rank,
                    trainer._fwd_round,
                )
                trainer._fwd_layer_state = None
                return

            # New forward round starting -- see _fwd_round's definition in
            # __init__ (round id for _tp_correct_exchange_sum).
            trainer._fwd_round += 1
            log.debug(
                "[BubbleTea] fwd_init: starting round %d (rank=%d)",
                trainer._fwd_round,
                trainer._tp_rank,
            )

            # Rendezvous: agree on a shared round number across TP ranks so
            # the TCPStore exchange keys match. Without this, the faster rank
            # (lighter expert traffic) pulls ahead and every exchange times out.
            if not trainer._sync_fwd_round():
                trainer._fwd_running.release()
                trainer._fwd_layer_state = None
                return

            # ── Initialise per-layer state ─────────────────────────────────
            try:
                with torch.no_grad():
                    dev = f"cuda:{trainer.device}"

                    # Each rank loads its own batch independently.  NCCL broadcasts
                    # from the bubble scheduler thread race with inference NCCL on
                    # the main thread (dual-group deadlock); both ranks use the same
                    # deterministic dataset iterator so they naturally train on the
                    # same sequences without cross-rank communication.
                    ids, labels = trainer._get_batch()

                    T = ids.shape[1]
                    positions = torch.arange(T, device=dev, dtype=torch.long)
                    tids = ids.squeeze(0)  # [T]
                    if hasattr(trainer._embed, "weight"):
                        # Bypass VocabParallelEmbedding.forward(), which calls
                        # tensor_model_parallel_all_reduce on the inference TP
                        # communicator — this races with inference TP all-reduces
                        # from bwd_stream and deadlocks.
                        # Instead compute local partial embeddings and correct
                        # them via the TCPStore-based exchange (see
                        # _tp_correct_exchange_sum), gated by VLLM_FT_TP_CORRECT.
                        embed_w = trainer._embed.weight.detach()  # [vocab_local, H]
                        v_local = embed_w.shape[0]
                        v_start = trainer._tp_rank * v_local
                        valid = (tids >= v_start) & (tids < v_start + v_local)
                        local_ids = (tids - v_start).clamp(0, v_local - 1)
                        hidden = embed_w[local_ids]  # [T, H]
                        hidden[~valid] = 0
                        # Partial embedding: tokens outside this rank's vocab shard
                        # are zeroed. Each token is nonzero on exactly one rank's
                        # shard, so summing across TP ranks reconstructs the full
                        # embedding row -- same correction as o_total in
                        # _attn_lora_forward, with the same per-round timeout +
                        # local-value fallback. Published async; layer 0's
                        # sub-op resolves the future.
                        if trainer._tp_correct:
                            hidden_fut = trainer._tp_correct_exchange_sum_async(
                                hidden.contiguous(), "embed", trainer._fwd_round
                            )
                        else:
                            hidden_fut = None
                    else:
                        hidden = trainer._embed(tids)  # mock / single-GPU
                        hidden_fut = None

                trainer._fwd_layer_state = {
                    "hidden": hidden,
                    "hidden_fut": hidden_fut,
                    "residual": None,
                    "positions": positions,
                    "labels": labels,
                    "ok": True,
                }
            except Exception as exc:
                log.debug("[BubbleTea LoRA] fwd_init error: %s", exc)
                trainer._fwd_layer_state = None
                trainer._fwd_running.release()

        def _make_layer_op(i: int):
            is_last = i == N - 1

            def _fwd_layer():
                state = trainer._fwd_layer_state
                if state is None or not state["ok"]:
                    # Init was skipped or a previous layer failed.
                    if is_last and state is not None:
                        trainer._fwd_running.release()
                    return
                try:
                    # Layer 0: consume the embed exchange published by
                    # _fwd_init (its collect ran during the inter-op gap).
                    if i == 0 and state.get("hidden_fut") is not None:
                        state["hidden"] = state.pop("hidden_fut").result()
                    # If the embed exchange timed out and fell back, rank 1
                    # ends up with all-zero hidden (its vocab shard has no
                    # tokens for this batch).  Propagating zeros through 48
                    # layers produces loss ≈ ln(V) with near-zero gradient —
                    # skip to avoid poisoning the accumulator.
                    if i == 0 and state["hidden"].abs().max().item() == 0.0:
                        log.warning(
                            "[BubbleTea] all-zero embedding (embed-exchange fallback "
                            "or empty vocab shard) — skipping round"
                        )
                        state["ok"] = False
                        return

                    # res_a = pre-norm sum for input LN (hidden + prev_residual).
                    # Captured before _layer_forward; needed by passthrough bwd.
                    _res = state["residual"]
                    res_a = (
                        (state["hidden"] + _res).detach()
                        if _res is not None
                        else state["hidden"].detach()
                    )

                    with torch.no_grad():
                        hidden, residual, x_norm, attn_out, x2, gate_up = (
                            trainer._layer_forward(
                                i,
                                trainer._layers[i],
                                state["hidden"],
                                state["residual"],
                                state["positions"],
                            )
                        )
                    state["hidden"] = hidden
                    state["residual"] = residual

                    if _NAN_DEBUG:
                        with torch.no_grad():
                            h_rows = torch.isnan(hidden).any(-1)
                            r_rows = torch.isnan(residual).any(-1)
                            a_rows = torch.isnan(attn_out).any(-1)
                            hn, rn, an = (
                                int(h_rows.sum()),
                                int(r_rows.sum()),
                                int(a_rows.sum()),
                            )
                            rm = residual.float().nan_to_num(0.0).abs().max().item()
                            hm = hidden.float().nan_to_num(0.0).abs().max().item()
                            am = attn_out.float().nan_to_num(0.0).abs().max().item()
                            prev_clean = state.get("_nan_clean", True)
                            now_nan = bool(hn or rn or an)
                            # Log the first layer where NaN appears, any layer
                            # with extreme magnitudes, and the last layer.
                            if (
                                (now_nan and prev_clean)
                                or rm > 1e6
                                or hm > 1e6
                                or is_last
                            ):
                                log.warning(
                                    "[BT-NANDBG] round=%d layer=%d "
                                    "nan_rows(hid=%d res=%d attn=%d)/%d "
                                    "absmax(hid=%.3e res=%.3e attn=%.3e)",
                                    trainer._fwd_round,
                                    i,
                                    hn,
                                    rn,
                                    an,
                                    hidden.shape[0],
                                    hm,
                                    rm,
                                    am,
                                )
                            state["_nan_clean"] = prev_clean and not now_nan

                    # Save activations needed by the backward sub-ops.
                    if "layers" not in trainer._fwd:
                        trainer._fwd["layers"] = {}
                    trainer._fwd["layers"][i] = {
                        "x_norm": x_norm,  # [T, H]   input to QKV
                        "attn_out": attn_out,  # [T, H]  O proj output
                        "x2": x2,  # [T, H]   input to FFN
                        "res_a": res_a,  # [T, H]  pre-norm sum for input LN
                        "gate_up": gate_up,  # {e: Tensor[n,2d]} SwiGLU inputs
                    }

                    if is_last:
                        import time as _t

                        # True final hidden = FFN output + accumulated residual.
                        # 'hidden' is just the last MoE delta; 'residual' carries
                        # the accumulated skip connection through all 48 layers.
                        # Without residual, logits are near-zero → CE ≈ log(vocab).
                        final = (hidden + residual).detach()
                        h_nan = torch.isnan(hidden).any().item()
                        r_nan = torch.isnan(residual).any().item()
                        if h_nan or r_nan:
                            log.warning(
                                "[BubbleTea] NaN in last fwd: hidden=%s residual=%s "
                                "— skipping round",
                                h_nan,
                                r_nan,
                            )
                            trainer._fwd_layer_state = None
                            trainer._fwd_running.release()
                            return
                        # If hidden_last contains Inf or extreme values that would
                        # cause float32 overflow (|x|^2 > float32_max) inside _rms_norm,
                        # skip this cycle to avoid propagating bad gradients.
                        _bad = (
                            torch.isinf(final).any().item()
                            or final.float().abs().max().item() > 1e18
                        )
                        if _bad:
                            log.warning(
                                "[BubbleTea] hidden_last has Inf/extreme values "
                                "(max=%.2e) — skipping this training sample",
                                final.float().abs().max().item(),
                            )
                            trainer._fwd_layer_state = None
                            trainer._fwd_running.release()
                            return
                        trainer._fwd["hidden_last"] = final
                        trainer._fwd["labels"] = state["labels"]
                        # Round id of the forward that produced hidden_last —
                        # _lm_head_op's vocab-parallel-CE exchange must use this,
                        # not whatever _fwd_round is by the time backward runs.
                        trainer._fwd["round"] = trainer._fwd_round
                        trainer._fwd_last_done = _t.time()
                        trainer._fwd_layer_state = None
                        trainer._fwd_running.release()
                except Exception as exc:
                    log.debug("[BubbleTea LoRA] fwd_layer_%d error: %s", i, exc)
                    state["ok"] = False
                    if is_last:
                        trainer._fwd_running.release()

            return _fwd_layer

        return [_fwd_init] + [_make_layer_op(i) for i in range(N)]

    def build_fwd_subops_cdbatch(self) -> list:
        """C+D_batch variant: attention-only forward sub-ops.

        Same structure as build_fwd_subops but each layer sub-op runs only
        LayerNorm + attention (no MoE FFN).  After computing x2 (the post-
        attention residual that is the MoE input), it records a CUDA event on
        bwd_stream and calls ft_moe_set_hidden(x2, event) so the main stream's
        inline fused MoE call can use the real training hidden state instead of
        the synthetic zeros from ft_moe_arm.

        The MoE for training tokens is handled by the inline batch injection in
        Qwen3MoeSparseMoeBlock.forward() — both inference and FT tokens are
        concatenated into one batch and routed through the same EP
        dispatch/combine all-to-all, so ft_moe_advance receives an
        EP-correct MoE output for the FT tokens at no extra communication
        cost. The next layer's sub-op (_fwd_attn_only / _fwd_finalize for the
        last layer) retrieves that output via ft_moe_get_moe_delta() and adds
        it to "hidden" before continuing — falling back to the previous
        x2-only approximation if it isn't ready yet (layer-index mismatch).
        _passthrough in the backward pass re-runs MoE locally (same as C+D),
        so gradient computation remains correct (forward value is EP-correct;
        the backward gradient through the MoE branch is still a local-experts
        approximation).
        """
        trainer = self
        N = len(trainer._layers)

        # Reuse the identical _fwd_init — it loads data, embeds tokens, and
        # stores the embedding in _fwd_layer_state.  We additionally push the
        # embedding into _ft_moe_state so layer 0's inline MoE starts with a
        # real (not zero) hidden state.
        def _fwd_init_cdbatch():
            # Run the standard init first (populates _fwd_layer_state).
            # We call _fwd_init by re-using build_fwd_subops' init logic inline.
            import time as _t

            from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
                ft_moe_reset_moe_delta,
                ft_moe_set_hidden,
            )

            if not trainer._data_thread_started:
                trainer._data_thread_started = True
                threading.Thread(
                    target=trainer._build_data_iter_bg,
                    args=(trainer._tokenizer_path, trainer._cache_dir),
                    daemon=True,
                    name="bt-lora-data-init",
                ).start()
            if not trainer._data_ready.is_set():
                trainer._fwd_layer_state = None
                return
            if _t.time() - trainer._fwd_last_done < trainer._FWD_COOLDOWN_S:
                trainer._fwd_layer_state = None
                return
            if not trainer._fwd_running.acquire(blocking=False):
                trainer._fwd_layer_state = None
                return

            # New forward round starting -- see _fwd_round's definition in
            # __init__ (round id for _tp_correct_exchange_sum). Without this,
            # item 1's o_total correction in _attn_lora_forward would be stuck
            # on a stale round_id in this (cdbatch) forward path.
            trainer._fwd_round += 1

            # Rendezvous: agree on a shared round number across TP ranks.
            if not trainer._sync_fwd_round():
                trainer._fwd_running.release()
                trainer._fwd_layer_state = None
                return

            # Clear any stale per-layer MoE delta left over from the previous
            # round (see ft_moe_reset_moe_delta's docstring) before layer 0's
            # _fwd_attn_only seeds a new one.
            ft_moe_reset_moe_delta()

            try:
                with torch.no_grad():
                    dev = f"cuda:{trainer.device}"
                    ids, labels = trainer._get_batch()
                    T = ids.shape[1]
                    positions = torch.arange(T, device=dev, dtype=torch.long)
                    tids = ids.squeeze(0)
                    if hasattr(trainer._embed, "weight"):
                        embed_w = trainer._embed.weight.detach()
                        v_local = embed_w.shape[0]
                        v_start = trainer._tp_rank * v_local
                        valid = (tids >= v_start) & (tids < v_start + v_local)
                        local_ids = (tids - v_start).clamp(0, v_local - 1)
                        hidden = embed_w[local_ids]
                        hidden[~valid] = 0
                        if trainer._tp_correct:
                            hidden = trainer._tp_correct_exchange_sum(
                                hidden.contiguous(), "embed", trainer._fwd_round
                            )
                    else:
                        hidden = trainer._embed(tids)
                trainer._fwd_layer_state = {
                    "hidden": hidden,
                    "residual": None,
                    "positions": positions,
                    "labels": labels,
                    "ok": True,
                }
                # Seed the inline MoE with the real embedding output.
                evt = torch.cuda.Event()
                evt.record()
                ft_moe_set_hidden(hidden, evt)
            except Exception as exc:
                log.debug("[BubbleTea C+D_batch] fwd_init error: %s", exc)
                trainer._fwd_layer_state = None
                trainer._fwd_running.release()

        def _make_attn_only_op(i: int):
            def _fwd_attn_only():
                from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
                    ft_moe_get_moe_delta,
                    ft_moe_get_moe_delta_event,
                    ft_moe_get_moe_delta_layer,
                    ft_moe_set_hidden,
                )

                state = trainer._fwd_layer_state
                if state is None or not state["ok"]:
                    return
                try:
                    # Fold in layer (i-1)'s EP-correct MoE output, computed
                    # for free via vLLM's existing inline batch injection /
                    # EP all-to-all (see ft_moe_advance in moe_runner.py).
                    # Layer 0's "hidden" is the raw embedding (no
                    # previous-layer delta to add). If the previous layer's
                    # delta isn't ready yet (layer-index mismatch -- e.g. the
                    # main stream hasn't reached that MoE block), fall back
                    # to the previous approximation (hidden == x2, MoE delta
                    # omitted) for this round only.
                    if i > 0 and ft_moe_get_moe_delta_layer() == i - 1:
                        delta = ft_moe_get_moe_delta()
                        if delta is not None and delta.shape == state["hidden"].shape:
                            evt = ft_moe_get_moe_delta_event()
                            if evt is not None:
                                torch.cuda.current_stream().wait_event(evt)
                            state["hidden"] = state["hidden"] + delta

                    _res = state["residual"]
                    res_a = (
                        (state["hidden"] + _res).detach()
                        if _res is not None
                        else state["hidden"].detach()
                    )
                    with torch.no_grad():
                        layer = trainer._layers[i]
                        # ── Input LayerNorm ────────────────────────────────────
                        if state["residual"] is None:
                            residual = state["hidden"]
                            x_norm = layer.input_layernorm(state["hidden"])
                        else:
                            x_norm, residual = layer.input_layernorm(
                                state["hidden"], state["residual"]
                            )
                        # ── Attention with LoRA (no MoE) ──────────────────────
                        attn_out = trainer._attn_lora_forward(
                            i, layer.self_attn, x_norm, state["positions"]
                        )
                        # ── Post-attention LayerNorm ───────────────────────────
                        x2, residual = layer.post_attention_layernorm(
                            attn_out, residual
                        )

                    # x2 is the MoE input for layer i.  Record a CUDA event on
                    # bwd_stream so the main stream can wait before using x2 as
                    # the inline MoE hidden state.
                    evt = torch.cuda.Event()
                    evt.record()  # records on bwd_stream (we are in its worker)
                    ft_moe_set_hidden(x2, evt)

                    # Advance _fwd_layer_state for the next layer's attention.
                    # hidden is set to x2 (the MoE input) — the MoE delta will
                    # be provided by the inline pass; _passthrough in backward
                    # re-derives it anyway when computing grad_hidden.
                    state["hidden"] = x2
                    state["residual"] = residual

                    # Save activations needed by backward sub-ops.
                    if "layers" not in trainer._fwd:
                        trainer._fwd["layers"] = {}
                    trainer._fwd["layers"][i] = {
                        "x_norm": x_norm,
                        "attn_out": attn_out,
                        "x2": x2,
                        "res_a": res_a,
                    }

                except Exception as exc:
                    log.debug(
                        "[BubbleTea C+D_batch] fwd_attn_only_%d error: %s", i, exc
                    )
                    state["ok"] = False

            _fwd_attn_only.__name__ = "_fwd_layer"  # same name for timing log
            return _fwd_attn_only

        def _fwd_finalize():
            """Final forward sub-op: compute hidden_last from layer N-1's x2

            plus its residual, folding in layer N-1's EP-correct MoE output
            if it's ready by now (one bubble after _make_attn_only_op(N-1)
            called ft_moe_set_hidden) -- otherwise falls back to the
            previous approximation (MoE delta omitted).
            """
            import time as _t

            from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
                ft_moe_get_moe_delta,
                ft_moe_get_moe_delta_event,
                ft_moe_get_moe_delta_layer,
            )

            state = trainer._fwd_layer_state
            if state is None:
                return
            if not state["ok"]:
                trainer._fwd_layer_state = None
                trainer._fwd_running.release()
                return
            try:
                x2_last = state["hidden"]
                residual_last = state["residual"]
                if ft_moe_get_moe_delta_layer() == N - 1:
                    delta = ft_moe_get_moe_delta()
                    if delta is not None and delta.shape == x2_last.shape:
                        evt = ft_moe_get_moe_delta_event()
                        if evt is not None:
                            torch.cuda.current_stream().wait_event(evt)
                        x2_last = x2_last + delta
                final = (x2_last + residual_last).detach()
                _bad = (
                    torch.isnan(final).any().item()
                    or torch.isinf(final).any().item()
                    or final.float().abs().max().item() > 1e18
                )
                if _bad:
                    log.warning("[BubbleTea C+D_batch] hidden_last bad — skipping")
                    trainer._fwd_layer_state = None
                    trainer._fwd_running.release()
                    return
                trainer._fwd["hidden_last"] = final
                trainer._fwd["labels"] = state["labels"]
                # Same round stamp as build_fwd_subops' last-layer op (see there).
                trainer._fwd["round"] = trainer._fwd_round
                trainer._fwd_last_done = _t.time()
                trainer._fwd_layer_state = None
                trainer._fwd_running.release()
            except Exception as exc:
                log.debug("[BubbleTea C+D_batch] fwd_finalize error: %s", exc)
                trainer._fwd_layer_state = None
                trainer._fwd_running.release()

        _fwd_finalize.__name__ = "_fwd_layer"  # same name for timing log

        return (
            [_fwd_init_cdbatch]
            + [_make_attn_only_op(i) for i in range(N)]
            + [_fwd_finalize]
        )

    def build_bwd_subops(self) -> list:
        """
        Return 4*N + 2 sub-ops for the bubble scheduler.

        Sub-op 0:        lm_head_grad  — seeds grad_hidden, logs CE loss.
        Sub-ops 1..4*N:  per-layer, layer N-1 down to 0, interleaved as:
                           4*(N-1-i)+1  layer_i_passthrough
                           4*(N-1-i)+2  layer_i_q_lora_grad
                           4*(N-1-i)+3  layer_i_v_lora_grad
                           4*(N-1-i)+4  layer_i_o_lora_grad
        Sub-op 4*N+1:    optimizer_step.
        """
        trainer = self
        N = len(trainer._layers)

        # ── Shared backward state ──────────────────────────────────────────
        # All sub-ops close over this single dict.
        #
        # ok          — False until lm_head sub-op confirms _fwd is ready;
        #               remaining sub-ops are no-ops when False.
        # grad_hidden — [T, H] gradient seed from lm_head_grad, updated in-place
        #               by each passthrough sub-op walking from layer N-1 to 0.
        # positions   — [T] torch.arange(T), recomputed once by lm_head sub-op.
        # loss        — CE scalar set by lm_head sub-op, logged by optimizer sub-op.
        # layers      — int → dict of per-layer intermediates written by each
        #               passthrough sub-op and read by the three LoRA grad sub-ops
        #               for that same layer:
        #                 grad_q       [T, q_sz]  for Q LoRA grads
        #                 grad_v       [T, kv_sz] for V LoRA grads
        #                 sdpa_out     [T, q_sz]  for O LoRA grads (forward activation)
        #                 grad_o_local [T, H]     for O LoRA grads (backward seed)
        bwd_state: dict = {
            "ok": False,
            "grad_hidden": None,
            "positions": None,
            "loss": 0.0,
            "layers": {},
            "_attn": {},  # layer_idx → scratch dict for split attn bwd sub-ops
            "_pending_gh": None,  # deferred grad_hidden: in-flight exchange future
            # + the linear tail ops postponed to consume time
        }

        def _resolve_pending_grad_hidden():
            """Consume a deferred grad_hidden (lmh_grad / attnbwd_{i} async
            exchange). The publisher postponed the LN backward and residual
            add to consume time — both linear in the gradient, so applying
            them after the cross-rank sum is equivalent. Runs at the top of
            every passthrough chunk (the first sub-op that needs
            grad_hidden); a no-op when nothing is pending."""
            pend = bwd_state["_pending_gh"]
            if pend is None:
                return
            bwd_state["_pending_gh"] = None
            g = pend["fut"].result()
            if pend["ln_w"] is not None:
                g = _rms_norm_bwd_x(g, pend["ln_x"], pend["ln_w"])
            if pend["add"] is not None:
                g = g + pend["add"]
            bwd_state["grad_hidden"] = g

        # ── Sub-op 0: lm_head_grad ────────────────────────────────────────────
        def _lm_head_op():
            fwd = trainer._fwd
            if "hidden_last" not in fwd or "labels" not in fwd:
                log.debug(
                    "[BubbleTea] bwd lm_head_op: fwd not ready (rank=%d)",
                    trainer._tp_rank,
                )
                return  # forward not ready — all downstream sub-ops stay no-ops
            log.debug(
                "[BubbleTea] bwd lm_head_op: starting (rank=%d round=%d)",
                trainer._tp_rank,
                fwd.get("round", -1),
            )
            try:
                hidden_last = fwd["hidden_last"]
                labels = fwd["labels"]
                T = hidden_last.shape[0]
                dev = f"cuda:{trainer.device}"

                # Apply the final RMSNorm before the LM head.  hidden_last is
                # the raw post-FFN activation from layer 47; skipping the norm
                # leaves activations unscaled after 48 residual layers and
                # produces logits with extreme magnitude (~60k CE loss).
                # Manual backward (no autograd): _rms_norm_bwd_x avoids holding
                # the Python GIL, which blocked inference shm_broadcast when we
                # tried .backward() from the bwd_stream worker thread.
                if hasattr(trainer._norm, "weight"):
                    # Ensure weight is on the same device/dtype as hidden_last.
                    # EP weight filtering can leave non-expert weights on CPU.
                    w_fn = trainer._norm.weight.detach().to(
                        device=hidden_last.device, dtype=hidden_last.dtype
                    )
                    hidden_normed = _rms_norm(hidden_last, w_fn)
                    grad_local, loss_scalar, grad_fut = trainer._lm_head_grad(
                        hidden_normed,
                        labels,
                        round_id=fwd.get("round"),
                        async_grad=True,
                    )
                    ln_x, ln_w = hidden_last, w_fn
                else:
                    # Unit-test mock norm has no weight — fall through unchanged.
                    grad_local, loss_scalar, grad_fut = trainer._lm_head_grad(
                        hidden_last, labels, round_id=fwd.get("round"), async_grad=True
                    )
                    ln_x = ln_w = None
                # _lm_head_grad returns None when this rank's vocab shard has
                # no valid target tokens (common on rank 1 for English text).
                # Skip the backward rather than propagating zero gradients.
                if grad_local is None:
                    return

                bwd_state["ok"] = True
                if grad_fut is not None:
                    # Vocab-parallel grad partial published async; layer N-1's
                    # first passthrough chunk resolves it. The final-norm
                    # backward is applied at consume time (linear in grad).
                    bwd_state["_pending_gh"] = {
                        "fut": grad_fut,
                        "ln_x": ln_x,
                        "ln_w": ln_w,
                        "add": None,
                    }
                    bwd_state["grad_hidden"] = None
                else:
                    g = grad_local
                    if ln_w is not None:
                        g = _rms_norm_bwd_x(g, ln_x, ln_w)
                    bwd_state["grad_hidden"] = g
                bwd_state["positions"] = torch.arange(T, device=dev, dtype=torch.long)
                bwd_state["loss"] = loss_scalar
            except Exception as exc:
                import traceback as _tb

                log.warning(
                    "[BubbleTea LoRA] bwd_lm_head error: %s\n%s", exc, _tb.format_exc()
                )
                bwd_state["ok"] = False

        # ── Per-layer passthrough sub-op factory ──────────────────────────────
        def _make_passthrough_chunk_ops(i: int):
            """Return a list of sub-ops for layer i's passthrough backward.

            Instead of one large sub-op that processes all N_local experts (and
            reads all their weights in one go), emit n_chunks smaller sub-ops
            that each handle N_local/n_chunks experts.  Each chunk reads only a
            fraction of the expert weight matrices, completing well within one
            EP bubble and preventing bwd_stream queue buildup.

            The last chunk also runs the post-attn / attention / input-LN
            backward steps that require the fully accumulated grad_x2.
            """
            layer = trainer._layers[i]
            n_ep = trainer._ep_num_local_experts
            n_c = max(1, trainer._bwd_passthrough_chunks)
            # Expert indices are split as evenly as possible.
            chunk_borders = [round(n_ep * k / n_c) for k in range(n_c + 1)]
            chunks = [
                (chunk_borders[k], chunk_borders[k + 1])
                for k in range(n_c)
                if chunk_borders[k] < chunk_borders[k + 1]
            ]
            # Always emit at least one sub-op so the attention/LN backward
            # runs even when there are no local MoE experts (e.g. unit tests
            # with dense-only mock models, or EP rank with 0 active experts).
            if not chunks:
                chunks = [(0, n_ep)]

            ops = []
            for chunk_idx, (e_start, e_end) in enumerate(chunks):
                is_last = chunk_idx == len(chunks) - 1

                def _moe_chunk(e_s=e_start, e_e=e_end, last=is_last):
                    if not bwd_state["ok"]:
                        return
                    try:
                        # Consume any deferred grad_hidden (lmh_grad for layer
                        # N-1, attnbwd_{i+1} for the rest) — its exchange ran
                        # while the previous layer's LoRA-grad sub-ops fired.
                        _resolve_pending_grad_hidden()

                        fwd_i = trainer._fwd["layers"][i]
                        x2 = fwd_i["x2"]
                        saved_gu = fwd_i.get("gate_up")
                        grad_hidden = bwd_state["grad_hidden"]

                        # Retrieve or initialise the partial grad_x2 accumulator.
                        partial = bwd_state.get("_grad_x2_partial")
                        if partial is None or partial.shape != x2.shape:
                            partial = torch.zeros_like(x2)
                            bwd_state["_grad_x2_partial"] = partial

                        # Accumulate this expert chunk's contribution.
                        trainer._moe_backward_passthrough(
                            x2,
                            grad_hidden,
                            layer.mlp,
                            saved_gate_up=saved_gu,
                            e_start=e_s,
                            e_end=e_e,
                            grad_x2_acc=partial,
                        )

                        if not last:
                            return  # more chunks to go; skip LN / attn backward

                        # ── All MoE chunks done — publish grad_x2 ──
                        # Attn backward is split into 4 separate bubble sub-ops
                        # (_attn_qkv_recompute, _attn_o_proj_bwd, _attn_sdpa_bwd,
                        # _attn_qkv_bwd) that follow this op in the sub-op list.
                        grad_x2 = partial
                        bwd_state["_grad_x2_partial"] = None  # free accumulator

                        # Shared expert backward (Qwen1.5-MoE / Qwen2-MoE only).
                        # The shared expert output is added to `output` in the
                        # forward, so its grad_x2 contribution is additive here.
                        se = getattr(layer.mlp, "shared_expert", None)
                        if se is not None:
                            saved_se_gu = (fwd_i.get("gate_up") or {}).get("shared")
                            gu_w = se.gate_up_proj.weight.detach()  # [2*d_local, H]
                            d_w = se.down_proj.weight.detach()  # [H, d_local]
                            if saved_se_gu is not None:
                                gate_up = saved_se_gu  # [T, 2*d_local]
                            else:
                                gate_up = x2 @ gu_w.T  # recompute
                            d_local = gate_up.shape[-1] // 2
                            gate = gate_up[:, :d_local]  # [T, d_local]
                            up = gate_up[:, d_local:]
                            se_mlp_out = F.linear(
                                F.silu(gate) * up, d_w
                            )  # [T, H] partial

                            if getattr(se, "expert_gate", None) is not None:
                                sg_w = se.expert_gate.weight.detach()  # [1, H]
                                gate_scalar = x2 @ sg_w.T  # [T, 1]
                                se_gate = torch.sigmoid(gate_scalar)  # [T, 1]
                                # grad through gating scalar applied to se_mlp_out
                                grad_mlp = grad_hidden * se_gate  # [T, H]
                                sig_d = se_gate * (1.0 - se_gate)
                                grad_sg = (grad_hidden * se_mlp_out).sum(
                                    -1, keepdim=True
                                ) * sig_d  # [T,1]
                                grad_x2 += grad_sg @ sg_w  # [T, H]
                            else:
                                grad_mlp = grad_hidden

                            # Backward through down_proj: grad_act = grad_mlp @ d_w
                            grad_act = (
                                grad_mlp @ d_w
                            )  # [T, H]@[H, d_local] = [T, d_local]
                            # Backward through SwiGLU
                            sig_g = torch.sigmoid(gate)
                            silu_deriv = sig_g * (1.0 + gate * (1.0 - sig_g))
                            grad_gate = grad_act * silu_deriv * up
                            grad_up = grad_act * F.silu(gate)
                            grad_gu = torch.cat(
                                [grad_gate, grad_up], dim=-1
                            )  # [T, 2*d_local]
                            # Backward through gate_up_proj: grad_x2 += grad_gu @ gu_w
                            grad_x2 += grad_gu @ gu_w  # [T, H]

                        # Item 9: grad_x2 only carries this rank's local
                        # experts' contributions (EP-sharded). Both ranks hold
                        # the full token batch, so the full gradient is the
                        # plain sum of the per-rank partials — no gradient
                        # all-to-all needed. Published async here; consumed by
                        # _attn_o_proj_bwd, with the gradient-independent
                        # _attn_qkv_recompute scheduled in between to hide the
                        # exchange. Without this exchange, grad_o_local (and
                        # everything propagated to lower layers) is partial,
                        # breaking the items-3-5 summation math.
                        if (
                            trainer._tp_correct
                            and trainer._ep_size > 1
                            and hasattr(layer.mlp, "gate")
                        ):
                            grad_x2_fut = trainer._tp_correct_exchange_sum_async(
                                grad_x2.contiguous(),
                                f"moebwd_{i}",
                                trainer._fwd.get("round", trainer._fwd_round),
                            )
                        else:
                            grad_x2_fut = _TpCorrectFuture.preset(grad_x2)

                        bwd_state["_attn"][i] = {"grad_x2_fut": grad_x2_fut}
                    except Exception as exc:
                        import traceback as _tb

                        log.warning(
                            "[BubbleTea LoRA] bwd_passthrough_%d"
                            " chunk %d-%d error: %s\n%s",
                            i,
                            e_s,
                            e_e,
                            exc,
                            _tb.format_exc(),
                        )
                        bwd_state["ok"] = False

                _moe_chunk.__name__ = "_passthrough"
                ops.append(_moe_chunk)
            return ops

        # ── Attention backward sub-op factory (4 ops per layer) ─────────────
        def _make_attn_bwd_ops(i: int):
            """Split _attn_backward_passthrough into 4 bubble-sized sub-ops.

            Each fits within the ~0.28ms p50 EP allreduce bubble:
              _attn_qkv_recompute ~0.18ms  — QKV + QK-norm + RoPE recompute
              _attn_o_proj_bwd    ~0.07ms  — grad_o_local + O projection gradient
              _attn_sdpa_bwd      ~0.20ms  — SDPA autograd + RoPE autograd
              _attn_qkv_bwd       ~0.14ms  — QK-norm bwd + QKV proj bwd + input LN
            Intermediates flow through bwd_state["_attn"][i].

            _attn_qkv_recompute is gradient-independent, so it runs FIRST:
            the moebwd_{i} exchange published by the last passthrough chunk
            completes behind it, and _attn_o_proj_bwd consumes the future.
            """
            layer = trainer._layers[i]
            layer_attn = layer.self_attn

            def _attn_o_proj_bwd():
                if not bwd_state["ok"]:
                    return
                try:
                    scaling = _LORA_ALPHA / 16
                    lora_d = trainer._lora.get(i, {})
                    scratch = bwd_state["_attn"][i]

                    # Consume the moebwd_{i} future, then finish what used to
                    # be the last chunk's tail: post-attn LN backward +
                    # residual to get grad_o_local.
                    grad_x2 = scratch.pop("grad_x2_fut").result()
                    fwd_i = trainer._fwd["layers"][i]
                    res_b = (fwd_i["attn_out"] + fwd_i["res_a"]).detach()
                    w_post = layer.post_attention_layernorm.weight.detach().to(
                        device=res_b.device, dtype=res_b.dtype
                    )
                    grad_res_b = _rms_norm_bwd_x(grad_x2, res_b, w_post)
                    grad_o_local = grad_res_b + bwd_state["grad_hidden"]
                    scratch["grad_o_local"] = grad_o_local

                    o_w = layer_attn.o_proj.weight.detach()
                    prev_o = getattr(trainer, f"_prev_delta_{i}_o_proj", None)
                    if prev_o is not None:
                        o_w = (o_w.float() - prev_o.float()).to(o_w.dtype)
                    A_o = lora_d.get("o_proj.lora_A")
                    B_o = lora_d.get("o_proj.lora_B")

                    grad_sdpa_out = grad_o_local.to(o_w.dtype) @ o_w
                    if A_o is not None and B_o is not None:
                        grad_z = grad_o_local.to(B_o.dtype) @ B_o * scaling
                        grad_sdpa_out = grad_sdpa_out + grad_z.to(A_o.dtype) @ A_o

                    scratch["grad_sdpa_out"] = grad_sdpa_out
                except Exception as exc:
                    log.warning("[BubbleTea LoRA] attn_o_proj_bwd_%d error: %s", i, exc)
                    bwd_state["ok"] = False

            _attn_o_proj_bwd.__name__ = "_attn_o_proj_bwd"

            def _attn_qkv_recompute():
                if not bwd_state["ok"]:
                    return
                try:
                    scaling = _LORA_ALPHA / 16
                    lora_d = trainer._lora.get(i, {})
                    x_norm = trainer._fwd["layers"][i]["x_norm"]
                    positions = bwd_state["positions"]
                    T = x_norm.shape[0]
                    nh = trainer._n_heads_local
                    nkv = trainer._n_kv_local
                    hd = _HEAD_DIM
                    q_sz = nh * hd
                    kv_sz = nkv * hd

                    qkv_w = layer_attn.qkv_proj.weight.detach()
                    q_base = F.linear(x_norm.detach(), qkv_w[:q_sz])
                    k_base = F.linear(x_norm.detach(), qkv_w[q_sz : q_sz + kv_sz])
                    v_base = F.linear(x_norm.detach(), qkv_w[q_sz + kv_sz :])

                    xf = x_norm.detach().float()
                    prev_q = getattr(trainer, f"_prev_delta_{i}_q_proj", None)
                    if prev_q is not None:
                        q_base = (q_base.float() - F.linear(xf, prev_q.float())).to(
                            q_base.dtype
                        )
                    prev_k = getattr(trainer, f"_prev_delta_{i}_k_proj", None)
                    if prev_k is not None:
                        k_base = (k_base.float() - F.linear(xf, prev_k.float())).to(
                            k_base.dtype
                        )
                    prev_v = getattr(trainer, f"_prev_delta_{i}_v_proj", None)
                    if prev_v is not None:
                        v_base = (v_base.float() - F.linear(xf, prev_v.float())).to(
                            v_base.dtype
                        )

                    A_q = lora_d.get("q_proj.lora_A")
                    B_q = lora_d.get("q_proj.lora_B")
                    A_k = lora_d.get("k_proj.lora_A")
                    B_k = lora_d.get("k_proj.lora_B")
                    A_v = lora_d.get("v_proj.lora_A")
                    B_v = lora_d.get("v_proj.lora_B")

                    xd = x_norm.detach()
                    q = q_base + (
                        (xd @ A_q.T.to(xd.dtype)) @ B_q.T.to(xd.dtype) * scaling
                        if A_q is not None and B_q is not None
                        else 0
                    )
                    k = k_base + (
                        (xd @ A_k.T.to(xd.dtype)) @ B_k.T.to(xd.dtype) * scaling
                        if A_k is not None and B_k is not None
                        else 0
                    )
                    v = v_base + (
                        (xd @ A_v.T.to(xd.dtype)) @ B_v.T.to(xd.dtype) * scaling
                        if A_v is not None and B_v is not None
                        else 0
                    )

                    if hasattr(layer_attn, "q_norm"):
                        q_normed = layer_attn.q_norm(q.view(T, nh, hd)).view(T, q_sz)
                        k_normed = layer_attn.k_norm(k.view(T, nkv, hd)).view(T, kv_sz)
                    else:
                        q_normed, k_normed = q, k
                    q_rot, k_rot = layer_attn.rotary_emb(positions, q_normed, k_normed)

                    scratch = bwd_state["_attn"][i]
                    scratch.update(
                        {
                            "q": q,
                            "k": k,
                            "v": v,
                            "k_rot": k_rot,
                            "q_normed": q_normed,
                            "k_normed": k_normed,
                            "q_rot": q_rot,
                        }
                    )
                except Exception as exc:
                    log.warning(
                        "[BubbleTea LoRA] attn_qkv_recompute_%d error: %s", i, exc
                    )
                    bwd_state["ok"] = False

            _attn_qkv_recompute.__name__ = "_attn_qkv_recompute"

            def _attn_sdpa_bwd():
                if not bwd_state["ok"]:
                    return
                try:
                    scratch = bwd_state["_attn"][i]
                    fwd_i = trainer._fwd["layers"][i]
                    T = fwd_i["x_norm"].shape[0]
                    nh = trainer._n_heads_local
                    nkv = trainer._n_kv_local
                    hd = _HEAD_DIM
                    q_sz = nh * hd
                    positions = bwd_state["positions"]

                    q_rot = scratch["q_rot"]
                    v = scratch["v"]
                    k_rot = scratch["k_rot"]
                    q_normed = scratch["q_normed"]
                    k_normed = scratch["k_normed"]
                    grad_sdpa_out = scratch["grad_sdpa_out"]

                    with torch.enable_grad():
                        q_leaf = q_rot.detach().requires_grad_(True)
                        k_leaf = k_rot.detach().requires_grad_(True)
                        v_leaf = v.detach().requires_grad_(True)

                        q3 = q_leaf.view(T, nh, hd).transpose(0, 1).unsqueeze(0)
                        k3 = k_leaf.view(T, nkv, hd).transpose(0, 1).unsqueeze(0)
                        v3 = v_leaf.view(T, nkv, hd).transpose(0, 1).unsqueeze(0)
                        if nkv < nh:
                            rep = nh // nkv
                            k3 = k3.repeat_interleave(rep, dim=1)
                            v3 = v3.repeat_interleave(rep, dim=1)

                        sdpa = F.scaled_dot_product_attention(
                            q3, k3, v3, is_causal=True
                        )
                        sdpa = sdpa.squeeze(0).transpose(0, 1).reshape(T, q_sz)
                        sdpa_out = sdpa.detach()
                        sdpa.backward(grad_sdpa_out.to(sdpa.dtype))

                    grad_q_rot = q_leaf.grad
                    grad_k_rot = k_leaf.grad
                    grad_v = v_leaf.grad

                    # RoPE backward (mini autograd, separate passes for Q and K)
                    with torch.enable_grad():
                        q_normed_leaf = q_normed.detach().requires_grad_(True)
                        q_rot_mini, _ = layer_attn.rotary_emb(
                            positions, q_normed_leaf, k_normed.detach()
                        )
                        q_rot_mini.backward(grad_q_rot.to(q_rot_mini.dtype))
                    grad_q_normed = q_normed_leaf.grad

                    with torch.enable_grad():
                        k_normed_leaf = k_normed.detach().requires_grad_(True)
                        _, k_rot_mini = layer_attn.rotary_emb(
                            positions, q_normed.detach(), k_normed_leaf
                        )
                        k_rot_mini.backward(grad_k_rot.to(k_rot_mini.dtype))
                    grad_k_normed = k_normed_leaf.grad

                    scratch.update(
                        {
                            "grad_v": grad_v,
                            "grad_q_normed": grad_q_normed,
                            "grad_k_normed": grad_k_normed,
                            "sdpa_out": sdpa_out,
                        }
                    )
                except Exception as exc:
                    log.warning("[BubbleTea LoRA] attn_sdpa_bwd_%d error: %s", i, exc)
                    bwd_state["ok"] = False

            _attn_sdpa_bwd.__name__ = "_attn_sdpa_bwd"

            def _attn_qkv_bwd():
                if not bwd_state["ok"]:
                    return
                try:
                    scratch = bwd_state["_attn"][i]
                    fwd_i = trainer._fwd["layers"][i]
                    x_norm = fwd_i["x_norm"]
                    res_a = fwd_i["res_a"]
                    lora_d = trainer._lora.get(i, {})
                    scaling = _LORA_ALPHA / 16
                    T = x_norm.shape[0]
                    nh = trainer._n_heads_local
                    nkv = trainer._n_kv_local
                    hd = _HEAD_DIM
                    q_sz = nh * hd
                    kv_sz = nkv * hd

                    grad_o_local = scratch["grad_o_local"]
                    grad_q_normed = scratch["grad_q_normed"]
                    grad_k_normed = scratch["grad_k_normed"]
                    grad_v = scratch["grad_v"]
                    sdpa_out = scratch["sdpa_out"]
                    q = scratch["q"]
                    k = scratch["k"]

                    A_q = lora_d.get("q_proj.lora_A")
                    B_q = lora_d.get("q_proj.lora_B")
                    A_k = lora_d.get("k_proj.lora_A")
                    B_k = lora_d.get("k_proj.lora_B")
                    A_v = lora_d.get("v_proj.lora_A")
                    B_v = lora_d.get("v_proj.lora_B")
                    qkv_w = layer_attn.qkv_proj.weight.detach()

                    # QK-norm backward for Q (skipped when no q_norm, e.g. Qwen1.5-MoE)
                    q_det = q.detach()
                    if hasattr(layer_attn, "q_norm") and hasattr(
                        layer_attn.q_norm, "weight"
                    ):
                        w_qn = layer_attn.q_norm.weight.detach().to(
                            q_det.device, q_det.dtype
                        )
                        grad_q_pre_norm = _rms_norm_bwd_x(
                            grad_q_normed.reshape(T * nh, hd),
                            q_det.reshape(T * nh, hd),
                            w_qn,
                        ).reshape(T, q_sz)
                    else:
                        grad_q_pre_norm = grad_q_normed

                    # QK-norm backward for K (skipped when no k_norm, e.g. Qwen1.5-MoE)
                    k_det = k.detach()
                    if hasattr(layer_attn, "k_norm") and hasattr(
                        layer_attn.k_norm, "weight"
                    ):
                        w_kn = layer_attn.k_norm.weight.detach().to(
                            k_det.device, k_det.dtype
                        )
                        grad_k_pre_norm = _rms_norm_bwd_x(
                            grad_k_normed.reshape(T * nkv, hd),
                            k_det.reshape(T * nkv, hd),
                            w_kn,
                        ).reshape(T, kv_sz)
                    else:
                        grad_k_pre_norm = grad_k_normed

                    # QKV projection backward
                    dt = qkv_w.dtype
                    grad_x_norm = (grad_q_pre_norm.to(dt) @ qkv_w[:q_sz]).to(
                        x_norm.dtype
                    )
                    grad_x_norm += (
                        grad_k_pre_norm.to(dt) @ qkv_w[q_sz : q_sz + kv_sz]
                    ).to(x_norm.dtype)
                    grad_x_norm += (grad_v.to(dt) @ qkv_w[q_sz + kv_sz :]).to(
                        x_norm.dtype
                    )
                    if A_q is not None and B_q is not None:
                        grad_z_q = grad_q_pre_norm.to(B_q.dtype) @ B_q * scaling
                        grad_x_norm += (grad_z_q @ A_q).to(x_norm.dtype)
                    if A_k is not None and B_k is not None:
                        grad_z_k = grad_k_pre_norm.to(B_k.dtype) @ B_k * scaling
                        grad_x_norm += (grad_z_k @ A_k).to(x_norm.dtype)
                    if A_v is not None and B_v is not None:
                        grad_z_v = grad_v.to(B_v.dtype) @ B_v * scaling
                        grad_x_norm += (grad_z_v @ A_v).to(x_norm.dtype)

                    # grad_x_norm carries this rank's local Q/K/V heads'
                    # contributions (qkv_proj is colwise-sharded). Exchange-sum so
                    # the grad_hidden propagated to layer i-1 is the full
                    # gradient — without this, every layer below the top gets
                    # a partial seed and the items-3-5 grad summation at
                    # optimizer_step no longer reconstructs the true LoRA
                    # grads for those layers. Published async; layer i-1's
                    # first passthrough chunk resolves it — this layer's
                    # three LoRA-grad sub-ops (which don't need grad_hidden)
                    # fire in between and hide the exchange. The input-LN
                    # backward + residual add are applied at consume time
                    # (linear in the gradient, so the order is equivalent).
                    # i == 0 has no consumer — no exchange at all.
                    res_a_det = res_a.detach()
                    w_in = (
                        trainer._layers[i]
                        .input_layernorm.weight.detach()
                        .to(res_a_det.device, res_a_det.dtype)
                    )
                    if i > 0 and trainer._tp_correct and trainer._tp_size > 1:
                        gx_fut = trainer._tp_correct_exchange_sum_async(
                            grad_x_norm.contiguous(),
                            f"attnbwd_{i}",
                            trainer._fwd.get("round", trainer._fwd_round),
                        )
                        bwd_state["_pending_gh"] = {
                            "fut": gx_fut,
                            "ln_x": res_a_det,
                            "ln_w": w_in,
                            "add": grad_o_local,
                        }
                        bwd_state["grad_hidden"] = None  # set at consume time
                    else:
                        grad_res_a = _rms_norm_bwd_x(grad_x_norm, res_a_det, w_in)
                        bwd_state["grad_hidden"] = grad_res_a + grad_o_local

                    # Store results for LoRA grad sub-ops; free attn scratch
                    bwd_state["layers"][i] = {
                        "grad_q": grad_q_pre_norm,
                        "grad_k": grad_k_pre_norm,
                        "grad_v": grad_v,
                        "sdpa_out": sdpa_out,
                        "grad_o_local": grad_o_local,
                    }
                    del bwd_state["_attn"][i]
                except Exception as exc:
                    log.warning("[BubbleTea LoRA] attn_qkv_bwd_%d error: %s", i, exc)
                    bwd_state["ok"] = False

            _attn_qkv_bwd.__name__ = "_attn_qkv_bwd"

            return [
                _attn_qkv_recompute,
                _attn_o_proj_bwd,
                _attn_sdpa_bwd,
                _attn_qkv_bwd,
            ]

        # ── Per-layer Q LoRA grad sub-op factory ─────────────────────────────
        def _make_q_lora_op(i: int):
            scaling = _LORA_ALPHA / 16

            def _q_lora():
                if not bwd_state["ok"]:
                    return
                try:
                    lora_d = trainer._lora.get(i, {})
                    A_q = lora_d.get("q_proj.lora_A")  # [r, H]   — replicated across TP
                    B_q = lora_d.get("q_proj.lora_B")  # [q_sz_local, r] — sharded
                    if A_q is None or B_q is None:
                        return
                    x_norm = trainer._fwd["layers"][i]["x_norm"]  # [T, H]
                    grad_q = bwd_state["layers"][i]["grad_q"]  # [T, q_sz_local]
                    with torch.no_grad():
                        z = x_norm.to(A_q.dtype) @ A_q.T  # [T, r]
                        gz = grad_q.to(B_q.dtype) @ B_q * scaling  # [T, r]
                        dA = (gz.T @ x_norm.to(gz.dtype)).to(A_q.dtype)  # [r, H]
                        dB = (grad_q.to(B_q.dtype).T @ z.to(B_q.dtype) * scaling).to(
                            B_q.dtype
                        )  # [q_sz_local, r]
                        # dA is partial (only rank 1's Q-head slice contributes) but
                        # all-reducing here would deadlock: backward runs only on the
                        # EP-light rank in the bubble scheduler thread.
                    A_q.grad = dA if A_q.grad is None else A_q.grad.add_(dA)
                    B_q.grad = dB if B_q.grad is None else B_q.grad.add_(dB)
                except Exception as exc:
                    log.debug("[BubbleTea LoRA] bwd_q_lora_%d error: %s", i, exc)
                    bwd_state["ok"] = False

            return _q_lora

        # ── Per-layer K LoRA grad sub-op factory ─────────────────────────────
        def _make_k_lora_op(i: int):
            scaling = _LORA_ALPHA / 16

            def _k_lora():
                if not bwd_state["ok"]:
                    return
                try:
                    lora_d = trainer._lora.get(i, {})
                    A_k = lora_d.get("k_proj.lora_A")  # [r, H]   — replicated across TP
                    B_k = lora_d.get("k_proj.lora_B")  # [kv_sz_local, r] — sharded
                    if A_k is None or B_k is None:
                        return
                    x_norm = trainer._fwd["layers"][i]["x_norm"]  # [T, H]
                    grad_k = bwd_state["layers"][i]["grad_k"]  # [T, kv_sz_local]
                    with torch.no_grad():
                        z = x_norm.to(A_k.dtype) @ A_k.T  # [T, r]
                        gz = grad_k.to(B_k.dtype) @ B_k * scaling  # [T, r]
                        dA = (gz.T @ x_norm.to(gz.dtype)).to(A_k.dtype)  # [r, H]
                        dB = (grad_k.to(B_k.dtype).T @ z.to(B_k.dtype) * scaling).to(
                            B_k.dtype
                        )
                    A_k.grad = dA if A_k.grad is None else A_k.grad.add_(dA)
                    B_k.grad = dB if B_k.grad is None else B_k.grad.add_(dB)
                except Exception as exc:
                    log.debug("[BubbleTea LoRA] bwd_k_lora_%d error: %s", i, exc)
                    bwd_state["ok"] = False

            return _k_lora

        # ── Per-layer V LoRA grad sub-op factory ─────────────────────────────
        def _make_v_lora_op(i: int):
            scaling = _LORA_ALPHA / 16

            def _v_lora():
                if not bwd_state["ok"]:
                    return
                try:
                    lora_d = trainer._lora.get(i, {})
                    A_v = lora_d.get("v_proj.lora_A")  # [r, H]
                    B_v = lora_d.get("v_proj.lora_B")  # [kv_sz, r]
                    if A_v is None or B_v is None:
                        return
                    x_norm = trainer._fwd["layers"][i]["x_norm"]  # [T, H]
                    grad_v = bwd_state["layers"][i]["grad_v"]  # [T, kv_sz]
                    with torch.no_grad():
                        z = x_norm.to(A_v.dtype) @ A_v.T  # [T, r]
                        gz = grad_v.to(B_v.dtype) @ B_v * scaling  # [T, r]
                        dA = (gz.T @ x_norm.to(gz.dtype)).to(A_v.dtype)  # [r, H]
                        dB = (grad_v.to(B_v.dtype).T @ z.to(B_v.dtype) * scaling).to(
                            B_v.dtype
                        )  # [kv_sz_local, r]
                        # dA partial — same asymmetric-backward constraint as Q.
                    A_v.grad = dA if A_v.grad is None else A_v.grad.add_(dA)
                    B_v.grad = dB if B_v.grad is None else B_v.grad.add_(dB)
                except Exception as exc:
                    log.debug("[BubbleTea LoRA] bwd_v_lora_%d error: %s", i, exc)
                    bwd_state["ok"] = False

            return _v_lora

        # ── Per-layer O LoRA grad sub-op factory ─────────────────────────────
        def _make_o_lora_op(i: int):
            scaling = _LORA_ALPHA / 16

            def _o_lora():
                if not bwd_state["ok"]:
                    return
                try:
                    lora_d = trainer._lora.get(i, {})
                    A_o = lora_d.get("o_proj.lora_A")  # [r, q_sz]
                    B_o = lora_d.get("o_proj.lora_B")  # [H, r]
                    if A_o is None or B_o is None:
                        return
                    layer_bwd = bwd_state["layers"][i]
                    sdpa_out = layer_bwd["sdpa_out"]  # [T, q_sz]
                    grad_o_local = layer_bwd["grad_o_local"]  # [T, H]
                    with torch.no_grad():
                        z_o = sdpa_out.to(A_o.dtype) @ A_o.T  # [T, r]
                        gz_o = grad_o_local.to(B_o.dtype) @ B_o * scaling  # [T, r]
                        dA = (gz_o.T @ sdpa_out.to(gz_o.dtype)).to(
                            A_o.dtype
                        )  # [r, q_sz_local]
                        dB = (
                            grad_o_local.to(B_o.dtype).T @ z_o.to(B_o.dtype) * scaling
                        ).to(B_o.dtype)  # [H, r]
                        # dB partial — same asymmetric-backward constraint.
                    A_o.grad = dA if A_o.grad is None else A_o.grad.add_(dA)
                    B_o.grad = dB if B_o.grad is None else B_o.grad.add_(dB)
                    # On rounds that end an accumulation cycle, stream this
                    # layer's replicated-LoRA grads out NOW (one small async
                    # exchange per layer) instead of one ~19 MB pack at
                    # optimizer time — by the time the sweep reaches layer 0,
                    # the upper layers' exchanges have completed behind it.
                    if (
                        trainer._tp_correct
                        and trainer._tp_size > 1
                        and (trainer._step + 1) % trainer.accum_steps == 0
                    ):
                        trainer._publish_layer_grads(i)
                    # Free this layer's backward intermediates now that all three
                    # LoRA grad ops for layer i have run.
                    del bwd_state["layers"][i]
                except Exception as exc:
                    log.debug("[BubbleTea LoRA] bwd_o_lora_%d error: %s", i, exc)
                    bwd_state["ok"] = False

            return _o_lora

        # ── Optimizer sub-op ──────────────────────────────────────────────────
        def _optimizer_op():
            try:
                if not bwd_state["ok"]:
                    log.debug(
                        "[BubbleTea] optimizer_op skipped: ok=False (rank=%d step=%d)",
                        trainer._tp_rank,
                        trainer._step,
                    )
                    return
                log.debug(
                    "[BubbleTea] optimizer_op: step=%d accum=%d/%d (rank=%d)",
                    trainer._step,
                    (trainer._step % trainer.accum_steps) + 1,
                    trainer.accum_steps,
                    trainer._tp_rank,
                )
                trainer.total_loss += bwd_state["loss"]
                trainer._step += 1
                if trainer._step % trainer.accum_steps == 0:
                    trainer.optimizer_step()
            except Exception as exc:
                import warnings as _w

                _w.warn(f"[BubbleTea LoRA] bwd_optimizer error: {exc}", stacklevel=1)
                log.debug("[BubbleTea LoRA] bwd_optimizer error: %s", exc)
            finally:
                trainer._fwd.clear()
                bwd_state["ok"] = False
                bwd_state["grad_hidden"] = None
                bwd_state["_pending_gh"] = None
                bwd_state["positions"] = None
                bwd_state["loss"] = 0.0
                bwd_state["layers"].clear()
                # Drop streamed grad futures left by a round that failed
                # before its optimizer step consumed them (a successful step
                # clears them itself in _all_reduce_replicated_grads).
                stale = getattr(trainer, "_pending_grad_futs", None)
                if stale:
                    stale.clear()

        # ── Assembly: (N_chunks+4+4)*N + 2 sub-ops in strict backward order ─────
        # Per layer: N_chunks passthrough + 4 attn bwd + 4 LoRA (Q/K/V/O) sub-ops
        return (
            [_lm_head_op]
            + [
                op
                for i in reversed(range(N))
                for op in (
                    *_make_passthrough_chunk_ops(i),
                    *_make_attn_bwd_ops(i),
                    _make_q_lora_op(i),
                    _make_k_lora_op(i),
                    _make_v_lora_op(i),
                    _make_o_lora_op(i),
                )
            ]
            + [_optimizer_op]
        )

    # ── Status for reporting ──────────────────────────────────────────────

    def training_stats(self) -> dict:
        """Return stats compatible with the compare_benchmark.py JSON schema."""
        return {
            "training_steps": self.completed_steps,
            "peft_samples_s": None,  # filled in by compare_benchmark after run
            "note": f"real LoRA (q/k/v/o, rank=16, alpaca-cleaned, t_ft={self.t_ft})",
        }
