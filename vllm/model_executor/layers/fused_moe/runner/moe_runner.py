# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

# ── Bubble profiling ──────────────────────────────────────────────────────────
def _bubble_profile_enabled() -> bool:
    return os.environ.get("VLLM_BUBBLE_PROFILE", "0") == "1"

# ── Pause-decode stall timing ─────────────────────────────────────────────────
# When VLLM_FT_TIMING=1, CUDA events bracket the wait_event() call at every
# prefill→decode transition so we can measure how long the main stream actually
# stalled waiting for bwd_stream to drain.  Events are collected at
# disarm_bubble_scheduler() time, after the backward pass completes.
_FT_TIMING: bool = os.environ.get("VLLM_FT_TIMING") == "1"
_TIMING_LOG: str = "/tmp/bt_timing.log"
_pause_timing_pairs: "list[tuple[torch.cuda.Event, torch.cuda.Event]]" = []


def _flush_pause_timings() -> None:
    """Collect pause_decode stall durations and append to the timing log.

    Called from disarm_bubble_scheduler() after the backward pass fully
    completes, so both CUDA events have been retired on the GPU.
    """
    torch.cuda.synchronize()   # ensure both events are done before elapsed_time
    ts = time.time()
    lines = []
    while _pause_timing_pairs:
        e0, e1 = _pause_timing_pairs.pop(0)
        try:
            ms = e0.elapsed_time(e1)
            lines.append(f"{ts:.3f},pause_decode,,pause_decode,{ms:.4f}\n")
        except Exception:
            pass
    if lines:
        try:
            with open(_TIMING_LOG, "a") as f:
                f.writelines(lines)
        except Exception:
            pass

# ── Bubble scheduler ──────────────────────────────────────────────────────────
import queue as _queue

_active_scheduler = None         # VllmBubbleScheduler | None
_sched_min_tokens: int = 512     # skip decode-scale batches
_sched_fills_per_trigger: int = 1  # fill_one() calls per MoE layer
_sched_trigger_rank: int | None = None  # only this rank fires; None = all ranks
# Tracks whether the scheduler was paused at the last prefill→decode boundary.
# Used to fire pause_decode() / resume_prefill() exactly once per transition.
_bwd_sched_in_decode: bool = False

# Job queue: submit_backward_job() pushes here; the first qualifying prefill
# dequeues and arms the job automatically.  Enables arming from any thread
# inside the worker process (e.g. a background SFT training thread).
_sched_job_queue: _queue.Queue = _queue.Queue()

# ── FT forward placement experiment ──────────────────────────────────────────
# VLLM_FT_FWD_MODE controls where the FT forward pass runs relative to
# inference:
#   off     — disabled (default)
#   bubble  — sub-ops dispatched into TP all_reduce bubbles (same mechanism as
#              backward bubble scheduler); measures TTFT overhead of forward
#   decode  — sub-ops dispatched on a low-priority stream during decode steps;
#              measures TPOT overhead of forward
#   prefill — sub-ops run synchronously on the main stream during prefill;
#              worst-case baseline: full FT forward latency added to TTFT
_fwd_mode: str = os.environ.get("VLLM_FT_FWD_MODE", "off")
_fwd_active_scheduler = None    # VllmBubbleScheduler | None (bubble / decode)
_fwd_job_queue: _queue.Queue = _queue.Queue()
_fwd_sched_min_tokens: int = 512
_fwd_fills_per_trigger: int = 1
_fwd_trigger_rank: int | None = None


class _FwdPrefillState:
    __slots__ = ("sub_ops", "cursor", "fills_per_trigger", "post_complete_fn")

    def __init__(self, sub_ops: list, fills_per_trigger: int = 9,
                 post_complete_fn=None) -> None:
        self.sub_ops = sub_ops
        self.cursor = 0
        self.fills_per_trigger = fills_per_trigger
        self.post_complete_fn = post_complete_fn


_fwd_prefill_state: "_FwdPrefillState | None" = None  # prefill-mode only

# ── Combined FT placement modes ───────────────────────────────────────────────
# VLLM_FT_COMBINED_MODE runs a full fwd+bwd training step pair with coordinated
# placement.  Overrides VLLM_FT_FWD_MODE and VLLM_BUBBLE_SCHED_DEMO.
#   B+D — fwd sync on main stream during prefill, bwd in TP all_reduce bubbles
#   C+D — fwd async on secondary stream during decode, bwd in TP bubbles
#   D+D — fwd+bwd concatenated into one bubble-scheduler job (sequential by
#          queue order; fwd portion drains first, then bwd)
# ── Real LoRA trainer (BubbleTea C+D with actual gradient updates) ───────────
# Set VLLM_FT_LORA_PATH to the adapter_model.safetensors directory to enable.
# When set, _submit_combined_fwd_job / _submit_combined_bwd_job use real ops
# from BubbleTeaLoRATrainer instead of the synthetic random-tensor sub-ops.
_bt_trainer = None   # type: "bt_lora_trainer.BubbleTeaLoRATrainer | None"

# Thread-local flag set by training sub-ops while they are executing.
# When a training sub-op calls layer.mlp(), that call re-enters this
# runner.  The flag prevents the reentrant call from triggering C+D
# scheduling code (fill_one, arm_scheduler, submit_job), which would
# cause all remaining sub-ops to chain-fire in rapid succession instead
# of waiting for individual EP bubbles.
import threading as _threading
_bt_subop_running = _threading.local()

def register_bt_trainer(trainer) -> None:
    """Called from qwen3_moe.py after model load when VLLM_FT_LORA_PATH is set."""
    global _bt_trainer
    _bt_trainer = trainer

_combined_mode: str = os.environ.get("VLLM_FT_COMBINED_MODE", "off")
if _combined_mode == "B+D":
    _fwd_mode = "prefill"
elif _combined_mode == "C+D":
    _fwd_mode = "decode"
elif _combined_mode == "D+D":
    _fwd_mode = "off"
# C+D_batch: FT attention on secondary stream (same as C+D), but MoE is
# batched into the main inference forward instead of running on a separate stream.
# _fwd_mode stays "decode" so attn sub-ops still dispatch on the secondary stream.
elif _combined_mode == "C+D_batch":
    _fwd_mode = "decode"

# ── FT MoE batch state ────────────────────────────────────────────────────────
# When C+D_batch is active, the MoE forward for FT tokens is injected directly
# into each Qwen3MoeSparseMoeBlock.forward() call instead of running on a
# secondary stream.  A single rolling hidden-state tensor flows through all
# 48 MoE layers alongside the inference tokens.
_FT_MOE_N_LAYERS: int = int(os.environ.get("VLLM_FT_MOE_N_LAYERS", "48"))
_ft_moe_state: dict = {
    "active":       False,
    "hidden":       None,   # torch.Tensor [T_ft, H] — current layer's FT hidden
    "hidden_event": None,   # CUDA Event: fires when "hidden" was last written on bwd_stream
                            # (C+D_batch real mode only; None = hidden is ready immediately)
    "layer_count":  0,      # number of MoE layers processed so far this pass
    "t_ft":         int(os.environ.get("VLLM_FT_COMBINED_T_FT", "128")),
    "pass_count":   0,      # completed FT passes (for throughput tracking)
    "real_training": False, # True when C+D_batch is using real trainer fwd sub-ops
    "moe_delta":       None, # torch.Tensor [T_ft, H] — most recently completed
                              # layer's EP-correct MoE output (combined_out[n_inf:]
                              # from ft_moe_advance), for _fwd_attn_only to fold
                              # into "hidden" (real-training mode only).
    "moe_delta_layer": -1,   # 0-indexed layer that produced "moe_delta"
    "moe_delta_event": None, # CUDA Event recorded on the main stream right after
                              # "moe_delta" was computed
}


def ft_moe_arm(hidden_dim: int, device: int) -> None:
    """Arm a new batched FT MoE pass.

    In real-training C+D_batch mode the initial hidden state is written by the
    _fwd_init_cdbatch sub-op via ft_moe_set_hidden(); we start with zeros here
    so the first layer's inline MoE has a safe (if imprecise) input until the
    sub-op fires.  In synthetic/demo mode we use randn as before.
    """
    import torch
    if _ft_moe_state["real_training"]:
        _ft_moe_state["hidden"] = torch.zeros(
            _ft_moe_state["t_ft"], hidden_dim,
            device=device, dtype=torch.bfloat16
        )
        _ft_moe_state["hidden_event"] = None
    else:
        _ft_moe_state["hidden"] = torch.randn(
            _ft_moe_state["t_ft"], hidden_dim,
            device=device, dtype=torch.bfloat16
        )
        _ft_moe_state["hidden_event"] = None
    _ft_moe_state["layer_count"] = 0
    _ft_moe_state["active"] = True


def ft_moe_set_hidden(hidden: "torch.Tensor", event: "torch.cuda.Event | None") -> None:
    """Update the FT hidden state from a bwd_stream sub-op.

    Called by _fwd_init_cdbatch and _fwd_attn_only_layer sub-ops after they
    compute the real training hidden state (embedding output or post-attention
    x2).  The CUDA event, recorded on bwd_stream, lets the main stream gate the
    next inline MoE call on this sub-op completing — keeping wait times near zero
    since attention (128 tokens) finishes well within the inter-layer gap.
    """
    _ft_moe_state["hidden"] = hidden.detach()
    _ft_moe_state["hidden_event"] = event


def ft_moe_get_hidden() -> "torch.Tensor | None":
    """Return current FT hidden state (or None if not active)."""
    if not _ft_moe_state["active"]:
        return None
    return _ft_moe_state["hidden"]


def ft_moe_get_hidden_event() -> "torch.cuda.Event | None":
    """Return the CUDA event gating the current FT hidden state, or None."""
    return _ft_moe_state.get("hidden_event")


def ft_moe_get_moe_delta() -> "torch.Tensor | None":
    """Return the most recently completed layer's EP-correct MoE output.

    This is combined_out[n_inf:] from Qwen3MoeSparseMoeBlock.forward() —
    the FT tokens' routed-expert output, computed via the same EP
    dispatch/combine all-to-all already used for inference tokens. None if
    no layer has completed yet (or not in real-training mode).
    """
    return _ft_moe_state.get("moe_delta")


def ft_moe_get_moe_delta_layer() -> int:
    """0-indexed layer that produced ft_moe_get_moe_delta(), or -1 if none."""
    return _ft_moe_state.get("moe_delta_layer", -1)


def ft_moe_get_moe_delta_event() -> "torch.cuda.Event | None":
    """CUDA event (recorded on the main stream) gating ft_moe_get_moe_delta()."""
    return _ft_moe_state.get("moe_delta_event")


def ft_moe_reset_moe_delta() -> None:
    """Clear the stashed MoE delta at the start of a new forward round.

    Called by _fwd_init_cdbatch before re-arming layer 0's hidden state.
    Without this, a layer's _fwd_attn_only / _fwd_finalize sub-op that runs
    before the main stream has produced *this* round's delta for the
    previous layer could otherwise see moe_delta_layer still pointing at
    that layer index from the *previous* round (stale tensor for the wrong
    tokens) and incorrectly treat it as ready.
    """
    _ft_moe_state["moe_delta"] = None
    _ft_moe_state["moe_delta_layer"] = -1
    _ft_moe_state["moe_delta_event"] = None


def ft_moe_advance(ft_out: "torch.Tensor") -> None:
    """Advance FT state after a MoE layer. Re-arms a new pass when complete."""
    import torch
    _ft_moe_state["layer_count"] += 1

    if _ft_moe_state["real_training"]:
        # Stash this layer's EP-correct MoE output -- computed via the same
        # all-to-all dispatch/combine vLLM already runs for inference tokens
        # (see Qwen3MoeSparseMoeBlock.forward()) -- so the next layer's
        # _fwd_attn_only sub-op can fold it into "hidden" instead of using x2
        # (the pre-MoE residual) as an approximation.
        evt = torch.cuda.Event()
        evt.record()
        _ft_moe_state["moe_delta"] = ft_out.detach()
        _ft_moe_state["moe_delta_layer"] = _ft_moe_state["layer_count"] - 1
        _ft_moe_state["moe_delta_event"] = evt

    if _ft_moe_state["layer_count"] >= _FT_MOE_N_LAYERS:
        _ft_moe_state["pass_count"] += 1
        # Log completion (same file as C+D completions)
        try:
            with open(_COMBINED_LOG, "a") as _f:
                import time as _t
                _f.write(f"{_t.time()}\n")
        except Exception:
            pass
        # Re-arm immediately for continuous training
        ft_out_dev = ft_out.device.index
        ft_moe_arm(ft_out.shape[-1], ft_out_dev)
    else:
        # In real-training mode the next hidden is set by the bwd_stream
        # attention sub-op; here we just store the MoE delta as a fallback
        # for synthetic/demo mode.
        if not _ft_moe_state["real_training"]:
            _ft_moe_state["hidden"] = ft_out.detach()
            _ft_moe_state["hidden_event"] = None

_combined_demo_armed: bool = False


# Cache for _ep_is_light_rank: keyed by (ep_rank, n_tok) so we compute once
# per prefill rather than once per layer.  All 48 MoE layers in one prefill
# share the same n_tok, so the result is reused until n_tok changes.
# This avoids repeating topk + GPU-CPU sync (.item()) 48× per prefill.
_ep_light_cache_key: "tuple[int, int] | None" = None
_ep_light_cache_val: bool = True


def _ep_is_light_rank(
    runner: "MoERunner",
    router_logits: "torch.Tensor | None",
    n_tok: int,
) -> bool:
    """True if this EP rank should dispatch training sub-ops this layer.

    Static path (VLLM_FT_COMBINED_TRIGGER_RANK set): dispatch only that rank.

    Dynamic path: dispatch only on the rank with fewer total expert-token
    assignments — the rank whose experts attracted fewer tokens this batch.
    That rank finishes FFN first, arrives at the TP all_reduce barrier first,
    and owns the genuine idle bubble.  The heavy rank has little or no bubble;
    dispatching there causes HBM contention and inflates TTFT.

    Load metric: total top-k expert-token assignments on this rank across all
    tokens.  For top-8 routing, a token that picks 7 experts on rank 1 and 1
    on rank 0 contributes 7× load to rank 1 — much heavier than top-1 would
    suggest.

    Cost: one topk + one GPU→CPU sync (.item()) PER PREFILL, not per layer.
    All 48 MoE layers in a prefill share the same n_tok, so the result is
    cached on the first call and reused for the remaining 47 calls.
    """
    global _ep_light_cache_key, _ep_light_cache_val

    if _COMBINED_TRIGGER_RANK is not None:
        return _sched_get_tp_rank() == _COMBINED_TRIGGER_RANK

    if (router_logits is None or
            not hasattr(runner, 'moe_config') or
            runner.moe_config.ep_size <= 1):
        return True

    ep_rank = runner.moe_config.ep_rank
    cache_key = (ep_rank, n_tok)
    if cache_key == _ep_light_cache_key:
        return _ep_light_cache_val   # same prefill — free cache hit

    # n_tok changed → new prefill.  Recompute (one topk + one .item()).
    n_local = runner.moe_config.num_local_experts
    exp_start = ep_rank * n_local
    top_k = runner.moe_config.experts_per_token

    with torch.no_grad():
        topk_ids = router_logits[:n_tok].topk(top_k, dim=-1).indices  # [T, k]
        local_assignments = int(
            ((topk_ids >= exp_start) & (topk_ids < exp_start + n_local)).sum()
        )

    result = local_assignments * runner.moe_config.ep_size <= n_tok * top_k
    _ep_light_cache_key = cache_key
    _ep_light_cache_val = result
    return result


def arm_bubble_scheduler(
    sched: "VllmBubbleScheduler",
    min_tokens: int = 512,
    fills_per_trigger: int = 1,
    trigger_rank: int | None = None,
) -> None:
    """Arm the bubble scheduler for the next prefill pass.

    Parameters
    ----------
    sched :
        Scheduler to arm.  Each TP rank should arm its own instance.
    min_tokens :
        Minimum token count to trigger fill_one().  Skips decode-scale calls.
    fills_per_trigger :
        fill_one() calls per MoE layer.  At T_in ≥ 8192 the p50 bubble fits
        2–3 sub-ops (~0.4 ms MoE chunk or ~1.1 ms attn bwd each).
    trigger_rank :
        If set, only this TP rank calls fill_one().  Use this to avoid the
        slower rank dispatching backward work that overlaps with its own FFN.
        For Qwen3-30B-A3B, rank 1 has bubbles on ~32/48 layers so
        trigger_rank=1 captures most of the available budget at zero FFN cost.
        None (default) fires on all ranks — higher throughput but more
        overlap with FFN on the non-bubbling rank.
    """
    global _active_scheduler, _sched_min_tokens, _sched_fills_per_trigger
    global _sched_trigger_rank
    _active_scheduler = sched
    _sched_min_tokens = min_tokens
    _sched_fills_per_trigger = fills_per_trigger
    _sched_trigger_rank = trigger_rank


def disarm_bubble_scheduler() -> None:
    """Disarm the bubble scheduler after the backward pass completes."""
    global _active_scheduler, _bwd_sched_in_decode
    _active_scheduler = None
    _bwd_sched_in_decode = False
    if _FT_TIMING and _pause_timing_pairs:
        _flush_pause_timings()


_sched_tp_rank: int | None = None


def _sched_get_tp_rank() -> int:
    global _sched_tp_rank
    if _sched_tp_rank is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            _sched_tp_rank = get_tensor_model_parallel_rank()
        except Exception:
            _sched_tp_rank = 0
    return _sched_tp_rank


def submit_backward_job(
    sub_ops: list,
    fills_per_trigger: int = 1,
    post_complete_fn=None,
    device: int | None = None,
    min_tokens: int = 512,
    trigger_rank: int | None = None,
) -> None:
    """Submit a backward pass job to drain through upcoming prefill bubbles.

    Thread-safe — call from any thread inside the worker process.  The job
    is picked up automatically at the start of the next qualifying prefill.

    Parameters
    ----------
    sub_ops :
        Backward sub-ops for this rank's local weights, in execution order.
        Build with ``backward_subops.build_qwen3_subops(rank, ...)``.
    fills_per_trigger :
        fill_one() calls per MoE layer.  Default 2 works well at T_in ≥ 8192.
    post_complete_fn :
        Called on the worker thread after all sub-ops and stream sync.
        Use for gradient allreduce or optimizer notification.
    device :
        CUDA device for the backward stream.  Defaults to current device.
    min_tokens :
        Minimum batch token count before this job activates.
    """
    _sched_job_queue.put((sub_ops, fills_per_trigger, post_complete_fn, device, min_tokens, trigger_rank))


def arm_fwd_scheduler(
    sched: "VllmBubbleScheduler",
    min_tokens: int = 512,
    fills_per_trigger: int = 1,
    trigger_rank: int | None = None,
) -> None:
    global _fwd_active_scheduler, _fwd_sched_min_tokens
    global _fwd_fills_per_trigger, _fwd_trigger_rank
    _fwd_active_scheduler = sched
    _fwd_sched_min_tokens = min_tokens
    _fwd_fills_per_trigger = fills_per_trigger
    _fwd_trigger_rank = trigger_rank


def disarm_fwd_scheduler() -> None:
    global _fwd_active_scheduler
    _fwd_active_scheduler = None


def _arm_fwd_prefill(sub_ops: list, fills_per_trigger: int,
                     post_complete_fn=None) -> None:
    global _fwd_prefill_state
    _fwd_prefill_state = _FwdPrefillState(sub_ops, fills_per_trigger,
                                          post_complete_fn)


def submit_forward_job(
    sub_ops: list,
    fills_per_trigger: int = 1,
    post_complete_fn=None,
    device: int | None = None,
    min_tokens: int = 512,
    trigger_rank: int | None = None,
) -> None:
    """Submit a forward pass job for the FT placement experiment.

    Behaviour depends on VLLM_FT_FWD_MODE:
      bubble / decode — pushed onto _fwd_job_queue; consumed by the next
                        qualifying MoE layer forward() call.
      prefill         — arms the synchronous prefill cursor immediately;
                        fills_per_trigger sub-ops are run per MoE layer on the
                        main stream (blocking TTFT for each fired op).

    Thread-safe — call from any thread inside the worker process.
    """
    if _fwd_mode == "prefill":
        _arm_fwd_prefill(sub_ops, fills_per_trigger, post_complete_fn)
    else:
        _fwd_job_queue.put((sub_ops, fills_per_trigger, post_complete_fn, device, min_tokens, trigger_rank))
# ─────────────────────────────────────────────────────────────────────────────

# Debug: write env state at import time to confirm what worker sees
try:
    with open("/tmp/vllm_worker_env.txt", "a") as _f:
        _f.write(f"pid={os.getpid()} VLLM_BUBBLE_PROFILE={os.environ.get('VLLM_BUBBLE_PROFILE','NOT_SET')} "
                 f"VLLM_BUBBLE_SCHED_DEMO={os.environ.get('VLLM_BUBBLE_SCHED_DEMO','NOT_SET')} "
                 f"VLLM_FT_FWD_MODE={os.environ.get('VLLM_FT_FWD_MODE','NOT_SET')} "
                 f"VLLM_FT_FWD_DEMO={os.environ.get('VLLM_FT_FWD_DEMO','NOT_SET')} "
                 f"VLLM_FT_SCHED_MODE={os.environ.get('VLLM_FT_SCHED_MODE','NOT_SET')} "
                 f"VLLM_FT_FWD_GREEN_CTX={os.environ.get('VLLM_FT_FWD_GREEN_CTX','NOT_SET')} "
                 f"VLLM_FT_GREEN_CTX_SMS={os.environ.get('VLLM_FT_GREEN_CTX_SMS','NOT_SET')} "
                 f"_fwd_demo_enabled={_fwd_demo_enabled} _fwd_mode={_fwd_mode!r}\n")
except Exception:
    pass

# ── Demo auto-arm (VLLM_BUBBLE_SCHED_DEMO=1) ─────────────────────────────────
# When set, each rank automatically submits a perpetual stream of demo backward
# jobs using real Qwen3-30B-A3B kernel shapes.  Used to measure scheduler
# overhead against a clean baseline without needing a real SFT training loop.
_sched_demo_enabled: bool = os.environ.get("VLLM_BUBBLE_SCHED_DEMO", "0") == "1"
_sched_demo_armed: bool = False   # set after first auto-submit


def _demo_resubmit() -> None:
    """post_complete_fn: immediately re-submit a new demo job when done."""
    if _sched_demo_enabled:
        _submit_demo_job()


def _submit_demo_job() -> None:
    """Build demo Qwen3 backward ops and push to job queue."""
    try:
        import sys
        import pathlib
        # backward_subops.py lives at the vllm repo root
        _root = str(pathlib.Path(__file__).parents[5])
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from backward_subops import build_qwen3_subops
        dev = torch.cuda.current_device()
        # Build outside inference_mode: backward sub-ops call
        # q.detach().requires_grad_(True) at execution time; that fails on
        # inference tensors.  inference_mode(False) temporarily exits the
        # context so tensors are regular (grad-trackable) tensors.
        with torch.inference_mode(False):
            sub_ops = build_qwen3_subops(t_ft=128, n_layers=48, moe_chunk_size=8, device=dev)
        _tr_str = os.environ.get("VLLM_BUBBLE_SCHED_RANK", "")
        _trigger_rank = int(_tr_str) if _tr_str.isdigit() else None
        submit_backward_job(
            sub_ops,
            fills_per_trigger=int(os.environ.get("VLLM_BUBBLE_SCHED_FILLS", "1")),
            trigger_rank=_trigger_rank,
            post_complete_fn=_demo_resubmit,
            device=dev,
            min_tokens=512,
        )
    except Exception as e:
        try:
            with open("/tmp/vllm_worker_env.txt", "a") as _f:
                _f.write(f"  demo_submit_error: {e}\n")
        except Exception:
            pass
# ─────────────────────────────────────────────────────────────────────────────

# ── Demo forward mode (VLLM_FT_FWD_DEMO=1) ───────────────────────────────────
# Auto-submits a perpetual stream of demo forward jobs, mirroring the backward
# demo mode.  Used to measure forward-placement overhead without a real SFT loop.
_fwd_demo_enabled: bool = os.environ.get("VLLM_FT_FWD_DEMO", "0") == "1"
_fwd_demo_armed: bool = False


def _fwd_demo_resubmit() -> None:
    # Log completion timestamp so wall-clock throughput can be measured externally.
    try:
        with open("/tmp/vllm_fwd_completions.log", "a") as _f:
            _f.write(f"{time.time():.6f}\n")
    except Exception:
        pass
    if _fwd_demo_enabled:
        _submit_fwd_demo_job()


def _submit_fwd_demo_job() -> None:
    try:
        import sys
        import pathlib
        _root = str(pathlib.Path(__file__).parents[5])
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from forward_subops import build_qwen3_fwd_subops
        dev = torch.cuda.current_device()
        sub_ops = build_qwen3_fwd_subops(t_ft=128, n_layers=48, moe_chunk_size=8, device=dev)
        _tr_str = os.environ.get("VLLM_FT_FWD_RANK", "")
        _trigger_rank = int(_tr_str) if _tr_str.isdigit() else None
        submit_forward_job(
            sub_ops,
            fills_per_trigger=int(os.environ.get("VLLM_FT_FWD_FILLS", "1")),
            trigger_rank=_trigger_rank,
            post_complete_fn=_fwd_demo_resubmit,
            device=dev,
            min_tokens=512,
        )
    except Exception as e:
        try:
            with open("/tmp/vllm_worker_env.txt", "a") as _f:
                _f.write(f"  fwd_demo_submit_error: {e}\n")
        except Exception:
            pass
# ─────────────────────────────────────────────────────────────────────────────

# ── Combined FT demo (VLLM_FT_COMBINED_MODE) ─────────────────────────────────
# Each training step = one forward pass + one backward pass.
# B+D / C+D: forward and backward run via separate schedulers, chained via
#            post_complete_fn callbacks (fwd_done → submit bwd; bwd_done → log + submit fwd).
# D+D:       forward and backward sub-ops concatenated into a single backward
#            scheduler job; sequential ordering is enforced by queue position.
_COMBINED_LOG = "/tmp/vllm_combined_completions.log"
_COMBINED_FWD_FILLS = int(os.environ.get("VLLM_FT_COMBINED_FWD_FILLS", "9"))
_COMBINED_BWD_FILLS = int(os.environ.get("VLLM_FT_COMBINED_BWD_FILLS", "2"))
_COMBINED_DD_FILLS  = int(os.environ.get("VLLM_FT_COMBINED_DD_FILLS",  "2"))
_COMBINED_T_FT      = int(os.environ.get("VLLM_FT_COMBINED_T_FT",      "128"))
# Only this TP rank dispatches fwd/bwd sub-ops (None = all ranks).
# Set to 1 to use only the light EP rank, which has genuine bubble windows
# with free HBM bandwidth.  Eliminates the heavy-rank HBM contention that
# causes TTFT overhead, at the cost of halving per-cycle gradient coverage.
_tr_env = os.environ.get("VLLM_FT_COMBINED_TRIGGER_RANK", "")
_COMBINED_TRIGGER_RANK: "int | None" = int(_tr_env) if _tr_env.isdigit() else None


def _log_combined_completion() -> None:
    try:
        with open(_COMBINED_LOG, "a") as _f:
            _f.write(f"{time.time():.6f}\n")
    except Exception:
        pass


def _combined_bwd_done() -> None:
    """Backward complete: resubmit forward.

    Completion logging is handled by bt_lora_trainer.optimizer_step() which
    writes to _COMBINED_LOG only when a real optimizer step fired.  Logging
    here would write a timestamp for every bwd scheduler drain, including
    no-op drains when the training forward was skipped.
    """
    if _combined_mode in ("B+D", "C+D", "C+D_batch"):
        _submit_combined_fwd_job()


def _combined_fwd_done() -> None:
    """Forward complete (B+D / C+D): now submit the backward job."""
    _submit_combined_bwd_job()


def _combined_dd_done() -> None:
    """Concatenated D+D job complete: resubmit.

    For real training, completion is logged by bt_lora_trainer.optimizer_step()
    so we only call _log_combined_completion() in synthetic/demo mode.
    """
    if _bt_trainer is None:
        _log_combined_completion()
    if _combined_mode == "D+D":
        _submit_combined_dd_job()


def _submit_combined_fwd_job() -> None:
    """Submit forward sub-ops for B+D or C+D mode.

    If a real LoRA trainer is registered (VLLM_FT_LORA_PATH is set), use real
    training ops.  Otherwise fall back to synthetic random-tensor sub-ops.

    Queue is capped at 1 item: if a fwd job is already pending, skip.
    Without this cap, the pre-data-ready no-op fwd/bwd cycle (which runs at
    inference speed) floods _fwd_job_queue with hundreds of items, causing
    hundreds of VllmBubbleScheduler instances (each with a new CUDA stream)
    to be created, exhausting CUDA stream resources and crashing the worker.
    """
    if not _fwd_job_queue.empty():
        return   # a fwd job is already waiting; don't queue another
    try:
        dev = torch.cuda.current_device()
        if _bt_trainer is not None:
            # ── Real LoRA training forward ────────────────────────────────
            # C+D_batch uses attention-only sub-ops; MoE runs inline in the
            # fused batch injection (Qwen3MoeSparseMoeBlock.forward).
            if _combined_mode == "C+D_batch":
                fwd_ops = _bt_trainer.build_fwd_subops_cdbatch()
            else:
                fwd_ops = _bt_trainer.build_fwd_subops()
        else:
            # ── Synthetic forward (original behaviour) ────────────────────
            import sys, pathlib
            _root = str(pathlib.Path(__file__).parents[5])
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from forward_subops import build_qwen3_fwd_subops
            fwd_ops = build_qwen3_fwd_subops(
                t_ft=_COMBINED_T_FT, n_layers=48, moe_chunk_size=8, device=dev
            )
        submit_forward_job(
            fwd_ops,
            fills_per_trigger=_COMBINED_FWD_FILLS,
            post_complete_fn=_combined_fwd_done,
            device=dev,
            min_tokens=512,
            trigger_rank=_COMBINED_TRIGGER_RANK,
        )
    except Exception as e:
        try:
            with open("/tmp/vllm_worker_env.txt", "a") as _f:
                _f.write(f"  combined_fwd_submit_error: {e}\n")
        except Exception:
            pass


def _submit_combined_bwd_job() -> None:
    """Submit backward sub-ops for B+D or C+D mode.

    If a real LoRA trainer is registered, use real backward + optimizer ops.
    Otherwise fall back to synthetic ops.

    Queue is capped at 1 item: if a bwd job is already pending, skip.
    """
    if not _sched_job_queue.empty():
        return   # a bwd job is already waiting; don't queue another
    try:
        dev = torch.cuda.current_device()
        if _bt_trainer is not None:
            # ── Real LoRA backward + optimizer ───────────────────────────
            bwd_ops = _bt_trainer.build_bwd_subops()
        else:
            # ── Synthetic backward (original behaviour) ───────────────────
            import sys, pathlib
            _root = str(pathlib.Path(__file__).parents[5])
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from backward_subops import build_qwen3_subops
            with torch.inference_mode(False):
                bwd_ops = build_qwen3_subops(
                    t_ft=128, n_layers=48, moe_chunk_size=8, device=dev
                )
        submit_backward_job(
            bwd_ops,
            fills_per_trigger=_COMBINED_BWD_FILLS,
            post_complete_fn=_combined_bwd_done,
            device=dev,
            min_tokens=512,
            trigger_rank=_COMBINED_TRIGGER_RANK,
        )
    except Exception as e:
        try:
            with open("/tmp/vllm_worker_env.txt", "a") as _f:
                _f.write(f"  combined_bwd_submit_error: {e}\n")
        except Exception:
            pass


def _submit_combined_dd_job() -> None:
    """Submit concatenated [fwd_ops + bwd_ops] as a single bubble-scheduler job.

    Both forward and backward sub-ops run exclusively in prefill EP bubbles
    (min_tokens=512 gates out decode-scale calls).  This eliminates all
    training activity during decode, removing the HBM-bandwidth competition
    that causes TPOT / TTFT overhead in C+D mode.
    """
    if not _sched_job_queue.empty():
        return  # a job is already pending; don't double-queue
    try:
        dev = torch.cuda.current_device()
        if _bt_trainer is not None:
            # ── Real LoRA training: forward then backward, all in prefill bubbles
            fwd_ops = _bt_trainer.build_fwd_subops()
            bwd_ops = _bt_trainer.build_bwd_subops()
        else:
            # ── Synthetic ops (demo / benchmark mode) ────────────────────────
            import sys, pathlib
            _root = str(pathlib.Path(__file__).parents[5])
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from forward_subops import build_qwen3_fwd_subops
            from backward_subops import build_qwen3_subops
            fwd_ops = build_qwen3_fwd_subops(
                t_ft=_COMBINED_T_FT, n_layers=48, moe_chunk_size=8, device=dev
            )
            with torch.inference_mode(False):
                bwd_ops = build_qwen3_subops(
                    t_ft=_COMBINED_T_FT, n_layers=48, moe_chunk_size=8, device=dev
                )
        submit_backward_job(
            fwd_ops + bwd_ops,
            fills_per_trigger=_COMBINED_DD_FILLS,
            post_complete_fn=_combined_dd_done,
            device=dev,
            min_tokens=512,
        )
    except Exception as e:
        try:
            with open("/tmp/vllm_worker_env.txt", "a") as _f:
                _f.write(f"  combined_dd_submit_error: {e}\n")
        except Exception:
            pass
# ─────────────────────────────────────────────────────────────────────────────

_BUBBLE_MIN_TOKENS = 100

def _bubble_out_path() -> str:
    try:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else os.getpid()
    except Exception:
        rank = os.getpid()
    return f"/tmp/vllm_bubble_rank{rank}.json"
_bubble_lock = threading.Lock()
_bubble_records: list[dict] = []
_bubble_pending: list[dict] = []   # unresolved event tuples, flushed after prefill sync
_bubble_calls = 0
_bubble_prefill_id = 0
_bubble_entry_t = 0.0   # wall-clock at entry to MoeRunner.forward, set per call
# ─────────────────────────────────────────────────────────────────────────────

from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerName,
    direct_register_custom_op,
)


def get_layer_from_name(layer_name: str) -> torch.nn.Module:
    forward_context: ForwardContext = get_forward_context()
    if not _USE_LAYERNAME and layer_name == "from_forward_context":
        all_moe_layers = forward_context.all_moe_layers
        assert all_moe_layers is not None
        moe_layer_index = forward_context.moe_layer_index
        if moe_layer_index >= len(all_moe_layers):
            raise AssertionError(
                "We expected the number of MOE layers in `all_moe_layers` "
                "to be equal to the number of "
                "{vllm.moe_forward, vllm.moe_forward_shared} calls."
            )
        layer_name = all_moe_layers[moe_layer_index]
        forward_context.moe_layer_index += 1
    return forward_context.no_compile_layers[layer_name]


# On torch >= 2.11, layer_name is a hoisted LayerName opaque object;
# on older versions it remains a plain str.
if TYPE_CHECKING:
    from typing import TypeAlias

    _layer_name_type: TypeAlias = str | LayerName
else:
    _layer_name_type = LayerName if _USE_LAYERNAME else str


@torch.compiler.assume_constant_result
def _resolve_layer_name(layer_name: str | LayerName) -> str:
    from torch._library.fake_class_registry import FakeScriptObject

    if isinstance(layer_name, LayerName):
        return layer_name.value
    elif isinstance(layer_name, FakeScriptObject):
        return layer_name.real_obj.value
    return layer_name


# Note: _moe_forward and _moe_forward_shared should not contain any
# implementation details, They should merely pass along control to
# the runner's '_forward_impl' method.
# These functions should never be called directly since they do not
# include all the functionality of the MoE layer.
def _moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return layer.runner._forward_impl(
        layer,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
    )


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    # `hidden_dim_unpadded > 0` only on the TRT-LLM MXFP4 path, where the
    # real kernel writes narrower than `hidden_states.shape[-1]`. Plumbed
    # as an op arg (not peeked from the layer registry) to keep the fake
    # a pure shape function of its inputs and preserve subgraph dedup.
    if hidden_dim_unpadded > 0:
        return hidden_states.new_empty((*hidden_states.shape[:-1], hidden_dim_unpadded))
    return torch.empty_like(hidden_states)


def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return layer.runner._forward_impl(
        layer,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
    )


def _moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # `fused_out`: see `_moe_forward_fake` for hidden_dim_unpadded semantics.
    # `shared_out`: matches `shared_experts_input` if provided (latent MoE),
    # else `hidden_states`.
    if hidden_dim_unpadded > 0:
        fused_out = hidden_states.new_empty(
            (*hidden_states.shape[:-1], hidden_dim_unpadded)
        )
    else:
        fused_out = torch.empty_like(hidden_states)
    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


direct_register_custom_op(
    op_name="moe_forward",
    op_func=_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=_moe_forward_shared,
    fake_impl=_moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _unpack(
    result: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if isinstance(result, tuple):
        return result
    else:
        return (None, result)


class MoERunner(MoERunnerInterface):
    """
    Standard MoE runner implementation for executing Mixture of Experts layers.

    This is the primary concrete implementation of MoE execution logic, providing
    comprehensive support for standard MoE operations. It handles:
    - Expert routing and token dispatching using various routing strategies
    - Shared experts computation with optional parallel execution using CUDA streams
    - Tensor model parallel and expert parallel operations
    - Multiple quantization methods and optimized kernel selection
    - Both monolithic and decomposed expert execution paths
    - Integration with various parallel execution modes (TP, EP, DP)

    The runner orchestrates the complete MoE forward pass including routing tokens
    to experts, executing expert computations in parallel, and combining results.
    It supports advanced features like overlapped execution of shared experts,
    optimized kernels for different parallel configurations, and seamless
    integration with vLLM's distributed execution framework.

    Eventually, this class may be split into more specialized implementations
    for different configurations (e.g., with/without shared experts, gates, etc.).
    """

    def __init__(
        self,
        layer_name: str,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_input_transform: torch.nn.Module | None,
        gate: torch.nn.Module | None,
        shared_experts: torch.nn.Module | None,
        quant_method: FusedMoEMethodBase,
        enable_dbo: bool,
        shared_expert_gate: torch.nn.Module | None = None,
        routed_output_transform: torch.nn.Module | None = None,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.router = router
        self.routed_input_transform = routed_input_transform
        self.routed_output_transform = routed_output_transform
        self.routed_scaling_factor = routed_scaling_factor
        self.gate = gate
        self.shared_expert_gate = shared_expert_gate
        self._quant_method = quant_method
        self.enable_dbo = enable_dbo

        # When both gates are present and FSE is enabled, fuse their
        # weight matrices into [num_experts + num_shared, hidden] so one
        # F.linear produces combined logits. The topk kernel can then
        # apply routing softmax and shared expert activation (sigmoid)
        # in a single launch.
        self._fse_fuse_gate = gate is not None and shared_expert_gate is not None
        self._combined_gate_weight: torch.Tensor | None = None

        self._shared_experts: SharedExperts | None = None
        if shared_experts is not None:
            self._shared_experts = SharedExperts(
                shared_experts,
                moe_config=moe_config,
                # Note: For now we must pass quant_method along to SharedExperts so it
                # can property determine where the shared experts are supposed to be
                # called, i.e. by a MK or by the MoERunner.
                # Once the MK can be created upfront, we can just pass in the proper
                # flags derived from the quant_method's MK.
                quant_method=quant_method,
                enable_dbo=enable_dbo,
            )

        # Needed for string -> FusedMoE layer lookup in custom ops.
        self.layer_name = layer_name

        self._forward_entry = self._select_forward()

    def _select_forward(self) -> Callable:
        if current_platform.is_tpu() or current_platform.is_cpu():
            # TODO: Once the OOM issue for the TPU backend is resolved, we
            # will switch to using the moe_forward custom op.
            # Note: CPU doesn't require wrapped _forward_impl.
            return _moe_forward if self._shared_experts is None else _moe_forward_shared

        return (
            torch.ops.vllm.moe_forward
            if self._shared_experts is None
            else torch.ops.vllm.moe_forward_shared
        )

    @property
    def shared_experts(self) -> SharedExperts | None:
        return self._shared_experts

    # TODO(bnell): temporary hack, do not call this method.
    def _replace_quant_method(self, quant_method: FusedMoEMethodBase):
        if self._shared_experts is not None:
            self._shared_experts._quant_method = quant_method
        self._quant_method = quant_method

    def _maybe_fuse_gate_weights(self):
        """Fuse router and shared expert gate weights on first call.

        Cannot be done at __init__ because gate weights are loaded after
        module construction (via weight_loader). Called once from
        _forward_impl before the first forward pass.
        """
        if self._combined_gate_weight is None:
            assert self.gate is not None and self.shared_expert_gate is not None
            self._combined_gate_weight = torch.cat(
                [self.gate.weight, self.shared_expert_gate.weight],
                dim=0,
            )

    def is_internal_router(self) -> bool:
        return self.gate is not None

    def apply_routed_input_transform(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply transform for routed experts (e.g., latent projection).

        This is called by FusedMoE.forward_native. The original hidden_states
        is saved separately so shared experts get [S, hidden_size] while
        routed experts get the transformed [S, moe_latent_size].

        Returns (possibly transformed) hidden states and the input for shared
        experts (or None if there are no shared experts).
        """
        if self.routed_input_transform is not None:
            result = self.routed_input_transform(hidden_states)
            # ReplicatedLinear returns (output, extra_bias) tuple.
            # We only need the output tensor; extra_bias is not used here.
            if isinstance(result, tuple):
                return result[0], hidden_states
            return result, hidden_states

        return (
            hidden_states,
            hidden_states if self._shared_experts is not None else None,
        )

    def apply_routed_output_transform(
        self,
        fused_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transform to routed expert output (e.g., latent to full dim).

        Used by latent MoE models (e.g., NemotronH) where routed experts
        operate in a compressed latent space and need projection back to
        the full hidden dimension before combining with shared expert output.
        """
        if self.routed_output_transform is not None:
            r = self.routed_output_transform(fused_output)
            fused_output = r[0] if isinstance(r, tuple) else r
        return fused_output

    def _maybe_apply_routed_scale_to_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Apply routed_scaling_factor to the output with FP16 overflow
        protection.

        Scale the fused expert output by routed_scaling_factor. For FP16,
        avoid overflow by dividing shared_output by the scale instead
        (the decoder layer compensates with matching divisions).
        """
        if self.routed_scaling_factor != 1.0:
            if fused_output.dtype != torch.float16 or shared_output is None:
                fused_output *= self.routed_scaling_factor
            elif shared_output is not None:
                shared_output *= 1.0 / self.routed_scaling_factor
        return shared_output, fused_output

    @property
    def _fused_output_is_reduced(self) -> bool:
        return (
            self._quant_method.moe_kernel is not None
            and self._quant_method.moe_kernel.output_is_reduced()
        )

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """All-reduce shared expert output when the combine kernel already
        reduced fused output.

        * If the combine kernel does the reduction for fused_output, reduce
          shared_output separately. O.w, reduce fused_output+shared_output later.
        * If we have SP (TP=N, DP=M, EP), there is a separate AG step handled
          in the model.
        """
        if (
            shared_output is not None
            and not self.moe_config.is_sequence_parallel
            and self._fused_output_is_reduced
        ):
            shared_output = tensor_model_parallel_all_reduce(shared_output)
        return shared_output

    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int,
    ) -> torch.Tensor:
        """Truncate padded dimensions and all-reduce the combined output.

        This is the "late" all-reduce path. When neither fused nor shared
        output was individually reduced, the combined sum is all-reduced
        here. Skipped when sequence-parallel is active (SP handles its
        own reduction) or when the early path already reduced both outputs.
        """
        # We don't need to reduce the final output if:
        # - We are not running with TP or DP
        # - The MK already reduced the fused output itself.
        if (
            not self.moe_config.is_sequence_parallel
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
            and not self._fused_output_is_reduced
        ):
            states = tensor_model_parallel_all_reduce(states)

        return states[..., :trunc_size]

    def _encode_layer_name(self) -> str | LayerName:
        if _USE_LAYERNAME:
            return LayerName(self.layer_name)
        # Can be unavailable or None in unittests
        if (
            is_forward_context_available()
            and get_forward_context().all_moe_layers is not None
        ):
            return "from_forward_context"
        return self.layer_name

    def _trtllm_mxfp4_unpadded_dim(self) -> int:
        """Return ``hidden_dim_unpadded`` when the active backend is TRT-LLM
        MXFP4 (whose kernel writes narrower than the padded
        ``hidden_states.shape[-1]``), else 0. Other MXFP4 backends (notably
        Cutlass MXFP4 MXFP8) write the full padded width, so
        ``moe_config.hidden_dim_unpadded`` alone is insufficient: it encodes
        the model's logical hidden, not whether the kernel narrows. Computed
        caller-side and passed as an op arg; doing the isinstance check
        inside the fake would specialize per ``layer_name`` and break
        subgraph dedup for identical-architecture models (e.g. Phi-MoE).
        """
        from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
            TrtLlmMxfp4ExpertsBase,
        )

        moe_kernel = getattr(self._quant_method, "moe_kernel", None)
        fused_experts = getattr(
            getattr(moe_kernel, "impl", None), "fused_experts", None
        )
        if isinstance(fused_experts, TrtLlmMxfp4ExpertsBase):
            return self.moe_config.hidden_dim_unpadded or self.moe_config.hidden_dim
        return 0

    def _maybe_pad_hidden_states(
        self,
        shared_experts_input: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Pad hidden_states to moe_config.hidden_dim and compute the
        original dimension for later truncation.

        For latent MoE, the routed hidden_states may be smaller than
        hidden_dim. Padding ensures uniform tensor sizes through the
        fused MoE kernel. The returned trunc_size is used by
        _maybe_reduce_final_output to strip the padding from the result.
        """
        shared_experts_hidden_dim = (
            shared_experts_input.shape[-1] if shared_experts_input is not None else 0
        )
        transformed_hidden_dim = hidden_states.shape[-1]
        if (
            not self._quant_method.skip_forward_padding
            and self.moe_config.hidden_dim != transformed_hidden_dim
        ):
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        if self.routed_output_transform is not None and shared_experts_hidden_dim > 0:
            orig_hidden_dims = shared_experts_hidden_dim
        else:
            orig_hidden_dims = transformed_hidden_dim

        return hidden_states, orig_hidden_dims

    def _maybe_apply_shared_experts(
        self,
        shared_experts_input: torch.Tensor | None,
        order: SharedExpertsOrder,
    ):
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts.apply(shared_experts_input, order)

    def _apply_quant_method(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run expert routing and the fused MoE kernel via the quant method.

        Orchestrates shared expert execution (before/after), expert selection
        via the router, and the actual fused MoE computation. Returns
        (shared_expert_output, fused_expert_output).
        """
        self._maybe_apply_shared_experts(
            shared_experts_input, SharedExpertsOrder.NO_OVERLAP
        )

        # Get routing replay buffer from persistent layer attribute
        # (set by bind_routing_capture_to_model during capturer init)
        routing_replay_out = getattr(layer, "_routing_replay_out", None)

        if self._quant_method.is_monolithic:
            fused_out = self._quant_method.apply_monolithic(
                layer=layer,
                x=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
        else:
            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )

            # Write routing data for non-monolithic path (Triton, etc.)
            if routing_replay_out is not None:
                routing_replay_out[: topk_ids.shape[0]].copy_(topk_ids.to(torch.int16))

            # Passing shared_experts_input in case SharedExpertsOrder is
            # MK_INTERNAL_OVERLAPPED.
            fused_out = self._quant_method.apply(
                layer=layer,
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        )

        return (
            self._shared_experts.output if self._shared_experts is not None else None,
            fused_out,
        )

    def _sequence_parallel_context(self):
        """Return a context manager for sequence-parallel token
        redistribution.

        When sequence parallelism is active, returns a context that handles
        local size tracking for proper token scatter/gather. Otherwise
        returns a no-op context.
        """
        ctx = get_forward_context()
        return (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

    def _maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor | None,
    ):
        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts.maybe_sync_shared_experts_stream(shared_experts_input)

    def _maybe_add_zero_expert_output(
        self,
        result: torch.Tensor,
    ) -> torch.Tensor:
        """Add the zero expert's contribution to the final result.

        When a ZeroExpertRouter is used, it computes a bias-like output
        from the "zero expert" that is added to the combined routed+shared
        expert output.
        """
        if isinstance(self.router, ZeroExpertRouter):
            zero_expert_output = self.router.zero_expert_output
            assert zero_expert_output is not None
            result = result + zero_expert_output
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Invoke the fused moe layer.

        Input:
        - hidden_states
        - router_logits

        Output:
        - The new hidden_states.

        Calling sequence
        - forward
          - self._forward_entry (_moe_forward or _moe_forward_shared custom op)
            - _forward_impl

        Note: The existence of _moe_forward and _moe_forward_shared custom ops are due
        to the following reason:
        1. pytorch cannot handle union types in custom op signatures so
           _moe_forward and _moe_forward_shared must be split.
        """

        # Apply transform for routed experts (e.g., latent projection
        # for latent MoE)
        hidden_states, shared_experts_input = self.apply_routed_input_transform(
            hidden_states
        )

        # Record before `_maybe_pad_hidden_states` pads activations to match
        # `moe_config.hidden_dim`, e.g. after `align_trtllm_fp4_moe_hidden_dim_for_fi`
        # so routed output can be trimmed before
        # shared+routed add / latent up proj if needed.
        routed_hidden_dim = hidden_states.shape[-1]
        hidden_states, og_hidden_dim = self._maybe_pad_hidden_states(
            shared_experts_input,
            hidden_states,
        )
        hidden_dim_was_padded = hidden_states.shape[-1] > routed_hidden_dim

        # ── bubble profiling ──────────────────────────────────────────────
        _profile_this = (
            _bubble_profile_enabled()
            and (self.moe_config.ep_size > 1 or self.moe_config.tp_size > 1)
            and hidden_states.shape[0] >= _BUBBLE_MIN_TOKENS
        )
        if _profile_this:
            global _bubble_calls, _bubble_prefill_id, _bubble_records, _bubble_pending, _bubble_entry_t
            _bubble_entry_t = time.perf_counter()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            with _bubble_lock:
                _bubble_calls += 1
                if _bubble_calls == 1:
                    _bubble_prefill_id += 1
                layer_idx = _bubble_calls
            e0.record()
        # ─────────────────────────────────────────────────────────────────

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self._trtllm_mxfp4_unpadded_dim(),
        )

        if _profile_this:
            e1.record()

        #
        # Note: there are two all-reduce points below. They are mutually
        # exclusive, controlled by _fused_output_is_reduced
        #  - When True: the combine kernel already reduced fused_output,
        #    so we reduce shared_output here to match, then skip the
        #    all-reduce in _maybe_reduce_final_output.
        #  - When False: neither output is reduced yet, so we combine
        #    them first and all-reduce the sum in _maybe_reduce_final_output.

        # Extract outputs from result
        shared_output, fused_output = _unpack(result)
        if (
            shared_output is not None or self.routed_output_transform is not None
        ) and hidden_dim_was_padded:
            fused_output = fused_output[..., :routed_hidden_dim]

        # If combine kernel already reduced fused, reduce shared to match.
        # See note above re: the two all-reduce points.
        shared_output = self._maybe_reduce_shared_expert_output(shared_output)

        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(
            shared_output, fused_output
        )

        # Apply output transform (e.g. latent -> full dim)
        fused_output = self.apply_routed_output_transform(fused_output)

        if shared_output is not None:
            result = shared_output + fused_output
        else:
            result = fused_output

        # Auto-arm the next pending job when no scheduler is active.
        # This runs on the first MoE layer of each new prefill, so the
        # scheduler is armed before any layer bubbles are triggered.
        n_tok = hidden_states.shape[0]

        # Skip all scheduling logic when called from inside a training sub-op.
        # A training sub-op calls layer.mlp() which re-enters this runner.
        # Without this guard, fill_one() fires from within the sub-op itself,
        # chaining all remaining sub-ops together and collapsing the per-bubble
        # design into a monolith.
        if getattr(_bt_subop_running, 'active', False):
            result = self._maybe_reduce_final_output(result, og_hidden_dim)
            return result

        # Demo mode: submit first job on the first qualifying prefill.
        global _sched_demo_armed, _fwd_demo_armed, _combined_demo_armed, _fwd_prefill_state
        if _sched_demo_enabled and not _sched_demo_armed and n_tok >= _sched_min_tokens:
            _sched_demo_armed = True
            _submit_demo_job()

        # FT forward demo mode: auto-submit first job on the first qualifying step.
        if _fwd_demo_enabled and not _fwd_demo_armed:
            if (_fwd_mode == "bubble" and n_tok >= _fwd_sched_min_tokens) or \
               (_fwd_mode == "decode" and n_tok < _fwd_sched_min_tokens) or \
               (_fwd_mode == "prefill" and n_tok >= _fwd_sched_min_tokens):
                _fwd_demo_armed = True
                _submit_fwd_demo_job()

        # C+D_batch: arm a new batched MoE pass on the first MoE layer of each
        # decode step (n_tok < _fwd_sched_min_tokens means it's a decode step).
        if _combined_mode == "C+D_batch" and not _ft_moe_state["active"]:
            if n_tok < _fwd_sched_min_tokens:  # decode step
                ft_moe_arm(hidden_states.shape[-1], hidden_states.device.index)

        # C+D_batch real training: submit the attention-only fwd job and the
        # backward job on the first qualifying step, then keep the chain going
        # via _combined_fwd_done / _combined_bwd_done (same callbacks as C+D).
        if _combined_mode == "C+D_batch" and not _combined_demo_armed and _bt_trainer is not None:
            _combined_demo_armed = True
            _ft_moe_state["real_training"] = True
            _submit_combined_fwd_job()

        # Combined mode demo: arm fwd→bwd chain (B+D / C+D) or concatenated job (D+D).
        if _combined_mode not in ("off", "C+D_batch") and not _combined_demo_armed:
            if _combined_mode == "D+D" and n_tok >= _sched_min_tokens:
                _combined_demo_armed = True
                _submit_combined_dd_job()
            elif _combined_mode in ("B+D", "C+D"):
                # B+D arms on first prefill; C+D arms on any step (fwd fires on decode).
                arm_on = (n_tok >= _sched_min_tokens) if _combined_mode == "B+D" else True
                if arm_on:
                    _combined_demo_armed = True
                    _submit_combined_fwd_job()

        if _active_scheduler is None and not _sched_job_queue.empty() and n_tok >= _sched_min_tokens:
            try:
                sub_ops, fills, post_fn, dev, min_tok, tr = _sched_job_queue.get_nowait()
                if n_tok >= min_tok:
                    from vllm.model_executor.layers.fused_moe.runner.bubble_scheduler import (
                        make_scheduler,
                    )
                    arm_bubble_scheduler(
                        make_scheduler(sub_ops, device=dev, post_complete_fn=post_fn, label="bwd"),
                        min_tokens=min_tok,
                        fills_per_trigger=fills,
                        trigger_rank=tr,
                    )
            except _queue.Empty:
                pass

        # Backward scheduling: prefill-only dispatch with decode drain.
        #
        # Prefill path (n_tok >= _sched_min_tokens):
        #   fill_one() is gated on a CUDA event recorded just before the TP
        #   all_reduce so sub-ops start only once the main stream is blocked
        #   at the NCCL barrier (the actual idle window).
        #
        # Decode path (n_tok < _sched_min_tokens):
        #   No new sub-ops are dispatched.  On the first decode call after a
        #   prefill (the prefill→decode transition) we call pause_decode(),
        #   which records a drain event on bwd_stream.  The main stream waits
        #   for that event so any sub-ops already queued during the last
        #   prefill complete before decode kernels start — eliminating the
        #   HBM bandwidth contention that caused +40% TPOT in C+D mode.
        global _bwd_sched_in_decode
        if _active_scheduler is not None and _active_scheduler.has_work():
            if n_tok >= _sched_min_tokens:
                # decode→prefill transition: re-enable dispatch.
                if _bwd_sched_in_decode:
                    _active_scheduler.resume_prefill()
                    _bwd_sched_in_decode = False
                if _ep_is_light_rank(self, router_logits, n_tok):
                    sync_evt = torch.cuda.Event()
                    sync_evt.record()
                    for _ in range(_sched_fills_per_trigger):
                        if not _active_scheduler.fill_one(in_bubble=True, sync_event=sync_evt):
                            break
            elif n_tok > 0:
                # prefill→decode transition: pause and drain bwd_stream once.
                if not _bwd_sched_in_decode:
                    drain_evt = _active_scheduler.pause_decode()
                    if drain_evt is not None:
                        if _FT_TIMING:
                            _e0 = torch.cuda.Event(enable_timing=True)
                            _e1 = torch.cuda.Event(enable_timing=True)
                            _e0.record()
                        torch.cuda.current_stream().wait_event(drain_evt)
                        if _FT_TIMING:
                            _e1.record()
                            _pause_timing_pairs.append((_e0, _e1))
                    _bwd_sched_in_decode = True
                # No dispatch during decode — bwd runs in prefill bubbles only.
            if not _active_scheduler.has_work():
                disarm_bubble_scheduler()
                _bwd_sched_in_decode = False

        # ── FT forward placement experiment ──────────────────────────────────
        if _fwd_mode == "bubble" and n_tok >= _fwd_sched_min_tokens:
            # Mode D: dispatch sub-ops into TP all_reduce bubbles.
            # Arm from job queue on first qualifying prefill.
            if _fwd_active_scheduler is None and not _fwd_job_queue.empty():
                try:
                    fwd_item = _fwd_job_queue.get_nowait()
                    fwd_ops, fwd_fills, fwd_post, fwd_dev, fwd_min, fwd_tr = fwd_item
                    if n_tok >= fwd_min:
                        from vllm.model_executor.layers.fused_moe.runner.bubble_scheduler import (
                            make_scheduler,
                        )
                        arm_fwd_scheduler(
                            make_scheduler(fwd_ops, device=fwd_dev, post_complete_fn=fwd_post, label="fwd_bubble", is_forward=True),
                            min_tokens=fwd_min,
                            fills_per_trigger=fwd_fills,
                            trigger_rank=fwd_tr,
                        )
                except _queue.Empty:
                    pass
            if (
                _fwd_active_scheduler is not None
                and _fwd_active_scheduler.has_work()
                and n_tok >= _fwd_sched_min_tokens
                and (_fwd_trigger_rank is None or _sched_get_tp_rank() == _fwd_trigger_rank)
            ):
                fwd_sync_evt = torch.cuda.Event()
                fwd_sync_evt.record()
                for _ in range(_fwd_fills_per_trigger):
                    if not _fwd_active_scheduler.fill_one(in_bubble=True, sync_event=fwd_sync_evt):
                        break
                if not _fwd_active_scheduler.has_work():
                    disarm_fwd_scheduler()

        elif _fwd_mode == "decode":
            # Mode C: dispatch sub-ops on secondary stream during decode.
            # ARM and FILL only on decode-scale batches (n_tok < threshold).
            # Prefill batches are excluded: their GPU load is too heavy for
            # concurrent secondary-stream training work.
            if _fwd_active_scheduler is None and not _fwd_job_queue.empty() \
                    and n_tok < _fwd_sched_min_tokens:
                try:
                    fwd_item = _fwd_job_queue.get_nowait()
                    fwd_ops, fwd_fills, fwd_post, fwd_dev, fwd_min, fwd_tr = fwd_item
                    from vllm.model_executor.layers.fused_moe.runner.bubble_scheduler import (
                        make_scheduler,
                    )
                    # Keep _fwd_sched_min_tokens unchanged (don't set to 0 —
                    # that would make the fill block unreachable).
                    arm_fwd_scheduler(
                        make_scheduler(fwd_ops, device=fwd_dev, post_complete_fn=fwd_post, label="fwd_decode", is_forward=True),
                        min_tokens=_fwd_sched_min_tokens,
                        fills_per_trigger=fwd_fills,
                        trigger_rank=fwd_tr,
                    )
                except _queue.Empty:
                    pass
            # FILL: only during decode-scale batches (small n_tok).
            # "C" in C+D means the training forward runs on the secondary
            # stream during Decode, not during prefill.  Firing fill_one()
            # during a large prefill (e.g. 7247 tokens) creates GPU work on
            # the secondary stream that competes with the prefill and causes
            # the EngineCore's sample_tokens RPC to time out.
            if (
                _fwd_active_scheduler is not None
                and _fwd_active_scheduler.has_work()
                and n_tok < _fwd_sched_min_tokens   # decode-scale only
                and (_fwd_trigger_rank is None or _sched_get_tp_rank() == _fwd_trigger_rank)
            ):
                for _ in range(_fwd_fills_per_trigger):
                    if not _fwd_active_scheduler.fill_one(in_bubble=False, sync_event=None):
                        break
                if not _fwd_active_scheduler.has_work():
                    disarm_fwd_scheduler()

        elif _fwd_mode == "prefill" and n_tok >= _fwd_sched_min_tokens and _fwd_prefill_state is not None:
            # Mode B: run sub-ops synchronously on the main stream during prefill.
            # fills_per_trigger sub-ops per MoE layer; cursor guards exhaustion.
            state = _fwd_prefill_state
            for _ in range(state.fills_per_trigger):
                if state.cursor >= len(state.sub_ops):
                    break
                state.sub_ops[state.cursor]()
                state.cursor += 1
            if state.cursor >= len(state.sub_ops):
                _fwd_prefill_state = None
                if state.post_complete_fn is not None:
                    state.post_complete_fn()

        result = self._maybe_reduce_final_output(result, og_hidden_dim)

        if _profile_this:
            e2.record()
            # Stash events for deferred resolution — no synchronize() here.
            # Calling synchronize() per-layer serializes NCCL across ranks and
            # inflates allreduce_ms by the full inter-rank FFN imbalance (~17ms/layer
            # on H100), making bubble estimates useless.  Instead we accumulate
            # event tuples and do ONE synchronize() after the final layer, then
            # resolve elapsed_time() for all layers at once.
            t_exit_approx = time.perf_counter()  # CPU-side approximation for gap analysis
            with _bubble_lock:
                _bubble_pending.append({
                    "e0": e0, "e1": e1, "e2": e2,
                    "prefill_id": _bubble_prefill_id,
                    "layer":      layer_idx,
                    "n_tokens":   hidden_states.shape[0],
                    "t_entry":    _bubble_entry_t,
                    "t_exit":     t_exit_approx,
                })
                if layer_idx % 48 == 0:
                    _bubble_calls = 0
                    # Single synchronize + resolve for the whole prefill.
                    torch.cuda.synchronize()
                    resolved = []
                    for p in _bubble_pending:
                        resolved.append({
                            "prefill_id":   p["prefill_id"],
                            "layer":        p["layer"],
                            "n_tokens":     p["n_tokens"],
                            "ffn_ms":       p["e0"].elapsed_time(p["e1"]),
                            "allreduce_ms": p["e1"].elapsed_time(p["e2"]),
                            "total_ms":     p["e0"].elapsed_time(p["e2"]),
                            "t_entry":      p["t_entry"],
                            "t_exit":       p["t_exit"],
                        })
                    _bubble_pending.clear()
                    _bubble_records.extend(resolved)
                    with open(_bubble_out_path(), "w") as f:
                        json.dump(_bubble_records, f, indent=2)

        return self._maybe_add_zero_expert_output(result)

    @property
    def do_naive_dispatch_combine(self) -> bool:
        return (
            self.moe_config.dp_size > 1 and not self._quant_method.supports_internal_mk
        )

    def _maybe_dispatch(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # For naive dispatch/combine Dp/Ep, dispatch the hidden states and
        # router logits to all experts.
        # NOTE: this will be removed once all kernels are migrated into the
        # MoEKernel framework.
        if self.do_naive_dispatch_combine:
            result = get_ep_group().dispatch_router_logits(
                hidden_states,
                router_logits,
                self.moe_config.is_sequence_parallel,
            )
            assert len(result) == 2
            hidden_states, router_logits = result

        # NOTE: Similar with DP, PCP also needs dispatch and combine. For
        # simplicity, AgRsAll2All was added separately for PCP here. Maybe
        # we should modify All2AllManager abstraction to better support PCP.
        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().all_gather(
                hidden_states,
                dim=0,
            )
            router_logits = get_pcp_group().all_gather(
                router_logits,
                dim=0,
            )

        return hidden_states, router_logits

    def _maybe_combine(
        self,
        shared_output: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if self.do_naive_dispatch_combine:
            hidden_states = get_ep_group().combine(
                hidden_states, self.moe_config.is_sequence_parallel
            )

        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().reduce_scatter(
                hidden_states,
                dim=0,
            )

        if self.shared_experts is not None:
            assert shared_output is not None
            return shared_output, hidden_states
        else:
            return hidden_states

    def _forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Entry point called by the custom op to run the MoE computation.

        Handles pre-dispatch setup (gate application, external shared expert
        triggering, quant config init) then performs the following steps
        within the sequence-parallel context.

        - Performs expert routing
        - fused MoE kernel execution
        - shared expert computation.

        Returns a single tensor of combined fused and shared output (if present).
        """
        # TODO(bnell): this can be removed after MK migration is complete.
        layer.ensure_moe_quant_config_init()

        # Sync aux and main stream for shared expert multi-stream overlap.
        self._maybe_sync_shared_experts_stream(shared_experts_input)

        # If the Runner holds the gate, apply it after the stream sync,
        # so it can run overlapped with the
        # NOTE: in future PR, MoE runner will always hold the gate.
        if self.gate is not None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = F.linear(hidden_states, self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)

        with self._sequence_parallel_context():
            # TODO(bnell): parts of the dispatch/combine steps will go away once
            # #32567 lands and the remaining kernels are made MKs.  The PCP
            # code will probably remain
            hidden_states, router_logits = self._maybe_dispatch(
                layer,
                hidden_states,
                router_logits,
            )

            shared_output, hidden_states = self._apply_quant_method(
                layer=layer,
                hidden_states=hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input,
                input_ids=input_ids,
            )

            return self._maybe_combine(
                shared_output,
                hidden_states,
            )
