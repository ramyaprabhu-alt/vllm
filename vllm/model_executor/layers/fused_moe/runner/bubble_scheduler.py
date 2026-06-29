# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
VllmBubbleScheduler — schedules SFT backward sub-ops into EP-imbalance
bubbles at TP all_reduce barriers (prefill) and during decode steps (fallback).

Background
----------
In vLLM TP=2 EP (Qwen3-30B-A3B), whichever rank finishes its expert FFN first
idles at the NCCL all_reduce barrier waiting for the other rank.  Which rank
is faster varies per layer and per input — with skewed routing (repeated
tokens) rank 1 is almost always faster; with real diverse text either rank can
lead in a given layer.

The idle window on the fast rank scales with routing imbalance and context:

    T_in= 8k → ~34 ms total, p50/layer=1.43 ms  (24/48 exploitable)
    T_in=16k → ~68 ms total, p50/layer=2.70 ms  (25/48 exploitable)
    T_in=32k → ~137ms total, p50/layer=4.43 ms  (31/48 exploitable)

Both ranks arm a scheduler instance.  During prefill, fill_one() is called
just before each layer's TP all_reduce; whichever rank reaches the barrier
first (the light EP rank) runs its sub-op uncontested, while the heavy
rank's sub-op competes at low priority.  During decode, per-token compute is
light enough that the low-priority bwd stream gets real GPU time on both
ranks — this decode fallback is the primary backward-progress path for the
rank that consistently holds the heavier EP side.

Typical backward sub-op costs (T_ft=128, Qwen3-30B-A3B per-rank shapes):
    Q/K/V/O proj bwd  ~0.05 ms each    → fits any bubble ≥ 0.1 ms
    Attn SDPA bwd     ~0.89 ms          → fits p50 at T_in ≥ 8k
    MoE 8-expert bwd  ~0.41 ms          → fits any bubble ≥ 0.5 ms
    Full attn bwd     ~1.09 ms/layer    → fits p50 at T_in ≥ 8k

Integration
-----------
The scheduler persists across multiple prefills.  MoeRunner.forward() calls
fill_one() on every MoE layer of every prefill while work remains, so the
sub-op queue drains purely through prefill bubbles — decode is never touched.

    # Each TP rank builds its own sub-op list (local weights only).
    sub_ops = build_backward_sub_ops(sft_engine, ...)
    sched = VllmBubbleScheduler(sub_ops, device=local_rank)

    # Arm once — both ranks arm independently, no coordination needed.
    arm_bubble_scheduler(sched)

    # Each subsequent prefill automatically continues draining via fill_one().
    # With 48 MoE layers/prefill and ~432 sub-ops, ~9 prefills clear the queue.
    for request in incoming_requests:
        run_prefill(request)         # fill_one() fires 48× per prefill
        run_decode_until_done(...)   # scheduler is silent during decode

    # When the backward pass must complete (e.g. before optimizer.step()),
    # disarm so no new fill_one() calls start, then wait for the worker.
    disarm_bubble_scheduler()
    sched.wait()

    print(f"Bubble util: {sched.bubble_utilization:.1%} "
          f"({sched.subops_in_bubble}/{sched.n_ops} sub-ops in-bubble)")

arm_bubble_scheduler / disarm_bubble_scheduler are defined in moe_runner.py
and imported from there so callers have a single import point.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from collections.abc import Callable

import torch

log = logging.getLogger(__name__)

# ── Green context scheduling mode ────────────────────────────────────────────
# VLLM_FT_SCHED_MODE controls the BACKWARD scheduler:
#   "bubble"    — VllmBubbleScheduler (default): lo-pri stream, bubble gated
#   "green_ctx" — GreenCtxScheduler: green ctx stream, runs continuously
_SCHED_MODE: str = os.environ.get("VLLM_FT_SCHED_MODE", "bubble")
# VLLM_FT_FWD_GREEN_CTX=1 puts the FORWARD scheduler on a green context
# stream with dedicated SMs. The forward runs during decode (competing with
# inference), so SM isolation helps. The backward runs in EP bubbles (GPU
# mostly idle), so it stays on a regular stream.
_FWD_GREEN_CTX: bool = os.environ.get("VLLM_FT_FWD_GREEN_CTX", "0") == "1"
# Number of SMs to dedicate to training in green_ctx modes.
_GREEN_CTX_SMS: int = int(os.environ.get("VLLM_FT_GREEN_CTX_SMS", "8"))

# ── Lightweight sub-op timing ─────────────────────────────────────────────────
# Set VLLM_FT_TIMING=1 to log per-sub-op GPU execution times to
# /tmp/bt_timing.log.  Uses CUDA events recorded on bwd_stream; timings are
# collected after bwd_stream.synchronize() so there is zero overhead on the
# main inference stream.
_FT_TIMING: bool = os.environ.get("VLLM_FT_TIMING") == "1"
_TIMING_LOG: str = "/tmp/bt_timing.log"


def _flush_subop_timings(
    label: str,
    events: "list[tuple[str, torch.cuda.Event, torch.cuda.Event]]",
) -> None:
    """Write collected CUDA-event timings to the log file.

    Called after bwd_stream.synchronize(), so elapsed_time() is safe.
    """
    import time as _time
    ts = _time.time()
    lines = []
    for name, t0, t1 in events:
        try:
            ms = t0.elapsed_time(t1)
            lines.append(f"{ts:.3f},subop,{label},{name},{ms:.4f}\n")
        except Exception:
            pass
    if lines:
        try:
            with open(_TIMING_LOG, "a") as f:
                f.writelines(lines)
        except Exception:
            pass


def _import_queue():
    return queue


class VllmBubbleScheduler:
    """Schedule backward sub-ops into EP-imbalance idle windows.

    Each call to fill_one() releases the background worker to process the
    next sub-op on a low-priority CUDA stream.  During prefill the sub-op
    is gated to the TP all_reduce idle window (where the light EP rank is
    blocked); during decode it fires ungated so the heavy EP rank can also
    make backward progress.

    Parameters
    ----------
    sub_ops :
        Backward sub-ops to execute, in order.  Each callable is invoked
        inside ``torch.cuda.stream(bwd_stream)`` on a background thread.
        Sub-ops must be independent of the current prefill's tensors (they
        operate on saved activations from a prior SFT forward pass).
    device :
        CUDA device to use.  Defaults to the current device at construction
        time.  Background threads do not inherit set_device() from the parent,
        so this must be set explicitly.
    """

    def __init__(
        self,
        sub_ops: list[Callable[[], None]],
        device: int | None = None,
        post_complete_fn: Callable[[], None] | None = None,
        label: str = "sched",
        stream: "torch.cuda.Stream | None" = None,
    ):
        """
        Parameters
        ----------
        sub_ops :
            Backward sub-ops in execution order.
        device :
            CUDA device index.  Background threads do not inherit
            set_device() from the parent, so this must be explicit.
        post_complete_fn :
            Optional callback invoked on the worker thread after all sub-ops
            and the backward stream synchronize.  Use for gradient allreduce,
            optimizer step notification, etc.
        stream :
            Optional external CUDA stream (e.g. from a green context).
            When provided, sub-ops run on this stream instead of a
            newly-created low-priority stream.
        """
        if device is not None:
            torch.cuda.set_device(device)
        self._device = device if device is not None else torch.cuda.current_device()

        self._sub_ops = list(sub_ops)
        self._n_ops = len(self._sub_ops)
        self._cursor = 0            # next un-signalled sub-op index
        self._post_complete_fn = post_complete_fn
        self._label = label

        self.subops_in_bubble = 0   # dispatched before fill_remaining()
        self.subops_after = 0       # dispatched by fill_remaining()

        self._trigger = threading.Semaphore(0)
        self._all_done = threading.Event()

        # Queue pairing each trigger with an optional CUDA sync event.
        # fill_one() records an event on the main stream immediately before
        # the all_reduce; the backward stream waits for it before starting
        # the sub-op, ensuring backward kernels start only after FFN is done
        # and the main stream is blocked at the NCCL barrier.
        self._event_queue: "queue.Queue[torch.cuda.Event | None]" = _import_queue().Queue()

        # Decode-pause flag.  Set by pause_decode() at the prefill→decode
        # transition; cleared by resume_prefill() at decode→prefill.
        # fill_one() returns False immediately when set, so no new sub-ops
        # are dispatched during decode steps.
        self._decode_paused: bool = False

        if stream is not None:
            self._bwd_stream = stream
        else:
            _lo, _hi = torch.cuda.Stream.priority_range()
            self._bwd_stream = torch.cuda.Stream(device=self._device, priority=_hi)

        self._worker = threading.Thread(
            target=self._run_worker,
            daemon=True,
        )
        self._worker.start()

    # ── Worker ────────────────────────────────────────────────────────────────

    def _run_worker(self) -> None:
        torch.cuda.set_device(self._device)
        _timing = _FT_TIMING
        _t_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        # vLLM workers run inside torch.inference_mode().  Backward passes
        # need grad computation, so we exit inference mode on this thread.
        with torch.inference_mode(False):
            for i, sub_op in enumerate(self._sub_ops):
                self._trigger.acquire()
                event: torch.cuda.Event | None = self._event_queue.get()
                with torch.cuda.stream(self._bwd_stream):
                    # Wait for the main stream to finish FFN before starting.
                    # The event was recorded just before the all_reduce, so
                    # waiting for it gates the sub-op to the actual idle window.
                    if event is not None:
                        self._bwd_stream.wait_event(event)
                    if _timing:
                        t0 = torch.cuda.Event(enable_timing=True)
                        t1 = torch.cuda.Event(enable_timing=True)
                        t0.record()
                        sub_op()
                        t1.record()
                        _t_events.append(
                            (getattr(sub_op, '__name__', f'op{i}'), t0, t1)
                        )
                    else:
                        sub_op()
            self._bwd_stream.synchronize()
        # Collect timings after sync — elapsed_time() is safe now.
        if _timing and _t_events:
            _flush_subop_timings(self._label, _t_events)
        if self._post_complete_fn is not None:
            self._post_complete_fn()
        self._all_done.set()

    # ── Scheduler interface ───────────────────────────────────────────────────

    @property
    def n_ops(self) -> int:
        return self._n_ops

    def has_work(self) -> bool:
        """True while sub-ops remain un-signalled."""
        return self._cursor < self._n_ops

    def fill_one(
        self,
        in_bubble: bool = True,
        sync_event: "torch.cuda.Event | None" = None,
    ) -> bool:
        """Signal the background worker to process the next sub-op.

        Called from MoeRunner.forward() just before the TP all_reduce.
        Pass sync_event (recorded on the main stream right before the
        all_reduce) so the backward stream waits for FFN completion before
        starting — pinning the sub-op to the actual idle window.

        Parameters
        ----------
        in_bubble :
            True when called from a layer trigger; False from fill_remaining().
        sync_event :
            CUDA event recorded on the main stream before the all_reduce.
            The backward stream waits for this event, ensuring the sub-op
            starts only after the main stream has entered the NCCL barrier.
            Pass None (default) to start immediately without gating.
        """
        if not self.has_work():
            return False
        if self._decode_paused:
            return False
        self._cursor += 1
        self._event_queue.put(sync_event)
        self._trigger.release()
        if in_bubble:
            self.subops_in_bubble += 1
        else:
            self.subops_after += 1
        return True

    def pause_decode(self) -> "torch.cuda.Event | None":
        """Called at the prefill→decode transition.

        Prevents fill_one() from dispatching new sub-ops and records a drain
        event on the backward stream.  The caller should have the main
        (inference) stream wait for this event before starting decode kernels,
        ensuring any sub-ops already queued on bwd_stream complete first.

        Returns the drain event, or None if the backward stream is already idle.
        Safe to call from any thread.  CUDA stream enqueue is internally
        serialised so the event will land after any work already on bwd_stream.
        """
        self._decode_paused = True
        drain = torch.cuda.Event()
        self._bwd_stream.record_event(drain)
        return drain

    def resume_prefill(self) -> None:
        """Called at the decode→prefill transition.  Re-enables fill_one()."""
        self._decode_paused = False

    def fill_remaining(self, sync_event: "torch.cuda.Event | None" = None) -> int:
        """Drain all remaining sub-ops after the prefill step completes.

        Returns the number of sub-ops that were dispatched here (not in a
        bubble).  These run on the backward stream after the prefill ends,
        in parallel with decode or the next request's queuing overhead.
        """
        count = 0
        while self.has_work():
            self.fill_one(in_bubble=False, sync_event=sync_event)
            count += 1
        return count

    def wait(self, timeout: float = 120.0) -> bool:
        """Block until all sub-ops complete.

        Also synchronizes bwd_stream → current stream so subsequent work on
        the current stream sees all backward writes (e.g. gradient updates).

        Returns True if all work completed within the timeout.
        """
        done = self._all_done.wait(timeout=timeout)
        torch.cuda.current_stream().wait_stream(self._bwd_stream)
        return done

    def is_complete(self) -> bool:
        """True once the worker has processed all sub-ops."""
        return self._all_done.is_set()

    @property
    def bubble_utilization(self) -> float:
        """Fraction of sub-ops dispatched inside a prefill bubble (0–1)."""
        if self._n_ops == 0:
            return 0.0
        return self.subops_in_bubble / self._n_ops

    def summary(self) -> str:
        return (
            f"VllmBubbleScheduler: "
            f"{self.subops_in_bubble}/{self._n_ops} in-bubble "
            f"({self.bubble_utilization:.1%}), "
            f"{self.subops_after} after"
        )

    def __repr__(self) -> str:
        return (
            f"VllmBubbleScheduler("
            f"n_ops={self._n_ops}, "
            f"cursor={self._cursor}, "
            f"in_bubble={self.subops_in_bubble}, "
            f"after={self.subops_after})"
        )


# ── Green context SM partitioning ────────────────────────────────────────────

_green_ctx_stream_cache: dict[int, torch.cuda.ExternalStream] = {}
_green_ctx_cache: dict[int, object] = {}


def _get_green_ctx_stream(device: int, n_sms: int) -> torch.cuda.ExternalStream:
    """Get or create a green context stream with `n_sms` dedicated SMs.

    Cached per device — all GreenCtxScheduler instances on the same device
    share one green context and stream, since only one scheduler is active
    at a time (enforced by arm/disarm in moe_runner.py).
    """
    if device in _green_ctx_stream_cache:
        return _green_ctx_stream_cache[device]

    from cuda.bindings import driver as drv

    err, = drv.cuInit(0)
    err, cu_dev = drv.cuDeviceGet(device)

    err, resource = drv.cuDeviceGetDevResource(
        cu_dev, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
    total_sms = resource.sm.smCount

    err, groups, actual, remainder = drv.cuDevSmResourceSplitByCount(
        1, resource, 0, n_sms)

    if err != drv.CUresult.CUDA_SUCCESS or actual < 1:
        log.warning(
            "[GreenCtx] cuDevSmResourceSplitByCount(%d SMs) failed: %s, "
            "falling back to full-resource green context", n_sms, err)
        err, desc = drv.cuDevResourceGenerateDesc([resource], 1)
        allocated_sms = total_sms
    else:
        trn_resource = groups[0]
        allocated_sms = trn_resource.sm.smCount
        err, desc = drv.cuDevResourceGenerateDesc([trn_resource], 1)

    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"[GreenCtx] cuDevResourceGenerateDesc failed: {err}")

    err, green_ctx = drv.cuGreenCtxCreate(
        desc, cu_dev,
        drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"[GreenCtx] cuGreenCtxCreate failed: {err}")

    err, cuda_ctx = drv.cuCtxFromGreenCtx(green_ctx)
    drv.cuCtxPushCurrent(cuda_ctx)
    err, raw_stream = drv.cuStreamCreate(0)
    drv.cuCtxPopCurrent()

    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"[GreenCtx] cuStreamCreate in green ctx failed: {err}")

    pt_stream = torch.cuda.ExternalStream(int(raw_stream))

    _green_ctx_stream_cache[device] = pt_stream
    _green_ctx_cache[device] = (green_ctx, raw_stream)

    log.info("[GreenCtx] Created green context: %d/%d SMs for training "
             "on device %d", allocated_sms, total_sms, device)
    return pt_stream


class GreenCtxScheduler:
    """Run training sub-ops continuously on a partitioned green context.

    Drop-in replacement for VllmBubbleScheduler. Instead of waiting for
    fill_one() triggers at EP bubble boundaries, sub-ops run back-to-back
    on a CUDA stream that lives inside a green context with dedicated SMs.

    The green context provides hardware-level SM partitioning: training
    kernels execute on their dedicated SMs without competing with inference
    for the GPU scheduler's attention. No bubble gating, no priority
    scheduling — just spatial partitioning.

    fill_one() / pause_decode() / resume_prefill() are kept as no-ops for
    interface compatibility (moe_runner.py calls them unconditionally).
    """

    def __init__(
        self,
        sub_ops: list[Callable[[], None]],
        device: int | None = None,
        post_complete_fn: Callable[[], None] | None = None,
        label: str = "green",
    ):
        if device is not None:
            torch.cuda.set_device(device)
        self._device = device if device is not None else torch.cuda.current_device()

        self._sub_ops = list(sub_ops)
        self._n_ops = len(self._sub_ops)
        self._post_complete_fn = post_complete_fn
        self._label = label

        self.subops_in_bubble = 0
        self.subops_after = self._n_ops

        self._all_done = threading.Event()
        self._bwd_stream = _get_green_ctx_stream(self._device, _GREEN_CTX_SMS)

        self._worker = threading.Thread(
            target=self._run_worker,
            daemon=True,
        )
        self._worker.start()

    def _run_worker(self) -> None:
        torch.cuda.set_device(self._device)
        _timing = _FT_TIMING
        _t_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        with torch.inference_mode(False):
            for i, sub_op in enumerate(self._sub_ops):
                with torch.cuda.stream(self._bwd_stream):
                    if _timing:
                        t0 = torch.cuda.Event(enable_timing=True)
                        t1 = torch.cuda.Event(enable_timing=True)
                        t0.record()
                        sub_op()
                        t1.record()
                        _t_events.append(
                            (getattr(sub_op, '__name__', f'op{i}'), t0, t1)
                        )
                    else:
                        sub_op()
            self._bwd_stream.synchronize()
        if _timing and _t_events:
            _flush_subop_timings(self._label, _t_events)
        if self._post_complete_fn is not None:
            self._post_complete_fn()
        self._all_done.set()

    # ── Interface compatibility with VllmBubbleScheduler ─────────────────────

    @property
    def n_ops(self) -> int:
        return self._n_ops

    def has_work(self) -> bool:
        return not self._all_done.is_set()

    def fill_one(
        self,
        in_bubble: bool = True,
        sync_event: "torch.cuda.Event | None" = None,
    ) -> bool:
        return False

    def pause_decode(self) -> "torch.cuda.Event | None":
        return None

    def resume_prefill(self) -> None:
        pass

    def fill_remaining(self, sync_event: "torch.cuda.Event | None" = None) -> int:
        return 0

    def wait(self, timeout: float = 120.0) -> bool:
        done = self._all_done.wait(timeout=timeout)
        torch.cuda.current_stream().wait_stream(self._bwd_stream)
        return done

    def is_complete(self) -> bool:
        return self._all_done.is_set()

    @property
    def bubble_utilization(self) -> float:
        return 0.0

    def summary(self) -> str:
        return (
            f"GreenCtxScheduler: "
            f"{self._n_ops} ops, "
            f"{'done' if self._all_done.is_set() else 'running'}, "
            f"{_GREEN_CTX_SMS} SMs"
        )

    def __repr__(self) -> str:
        return (
            f"GreenCtxScheduler("
            f"n_ops={self._n_ops}, "
            f"sms={_GREEN_CTX_SMS}, "
            f"done={self._all_done.is_set()})"
        )


# ── Factory ──────────────────────────────────────────────────────────────────

def make_scheduler(
    sub_ops: list[Callable[[], None]],
    device: int | None = None,
    post_complete_fn: Callable[[], None] | None = None,
    label: str = "sched",
    is_forward: bool = False,
) -> VllmBubbleScheduler | GreenCtxScheduler:
    """Create the appropriate scheduler.

    Backward scheduler type is controlled by VLLM_FT_SCHED_MODE.
    Forward scheduler uses a green context stream when VLLM_FT_FWD_GREEN_CTX=1
    (the forward runs during decode, competing with inference — SM isolation
    helps there; the backward runs in EP bubbles where the GPU is idle).
    """
    # Backward: green_ctx mode → continuous execution, no bubble gating
    if not is_forward and _SCHED_MODE == "green_ctx":
        return GreenCtxScheduler(
            sub_ops, device=device,
            post_complete_fn=post_complete_fn, label=label,
        )

    # Forward with green context: run continuously on a partitioned stream.
    # SM isolation means no bubble gating needed — the dedicated SMs can't
    # interfere with inference beyond their partition.
    if is_forward and _FWD_GREEN_CTX:
        return GreenCtxScheduler(
            sub_ops, device=device,
            post_complete_fn=post_complete_fn, label=label,
        )

    # Default: regular low-priority stream with bubble gating
    return VllmBubbleScheduler(
        sub_ops, device=device,
        post_complete_fn=post_complete_fn, label=label,
    )
