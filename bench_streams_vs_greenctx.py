#!/usr/bin/env python3
"""
Microbenchmark: CUDA Streams vs Green Contexts for concurrent kernel placement.

Runs two matmuls concurrently — one memory-bound, one compute-bound — and
compares overlap behavior under:

  1. Serial baseline (each kernel alone)
  2. Two CUDA streams (same priority, same context)
  3. Two CUDA streams with different priorities (BubbleTea's current approach)
  4. Two green contexts with SM partitioning (via cuDevSmResourceSplitByCount)

SM partitioning on A100: minimum 4 SMs per partition (DEFAULT) or 2 SMs
(IGNORE_SM_COSCHEDULING). The remainder resource gets its own green context.

Usage:
    python bench_streams_vs_greenctx.py [--device 0] [--warmup 20] [--iters 100]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from cuda.bindings import driver as drv


# ── Helpers ──────────────────────────────────────────────────────────────────

def check(err, msg="CUDA driver error"):
    if isinstance(err, tuple):
        err = err[0]
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{msg}: {err}")


@dataclass
class KernelSpec:
    name: str
    M: int
    K: int
    N: int

    def describe(self) -> str:
        flops = 2 * self.M * self.K * self.N
        bytes_rw = (self.M * self.K + self.K * self.N + self.M * self.N) * 2
        ai = flops / bytes_rw
        return (f"{self.name}: [{self.M},{self.K}]×[{self.K},{self.N}]  "
                f"FLOPs={flops/1e9:.2f}G  bytes={bytes_rw/1e6:.1f}MB  "
                f"AI={ai:.1f}")


def make_tensors(spec: KernelSpec, device: torch.device):
    A = torch.randn(spec.M, spec.K, device=device, dtype=torch.bfloat16)
    B = torch.randn(spec.K, spec.N, device=device, dtype=torch.bfloat16)
    return A, B


def stats(xs):
    m = sum(xs) / len(xs)
    s = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return m, s


def timed_matmul(A, B, stream, n_warmup=5, n_iters=50):
    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            torch.mm(A, B)
    stream.synchronize()

    t0s, t1s = [], []
    for _ in range(n_iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            t0.record(stream)
            torch.mm(A, B)
            t1.record(stream)
        t0s.append(t0)
        t1s.append(t1)
    stream.synchronize()
    return stats([a.elapsed_time(b) for a, b in zip(t0s, t1s)])


# ── Green context with SM partitioning ───────────────────────────────────────

def get_num_sms(dev: int) -> int:
    err, n = drv.cuDeviceGetAttribute(
        drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)
    check(err)
    return n


class PartitionedGreenCtx:
    """A green context created from a split SM resource."""

    def __init__(self, dev: int, sm_resource: "drv.CUdevResource", label: str = ""):
        self.label = label
        self.sm_count = sm_resource.sm.smCount

        err, desc = drv.cuDevResourceGenerateDesc([sm_resource], 1)
        check(err, f"cuDevResourceGenerateDesc ({label})")

        err, self._green_ctx = drv.cuGreenCtxCreate(
            desc, dev, drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM)
        check(err, f"cuGreenCtxCreate ({label})")

        err, self._cuda_ctx = drv.cuCtxFromGreenCtx(self._green_ctx)
        check(err, "cuCtxFromGreenCtx")

        drv.cuCtxPushCurrent(self._cuda_ctx)
        err, self._raw_stream = drv.cuStreamCreate(0)
        check(err, "cuStreamCreate")
        drv.cuCtxPopCurrent()

        self.pt_stream = torch.cuda.ExternalStream(int(self._raw_stream))

    def destroy(self):
        drv.cuStreamDestroy(self._raw_stream)
        drv.cuGreenCtxDestroy(self._green_ctx)


def split_sms(dev: int, n_groups: int, min_count: int,
              ignore_cosched: bool = False):
    """Split device SMs into n_groups + remainder.

    Returns (list[CUdevResource], remainder_CUdevResource).
    """
    err, resource = drv.cuDeviceGetDevResource(
        dev, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
    check(err)

    flags = (drv.CUdevSmResourceSplit_flags.CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING
             if ignore_cosched else 0)

    err, groups, actual, remainder = drv.cuDevSmResourceSplitByCount(
        n_groups, resource, flags, min_count)
    check(err, "cuDevSmResourceSplitByCount")

    return groups[:actual], remainder, actual


# ── Benchmark routines ───────────────────────────────────────────────────────

def bench_concurrent(A1, B1, A2, B2, stream1, stream2, n_warmup, n_iters):
    for _ in range(n_warmup):
        with torch.cuda.stream(stream1):
            torch.mm(A1, B1)
        with torch.cuda.stream(stream2):
            torch.mm(A2, B2)
    stream1.synchronize()
    stream2.synchronize()

    wall_times, k1_times, k2_times = [], [], []
    for _ in range(n_iters):
        t0_1 = torch.cuda.Event(enable_timing=True)
        t1_1 = torch.cuda.Event(enable_timing=True)
        t0_2 = torch.cuda.Event(enable_timing=True)
        t1_2 = torch.cuda.Event(enable_timing=True)
        wall_start = torch.cuda.Event(enable_timing=True)
        wall_end_1 = torch.cuda.Event(enable_timing=True)
        wall_end_2 = torch.cuda.Event(enable_timing=True)

        wall_start.record()

        with torch.cuda.stream(stream1):
            stream1.wait_event(wall_start)
            t0_1.record(stream1)
            torch.mm(A1, B1)
            t1_1.record(stream1)
            wall_end_1.record(stream1)

        with torch.cuda.stream(stream2):
            stream2.wait_event(wall_start)
            t0_2.record(stream2)
            torch.mm(A2, B2)
            t1_2.record(stream2)
            wall_end_2.record(stream2)

        stream1.synchronize()
        stream2.synchronize()

        k1_times.append(t0_1.elapsed_time(t1_1))
        k2_times.append(t0_2.elapsed_time(t1_2))
        wall_times.append(max(
            wall_start.elapsed_time(wall_end_1),
            wall_start.elapsed_time(wall_end_2),
        ))

    return stats(k1_times), stats(k2_times), stats(wall_times)


def print_result(label1, k1m, k1s, b1, label2, k2m, k2s, b2, wm, ws, serial_sum):
    overlap = 1.0 - wm / serial_sum if serial_sum > 0 else 0
    print(f"    {label1:25s}: {k1m:7.3f} ± {k1s:.3f} ms  (vs solo: {k1m/b1:.2f}×)")
    print(f"    {label2:25s}: {k2m:7.3f} ± {k2s:.3f} ms  (vs solo: {k2m/b2:.2f}×)")
    print(f"    {'Wall time':25s}: {wm:7.3f} ± {ws:.3f} ms")
    print(f"    {'Overlap':25s}: {overlap:7.1%}  "
          f"(serial={serial_sum:.3f}ms, perfect={max(b1,b2):.3f}ms)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    dev = args.device
    n_warmup = args.warmup
    n_iters = args.iters

    err, = drv.cuInit(0)
    check(err)
    err, cu_dev = drv.cuDeviceGet(dev)
    check(err)

    torch.cuda.set_device(dev)
    torch.cuda.init()
    _ = torch.zeros(1, device=f"cuda:{dev}")

    num_sms = get_num_sms(cu_dev)
    err, cc_major = drv.cuDeviceGetAttribute(
        drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_dev)
    err, cc_minor = drv.cuDeviceGetAttribute(
        drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_dev)
    print(f"Device {dev}: {torch.cuda.get_device_name(dev)}, "
          f"cc={cc_major}.{cc_minor}, {num_sms} SMs")
    print(f"Warmup={n_warmup}, Iters={n_iters}")
    print()

    # ── Kernel specs ─────────────────────────────────────────────────────────
    mem = KernelSpec("mem_bound", M=16384, K=64, N=64)
    comp = KernelSpec("compute_bound", M=4096, K=4096, N=4096)

    print(mem.describe())
    print(comp.describe())
    print()

    Am, Bm = make_tensors(mem, torch.device(f"cuda:{dev}"))
    Ac, Bc = make_tensors(comp, torch.device(f"cuda:{dev}"))

    default_stream = torch.cuda.default_stream(dev)

    # ── 1. Serial baseline ───────────────────────────────────────────────────
    print("=" * 76)
    print("1. SERIAL BASELINE")
    print("=" * 76)
    bm, bm_s = timed_matmul(Am, Bm, default_stream, n_warmup, n_iters)
    bc, bc_s = timed_matmul(Ac, Bc, default_stream, n_warmup, n_iters)
    serial_sum = bm + bc
    print(f"  mem_bound:     {bm:7.3f} ± {bm_s:.3f} ms")
    print(f"  compute_bound: {bc:7.3f} ± {bc_s:.3f} ms")
    print(f"  Sum:           {serial_sum:7.3f} ms")
    print()

    # ── 2. Two streams, same priority ────────────────────────────────────────
    print("=" * 76)
    print("2. TWO STREAMS — same priority")
    print("=" * 76)
    s_a = torch.cuda.Stream(dev)
    s_b = torch.cuda.Stream(dev)
    (k1m, k1s), (k2m, k2s), (wm, ws) = bench_concurrent(
        Ac, Bc, Am, Bm, s_a, s_b, n_warmup, n_iters)
    print_result("compute_bound", k1m, k1s, bc,
                 "mem_bound", k2m, k2s, bm, wm, ws, serial_sum)
    print()

    # ── 3. Two streams, hi/lo priority ───────────────────────────────────────
    print("=" * 76)
    print("3. TWO STREAMS — hi-pri (inference) + lo-pri (training)")
    print("=" * 76)
    lo, hi = torch.cuda.Stream.priority_range()
    s_hipri = torch.cuda.Stream(dev, priority=lo)
    s_lopri = torch.cuda.Stream(dev, priority=hi)
    print(f"  Priority range: [{lo}, {hi}]")
    print()
    print("  [compute=hi-pri, mem=lo-pri]")
    (k1m, k1s), (k2m, k2s), (wm, ws) = bench_concurrent(
        Ac, Bc, Am, Bm, s_hipri, s_lopri, n_warmup, n_iters)
    print_result("compute (hi-pri)", k1m, k1s, bc,
                 "mem (lo-pri)", k2m, k2s, bm, wm, ws, serial_sum)
    print()

    # ── 4. Green contexts with SM partitioning ───────────────────────────────
    print("=" * 76)
    print("4. GREEN CONTEXTS — SM partitioning via cuDevSmResourceSplitByCount")
    print("=" * 76)

    # Configurations: (training_groups, training_minCount, ignore_cosched)
    # We give training N groups of M SMs, inference gets the remainder.
    configs = [
        (1, 4,  False, "training=1×4=4 SMs"),
        (2, 4,  False, "training=2×4=8 SMs"),
        (1, 8,  False, "training=1×8=8 SMs"),
        (1, 16, False, "training=1×16=16 SMs"),
        (1, 32, False, "training=1×32=32 SMs"),
        (2, 4,  True,  "training=2×2=4 SMs (no cosched)"),
    ]

    for n_trn_groups, min_count, ignore_cosched, desc_str in configs:
        # Split: n_trn_groups partitions for training, remainder for inference.
        # We request n_trn_groups+1 total so we can assign remainder to inference.
        try:
            groups, remainder, actual = split_sms(
                cu_dev, n_trn_groups, min_count, ignore_cosched)

            if actual < n_trn_groups:
                print(f"  [{desc_str}]: only got {actual}/{n_trn_groups} groups, skipping")
                continue

            trn_sms = sum(g.sm.smCount for g in groups[:n_trn_groups])
            inf_sms = remainder.sm.smCount

            print(f"  [{desc_str}]  inference={inf_sms} SMs (remainder), "
                  f"training={trn_sms} SMs ({n_trn_groups}×{groups[0].sm.smCount})")

            # Create green contexts
            gc_inf = PartitionedGreenCtx(cu_dev, remainder, label="inference")

            # For training, use the first group (if multiple groups, they'd need
            # merging or separate contexts; we use just group[0] for simplicity)
            gc_trn = PartitionedGreenCtx(cu_dev, groups[0], label="training")

            (k1m, k1s), (k2m, k2s), (wm, ws) = bench_concurrent(
                Ac, Bc, Am, Bm,
                gc_inf.pt_stream, gc_trn.pt_stream,
                n_warmup, n_iters)
            print_result(f"compute (inf {inf_sms}SM)", k1m, k1s, bc,
                         f"mem (trn {gc_trn.sm_count}SM)", k2m, k2s, bm,
                         wm, ws, serial_sum)

            gc_inf.destroy()
            gc_trn.destroy()

        except RuntimeError as e:
            print(f"  [{desc_str}]: FAILED — {e}")
        print()

    # ── 5. Reversed: compute on small partition, mem on large ────────────────
    print("=" * 76)
    print("5. GREEN CONTEXTS — reversed (compute on small, mem on large)")
    print("=" * 76)
    for n_trn_groups, min_count, ignore_cosched, desc_str in configs[:4]:
        try:
            groups, remainder, actual = split_sms(
                cu_dev, n_trn_groups, min_count, ignore_cosched)
            if actual < n_trn_groups:
                continue

            small_sms = groups[0].sm.smCount
            large_sms = remainder.sm.smCount

            gc_small = PartitionedGreenCtx(cu_dev, groups[0], label="small")
            gc_large = PartitionedGreenCtx(cu_dev, remainder, label="large")

            print(f"  [compute on {small_sms} SMs, mem on {large_sms} SMs]")
            (k1m, k1s), (k2m, k2s), (wm, ws) = bench_concurrent(
                Ac, Bc, Am, Bm,
                gc_small.pt_stream, gc_large.pt_stream,
                n_warmup, n_iters)
            print_result(f"compute ({small_sms}SM)", k1m, k1s, bc,
                         f"mem ({large_sms}SM)", k2m, k2s, bm,
                         wm, ws, serial_sum)

            gc_small.destroy()
            gc_large.destroy()
        except RuntimeError as e:
            print(f"  FAILED: {e}")
        print()

    # ── 6. Kernel size sweep ─────────────────────────────────────────────────
    print("=" * 76)
    print("6. KERNEL SIZE SWEEP — streams vs green ctx (training=4 SMs, inference=100 SMs)")
    print("=" * 76)

    # Create partitioned contexts for the sweep
    groups, remainder, _ = split_sms(cu_dev, 1, 4, False)
    gc_inf_sweep = PartitionedGreenCtx(cu_dev, remainder, label="inf_sweep")
    gc_trn_sweep = PartitionedGreenCtx(cu_dev, groups[0], label="trn_sweep")

    print(f"  {'mem_shape':>22s}  {'approach':>14s}  {'mem_ms':>8s}  "
          f"{'comp_ms':>8s}  {'wall_ms':>8s}  {'overlap':>8s}")
    print("  " + "-" * 80)

    mem_specs = [
        KernelSpec("mem_tiny",  M=1024,  K=64,  N=64),
        KernelSpec("mem_small", M=4096,  K=64,  N=64),
        KernelSpec("mem_med",   M=16384, K=64,  N=64),
        KernelSpec("mem_large", M=16384, K=256, N=256),
        KernelSpec("mem_xlg",   M=16384, K=512, N=512),
    ]

    for mspec in mem_specs:
        Am_s, Bm_s = make_tensors(mspec, torch.device(f"cuda:{dev}"))
        bm_s, _ = timed_matmul(Am_s, Bm_s, default_stream, n_warmup, n_iters)
        ss = bm_s + bc

        # Streams (same priority)
        (k1m, _), (k2m, _), (wm, _) = bench_concurrent(
            Ac, Bc, Am_s, Bm_s, s_a, s_b, n_warmup, n_iters)
        ov = 1.0 - wm / ss if ss > 0 else 0
        shape_str = f"[{mspec.M},{mspec.K}]×[{mspec.K},{mspec.N}]"
        print(f"  {shape_str:>22s}  {'streams':>14s}  {k2m:7.3f}  "
              f"{k1m:7.3f}  {wm:7.3f}  {ov:7.1%}")

        # Streams (priority)
        (k1m, _), (k2m, _), (wm, _) = bench_concurrent(
            Ac, Bc, Am_s, Bm_s, s_hipri, s_lopri, n_warmup, n_iters)
        ov = 1.0 - wm / ss if ss > 0 else 0
        print(f"  {'':>22s}  {'streams+pri':>14s}  {k2m:7.3f}  "
              f"{k1m:7.3f}  {wm:7.3f}  {ov:7.1%}")

        # Green ctx (4 SM training, 100 SM inference)
        (k1m, _), (k2m, _), (wm, _) = bench_concurrent(
            Ac, Bc, Am_s, Bm_s,
            gc_inf_sweep.pt_stream, gc_trn_sweep.pt_stream,
            n_warmup, n_iters)
        ov = 1.0 - wm / ss if ss > 0 else 0
        print(f"  {'':>22s}  {'green 100/4':>14s}  {k2m:7.3f}  "
              f"{k1m:7.3f}  {wm:7.3f}  {ov:7.1%}")
        print()

    gc_inf_sweep.destroy()
    gc_trn_sweep.destroy()

    print("Done.")


if __name__ == "__main__":
    main()
