"""
Safety test for VLLM_FT_TP_CORRECT's TCPStore-based exchange
(_tp_correct_exchange_sum, see bt_lora_trainer.py).

Validates the properties that motivate this design over a persistent
NCCL/Gloo ProcessGroup all-reduce on _train_tp_pg (which the bubble
scheduler's asymmetric per-rank sub-op firing could otherwise hang or
permanently desync):

  1. A symmetric call (both ranks publish the same round_id) produces the
     correct sum.
  2. A one-sided call (peer never publishes for this tag) times out within
     VLLM_FT_TP_CORRECT_TIMEOUT_MS and falls back to the local value, without
     hanging.
  3. A round_id mismatch (peer published a *different* round than ours --
     e.g. it's ahead of or behind us) is detected and falls back to the
     local value, rather than incorrectly summing values from different
     rounds.
  4. None of the above leaves the exchange permanently broken: a later
     symmetric call (reusing the same per-(tag,rank) keys with a new
     round_id) succeeds again.
  5. The default ProcessGroup (used by _train_tp_pg/sync_replicated_params
     in production) is completely unaffected throughout.

Run (CPU, no GPU needed):
    torchrun --nproc_per_node=2 test_bt_lora_tp_correct_safety.py
"""
import os

# Short timeout so the timeout-path check (step 2) doesn't take long. Must be
# set before importing bt_lora_trainer (read at module import time).
os.environ.setdefault("VLLM_FT_TP_CORRECT_TIMEOUT_MS", "200")

import sys
import time

sys.path.insert(0, '/mnt/nfs/home/ramya/vllm')

import torch
import torch.distributed as dist

import bt_lora_trainer as blt

TIMEOUT_S = blt._TP_CORRECT_TIMEOUT_MS / 1000.0


def _make_trainer(tp_rank, store):
    trainer = object.__new__(blt.BubbleTeaLoRATrainer)
    trainer._tp_rank = tp_rank
    trainer._tp_size = 2
    trainer._tp_correct_store = store
    return trainer


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "this test requires --nproc_per_node=2"

    dist.init_process_group(backend="gloo")
    dist.barrier()
    store = dist.distributed_c10d._get_default_store()
    trainer = _make_trainer(rank, store)
    ok = True

    # ── 1. Symmetric exchange ────────────────────────────────────────────
    t1 = torch.full((4,), float(rank + 1))
    out1 = trainer._tp_correct_exchange_sum(t1.clone(), "t1", round_id=0)
    expected1 = torch.full((4,), 3.0)  # (0+1) + (1+1)
    if not torch.allclose(out1, expected1):
        print(f"[rank {rank}] FAIL step1: got {out1}, expected {expected1}")
        ok = False
    else:
        print(f"[rank {rank}] step1 OK: {out1}")

    dist.barrier()

    # ── 2. One-sided call: peer never publishes for this tag ───────────────
    t2 = torch.full((4,), 7.0)
    if rank == 0:
        start = time.monotonic()
        out2 = trainer._tp_correct_exchange_sum(t2.clone(), "t2", round_id=5)
        elapsed = time.monotonic() - start
        if not torch.equal(out2, t2):
            print(f"[rank 0] FAIL step2: expected local fallback {t2}, got {out2}")
            ok = False
        elif elapsed > TIMEOUT_S * 5:
            print(f"[rank 0] FAIL step2: took {elapsed:.3f}s, expected ~{TIMEOUT_S}s")
            ok = False
        else:
            print(f"[rank 0] step2 OK: fell back to local value in {elapsed:.3f}s")
    else:
        # Deliberately skip the "t2" exchange to simulate skew, sleeping past
        # rank 0's timeout before continuing.
        time.sleep(TIMEOUT_S * 3)
        print("[rank 1] step2 OK: skipped exchange (simulating skew)")

    dist.barrier()

    # ── 3. round_id mismatch: both call, but with different round ids ──────
    t3 = torch.full((4,), 100.0 + rank)
    round_id3 = 10 if rank == 0 else 11
    start = time.monotonic()
    out3 = trainer._tp_correct_exchange_sum(t3.clone(), "t3", round_id=round_id3)
    elapsed = time.monotonic() - start
    if not torch.equal(out3, t3):
        print(f"[rank {rank}] FAIL step3: expected local fallback {t3}, got {out3}")
        ok = False
    elif elapsed > TIMEOUT_S * 5:
        print(f"[rank {rank}] FAIL step3: took {elapsed:.3f}s "
              f"(round-id mismatch should be detected without waiting)")
        ok = False
    else:
        print(f"[rank {rank}] step3 OK: fell back on round_id mismatch "
              f"in {elapsed:.3f}s")

    dist.barrier()

    # ── 4. Re-use "t1"'s keys with a new round_id -- must still work ───────
    t4 = torch.full((4,), float(rank + 10))
    out4 = trainer._tp_correct_exchange_sum(t4.clone(), "t1", round_id=1)
    expected4 = torch.full((4,), 21.0)  # (0+10) + (1+10)
    if not torch.allclose(out4, expected4):
        print(f"[rank {rank}] FAIL step4: got {out4}, expected {expected4}")
        ok = False
    else:
        print(f"[rank {rank}] step4 OK: {out4}")

    dist.barrier()

    # ── 5. Default ProcessGroup (train_tp_pg-equivalent) still works ───────
    x = torch.tensor([float(rank + 1)])
    start = time.monotonic()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    elapsed = time.monotonic() - start
    if not torch.equal(x, torch.tensor([3.0])) or elapsed > 5.0:
        print(f"[rank {rank}] FAIL step5: x={x}, elapsed={elapsed:.3f}s")
        ok = False
    else:
        print(f"[rank {rank}] step5 OK: default PG all-reduce succeeded "
              f"({x}, {elapsed:.3f}s)")

    dist.barrier()
    dist.destroy_process_group()

    print(f"[rank {rank}] " + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
