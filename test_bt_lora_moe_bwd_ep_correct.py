"""
Standalone TP=2/EP=2 correctness test for bt_lora_trainer.py item 9 — MoE
backward EP-correctness — plus the two companion gaps found while
implementing it (see Sessions_10_6_2026.md):

  - "moefwd_{i}":  _training_moe_forward returns a local-experts-only
                   partial output in the non-C+D_batch path (item 8 only
                   fixed the C+D_batch path) — exchange-summed.
  - "moebwd_{i}":  _moe_backward_passthrough's grad_x2 only accumulates the
                   local experts' contributions — exchange-summed in the
                   last passthrough chunk op (item 9 proper).
  - "attnbwd_{i}": _attn_qkv_bwd's grad_x_norm only carries the local Q/V
                   heads' contributions, so grad_hidden propagated below the
                   top layer was partial even with items 1-8 ON (gap not in
                   the original 14-item inventory).

Unlike test_bt_lora_tp_correctness.py (autograd `_forward_loss`, which only
exercises the LAST layer's LoRA grads), this test drives the full *manual*
production sub-op chain — build_fwd_subops() then build_bwd_subops() — on a
toy 2-layer MoE model with a vocab-sharded lm_head, and compares EVERY
layer's LoRA gradients (and the CE loss) against a TP=1/EP=1 reference
trainer running the same manual chain with the unsharded weights and all
experts local.

  - VLLM_FT_TP_CORRECT=ON:  all layers' reconstructed grads match the ref.
  - VLLM_FT_TP_CORRECT=OFF: lower-layer grads diverge (fix is load-bearing).

Run (2 GPUs, or CPU via gloo):
    torchrun --nproc_per_node=2 test_bt_lora_moe_bwd_ep_correct.py
"""
import os

# Generous exchange timeout: the two test processes drift while driving the
# long sub-op chains; production uses 50ms, correctness is what's under test.
os.environ.setdefault("VLLM_FT_TP_CORRECT_TIMEOUT_MS", "5000")

import sys
import tempfile
import types

sys.path.insert(0, '/mnt/nfs/home/ramya/vllm')

import torch
import torch.distributed as dist

from test_bt_lora_unit import (
    _make_toy_safetensors, _PatchToyDims,
    TOY_H, TOY_HEADS, TOY_KV, TOY_HD, TOY_LAYERS, TOY_VOCAB, TOY_T,
)
from test_bt_lora_tp_correctness import _build_trainer, Q_SZ

SEED    = 4321
SCALE   = 0.2    # keep activations O(1) through the un-normalised mock norms
MOE_SCALE = 0.05
E       = 4      # total experts (2 per rank under EP=2)
TOPK    = 2
D_INTER = 16


def _make_moe(gate_w, w13, w2):
    return types.SimpleNamespace(
        gate=types.SimpleNamespace(weight=gate_w),
        experts=types.SimpleNamespace(top_k=TOPK, w13_weight=w13, w2_weight=w2),
    )


def _attach_moe(trainer, gate_w, w13_full, w2_full, ep_rank, ep_size):
    e_local = E // ep_size
    for layer in trainer._layers:
        layer.mlp = _make_moe(
            gate_w,
            w13_full[ep_rank * e_local:(ep_rank + 1) * e_local].contiguous(),
            w2_full[ep_rank * e_local:(ep_rank + 1) * e_local].contiguous(),
        )
    trainer._ep_size              = ep_size
    trainer._ep_expert_start      = ep_rank * e_local
    trainer._ep_num_local_experts = e_local


def _run_manual_chain(trainer) -> float:
    """Drive the production manual path end-to-end: forward sub-op chain,
    backward sub-op chain, and the real optimizer trigger (accum_steps=1) so
    the streamed per-layer grad exchanges (_publish_layer_grads → futures
    consumed in _all_reduce_replicated_grads) and the TP-consistent grad
    clip run exactly as in production. optimizer.step / zero_grad are
    no-op'd so the post-exchange grads survive for comparison; the CE loss
    is captured before optimizer_step resets total_loss."""
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.accum_steps = 1
    trainer.optimizer.step = lambda *a, **k: None
    trainer.optimizer.zero_grad = lambda *a, **k: None
    captured = {}
    orig_step = trainer.optimizer_step

    def _capturing_step():
        captured["loss"] = trainer.total_loss
        orig_step()
    trainer.optimizer_step = _capturing_step

    for op in trainer.build_fwd_subops():
        op()
    assert "hidden_last" in trainer._fwd, "forward sub-op chain did not complete"
    for op in trainer.build_bwd_subops():
        op()
    assert "loss" in captured, "optimizer_step was never triggered"
    return captured["loss"]


def _all_grads(trainer) -> dict:
    out = {}
    for li in sorted(trainer._lora.keys()):
        for key, p in trainer._lora[li].items():
            out[f"{li}/{key}"] = (p.grad.detach().cpu().clone()
                                   if p.grad is not None
                                   else torch.zeros_like(p).cpu())
    return out


def _full_grad(g0: dict, g1: dict, key: str) -> torch.Tensor:
    """Reconstruct the TP=1-equivalent gradient from the two ranks' grads
    (same sharding convention as test_bt_lora_tp_correctness._full_grad,
    keyed by 'layer/proj.lora_X')."""
    proj_key = key.split("/", 1)[1]
    if proj_key in ("q_proj.lora_B", "k_proj.lora_B", "v_proj.lora_B"):
        return torch.cat([g0[key], g1[key]], dim=0)   # row-sharded
    if proj_key == "o_proj.lora_A":
        return torch.cat([g0[key], g1[key]], dim=1)   # column-sharded
    # Replicated (q/k/v_proj.lora_A, o_proj.lora_B): identical post-exchange.
    return g0[key]


def _check(results: dict) -> bool:
    ok = True
    for tag, r in results.items():
        print(f"\n=== VLLM_FT_TP_CORRECT={tag} (manual sub-op chain, MoE) ===")
        print(f"  loss: ref={r['ref_loss']:.6f}  "
              f"tp2_r0={r['tp2'][0]['loss']:.6f}  tp2_r1={r['tp2'][1]['loss']:.6f}")
        g0, g1, gref = r["tp2"][0]["grads"], r["tp2"][1]["grads"], r["ref_grads"]
        all_match = True
        any_nonzero_ref = False
        for key in sorted(gref):
            full  = _full_grad(g0, g1, key)
            match = torch.allclose(full, gref[key], atol=1e-4, rtol=1e-3)
            all_match &= match
            any_nonzero_ref |= bool(gref[key].abs().max().item() > 0)
            diff = (full - gref[key]).abs().max().item()
            print(f"  {key:22s} match={match!s:5}  max_abs_diff={diff:.3e}")
        loss_match = (abs(r["tp2"][0]["loss"] - r["ref_loss"]) < 1e-4 and
                      abs(r["tp2"][1]["loss"] - r["ref_loss"]) < 1e-4)

        if not any_nonzero_ref:
            ok = False
            print("  FAIL: reference grads are all zero — test is vacuous")
        elif tag == "ON":
            if not (all_match and loss_match):
                ok = False
                print("  FAIL: expected all layers' grads + loss to match the "
                       "TP=1 reference with VLLM_FT_TP_CORRECT=1")
            else:
                print("  PASS")
        else:
            if all_match and loss_match:
                ok = False
                print("  FAIL: expected MISMATCH with VLLM_FT_TP_CORRECT=0 "
                       "(fix would be vacuous)")
            else:
                print("  PASS (mismatch confirmed, as expected without the fix)")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "this test requires --nproc_per_node=2"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend)
    dist.barrier()
    train_tp_pg = dist.new_group(ranks=[0, 1], backend=backend)
    tp_correct_store = dist.distributed_c10d._get_default_store()

    with _PatchToyDims():
        # ── Shared (seeded) full weights — identical across ranks ───────────
        g = torch.Generator().manual_seed(SEED)
        qkv_full = [(torch.randn(Q_SZ + 2 * (TOY_KV * TOY_HD), TOY_H, generator=g) * SCALE).to(device)
                     for _ in range(TOY_LAYERS)]
        o_full   = [(torch.randn(TOY_H, Q_SZ, generator=g) * SCALE).to(device)
                     for _ in range(TOY_LAYERS)]
        embed_w  = (torch.randn(TOY_VOCAB, TOY_H, generator=g) * SCALE).to(device)
        lm_w     = (torch.randn(TOY_VOCAB, TOY_H, generator=g) * SCALE).to(device)
        gate_w   = (torch.randn(E, TOY_H, generator=g) * SCALE).to(device)
        w13_full = (torch.randn(E, 2 * D_INTER, TOY_H, generator=g) * MOE_SCALE).to(device)
        w2_full  = (torch.randn(E, TOY_H, D_INTER, generator=g) * MOE_SCALE).to(device)

        # ── TP=2 shards for this rank ────────────────────────────────────────
        nh_local, nkv_local = TOY_HEADS // 2, max(1, TOY_KV // 2)
        q_sz_local, kv_sz_local = nh_local * TOY_HD, nkv_local * TOY_HD
        kv_sz = TOY_KV * TOY_HD

        qkv_local, o_local = [], []
        for i in range(TOY_LAYERS):
            Qf = qkv_full[i][:Q_SZ]
            Kf = qkv_full[i][Q_SZ:Q_SZ + kv_sz]
            Vf = qkv_full[i][Q_SZ + kv_sz:]
            Qr = Qf[rank * q_sz_local:(rank + 1) * q_sz_local]
            Kr = Kf[rank * kv_sz_local:(rank + 1) * kv_sz_local]
            Vr = Vf[rank * kv_sz_local:(rank + 1) * kv_sz_local]
            qkv_local.append(torch.cat([Qr, Kr, Vr], dim=0).contiguous())
            o_local.append(o_full[i][:, rank * q_sz_local:(rank + 1) * q_sz_local].contiguous())

        # Vocab-sharded lm_head for TP=2 (production ParallelLMHead layout —
        # exercises item 7 inside the chain); the TP=1 ref keeps the full lm_w.
        v_half     = TOY_VOCAB // 2
        lm_w_local = lm_w[rank * v_half:(rank + 1) * v_half].contiguous()

        adapter_path = os.path.join(tempfile.gettempdir(),
                                     "bt_lora_moe_bwd_test_adapter.safetensors")
        if rank == 0:
            _make_toy_safetensors(adapter_path, nonzero_b=True)
        dist.barrier()

        # ── Identical batch (broadcast from rank 0, seeded for determinism) ──
        if rank == 0:
            batch = torch.randint(0, TOY_VOCAB, (1, TOY_T), generator=g).to(device)
        else:
            batch = torch.empty((1, TOY_T), dtype=torch.long, device=device)
        dist.broadcast(batch, src=0, group=train_tp_pg)
        input_ids, labels = batch, batch.clone()

        results = {}
        for tp_correct in (True, False):
            tag = "ON" if tp_correct else "OFF"

            t2 = _build_trainer(2, rank, qkv_local, o_local, embed_w, lm_w_local,
                                 adapter_path, device, input_ids, labels,
                                 train_tp_pg=train_tp_pg,
                                 tp_correct_store=tp_correct_store,
                                 tp_correct=tp_correct)
            _attach_moe(t2, gate_w, w13_full, w2_full, ep_rank=rank, ep_size=2)
            loss2  = _run_manual_chain(t2)
            grads2 = _all_grads(t2)

            gathered: list = [None, None]
            dist.all_gather_object(
                gathered, {"loss": loss2, "grads": grads2}, group=train_tp_pg)

            if rank == 0:
                t1 = _build_trainer(1, 0, qkv_full, o_full, embed_w, lm_w,
                                     adapter_path, device, input_ids, labels,
                                     train_tp_pg=None, tp_correct=False)
                _attach_moe(t1, gate_w, w13_full, w2_full, ep_rank=0, ep_size=1)
                loss1  = _run_manual_chain(t1)
                grads1 = _all_grads(t1)
                results[tag] = {"ref_loss": loss1, "ref_grads": grads1,
                                 "tp2": gathered}
            dist.barrier()

        ok = _check(results) if rank == 0 else True

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0 and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
