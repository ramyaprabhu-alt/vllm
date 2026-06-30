# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone TP=2 correctness test for bt_lora_trainer.py items 1, 3-6 and 7
(see /mnt/nfs/home/ramya/slora-plus/Sessions_10_6_2026.md):

  1. O-projection LoRA output all-reduce in _attn_lora_forward
  3-6. All-reduce of gradients for TP-replicated LoRA params
        (q/k/v_proj.lora_A, o_proj.lora_B) before the optimizer step
  7. Vocab-parallel cross-entropy in _lm_head_grad (sharded lm_head):
        full-vocab softmax denominator + grad_hidden partial-sum via the
        TCPStore exchange, vs. the local-shard-only CE approximation

All are gated by VLLM_FT_TP_CORRECT / trainer._tp_correct.

For each setting of _tp_correct (True, False), runs one forward+backward step
on a TP=2 toy model (one trainer instance per rank, sharded consistently with
_load_lora's TP convention) and compares the reconstructed full gradients
against a TP=1 reference trainer with the unsharded weights and identical
batch:

  - With _tp_correct=True:  TP=2 (reconstructed) == TP=1 reference.
  - With _tp_correct=False: TP=2 (reconstructed) != TP=1 reference
                             (demonstrates the fix matters).

Run (2 GPUs, or CPU via gloo if no GPU):
    torchrun --nproc_per_node=2 test_bt_lora_tp_correctness.py
"""

import os
import sys
import tempfile
import threading
import types

sys.path.insert(0, "/mnt/nfs/home/ramya/vllm")

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from test_bt_lora_unit import (
    TOY_H,
    TOY_HD,
    TOY_HEADS,
    TOY_KV,
    TOY_LAYERS,
    TOY_T,
    TOY_VOCAB,
    _make_toy_safetensors,
    _MockFusedNorm,
    _PatchToyDims,
)

import bubbletea.trainer as blt

Q_SZ = TOY_HEADS * TOY_HD  # 64
KV_SZ = TOY_KV * TOY_HD  # 32
SEED = 1234
LAST_LAYER = TOY_LAYERS - 1  # only this layer's LoRA params get gradient
# (see _forward_loss: loss depends only on the
# last layer's attn_out)


def _make_layer(qkv_w, o_w, device):
    return types.SimpleNamespace(
        self_attn=types.SimpleNamespace(
            qkv_proj=types.SimpleNamespace(weight=qkv_w),
            o_proj=types.SimpleNamespace(weight=o_w),
            q_norm=lambda x: x,
            k_norm=lambda x: x,
            rotary_emb=lambda pos, q, k: (q, k),
        ),
        input_layernorm=_MockFusedNorm(TOY_H, device),
        post_attention_layernorm=_MockFusedNorm(TOY_H, device),
        mlp=lambda x: x,
    )


class _Embed:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, ids):
        return self.weight[ids]


def _build_trainer(
    tp_size,
    tp_rank,
    qkv_layers,
    o_layers,
    embed_w,
    lm_w,
    adapter_path,
    device,
    input_ids,
    labels,
    train_tp_pg=None,
    tp_correct_store=None,
    tp_correct=False,
):
    """Construct a BubbleTeaLoRATrainer via object.__new__ (bypassing
    __init__'s vLLM-distributed dependencies), mirroring
    test_bt_lora_unit.py's _make_bare_trainer but parameterised for TP."""
    trainer = object.__new__(blt.BubbleTeaLoRATrainer)

    trainer.device = device.index if device.type == "cuda" else 0
    trainer._tp_size = tp_size
    trainer._tp_rank = tp_rank
    trainer._tp_group = None
    trainer._train_tp_pg = train_tp_pg
    trainer._tp_correct_store = tp_correct_store
    trainer._fwd_round = 0
    trainer._tp_correct = tp_correct
    trainer._do_param_sync = False
    trainer._pending_param_sync = False
    trainer._n_heads_local = TOY_HEADS // tp_size
    trainer._n_kv_local = max(1, TOY_KV // tp_size)
    trainer._step = 0
    trainer.accum_steps = 10**9  # never trigger optimizer_step here
    trainer.completed_steps = 0
    trainer.total_loss = 0.0
    trainer.t_ft = TOY_T
    trainer._lock = threading.Lock()
    trainer._fwd = {}
    trainer._lora = {}
    trainer._fwd_layer_state = None
    trainer._ep_expert_start = 0
    trainer._ep_num_local_experts = 0
    trainer._ep_size = 1
    trainer._bwd_passthrough_chunks = 1
    trainer._data_ready = threading.Event()
    trainer._data_ready.set()
    trainer._data_thread_started = True
    trainer._fwd_running = threading.Lock()
    trainer._fwd_last_done = 0.0
    trainer._FWD_COOLDOWN_S = 0.0

    layers = [
        _make_layer(qkv_layers[i], o_layers[i], device) for i in range(TOY_LAYERS)
    ]
    embed = _Embed(embed_w)
    lm_head = types.SimpleNamespace(weight=lm_w)
    trainer._layers = layers
    trainer._embed = embed
    trainer._norm = lambda x: x
    trainer._lm_head = lm_head
    trainer._model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            layers=layers, embed_tokens=embed, norm=trainer._norm
        ),
        lm_head=lm_head,
    )

    def _get_batch_mock():
        return input_ids.clone(), labels.clone()

    trainer._get_batch = _get_batch_mock

    # Patch load_file to load the (cpu-saved) adapter, like _make_bare_trainer.
    orig_load = blt.load_file
    try:
        blt.load_file = lambda p, device: load_file(p, device="cpu")
        trainer._load_lora(adapter_path)
    finally:
        blt.load_file = orig_load
    for layer_d in trainer._lora.values():
        for k in list(layer_d):
            layer_d[k] = layer_d[k].detach().to(device).requires_grad_(True)

    all_params = [p for d in trainer._lora.values() for p in d.values()]
    trainer.optimizer = torch.optim.AdamW(all_params, lr=2e-4, weight_decay=0.01)
    return trainer


def _run_step(trainer) -> float:
    trainer.optimizer.zero_grad(set_to_none=True)
    loss = trainer._forward_loss()
    loss.backward()
    trainer._all_reduce_replicated_grads()
    return loss.item()


def _grads(trainer) -> dict:
    out = {}
    for key, p in trainer._lora[LAST_LAYER].items():
        out[key] = (
            p.grad.detach().cpu().clone()
            if p.grad is not None
            else torch.zeros_like(p).cpu()
        )
    return out


def _full_grad(g0: dict, g1: dict, key: str) -> torch.Tensor:
    """Reconstruct the TP=1-equivalent gradient from the two ranks' TP=2 grads."""
    if key in ("q_proj.lora_B", "k_proj.lora_B", "v_proj.lora_B"):
        return torch.cat([g0[key], g1[key]], dim=0)  # row-sharded
    if key == "o_proj.lora_A":
        return torch.cat([g0[key], g1[key]], dim=1)  # column-sharded
    # Replicated (q/k/v_proj.lora_A, o_proj.lora_B): both ranks should hold
    # the same value once all-reduced.
    return g0[key]


def _check(results: dict) -> bool:
    ok = True
    for tag, r in results.items():
        print(f"\n=== VLLM_FT_TP_CORRECT={tag} ===")
        ref_loss = r["ref_loss"]
        loss_r0 = r["tp2"][0]["loss"]
        loss_r1 = r["tp2"][1]["loss"]
        print(f"  loss: ref={ref_loss:.6f}  tp2_r0={loss_r0:.6f}  tp2_r1={loss_r1:.6f}")

        g0, g1, gref = r["tp2"][0]["grads"], r["tp2"][1]["grads"], r["ref_grads"]
        all_match = True
        for key in gref:
            full = _full_grad(g0, g1, key)
            match = torch.allclose(full, gref[key], atol=1e-4, rtol=1e-3)
            all_match &= match
            diff = (full - gref[key]).abs().max().item()
            print(f"  {key:18s} match={match!s:5}  max_abs_diff={diff:.3e}")

        # Item 2: embedding all-reduce correction (_fwd_init).
        h0, h1, href = r["tp2"][0]["hidden"], r["tp2"][1]["hidden"], r["ref_hidden"]
        h0_match = torch.allclose(h0, href, atol=1e-4, rtol=1e-3)
        h1_match = torch.allclose(h1, href, atol=1e-4, rtol=1e-3)
        diff0 = (h0 - href).abs().max().item()
        diff1 = (h1 - href).abs().max().item()
        print(f"  {'embed (rank0)':18s} match={h0_match!s:5}  max_abs_diff={diff0:.3e}")
        print(f"  {'embed (rank1)':18s} match={h1_match!s:5}  max_abs_diff={diff1:.3e}")

        # Item 7: vocab-parallel CE (_lm_head_grad with a sharded lm_head).
        ref_lg, ref_ll = r["ref_lm_grad"], r["ref_lm_loss"]
        lm_match = []
        for rk in (0, 1):
            lg, ll = r["tp2"][rk]["lm_grad"], r["tp2"][rk]["lm_loss"]
            if lg is None:
                g_match, g_diff = False, float("nan")
            else:
                g_match = torch.allclose(lg, ref_lg, atol=1e-4, rtol=1e-3)
                g_diff = (lg - ref_lg).abs().max().item()
            l_match = abs(ll - ref_ll) < 1e-4
            lm_match.append(g_match and l_match)
            print(
                f"  {f'lm_head (rank{rk})':18s} match={g_match!s:5}  "
                f"max_abs_diff={g_diff:.3e}  "
                f"loss={ll:.6f} (ref={ref_ll:.6f}, match={l_match})"
            )

        if tag == "ON":
            if not all_match:
                ok = False
                print("  FAIL: expected match with VLLM_FT_TP_CORRECT=1")
            elif not (h0_match and h1_match):
                ok = False
                print(
                    "  FAIL: expected both ranks' embeddings to match the "
                    "TP=1 reference with VLLM_FT_TP_CORRECT=1"
                )
            elif not (lm_match[0] and lm_match[1]):
                ok = False
                print(
                    "  FAIL: expected both ranks' lm_head grad/loss to match "
                    "the TP=1 reference with VLLM_FT_TP_CORRECT=1"
                )
            else:
                print("  PASS")
        else:
            if all_match:
                ok = False
                print(
                    "  FAIL: expected MISMATCH with VLLM_FT_TP_CORRECT=0 "
                    "(fix would be vacuous)"
                )
            elif h1_match:
                ok = False
                print(
                    "  FAIL: expected rank 1's embedding to be zeroed "
                    "(mismatch) with VLLM_FT_TP_CORRECT=0"
                )
            elif lm_match[0] or lm_match[1]:
                ok = False
                print(
                    "  FAIL: expected both ranks' local-shard CE to diverge "
                    "from the full-vocab reference with VLLM_FT_TP_CORRECT=0"
                )
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
    # Default rendezvous TCPStore, used by the production VLLM_FT_TP_CORRECT
    # exchange (_tp_correct_exchange_sum). This test's calls are synchronous
    # and symmetric (round_id=0 for both ranks every time), so we're only
    # checking that the store-based path reproduces the same numeric result
    # as the NCCL path did, not exercising the timeout/fallback behavior.
    tp_correct_store = dist.distributed_c10d._get_default_store()

    with _PatchToyDims():
        # ── Shared (seeded) full weights — identical across ranks ───────────
        # Scaled down (0.05) so the un-normalised mock RMSNorm layers don't
        # compound activations into huge logits/loss across TOY_LAYERS — keeps
        # gradient magnitudes O(1) so the allclose tolerances below are
        # meaningful.
        SCALE = 0.3
        g = torch.Generator().manual_seed(SEED)
        qkv_full = [
            (torch.randn(Q_SZ + 2 * KV_SZ, TOY_H, generator=g) * SCALE).to(device)
            for _ in range(TOY_LAYERS)
        ]
        o_full = [
            (torch.randn(TOY_H, Q_SZ, generator=g) * SCALE).to(device)
            for _ in range(TOY_LAYERS)
        ]
        embed_w = (torch.randn(TOY_VOCAB, TOY_H, generator=g) * SCALE).to(device)
        lm_w = (torch.randn(TOY_VOCAB, TOY_H, generator=g) * SCALE).to(device)

        # ── TP=2 shards for this rank ────────────────────────────────────────
        nh_local, nkv_local = TOY_HEADS // 2, max(1, TOY_KV // 2)
        q_sz_local, kv_sz_local = nh_local * TOY_HD, nkv_local * TOY_HD

        qkv_local, o_local = [], []
        for i in range(TOY_LAYERS):
            Qf = qkv_full[i][:Q_SZ]
            Kf = qkv_full[i][Q_SZ : Q_SZ + KV_SZ]
            Vf = qkv_full[i][Q_SZ + KV_SZ :]
            Qr = Qf[rank * q_sz_local : (rank + 1) * q_sz_local]
            Kr = Kf[rank * kv_sz_local : (rank + 1) * kv_sz_local]
            Vr = Vf[rank * kv_sz_local : (rank + 1) * kv_sz_local]
            qkv_local.append(torch.cat([Qr, Kr, Vr], dim=0).contiguous())
            o_local.append(
                o_full[i][:, rank * q_sz_local : (rank + 1) * q_sz_local].contiguous()
            )

        # ── Toy LoRA adapter (same file for TP=1 and TP=2 — _load_lora shards
        #    it per _tp_rank/_tp_size) ────────────────────────────────────────
        adapter_path = os.path.join(
            tempfile.gettempdir(), "bt_lora_tp_test_adapter.safetensors"
        )
        if rank == 0:
            _make_toy_safetensors(adapter_path, nonzero_b=True)
        dist.barrier()

        # ── Identical batch (broadcast from rank 0) ──────────────────────────
        if rank == 0:
            batch = torch.randint(0, TOY_VOCAB, (1, TOY_T), device=device)
        else:
            batch = torch.empty((1, TOY_T), dtype=torch.long, device=device)
        dist.broadcast(batch, src=0, group=train_tp_pg)
        input_ids, labels = batch, batch.clone()

        # ── Item 7 fixture: identical on both ranks (seeded), labels chosen so
        # both vocab shards own some targets and one position is ignored ─────
        gh = torch.Generator().manual_seed(SEED + 7)
        hidden_lm = (torch.randn(TOY_T, TOY_H, generator=gh) * SCALE).to(device)
        labels_lm = torch.tensor(
            [[5, 20, 3, -100, 28, 9, 17, 2]], dtype=torch.long, device=device
        )
        assert labels_lm.shape[1] == TOY_T
        v_half = TOY_VOCAB // 2

        results = {}
        for tp_correct in (True, False):
            tag = "ON" if tp_correct else "OFF"

            t2 = _build_trainer(
                2,
                rank,
                qkv_local,
                o_local,
                embed_w,
                lm_w,
                adapter_path,
                device,
                input_ids,
                labels,
                train_tp_pg=train_tp_pg,
                tp_correct_store=tp_correct_store,
                tp_correct=tp_correct,
            )
            loss2 = _run_step(t2)
            grads2 = _grads(t2)

            # Item 2: run _fwd_init (build_fwd_subops' first sub-op) to exercise
            # the embedding all-reduce correction. The exchange is async now —
            # resolve the future the same way the layer-0 sub-op (the real
            # consumer) does.
            fwd_ops2 = t2.build_fwd_subops()
            fwd_ops2[0]()
            state2 = t2._fwd_layer_state
            if state2.get("hidden_fut") is not None:
                state2["hidden"] = state2.pop("hidden_fut").result()
            hidden2 = state2["hidden"].detach().cpu().clone()

            # Item 7: swap in a vocab-sharded lm_head (the production
            # ParallelLMHead layout) and call _lm_head_grad directly. Done
            # after _run_step/_fwd_init so those keep the full lm_w (they
            # rely on a replicated lm_head to match the TP=1 reference).
            t2._lm_head = types.SimpleNamespace(
                weight=lm_w[rank * v_half : (rank + 1) * v_half].contiguous()
            )
            lm_grad2, lm_loss2 = t2._lm_head_grad(hidden_lm, labels_lm, round_id=7)
            if lm_grad2 is not None:
                lm_grad2 = lm_grad2.detach().cpu().clone()

            gathered: list = [None, None]
            dist.all_gather_object(
                gathered,
                {
                    "loss": loss2,
                    "grads": grads2,
                    "hidden": hidden2,
                    "lm_grad": lm_grad2,
                    "lm_loss": lm_loss2,
                },
                group=train_tp_pg,
            )

            if rank == 0:
                t1 = _build_trainer(
                    1,
                    0,
                    qkv_full,
                    o_full,
                    embed_w,
                    lm_w,
                    adapter_path,
                    device,
                    input_ids,
                    labels,
                    train_tp_pg=None,
                    tp_correct=False,
                )
                loss1 = _run_step(t1)
                grads1 = _grads(t1)
                fwd_ops1 = t1.build_fwd_subops()
                fwd_ops1[0]()
                hidden1 = t1._fwd_layer_state["hidden"].detach().cpu().clone()
                # Item 7 reference: t1 holds the full (unsharded) lm_w, so its
                # local-shard path is exactly the full-vocab CE.
                lm_grad1, lm_loss1 = t1._lm_head_grad(hidden_lm, labels_lm)
                results[tag] = {
                    "ref_loss": loss1,
                    "ref_grads": grads1,
                    "ref_hidden": hidden1,
                    "ref_lm_grad": lm_grad1.detach().cpu().clone(),
                    "ref_lm_loss": lm_loss1,
                    "tp2": gathered,
                }

        ok = _check(results) if rank == 0 else True

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0 and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
