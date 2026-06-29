"""
Standalone test for item 8 (EP-correct MoE forward delta wiring) in
build_fwd_subops_cdbatch (see /mnt/nfs/home/ramya/slora-plus/Sessions_10_6_2026.md).

Qwen3MoeSparseMoeBlock.forward() concatenates the FT hidden state (seeded via
ft_moe_set_hidden) onto the inference batch and runs ONE self.experts(...)
call -- so the FT tokens get an EP-correct MoE output (combined_out[n_inf:])
via the SAME all-to-all dispatch/combine vLLM already does for inference, at
no extra communication cost.  ft_moe_advance(combined_out[n_inf:]) is called
with that output.

This test simulates that main-stream call by invoking
moe_runner.ft_moe_advance(fake_delta) directly between forward sub-ops, and
checks that:

  1. _fwd_attn_only(i) for i >= 1 folds layer (i-1)'s delta into "hidden"
     (visible via the saved "res_a" activation) when ft_moe_advance was
     called for layer i-1 since the last reset.
  2. _fwd_finalize folds layer (N-1)'s delta into hidden_last similarly.
  3. When the previous layer's delta is NOT available yet (ft_moe_advance not
     called, or stale from a previous round), the sub-op gracefully falls
     back to the pre-existing approximation (no delta added) -- in
     particular, ft_moe_reset_moe_delta() (called by _fwd_init_cdbatch at the
     start of each round) prevents a stale delta from a previous round being
     incorrectly reused.

Run (single GPU):
    .venv/bin/python test_bt_lora_cdbatch_moe_delta.py
"""
import os
import sys
import tempfile

sys.path.insert(0, '/mnt/nfs/home/ramya/vllm')

import torch

import bt_lora_trainer as blt
from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
from test_bt_lora_unit import (
    _make_bare_trainer, _make_mock_model, _make_toy_safetensors,
    _PatchToyDims, TOY_LAYERS, DEVICE,
)


def _reset_ft_moe_state():
    mr._ft_moe_state["real_training"] = True
    mr._ft_moe_state["layer_count"] = 0
    mr._ft_moe_state["moe_delta"] = None
    mr._ft_moe_state["moe_delta_layer"] = -1
    mr._ft_moe_state["moe_delta_event"] = None


def main() -> None:
    assert torch.cuda.is_available(), "this test requires a CUDA device"

    adapter_path = os.path.join(tempfile.gettempdir(),
                                 "bt_lora_cdbatch_moe_delta_adapter.safetensors")
    _make_toy_safetensors(adapter_path, nonzero_b=True)

    ctx = _PatchToyDims()
    ctx.__enter__()

    model = _make_mock_model(DEVICE)
    trainer = _make_bare_trainer(model, adapter_path, device_idx=0)

    N = TOY_LAYERS
    fwd_ops = trainer.build_fwd_subops_cdbatch()
    assert len(fwd_ops) == N + 2, f"expected {N + 2} sub-ops, got {len(fwd_ops)}"

    ok = True

    # ── Round 1: ft_moe_advance fires for every layer's delta ────────────────
    _reset_ft_moe_state()
    fwd_ops[0]()  # _fwd_init_cdbatch
    state = trainer._fwd_layer_state
    assert state is not None and state["ok"], "round 1 init failed"

    for i in range(N):
        if i > 0:
            prev_x2 = trainer._fwd["layers"][i - 1]["x2"]
            fake_delta = torch.full_like(prev_x2, fill_value=10.0 * i)
            mr.ft_moe_advance(fake_delta)

        hidden_in = trainer._fwd_layer_state["hidden"].clone()
        residual_in = trainer._fwd_layer_state["residual"]
        residual_in = residual_in.clone() if residual_in is not None else None

        fwd_ops[i + 1]()  # _fwd_attn_only(i)

        expected_hidden = hidden_in + fake_delta if i > 0 else hidden_in
        expected_res_a = (expected_hidden + residual_in
                           if residual_in is not None else expected_hidden)
        actual_res_a = trainer._fwd["layers"][i]["res_a"]
        match = torch.allclose(actual_res_a, expected_res_a)
        ok &= match
        print(f"round1 layer {i}: delta folded in res_a -> match={match}")
        if not match:
            print(f"  expected[:2,:4]={expected_res_a[:2, :4]}")
            print(f"  actual[:2,:4]  ={actual_res_a[:2, :4]}")

    # _fwd_finalize: layer N-1's delta should fold into hidden_last.
    prev_x2 = trainer._fwd["layers"][N - 1]["x2"]
    fake_delta_last = torch.full_like(prev_x2, fill_value=999.0)
    mr.ft_moe_advance(fake_delta_last)

    hidden_in = trainer._fwd_layer_state["hidden"].clone()
    residual_in = trainer._fwd_layer_state["residual"].clone()

    fwd_ops[N + 1]()  # _fwd_finalize

    expected_final = hidden_in + fake_delta_last + residual_in
    actual_final = trainer._fwd["hidden_last"]
    match = torch.allclose(actual_final, expected_final)
    ok &= match
    print(f"round1 finalize: delta folded in hidden_last -> match={match}")

    assert trainer._fwd_layer_state is None, "finalize must clear _fwd_layer_state"
    assert not trainer._fwd_running.locked(), "finalize must release _fwd_running"

    # ── Round 2: ft_moe_advance does NOT fire -- must fall back gracefully,
    #    and must NOT reuse round 1's stale moe_delta_layer == N-1 from
    #    finalize (this is what ft_moe_reset_moe_delta guards against). ──────
    fwd_ops[0]()  # _fwd_init_cdbatch (resets moe_delta via ft_moe_reset_moe_delta)
    state = trainer._fwd_layer_state
    assert state is not None and state["ok"], "round 2 init failed"
    assert mr.ft_moe_get_moe_delta_layer() == -1, \
        "ft_moe_reset_moe_delta should clear moe_delta_layer at round start"

    for i in range(N):
        hidden_in = trainer._fwd_layer_state["hidden"].clone()
        residual_in = trainer._fwd_layer_state["residual"]
        residual_in = residual_in.clone() if residual_in is not None else None

        fwd_ops[i + 1]()  # _fwd_attn_only(i), no ft_moe_advance called

        expected_res_a = (hidden_in + residual_in
                           if residual_in is not None else hidden_in)
        actual_res_a = trainer._fwd["layers"][i]["res_a"]
        match = torch.allclose(actual_res_a, expected_res_a)
        ok &= match
        print(f"round2 layer {i}: no delta available -> fallback match={match}")

    hidden_in = trainer._fwd_layer_state["hidden"].clone()
    residual_in = trainer._fwd_layer_state["residual"].clone()
    fwd_ops[N + 1]()  # _fwd_finalize, no ft_moe_advance called for layer N-1

    expected_final = hidden_in + residual_in
    actual_final = trainer._fwd["hidden_last"]
    match = torch.allclose(actual_final, expected_final)
    ok &= match
    print(f"round2 finalize: no delta available -> fallback match={match}")

    ctx.__exit__(None, None, None)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
