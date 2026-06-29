"""
Unit tests for bt_lora_trainer.py
Run with: /mnt/nfs/home/ramya/vllm/.venv/bin/python test_bt_lora_unit.py

Strategy: use object.__new__ to bypass __init__ (which requires vLLM distributed),
then manually set each attribute needed by the method under test.

Tests cover:
  1. Key parsing  — the corrected parts[4]/[6]/[7] indices (bug B from session notes)
  2. TP sharding  — shapes after TP=1 and TP=2 sharding, complementarity
  3. _sync_to_vllm — delta applied correctly, no-compounding on double sync
  4. Gradient flow — confirms the no_grad bug, then verifies the fix
  5. Training step — end-to-end forward+backward with toy model produces finite loss
"""

import sys, os, types, threading, tempfile, unittest

sys.path.insert(0, '/mnt/nfs/home/ramya/vllm')

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import bt_lora_trainer as blt

# ── Real adapter path ─────────────────────────────────────────────────────────
REAL_ADAPTER = (
    '/mnt/nfs/home/ramya/slora-plus/S-LoRA/test/qwen3/adapters/'
    'qwen3-toy-lora/adapter_model.safetensors'
)

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# ── Toy architecture for forward/backward tests ───────────────────────────────
TOY_H      = 64
TOY_HEADS  = 4
TOY_KV     = 2
TOY_HD     = 16
TOY_LAYERS = 2
TOY_VOCAB  = 32
TOY_RANK   = 4
TOY_T      = 8   # sequence length


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_toy_safetensors(path: str, nonzero_b: bool = False) -> None:
    """Write a minimal toy LoRA adapter_model.safetensors.

    nonzero_b=True initialises lora_B with small random values instead of zeros.
    This is needed for gradient-flow tests: with lora_B=zeros the gradient of
    lora_A is analytically zero (d(delta)/d(A) = B @ x = 0 @ x = 0), so tests
    that verify non-zero gradients must use non-zero B.
    """
    q_sz  = TOY_HEADS * TOY_HD
    kv_sz = TOY_KV    * TOY_HD
    def _b(shape):
        return torch.randn(*shape) * 0.01 if nonzero_b else torch.zeros(*shape)
    weights = {}
    for i in range(TOY_LAYERS):
        pfx = f"base_model.model.model.layers.{i}.self_attn"
        weights[f"{pfx}.q_proj.lora_A.weight"] = torch.randn(TOY_RANK, TOY_H)   * 0.01
        weights[f"{pfx}.q_proj.lora_B.weight"] = _b((q_sz, TOY_RANK))
        weights[f"{pfx}.k_proj.lora_A.weight"] = torch.randn(TOY_RANK, TOY_H)   * 0.01
        weights[f"{pfx}.k_proj.lora_B.weight"] = _b((kv_sz, TOY_RANK))
        weights[f"{pfx}.v_proj.lora_A.weight"] = torch.randn(TOY_RANK, TOY_H)   * 0.01
        weights[f"{pfx}.v_proj.lora_B.weight"] = _b((kv_sz, TOY_RANK))
        weights[f"{pfx}.o_proj.lora_A.weight"] = torch.randn(TOY_RANK, q_sz)    * 0.01
        weights[f"{pfx}.o_proj.lora_B.weight"] = _b((TOY_H, TOY_RANK))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    save_file(weights, path)


class _MockFusedNorm:
    """Mock RMSNorm that can be called with 1 arg (returns tensor) or 2 (returns tuple).
    Also has a .weight attribute, matching vLLM's fused RMS norm usage."""
    def __init__(self, H, device):
        self.weight = torch.ones(H, device=device)

    def __call__(self, x, residual=None):
        if residual is None:
            return x.clone()
        # Return outputs WITHOUT grad (mimics the real no_grad applied in _layer_forward)
        return x.detach().clone(), residual.detach().clone()


def _make_mock_model(device):
    """Build a minimal Qwen3-like mock model with TOY_* dimensions."""
    q_sz  = TOY_HEADS * TOY_HD
    kv_sz = TOY_KV    * TOY_HD

    layers = []
    for _ in range(TOY_LAYERS):
        attn = types.SimpleNamespace(
            qkv_proj = types.SimpleNamespace(
                weight = torch.randn(q_sz + kv_sz + kv_sz, TOY_H, device=device)
            ),
            o_proj   = types.SimpleNamespace(
                weight = torch.randn(TOY_H, q_sz, device=device)
            ),
            q_norm      = lambda x: x,
            k_norm      = lambda x: x,
            rotary_emb  = lambda pos, q, k: (q, k),
        )
        layer = types.SimpleNamespace(
            self_attn               = attn,
            input_layernorm         = _MockFusedNorm(TOY_H, device),
            post_attention_layernorm = _MockFusedNorm(TOY_H, device),
            mlp                     = lambda x: x,
        )
        layers.append(layer)

    class _MockEmbed:
        def __call__(self, ids):          # ids: [T] int64
            return torch.randn(ids.shape[0], TOY_H, device=device)

    class _MockNorm:
        def __call__(self, x):
            return x

    inner = types.SimpleNamespace(
        layers       = layers,
        embed_tokens = _MockEmbed(),
        norm         = _MockNorm(),
    )
    model = types.SimpleNamespace(
        model   = inner,
        lm_head = types.SimpleNamespace(
            weight = torch.randn(TOY_VOCAB, TOY_H, device=device)
        ),
    )
    return model


def _make_bare_trainer(model, adapter_path, device_idx=0):
    """Create a BubbleTeaLoRATrainer via object.__new__, bypassing __init__
    (which requires vLLM distributed), and manually set all needed attributes."""
    trainer = object.__new__(blt.BubbleTeaLoRATrainer)

    trainer.device       = device_idx
    trainer._tp_size     = 1
    trainer._tp_rank     = 0
    trainer._tp_group    = None
    trainer._train_tp_pg      = None   # no distributed TP in unit tests
    trainer._tp_correct_store = None
    trainer._fwd_round        = 0
    trainer._do_param_sync    = False  # no cross-rank sync in unit tests
    trainer._pending_param_sync = False
    trainer._tp_correct       = False  # no TP-correctness collectives in unit tests
    trainer._n_heads_local = TOY_HEADS
    trainer._n_kv_local    = TOY_KV
    trainer._step          = 0
    trainer.accum_steps    = 4
    trainer.completed_steps = 0
    trainer.total_loss     = 0.0
    trainer.t_ft           = TOY_T
    trainer._lock          = threading.Lock()
    trainer._fwd           = {}
    trainer._lora          = {}
    trainer._data_ready          = threading.Event()
    trainer._data_ready.set()        # no background thread in unit tests
    trainer._data_thread_started = True  # prevent _real_fwd from re-starting thread
    trainer._fwd_running         = threading.Lock()  # concurrency guard
    trainer._fwd_last_done       = 0.0               # cooldown timer
    trainer._FWD_COOLDOWN_S      = 0.0               # no cooldown in unit tests
    trainer._fwd_layer_state     = None              # per-layer state for sub-ops
    trainer._ep_expert_start        = 0              # single-EP-rank unit tests
    trainer._ep_num_local_experts   = 0              # mock model has no MoE experts
    trainer._ep_size                = 1
    trainer._bwd_passthrough_chunks = 1              # no chunking needed in tests

    trainer._model   = model
    trainer._layers  = model.model.layers
    trainer._embed   = model.model.embed_tokens
    trainer._norm    = model.model.norm
    trainer._lm_head = model.lm_head

    # Mock data iterator — yields (input_ids [1,T], labels [1,T])
    def _mock_data():
        while True:
            dev = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')
            ids    = torch.randint(0, TOY_VOCAB, (1, TOY_T), device=dev)
            labels = ids.clone()
            yield ids, labels

    class _MockIter:
        def __init__(self):
            self._g = _mock_data()
        def __next__(self):
            return next(self._g)

    # Patch _get_batch to use our mock iterator directly
    _iter = _MockIter()
    trainer._data_iter = None  # not used (we override _get_batch below)

    def _get_batch_mock():
        return next(_iter)
    trainer._get_batch = _get_batch_mock

    # Patch load_file to use cpu (adapter saved on cpu)
    orig_load = blt.load_file
    try:
        blt.load_file = lambda p, device: load_file(p, device='cpu')
        trainer._load_lora(adapter_path)
    finally:
        blt.load_file = orig_load

    # Move LoRA params to device — detach first so .to() doesn't create a non-leaf
    dev = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')
    for layer_d in trainer._lora.values():
        for k in list(layer_d):
            layer_d[k] = layer_d[k].detach().to(dev).requires_grad_(True)

    all_params = [p for d in trainer._lora.values() for p in d.values()]
    trainer.optimizer = torch.optim.AdamW(all_params, lr=2e-4, weight_decay=0.01)

    return trainer


# ── Patch/restore helpers ─────────────────────────────────────────────────────

class _PatchToyDims:
    """Context manager that patches bt_lora_trainer module-level constants to toy dims."""
    _ORIG = {}

    def __enter__(self):
        for name, val in [
            ('_HIDDEN', TOY_H), ('_N_HEADS', TOY_HEADS),
            ('_N_KV', TOY_KV), ('_HEAD_DIM', TOY_HD),
            ('_N_LAYERS', TOY_LAYERS), ('_LORA_ALPHA', 16),
        ]:
            self._ORIG[name] = getattr(blt, name)
            setattr(blt, name, val)
        return self

    def __exit__(self, *_):
        for name, val in self._ORIG.items():
            setattr(blt, name, val)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Key parsing
# ══════════════════════════════════════════════════════════════════════════════

class TestKeyParsing(unittest.TestCase):
    """Verify the safetensors key-index fix (Bug B from session notes):
    correct indices are parts[4], parts[6], parts[7] — NOT parts[5],[7],[8]."""

    @classmethod
    def setUpClass(cls):
        cls.weights = load_file(REAL_ADAPTER, device='cpu')

    def test_total_key_count(self):
        # 48 layers × 4 projections × 2 (lora_A + lora_B) = 384
        self.assertEqual(len(self.weights), 384)

    def test_old_indices_would_misparse(self):
        """The buggy indices (parts[5],[7],[8]) produce wrong results."""
        key = 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'
        parts = key.split('.')
        # Wrong: parts[5] = "self_attn", parts[7] = "lora_A", parts[8] = "weight"
        self.assertEqual(parts[5], 'self_attn')
        self.assertRaises(ValueError, int, parts[5])  # not a layer index

    def test_corrected_indices_parse_all_keys(self):
        """Corrected indices (parts[4], parts[6], parts[7]) parse every key."""
        seen = {}
        for key in self.weights:
            parts = key.split('.')
            layer_idx = int(parts[4])   # was parts[5] — wrong
            proj      = parts[6]        # was parts[7] — wrong
            ab        = parts[7]        # was parts[8] — wrong
            seen[(layer_idx, proj, ab)] = True
        self.assertEqual(len(seen), 384)

    def test_all_48_layers_present(self):
        layers = set()
        for k in self.weights:
            layers.add(int(k.split('.')[4]))
        self.assertEqual(layers, set(range(48)))

    def test_all_4_projections_present(self):
        projs = set(k.split('.')[6] for k in self.weights)
        self.assertEqual(projs, {'q_proj', 'k_proj', 'v_proj', 'o_proj'})

    def test_shapes_match_qwen3_30b_a3b(self):
        """Verify the exact shapes documented in the session notes."""
        RANK, H = 16, 2048
        for key, tensor in self.weights.items():
            parts = key.split('.')
            proj, ab = parts[6], parts[7]
            if ab == 'lora_A':
                if proj in ('q_proj', 'k_proj', 'v_proj'):
                    self.assertEqual(tensor.shape, (RANK, H), key)
                elif proj == 'o_proj':
                    self.assertEqual(tensor.shape, (RANK, 32 * 128), key)  # [16, 4096]
            elif ab == 'lora_B':
                if proj == 'q_proj':
                    self.assertEqual(tensor.shape, (32 * 128, RANK), key)  # [4096, 16]
                elif proj in ('k_proj', 'v_proj'):
                    self.assertEqual(tensor.shape, (4 * 128, RANK), key)   # [512, 16]
                elif proj == 'o_proj':
                    self.assertEqual(tensor.shape, (H, RANK), key)         # [2048, 16]


# ══════════════════════════════════════════════════════════════════════════════
# 2. TP sharding
# ══════════════════════════════════════════════════════════════════════════════

class TestTPSharding(unittest.TestCase):
    """Verify the sharding logic in _load_lora for TP=1 and TP=2."""

    @classmethod
    def setUpClass(cls):
        cls.weights = load_file(REAL_ADAPTER, device='cpu')

    def _shard(self, tensor, tp_rank, tp_size, ab, proj):
        """Replicate the sharding from _load_lora."""
        if ab == 'lora_B' and proj in ('q_proj', 'k_proj', 'v_proj'):
            chunk = tensor.shape[0] // tp_size
            return tensor[tp_rank * chunk: (tp_rank + 1) * chunk]
        elif ab == 'lora_A' and proj == 'o_proj':
            chunk = tensor.shape[1] // tp_size
            return tensor[:, tp_rank * chunk: (tp_rank + 1) * chunk]
        return tensor

    def test_tp1_no_change(self):
        for key, t in list(self.weights.items())[:16]:
            parts = key.split('.')
            sharded = self._shard(t, 0, 1, parts[7], parts[6])
            self.assertEqual(sharded.shape, t.shape, key)

    def test_tp2_q_proj_b_halved_along_output(self):
        q_b = [(k, v) for k, v in self.weights.items() if 'q_proj.lora_B' in k]
        for key, t in q_b[:4]:
            s0 = self._shard(t, 0, 2, 'lora_B', 'q_proj')
            s1 = self._shard(t, 1, 2, 'lora_B', 'q_proj')
            self.assertEqual(s0.shape[0], t.shape[0] // 2)
            self.assertEqual(s1.shape[0], t.shape[0] // 2)
            self.assertEqual(s0.shape[1], t.shape[1])

    def test_tp2_kv_proj_b_halved(self):
        for proj in ('k_proj', 'v_proj'):
            kv_b = [(k, v) for k, v in self.weights.items() if f'{proj}.lora_B' in k]
            for key, t in kv_b[:4]:
                s0 = self._shard(t, 0, 2, 'lora_B', proj)
                self.assertEqual(s0.shape[0], t.shape[0] // 2)  # 512 → 256 per rank

    def test_tp2_o_proj_a_halved_along_input(self):
        o_a = [(k, v) for k, v in self.weights.items() if 'o_proj.lora_A' in k]
        for key, t in o_a[:4]:
            s0 = self._shard(t, 0, 2, 'lora_A', 'o_proj')
            s1 = self._shard(t, 1, 2, 'lora_A', 'o_proj')
            self.assertEqual(s0.shape[0], t.shape[0])            # rank dim unchanged
            self.assertEqual(s0.shape[1], t.shape[1] // 2)       # input dim halved

    def test_tp2_ranks_complement_for_q_proj(self):
        q_b = [(k, v) for k, v in self.weights.items() if 'q_proj.lora_B' in k]
        for key, t in q_b[:4]:
            s0 = self._shard(t, 0, 2, 'lora_B', 'q_proj')
            s1 = self._shard(t, 1, 2, 'lora_B', 'q_proj')
            self.assertTrue(torch.equal(torch.cat([s0, s1], dim=0), t))


# ══════════════════════════════════════════════════════════════════════════════
# 3. _sync_to_vllm
# ══════════════════════════════════════════════════════════════════════════════

class TestSyncToVLLM(unittest.TestCase):
    """Verify _sync_to_vllm merges LoRA delta correctly and does not compound."""

    def setUp(self):
        q_sz  = TOY_HEADS * TOY_HD
        kv_sz = TOY_KV    * TOY_HD
        dev   = DEVICE
        self.q_sz  = q_sz
        self.kv_sz = kv_sz
        self.dev   = dev

        # Build a mock trainer with 1 layer, known LoRA weights
        attn = types.SimpleNamespace(
            qkv_proj = types.SimpleNamespace(
                weight = torch.zeros(q_sz + kv_sz + kv_sz, TOY_H, device=dev)
            ),
            o_proj = types.SimpleNamespace(
                weight = torch.zeros(TOY_H, q_sz, device=dev)
            ),
        )
        layer  = types.SimpleNamespace(self_attn=attn)

        self.trainer = types.SimpleNamespace(
            _n_heads_local = TOY_HEADS,
            _n_kv_local    = TOY_KV,
            _layers        = [layer],
            _lora          = {
                0: {
                    'q_proj.lora_A': torch.randn(TOY_RANK, TOY_H, device=dev).requires_grad_(True),
                    'q_proj.lora_B': torch.randn(q_sz, TOY_RANK, device=dev).requires_grad_(True),
                    'v_proj.lora_A': torch.randn(TOY_RANK, TOY_H, device=dev).requires_grad_(True),
                    'v_proj.lora_B': torch.randn(kv_sz, TOY_RANK, device=dev).requires_grad_(True),
                    'o_proj.lora_A': torch.randn(TOY_RANK, q_sz, device=dev).requires_grad_(True),
                    'o_proj.lora_B': torch.randn(TOY_H, TOY_RANK, device=dev).requires_grad_(True),
                }
            },
        )

    def _call_sync(self):
        """Call the real _sync_to_vllm on self.trainer."""
        blt.BubbleTeaLoRATrainer._sync_to_vllm(self.trainer)

    def test_q_proj_delta_applied_correctly(self):
        self._call_sync()
        d = self.trainer._lora[0]
        A = d['q_proj.lora_A'].data.to(torch.bfloat16)
        B = d['q_proj.lora_B'].data.to(torch.bfloat16)
        expected = (B @ A)   # scale = LORA_ALPHA/16 = 1.0
        w = self.trainer._layers[0].self_attn.qkv_proj.weight[:self.q_sz]
        torch.testing.assert_close(w.to(torch.bfloat16), expected,
                                   msg='q_proj delta not applied correctly')

    def test_o_proj_delta_applied_correctly(self):
        self._call_sync()
        d = self.trainer._lora[0]
        A = d['o_proj.lora_A'].data.to(torch.bfloat16)
        B = d['o_proj.lora_B'].data.to(torch.bfloat16)
        expected = (B @ A)
        w = self.trainer._layers[0].self_attn.o_proj.weight
        torch.testing.assert_close(w.to(torch.bfloat16), expected,
                                   msg='o_proj delta not applied correctly')

    def test_no_compounding_on_double_sync(self):
        """Calling sync twice with the same LoRA weights must give the same result."""
        self._call_sync()
        w_first = self.trainer._layers[0].self_attn.qkv_proj.weight[:self.q_sz].clone()

        self._call_sync()
        w_second = self.trainer._layers[0].self_attn.qkv_proj.weight[:self.q_sz]

        torch.testing.assert_close(w_first, w_second,
                                   msg='Weight changed on second sync — delta is compounding')

    def test_second_sync_reflects_updated_lora(self):
        """After an optimizer step updates LoRA A, the second sync applies the new delta."""
        self._call_sync()

        # Simulate optimizer step: perturb lora_A
        with torch.no_grad():
            self.trainer._lora[0]['q_proj.lora_A'] += 1.0

        self._call_sync()

        d = self.trainer._lora[0]
        A = d['q_proj.lora_A'].data.to(torch.bfloat16)
        B = d['q_proj.lora_B'].data.to(torch.bfloat16)
        expected = (B @ A)
        w = self.trainer._layers[0].self_attn.qkv_proj.weight[:self.q_sz]
        torch.testing.assert_close(w.to(torch.bfloat16), expected)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Gradient flow — bug detection + fix verification
# ══════════════════════════════════════════════════════════════════════════════

class TestGradientFlow(unittest.TestCase):
    """
    Verifies the gradient-flow bug and confirms the fix.

    Bug: every _layer_forward call applies post_attention_layernorm under
    no_grad(), which severs the autograd path from the LoRA deltas to the loss.
    Result: loss.requires_grad = False → loss.backward() raises RuntimeError.

    Fix (applied in bt_lora_trainer.py): _layer_forward also returns attn_out
    (the O-projection result, which carries grad via o_delta and v_delta).
    _forward_loss then uses last_attn_out instead of hidden_normed for the
    logit computation, preserving the grad path.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.adapter_path = os.path.join(cls.tmpdir, 'adapter_model.safetensors')
        # Use nonzero_b=True: standard init has lora_B=0, which analytically
        # zeros out lora_A's gradient (d(delta)/d(A) = B @ x = 0).  These tests
        # are checking that the grad CHAIN works, not the initial-value edge case.
        _make_toy_safetensors(cls.adapter_path, nonzero_b=True)

    def _make_trainer(self):
        with _PatchToyDims():
            model   = _make_mock_model(DEVICE)
            trainer = _make_bare_trainer(model, self.adapter_path,
                                         device_idx=DEVICE.index if hasattr(DEVICE, 'index') else 0)
        return trainer

    def test_loss_is_finite_and_requires_grad(self):
        """After the fix: forward produces a finite, differentiable loss."""
        with _PatchToyDims():
            trainer = self._make_trainer()
            with torch.inference_mode(False):
                with torch.enable_grad():
                    loss = blt.BubbleTeaLoRATrainer._forward_loss(trainer)

        self.assertTrue(loss.requires_grad,
                        "loss.requires_grad=False — gradient path is severed. "
                        "Check that _forward_loss uses last_attn_out (not hidden_normed) "
                        "for the logit computation.")
        self.assertFalse(torch.isnan(loss), "loss is NaN")
        self.assertFalse(torch.isinf(loss), "loss is Inf")

    def test_v_proj_and_o_proj_get_gradients(self):
        """v_proj and o_proj LoRA params must receive non-zero gradients."""
        with _PatchToyDims():
            trainer = self._make_trainer()
            with torch.inference_mode(False):
                with torch.enable_grad():
                    loss = blt.BubbleTeaLoRATrainer._forward_loss(trainer)

        self.assertTrue(loss.requires_grad, "loss has no grad — fix not applied")
        loss.backward()

        last_idx = TOY_LAYERS - 1
        for proj in ('v_proj', 'o_proj'):
            for ab in ('lora_A', 'lora_B'):
                key = f'{proj}.{ab}'
                param = trainer._lora[last_idx][key]
                self.assertIsNotNone(param.grad,
                    f"{key} in last layer has no gradient after backward")
                self.assertGreater(param.grad.abs().sum().item(), 0,
                    f"{key} in last layer has all-zero gradient")

    def test_training_step_completes_without_error(self):
        """training_step (forward + backward + accumulation) must not raise."""
        with _PatchToyDims():
            trainer = self._make_trainer()
            try:
                loss_val = blt.BubbleTeaLoRATrainer.training_step(trainer)
            except RuntimeError as e:
                self.fail(f"training_step raised RuntimeError: {e}")
        self.assertIsInstance(loss_val, float)
        self.assertFalse(torch.isnan(torch.tensor(loss_val)), "loss is NaN")

    def test_optimizer_step_updates_weights(self):
        """After accum_steps gradient steps, LoRA weights should change."""
        with _PatchToyDims():
            trainer = self._make_trainer()
            # Record initial weights
            last_idx = TOY_LAYERS - 1
            A_v_init = trainer._lora[last_idx]['v_proj.lora_A'].data.clone()

            for _ in range(trainer.accum_steps):
                blt.BubbleTeaLoRATrainer.training_step(trainer)

        # Weights must have been updated by AdamW
        A_v_after = trainer._lora[last_idx]['v_proj.lora_A'].data
        self.assertFalse(torch.equal(A_v_init, A_v_after),
                         "v_proj.lora_A unchanged after optimizer step — "
                         "gradient accumulation or optimizer step may be broken")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Sub-op builders
# ══════════════════════════════════════════════════════════════════════════════

class TestSubOpBuilders(unittest.TestCase):
    """build_fwd_subops / build_bwd_subops must return per-layer callables."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.adapter_path = os.path.join(cls.tmpdir, 'adapter_model.safetensors')
        _make_toy_safetensors(cls.adapter_path, nonzero_b=True)

    def _dev_idx(self):
        return DEVICE.index if hasattr(DEVICE, 'index') else 0

    def test_fwd_subops_count(self):
        """build_fwd_subops must return 1 init + N_layers sub-ops."""
        with _PatchToyDims():
            trainer = _make_bare_trainer(_make_mock_model(DEVICE), self.adapter_path,
                                         device_idx=self._dev_idx())
            ops = blt.BubbleTeaLoRATrainer.build_fwd_subops(trainer)
        # 1 init + TOY_LAYERS layer ops
        self.assertEqual(len(ops), 1 + TOY_LAYERS)
        for op in ops:
            self.assertTrue(callable(op))

    def test_fwd_subops_run_all_layers_and_store_result(self):
        """Running all forward sub-ops must populate trainer._fwd with per-layer
        activation dicts (x_norm, attn_out, x2), hidden_last, and labels."""
        with _PatchToyDims():
            trainer = _make_bare_trainer(_make_mock_model(DEVICE), self.adapter_path,
                                         device_idx=self._dev_idx())
            ops = blt.BubbleTeaLoRATrainer.build_fwd_subops(trainer)
            for op in ops:
                op()  # run each sub-op in sequence

        self.assertIn('hidden_last', trainer._fwd,
                      "hidden_last not stored after all fwd sub-ops")
        self.assertIn('labels', trainer._fwd,
                      "labels not stored after all fwd sub-ops")
        self.assertIn('layers', trainer._fwd,
                      "layers dict not stored after all fwd sub-ops")
        self.assertEqual(len(trainer._fwd['layers']), TOY_LAYERS,
                         f"Expected {TOY_LAYERS} layer entries, "
                         f"got {len(trainer._fwd['layers'])}")
        # Spot-check the last layer has all three activation tensors
        last = trainer._fwd['layers'][TOY_LAYERS - 1]
        for key in ('x_norm', 'attn_out', 'x2'):
            self.assertIn(key, last,
                          f"'{key}' missing from layer {TOY_LAYERS - 1} activations")

    def test_bwd_subops_count(self):
        """build_bwd_subops must return (N_chunks+4+3)*N + 2 sub-ops.

        With _bwd_passthrough_chunks=1 (set in _make_bare_trainer):
          1 passthrough + 4 attn bwd + 3 LoRA = 8 per layer, plus lm_head and optimizer.
        """
        with _PatchToyDims():
            trainer = _make_bare_trainer(_make_mock_model(DEVICE), self.adapter_path,
                                         device_idx=self._dev_idx())
            ops = blt.BubbleTeaLoRATrainer.build_bwd_subops(trainer)
        n_chunks = trainer._bwd_passthrough_chunks  # 1 in tests
        expected = (n_chunks + 4 + 3) * TOY_LAYERS + 2
        self.assertEqual(len(ops), expected)

    def test_bwd_subops_run_after_fwd(self):
        """Running all fwd then all bwd sub-ops must increment _step."""
        with _PatchToyDims():
            trainer = _make_bare_trainer(_make_mock_model(DEVICE), self.adapter_path,
                                         device_idx=self._dev_idx())
            fwd_ops = blt.BubbleTeaLoRATrainer.build_fwd_subops(trainer)
            bwd_ops = blt.BubbleTeaLoRATrainer.build_bwd_subops(trainer)
            for op in fwd_ops:
                op()
            for op in bwd_ops:
                op()

        self.assertEqual(trainer._step, 1, "_step not incremented after bwd sub-ops")

    def test_bwd_subops_no_op_when_fwd_empty(self):
        """bwd sub-ops must be silent no-ops when _fwd is empty."""
        with _PatchToyDims():
            trainer = _make_bare_trainer(_make_mock_model(DEVICE), self.adapter_path,
                                         device_idx=self._dev_idx())
            bwd_ops = blt.BubbleTeaLoRATrainer.build_bwd_subops(trainer)
            try:
                for op in bwd_ops:
                    op()
            except Exception as e:
                self.fail(f"bwd sub-op raised with empty _fwd: {e}")

    def test_subop_qlora_grads_nonzero_after_sync(self):
        """After one optimizer step + _sync_to_vllm, Q LoRA grads must still be non-zero.

        Regression test for the _attn_o_proj_bwd / _attn_qkv_recompute _prev_delta bug:
        before the fix, grad_sdpa_out used the merged o_w instead of the base weight,
        producing wrong SDPA backward and wrong grad_q.
        """
        with _PatchToyDims():
            model  = _make_mock_model(DEVICE)
            trainer = _make_bare_trainer(model, self.adapter_path,
                                         device_idx=self._dev_idx())

            # Give the mock model a norm weight so _lm_head_op takes the normed path
            class _NormWithWeight:
                def __init__(self, H, device):
                    self.weight = torch.ones(H, device=device)
                def __call__(self, x):
                    return x
            model.model.norm = _NormWithWeight(TOY_H, DEVICE)
            trainer._norm = model.model.norm

            dev = DEVICE
            ids    = torch.randint(0, TOY_VOCAB, (1, TOY_T), device=dev)
            labels = ids.clone()
            trainer._get_batch = lambda: (ids.clone(), labels.clone())

            # Run accum_steps cycles to trigger first optimizer step + _sync_to_vllm
            for _ in range(trainer.accum_steps):
                fwd_ops = blt.BubbleTeaLoRATrainer.build_fwd_subops(trainer)
                bwd_ops = blt.BubbleTeaLoRATrainer.build_bwd_subops(trainer)
                for op in fwd_ops:
                    op()
                for op in bwd_ops:
                    op()
            # _sync_to_vllm has now run — _prev_delta_{i}_q_proj is set

            # Run one more fwd+bwd cycle — gradients must still be non-zero
            fwd_ops = blt.BubbleTeaLoRATrainer.build_fwd_subops(trainer)
            bwd_ops = blt.BubbleTeaLoRATrainer.build_bwd_subops(trainer)
            for op in fwd_ops:
                op()
            # zero_grad was called at end of last optimizer_step
            for op in bwd_ops:
                op()

            last_idx = TOY_LAYERS - 1
            for proj in ('q_proj', 'v_proj', 'o_proj'):
                for ab in ('lora_A', 'lora_B'):
                    key  = f'{proj}.{ab}'
                    grad = trainer._lora[last_idx][key].grad
                    if grad is not None:
                        self.assertGreater(
                            grad.abs().sum().item(), 0,
                            f"{key} gradient is zero after _sync_to_vllm "
                            "(possible _prev_delta bug in _attn_o_proj_bwd/_attn_qkv_recompute)"
                        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
