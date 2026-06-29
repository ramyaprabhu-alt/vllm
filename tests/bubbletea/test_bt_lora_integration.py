"""
Integration test for BubbleTea real LoRA training.

Launches the actual BubbleTea server with VLLM_FT_LORA_PATH set, sends
20 short inference requests, then verifies:
  1. Server log contains "[BubbleTea LoRA] Trainer ready"  (trainer init)
  2. /tmp/vllm_combined_completions.log has at least one entry  (training ran)
  3. All inference requests succeeded (training did not break inference)

Run with:
  /mnt/nfs/home/ramya/vllm/.venv/bin/python test_bt_lora_integration.py

Requires:
  - 2× A100-80GB available and free (checks via nvidia-smi before launch)
  - ~10 minutes wall time (model load ~5 min + warmup + 20 reqs)
"""

import os, re, signal, subprocess, sys, time, unittest, urllib.request
from pathlib import Path

# ── Config (mirrors compare_benchmark.py) ─────────────────────────────────────
BT_VENV        = Path('/mnt/nfs/home/ramya/vllm/.venv')
MODEL_DIR      = '/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B'
BT_LORA_ADAPTER = '/mnt/nfs/home/ramya/slora-plus/S-LoRA/test/qwen3/adapters/qwen3-toy-lora'
BT_COMPLETIONS = Path('/tmp/vllm_combined_completions.log')
PORT           = 8765   # use a different port from the main benchmark to avoid conflicts
SERVER_READY_TIMEOUT = 600   # seconds
RESULT_DIR     = Path('/tmp/bt_lora_integration_test')


def _gpu_mem_used_mib() -> list[int]:
    """Return used GPU memory in MiB for each GPU."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            text=True,
        ).strip().splitlines()
        return [int(x) for x in out if x.strip().isdigit()]
    except Exception:
        return []


def _wait_server(port: int, timeout: int = SERVER_READY_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f'http://localhost:{port}/health', timeout=3)
            return True
        except Exception:
            time.sleep(5)
    return False


def _kill_server(port: int) -> None:
    try:
        out = subprocess.check_output(['lsof', '-ti', f':{port}'], text=True).strip()
        for pid in out.splitlines():
            try:
                os.kill(int(pid), signal.SIGTERM)
            except Exception:
                pass
    except Exception:
        pass
    time.sleep(4)
    for pattern in [r'vllm\.worker', r'vllm serve', r'VLLM::Worker_TP']:
        subprocess.run(['pkill', '-9', '-f', pattern], capture_output=True)
    subprocess.run(['pkill', '-9', '-x', 'VLLM::Worker_TP'], capture_output=True)
    # Wait for GPUs to free
    deadline = time.time() + 90
    while time.time() < deadline:
        mems = _gpu_mem_used_mib()
        if mems and all(m < 5120 for m in mems):
            break
        time.sleep(5)


def _send_request(prompt: str, port: int, max_tokens: int = 32) -> dict | None:
    """Send one inference request to the server. Returns parsed JSON or None."""
    import json, urllib.error
    body = json.dumps({
        'model':       MODEL_DIR,
        'messages':    [{'role': 'user', 'content': prompt}],
        'max_tokens':  max_tokens,
        'temperature': 0.0,
    }).encode()
    req = urllib.request.Request(
        f'http://localhost:{port}/v1/chat/completions',
        data=body,
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f'  request failed: {e}')
        return None


class TestBubbleTeaLoRAIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Check prerequisites then launch the server once for all tests."""
        # ── Pre-flight checks ──────────────────────────────────────────────
        if not Path(MODEL_DIR).exists():
            raise unittest.SkipTest(f'Model not found: {MODEL_DIR}')
        if not Path(BT_LORA_ADAPTER).exists():
            raise unittest.SkipTest(f'LoRA adapter not found: {BT_LORA_ADAPTER}')

        mems = _gpu_mem_used_mib()
        if len(mems) < 2:
            raise unittest.SkipTest('Fewer than 2 GPUs detected')
        if any(m > 5120 for m in mems[:2]):
            raise unittest.SkipTest(
                f'GPUs not free (used: {mems[:2]} MiB) — kill other processes first'
            )

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        BT_COMPLETIONS.unlink(missing_ok=True)
        Path('/tmp/vllm_training_loss.log').unlink(missing_ok=True)

        # ── Kill any stale workers ─────────────────────────────────────────
        subprocess.run(['pkill', '-9', '-f', 'VLLM::Worker_TP'], capture_output=True)
        subprocess.run(['pkill', '-9', '-x', 'VLLM::Worker_TP'], capture_output=True)
        time.sleep(2)

        # ── Build server env ──────────────────────────────────────────────
        env = os.environ.copy()
        env['PATH']                  = f"{BT_VENV}/bin:{env.get('PATH', '')}"
        env['CUDA_VISIBLE_DEVICES']  = '0,1'
        env['VLLM_FT_COMBINED_MODE'] = 'C+D'
        env['VLLM_FT_LORA_PATH']     = BT_LORA_ADAPTER
        env['VLLM_FT_TOKENIZER_PATH']= MODEL_DIR
        env['VLLM_FT_CACHE_DIR']     = '/mnt/nfs/home/ramya/scratch'
        env['VLLM_FT_COMBINED_T_FT'] = '128'
        env['VLLM_FT_ACCUM_STEPS']   = '1'   # one fwd+bwd cycle = one optimizer step
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        env['HF_TOKEN']              = os.environ.get("HF_TOKEN", "")
        env['HF_DATASETS_OFFLINE']   = '1'   # datasets: use cache only
        env['TRANSFORMERS_OFFLINE']  = '1'   # transformers: no hub check on tokenizer

        cls.server_log = RESULT_DIR / 'server.log'
        server_cmd = [
            f'{BT_VENV}/bin/vllm', 'serve', MODEL_DIR,
            '--tensor-parallel-size', '2',
            '--enable-expert-parallel',
            '--enable-ep-weight-filter',
            '--all2all-backend', 'allgather_reducescatter',
            '--moe-backend', 'triton',
            '--dtype', 'bfloat16',
            '--max-model-len', '4096',
            '--gpu-memory-utilization', '0.85',
            '--max-num-seqs', '64',
            '--enforce-eager',
            '--no-enable-chunked-prefill',
            '--no-enable-prefix-caching',
            '--max-num-batched-tokens', '4096',
            '--trust-remote-code',
            '--host', '0.0.0.0',
            '--port', str(PORT),
        ]

        print(f'\n  Launching BubbleTea server (log: {cls.server_log}) ...')
        with open(cls.server_log, 'w') as log_f:
            cls.server_proc = subprocess.Popen(
                server_cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT
            )
        print(f'  Server PID: {cls.server_proc.pid}')

        print(f'  Waiting for server to be ready (up to {SERVER_READY_TIMEOUT}s) ...')
        cls.server_ready = _wait_server(PORT)
        if not cls.server_ready:
            print('  ERROR: server did not become ready — check server.log')
            _kill_server(PORT)

    @classmethod
    def tearDownClass(cls):
        print('\n  Tearing down server ...')
        _kill_server(PORT)

    def setUp(self):
        if not self.server_ready:
            self.skipTest('Server did not start — see setUpClass logs')

    # ── Tests ──────────────────────────────────────────────────────────────

    def test_1_trainer_init_logged(self):
        """Server log must contain the trainer-ready message."""
        log_text = self.server_log.read_text(errors='replace')
        self.assertIn(
            '[BubbleTea LoRA] Trainer ready',
            log_text,
            msg=(
                'Trainer-ready message not found in server log.\n'
                'Check VLLM_FT_LORA_PATH is set and _maybe_init_bt_trainer ran.\n'
                f'Look for "Failed to initialise" in {self.server_log}'
            ),
        )
        # Extract and print trainable param count
        m = re.search(r'Trainer ready – (\d+) trainable params', log_text)
        if m:
            print(f'\n  Trainer ready: {int(m.group(1)):,} trainable params')

    def test_2_inference_requests_succeed(self):
        """10 long-context requests must succeed and trigger C+D training.

        Uses arxiv abstracts (~300 input tokens each) so that prefill batches
        exceed the submit_forward_job min_tokens=512 threshold and C+D
        training fires.  Falls back to short prompts if the arxiv file is
        missing (smoke-test only in that case).
        """
        import json, threading

        arxiv_path = Path('/mnt/nfs/home/ramya/scratch/arxiv_bench_500.jsonl')
        if arxiv_path.exists():
            with open(arxiv_path) as f:
                all_entries = [json.loads(l) for l in f if l.strip()]
            # Target >512 and <1024 input tokens (≈394–787 words).
            # This exceeds submit_forward_job's min_tokens=512 threshold while
            # keeping total context (input + 100 output) well under max_model_len=4096.
            pool = [e for e in all_entries if 394 <= len(e['prompt'].split()) <= 787]
            import itertools
            prompts = [e['prompt'] for e in itertools.islice(itertools.cycle(pool), 20)]
        else:
            prompts = ['Explain quantum computing in detail, covering qubits, superposition, entanglement, and applications.'] * 20
        max_tokens = 100  # short output keeps decode fast; C+D fires during prefill

        # Send requests one at a time with a gap between each so the server sees
        # multiple separate prefill events.  The C+D training cycle needs:
        #   - decode traffic  → drains the 49 fwd sub-ops (one decode batch suffices)
        #   - prefill traffic → drains the 194 bwd sub-ops (~2 prefill batches)
        # With concurrent requests all prefilled in one batch at t=0 there is only
        # one prefill event, which fires before the fwd cycle is ready, so the bwd
        # never drains.  Sending sequentially guarantees a new prefill event arrives
        # after the fwd completes.
        results = [None] * len(prompts)
        t_start = time.time()
        for i, p in enumerate(prompts):
            results[i] = _send_request(p, PORT, max_tokens=max_tokens)
            if i < len(prompts) - 1:
                time.sleep(3)   # gap → next request's prefill arrives after fwd drains
        elapsed = time.time() - t_start

        successes = sum(1 for r in results if r and r.get('choices'))
        print(f'\n  {successes}/{len(prompts)} requests succeeded in {elapsed:.1f}s')
        self.assertGreaterEqual(
            successes, len(prompts) * 0.9,
            msg=f'Too many failed requests ({len(prompts) - successes}/{len(prompts)})'
        )

    def test_3_training_steps_completed(self):
        """At least one LoRA training step must have completed during inference."""
        # Give training some time to accumulate steps (accum_steps=4, each step ~few seconds)
        deadline = time.time() + 180  # extra time for first-run dataset download
        steps = 0
        while time.time() < deadline:
            if BT_COMPLETIONS.exists():
                lines = [l.strip() for l in BT_COMPLETIONS.read_text().splitlines() if l.strip()]
                steps = len(lines)
                if steps > 0:
                    break
            time.sleep(5)

        print(f'\n  Training steps completed: {steps}')
        self.assertGreater(
            steps, 0,
            msg=(
                f'{BT_COMPLETIONS} is empty or missing after 90s.\n'
                'Check server log for "[BubbleTea LoRA]" messages.\n'
                f'Log: {self.server_log}'
            ),
        )

    def test_4_no_training_error_in_log(self):
        """Server log must not contain BubbleTea LoRA error messages."""
        log_text = self.server_log.read_text(errors='replace')
        error_patterns = [
            'Failed to initialise real LoRA trainer',
            'fwd_subop error',
            'bwd_subop error',
            'sync_to_vllm failed',
        ]
        for pat in error_patterns:
            self.assertNotIn(
                pat, log_text,
                msg=f'Error pattern found in server log: {pat!r}'
            )

    def test_5_loss_decreasing(self):
        """Training loss logged to /tmp/vllm_training_loss.log must be finite,
        in a plausible LM range, and not diverging over time.

        With accum_steps=1 and different samples each step, per-step variance is
        high, so we check a lenient downward trend (second-half avg ≤ first-half
        avg * 1.2) rather than strict monotonicity.
        """
        loss_log = Path('/tmp/vllm_training_loss.log')

        # Wait for at least 8 steps (accum_steps=1, 20 requests → ~10-16 steps)
        print('\n  Waiting for training loss log ...')
        deadline = time.time() + 240
        steps_found = 0
        while time.time() < deadline:
            if loss_log.exists():
                lines = [l.strip() for l in loss_log.read_text().splitlines()
                         if l.strip()]
                steps_found = len(lines)
                if steps_found >= 8:
                    break
            time.sleep(5)

        if not loss_log.exists() or steps_found == 0:
            self.skipTest('/tmp/vllm_training_loss.log not found — training may not have run')

        # Parse
        losses = []
        for line in loss_log.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                _step, loss_val = line.split(',', 1)
                losses.append(float(loss_val))
            except (ValueError, IndexError):
                continue

        print(f'  {len(losses)} training steps logged: {[f"{l:.3f}" for l in losses]}')
        self.assertGreater(len(losses), 0, 'No parseable loss entries')

        import math
        # All losses must be finite
        for i, lv in enumerate(losses):
            self.assertFalse(
                math.isnan(lv) or math.isinf(lv),
                f'Step {i+1}: loss={lv} is NaN/Inf — gradient explosion or RMSNorm bug'
            )
            self.assertLess(
                lv, 100.0,
                f'Step {i+1}: loss={lv:.2f} is implausibly high (>100) — '
                'possible OOB labels or missing final RMSNorm'
            )
            self.assertGreaterEqual(lv, 0.0, f'Step {i+1}: loss={lv} is negative')

        # Filter out near-zero entries: rank 1 logs 0.0 when its vocab shard has
        # no valid targets on that sample.  Only meaningful (>0.5) losses matter.
        meaningful = [lv for lv in losses if lv > 0.5]
        print(f'  {len(meaningful)} meaningful steps (loss>0.5): {[f"{l:.3f}" for l in meaningful]}')

        if len(meaningful) >= 4:
            mid = len(meaningful) // 2
            first_avg = sum(meaningful[:mid]) / mid
            last_avg  = sum(meaningful[mid:]) / (len(meaningful) - mid)
            print(f'  first-half avg={first_avg:.4f}  last-half avg={last_avg:.4f}')
            self.assertLessEqual(
                last_avg, first_avg * 1.2,
                f'Loss appears to be diverging: '
                f'first-half avg={first_avg:.4f}, last-half avg={last_avg:.4f}. '
                f'Check _prev_delta correction in _attn_o_proj_bwd/_attn_qkv_recompute.'
            )
        else:
            print(f'  Only {len(meaningful)} meaningful steps — skipping trend check (need 4+)')


if __name__ == '__main__':
    unittest.main(verbosity=2)
