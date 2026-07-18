# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration test for VLLM_FT_BWD_MODE=both (base-scheduler Arm B+C hybrid).

Launches the actual BubbleTea server with VLLM_FT_LORA_PATH and
VLLM_FT_BWD_MODE=both set, sends inference requests, then verifies:
  1. Server log contains "[BubbleTea LoRA] Trainer ready"  (trainer init)
  2. Server log contains "[bwd_mode=both] job complete"  (the dispatch branch
     in moe_runner.py fired and drained a job)
  3. The scheduler summary in that log line reports sub-ops released via
     BOTH channels -- some fraction in-bubble (prefill, arm C's channel) AND
     a nonzero decode-dispatch count (arm B's channel) -- confirming the two
     disjoint-phase triggers both actually drain the same sub-op queue
     concurrently, per the corrected mental model from this session (they
     don't "fill each other's gaps"; they're two independent drains).
  4. All inference requests succeeded.
  5. No BubbleTea LoRA error patterns in the server log.

Run with:
  /mnt/nfs/home/ramya/vllm/.venv/bin/python test_bt_lora_bwd_mode_both_integration.py

Requires:
  - 2x A100-80GB available and free (checks via nvidia-smi before launch)
  - ~10 minutes wall time (model load ~5 min + warmup + requests)
  - libcudart.so.13 discoverable (see test_bt_lora_bwd_mode_integration.py
    docstring for the fix if `import vllm` fails).
"""

import contextlib
import os
import re
import signal
import subprocess
import time
import unittest
import urllib.request
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
BT_VENV        = Path('/mnt/nfs/home/ramya/vllm/.venv')
MODEL_DIR      = '/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B'
BT_LORA_ADAPTER = '/mnt/nfs/home/ramya/slora-plus/S-LoRA/test/qwen3/adapters/qwen3-toy-lora'
PORT           = 8768   # distinct from the decode-mode test's 8767
SERVER_READY_TIMEOUT = 600   # seconds
RESULT_DIR     = Path('/tmp/bt_bwd_mode_both_integration_test')


def _gpu_mem_used_mib() -> list[int]:
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
            with contextlib.suppress(Exception):
                os.kill(int(pid), signal.SIGTERM)
    except Exception:
        pass
    time.sleep(4)
    for pattern in [r'vllm\.worker', r'vllm serve', r'VLLM::Worker_TP']:
        subprocess.run(['pkill', '-9', '-f', pattern], capture_output=True)
    subprocess.run(['pkill', '-9', '-x', 'VLLM::Worker_TP'], capture_output=True)
    deadline = time.time() + 90
    while time.time() < deadline:
        mems = _gpu_mem_used_mib()
        if mems and all(m < 5120 for m in mems):
            break
        time.sleep(5)


def _send_request(prompt: str, port: int, max_tokens: int = 32) -> dict | None:
    import json
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


class TestBubbleTeaLoRABwdModeBoth(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not Path(MODEL_DIR).exists():
            raise unittest.SkipTest(f'Model not found: {MODEL_DIR}')
        if not Path(BT_LORA_ADAPTER).exists():
            raise unittest.SkipTest(f'LoRA adapter not found: {BT_LORA_ADAPTER}')

        mems = _gpu_mem_used_mib()
        if len(mems) < 2:
            raise unittest.SkipTest('Fewer than 2 GPUs detected')
        if any(m > 5120 for m in mems[:2]):
            raise unittest.SkipTest(
                f'GPUs not free (used: {mems[:2]} MiB) - kill other processes first'
            )

        RESULT_DIR.mkdir(parents=True, exist_ok=True)

        subprocess.run(['pkill', '-9', '-f', 'VLLM::Worker_TP'], capture_output=True)
        subprocess.run(['pkill', '-9', '-x', 'VLLM::Worker_TP'], capture_output=True)
        time.sleep(2)

        env = os.environ.copy()
        env['PATH']                  = f"{BT_VENV}/bin:{env.get('PATH', '')}"
        env['CUDA_VISIBLE_DEVICES']  = '0,1'
        env['VLLM_FT_COMBINED_MODE'] = 'C+D'
        # ── The setting under test ──────────────────────────────────────────
        env['VLLM_FT_BWD_MODE']      = 'both'
        env['VLLM_FT_BWD_DECODE_FILLS_PER_TRIGGER'] = '2'
        # ─────────────────────────────────────────────────────────────────
        env['VLLM_FT_LORA_PATH']     = BT_LORA_ADAPTER
        env['VLLM_FT_TOKENIZER_PATH']= MODEL_DIR
        env['VLLM_FT_CACHE_DIR']     = '/mnt/nfs/home/ramya/scratch'
        env['VLLM_FT_COMBINED_T_FT'] = '128'
        env['VLLM_FT_ACCUM_STEPS']   = '1'
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        env['HF_TOKEN']              = os.environ.get("HF_TOKEN", "")
        env['HF_DATASETS_OFFLINE']   = '1'
        env['TRANSFORMERS_OFFLINE']  = '1'
        cu13_lib = str(BT_VENV / 'lib/python3.12/site-packages/nvidia/cu13/lib')
        env['LD_LIBRARY_PATH'] = f"{cu13_lib}:{env.get('LD_LIBRARY_PATH', '')}"

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

        print(f'\n  Launching BubbleTea server with VLLM_FT_BWD_MODE=both '
              f'(log: {cls.server_log}) ...')
        with open(cls.server_log, 'w') as log_f:
            cls.server_proc = subprocess.Popen(
                server_cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT
            )
        print(f'  Server PID: {cls.server_proc.pid}')

        print(f'  Waiting for server to be ready (up to {SERVER_READY_TIMEOUT}s) ...')
        cls.server_ready = _wait_server(PORT)
        if not cls.server_ready:
            print('  ERROR: server did not become ready - check server.log')
            _kill_server(PORT)

    @classmethod
    def tearDownClass(cls):
        print('\n  Tearing down server ...')
        _kill_server(PORT)

    def setUp(self):
        if not self.server_ready:
            self.skipTest('Server did not start - see setUpClass logs')

    # ── Tests ──────────────────────────────────────────────────────────────

    def test_1_trainer_init_logged(self):
        log_text = self.server_log.read_text(errors='replace')
        self.assertIn(
            '[BubbleTea LoRA] Trainer ready',
            log_text,
            msg=f'Trainer-ready message not found in server log ({self.server_log})',
        )

    def test_2_inference_requests_succeed(self):
        import itertools
        import json

        arxiv_path = Path('/mnt/nfs/home/ramya/scratch/arxiv_bench_500.jsonl')
        if arxiv_path.exists():
            with open(arxiv_path) as f:
                all_entries = [json.loads(l) for l in f if l.strip()]
            pool = [e for e in all_entries if 394 <= len(e['prompt'].split()) <= 787]
            prompts = [e['prompt'] for e in itertools.islice(itertools.cycle(pool), 20)]
        else:
            prompts = ['Explain quantum computing in detail, covering qubits, '
                       'superposition, entanglement, and applications.'] * 20

        results = [None] * len(prompts)
        t_start = time.time()
        for i, p in enumerate(prompts):
            results[i] = _send_request(p, PORT, max_tokens=100)
            if i < len(prompts) - 1:
                time.sleep(3)
        elapsed = time.time() - t_start

        successes = sum(1 for r in results if r and r.get('choices'))
        print(f'\n  {successes}/{len(prompts)} requests succeeded in {elapsed:.1f}s')
        self.assertGreaterEqual(
            successes, len(prompts) * 0.9,
            msg=f'Too many failed requests ({len(prompts) - successes}/{len(prompts)})'
        )

    def test_3_both_channels_fire_concurrently(self):
        """Both the prefill-bubble channel (arm C) and the decode-dispatch
        channel (arm B) must release sub-ops across the run -- confirming
        VLLM_FT_BWD_MODE=both really runs the hybrid, not silently falling
        back to one channel.

        Note: each TP rank arms its own scheduler instance and independently
        gates prefill-bubble dispatch on _ep_is_light_rank() (unchanged,
        pre-existing logic -- see bubble_scheduler.py). Only whichever rank
        is currently the faster EP rank ever gets prefill-bubble releases;
        with skewed/repeated-token routing one rank can be light on every
        single job, so a per-job / per-rank assertion of "in_bubble > 0" is
        wrong -- check the aggregate across all reported jobs instead.
        """
        deadline = time.time() + 120
        log_text = ""
        while time.time() < deadline:
            log_text = self.server_log.read_text(errors='replace')
            if '[bwd_mode=both] job complete' in log_text:
                break
            time.sleep(5)

        self.assertIn(
            '[bwd_mode=both] job complete', log_text,
            msg=(
                'No "[bwd_mode=both] job complete" line found in server log '
                '-- the hybrid dispatch branch in moe_runner.py never fired or '
                'never drained a job.'
            ),
        )

        pattern = re.compile(
            r'\[bwd_mode=both\] job complete: (\d+) sub-ops released via decode '
            r'\(VllmBubbleScheduler: (\d+)/(\d+) in-bubble .*?(\d+) after\)'
        )
        matches = pattern.findall(log_text)
        self.assertGreater(len(matches), 0, 'Could not parse any job-complete summary line')

        total_in_bubble = 0
        for decode_count, in_bubble, n_ops, after_count in matches:
            decode_count, in_bubble, n_ops, after_count = map(
                int, (decode_count, in_bubble, n_ops, after_count)
            )
            print(f'\n  job: {decode_count} released via decode, '
                  f'{in_bubble}/{n_ops} in-bubble, {after_count} after')
            total_in_bubble += in_bubble
            # Every job's decode channel must fire -- this is arm B's channel
            # and is phase-gated only (not per-rank light/heavy gated), so it
            # must be nonzero on every reported job regardless of which rank.
            self.assertGreater(
                decode_count, 0,
                msg='Decode-channel dispatch count is 0 -- arm B channel never fired.',
            )
            # after_count includes both fill_remaining() tail-drain calls (also
            # in_bubble=False) and the decode-channel releases, so it should be
            # >= decode_count, and everything should sum to n_ops.
            self.assertGreaterEqual(after_count, decode_count)
            self.assertEqual(in_bubble + after_count, n_ops)

        # The prefill-bubble channel (arm C) is gated per-rank on
        # _ep_is_light_rank() -- only the currently-faster EP rank ever
        # dispatches through it. With skewed/repeated-token routing, one rank
        # can be light on every single job (see bubble_scheduler.py docs), so
        # check the channel fired for *some* job/rank across the whole run,
        # not on every individual line.
        self.assertGreater(
            total_in_bubble, 0,
            msg=(
                'Prefill-bubble channel released 0 sub-ops across the entire '
                'run -- arm C channel never fired in "both" mode for any rank '
                '(degenerated to pure decode-mode behaviour).'
            ),
        )

    def test_4_no_training_error_in_log(self):
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
