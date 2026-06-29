#!/usr/bin/env python3
"""
burstgpt_trace_replay.py — replay a BurstGPT trace window's *real* request
arrival timing and token-length distribution against a running
OpenAI-completions-compatible server (vLLM / BubbleTea / LLMStation).

Why this exists
----------------
`vllm bench serve --dataset-name burstgpt` only uses the BurstGPT CSV's
`Request tokens` / `Response tokens` columns to sample synthetic prompts; the
`Timestamp` column (the actual arrival times) is discarded, and requests are
instead dispatched on a synthetic Poisson/gamma process driven by
`--request-rate`/`--burstiness`. This script instead replays the *actual*
relative inter-arrival deltas of a contiguous slice of the trace (see
`compare_results/burstgpt_traces/*.json`, produced from
`BurstGPT_without_fails_2.csv`).

Prompt construction mirrors vLLM's BurstGPTDataset: for request index i with
recorded prompt length L, the prompt is `tokenizer.decode([(i + j) %
vocab_size for j in range(L)])` — synthetic content, real length.

Output is printed in the same "Serving Benchmark Result" text format that
`vllm bench serve` produces, so `compare_benchmark.py`'s `_parse_bench_log`
regexes work unchanged on this script's stdout.

Usage
-----
    python burstgpt_trace_replay.py \
        --model /path/to/model \
        --port 8000 \
        --trace-file compare_results/burstgpt_traces/window1_busiest_hour.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm.benchmarks.lib.endpoint_request_func import (  # noqa: E402
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)
from vllm.tokenizers import get_tokenizer  # noqa: E402


def load_trace(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    requests = data["requests"]
    print(f"Loaded trace '{data.get('name', path.name)}': {len(requests)} requests, "
          f"span={requests[-1]['arrival_s']:.1f}s "
          f"(compress={data.get('compress', 1.0)})")
    if "description" in data:
        print(f"  {data['description']}")
    return requests


def build_prompts(requests: list[dict], tokenizer) -> list[str]:
    vocab_size = tokenizer.vocab_size
    prompts = []
    for i, req in enumerate(requests):
        token_ids = [(i + j) % vocab_size for j in range(req["prompt_len"])]
        prompts.append(tokenizer.decode(token_ids))
    return prompts


async def run_one(
    request_func,
    session: aiohttp.ClientSession,
    req_input: RequestFuncInput,
    target_time: float,
    t_start: float,
    pbar,
) -> RequestFuncOutput:
    delay = target_time - (time.perf_counter() - t_start)
    if delay > 0:
        await asyncio.sleep(delay)
    return await request_func(request_func_input=req_input, session=session, pbar=pbar)


async def main_async(args: argparse.Namespace) -> dict:
    requests = load_trace(args.trace_file)

    tokenizer = get_tokenizer(args.tokenizer or args.model,
                               trust_remote_code=args.trust_remote_code)
    prompts = build_prompts(requests, tokenizer)

    request_func = ASYNC_REQUEST_FUNCS[args.backend]
    api_url = f"{args.base_url}{args.endpoint}"

    connector = aiohttp.TCPConnector(
        limit=0, limit_per_host=0, ttl_dns_cache=300, use_dns_cache=True,
        keepalive_timeout=60, enable_cleanup_closed=True, force_close=False,
        ssl=False,
    )
    session = aiohttp.ClientSession(
        connector=connector, trust_env=True,
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
    )

    try:
        # Warm-up / readiness probe with the first request's shape.
        print("Starting initial single prompt test run...")
        test_input = RequestFuncInput(
            model=args.model, model_name=args.model,
            prompt=prompts[0], api_url=api_url,
            prompt_len=requests[0]["prompt_len"],
            output_len=min(requests[0]["output_len"], 16),
            ignore_eos=args.ignore_eos,
        )
        test_output = await request_func(request_func_input=test_input, session=session)
        if not test_output.success:
            raise RuntimeError(f"Initial test run failed: {test_output.error}")
        print("Initial test run completed. Starting trace replay...")

        from tqdm.asyncio import tqdm
        pbar = None if args.disable_tqdm else tqdm(total=len(requests))

        t_start = time.perf_counter()
        tasks = []
        for i, (req, prompt) in enumerate(zip(requests, prompts)):
            req_input = RequestFuncInput(
                model=args.model, model_name=args.model,
                prompt=prompt, api_url=api_url,
                prompt_len=req["prompt_len"],
                output_len=req["output_len"],
                ignore_eos=args.ignore_eos,
            )
            tasks.append(asyncio.create_task(
                run_one(request_func, session, req_input, req["arrival_s"], t_start, pbar)
            ))

        outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
        duration = time.perf_counter() - t_start
        if pbar is not None:
            pbar.close()
    finally:
        await session.close()

    return summarize(requests, outputs, duration, tokenizer)


def summarize(requests, outputs: list[RequestFuncOutput], duration: float, tokenizer) -> dict:
    completed = 0
    total_input = 0
    actual_output_lens = []
    ttfts, tpots, itls, e2els = [], [], [], []
    for req, out in zip(requests, outputs):
        if not out.success:
            actual_output_lens.append(0)
            continue
        output_len = out.output_tokens
        if not output_len:
            output_len = len(tokenizer(out.generated_text, add_special_tokens=False).input_ids)
        actual_output_lens.append(output_len)
        total_input += out.prompt_len
        if output_len > 1:
            tpots.append((out.latency - out.ttft) / (output_len - 1))
        itls.extend(out.itl)
        ttfts.append(out.ttft)
        e2els.append(out.latency)
        completed += 1

    total_output = sum(actual_output_lens)

    def pct(vals, p):
        return float(np.percentile(vals, p)) * 1000 if vals else 0.0

    def stat(vals, fn):
        return float(fn(vals)) * 1000 if vals else 0.0

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", duration))
    print("{:<40} {:<10}".format("Total input tokens:", total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", completed / duration))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", total_output / duration))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                     (total_input + total_output) / duration))

    result = {
        "duration": duration,
        "completed": completed,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": completed / duration,
        "output_throughput": total_output / duration,
        "ttfts": ttfts,
        "itls": itls,
        "errors": [o.error for o in outputs],
    }

    for attr_name, label, header, vals in [
        ("ttft", "TTFT", "Time to First Token", ttfts),
        ("tpot", "TPOT", "Time per Output Token (excl. 1st token)", tpots),
        ("itl", "ITL", "Inter-token Latency", itls),
        ("e2el", "E2EL", "End-to-end Latency", e2els),
    ]:
        print("{s:{c}^{n}}".format(s=header, n=50, c="-"))
        mean_v = stat(vals, np.mean)
        med_v = stat(vals, np.median)
        std_v = stat(vals, np.std)
        p99_v = pct(vals, 99)
        print("{:<40} {:<10.2f}".format(f"Mean {label} (ms):", mean_v))
        print("{:<40} {:<10.2f}".format(f"Median {label} (ms):", med_v))
        print("{:<40} {:<10.2f}".format(f"P99 {label} (ms):", p99_v))
        result[f"mean_{attr_name}_ms"] = mean_v
        result[f"median_{attr_name}_ms"] = med_v
        result[f"std_{attr_name}_ms"] = std_v
        result[f"p99_{attr_name}_ms"] = p99_v

    print("=" * 50)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--backend", default="vllm", choices=list(ASYNC_REQUEST_FUNCS.keys()))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--endpoint", default="/v1/completions")
    ap.add_argument("--trace-file", type=Path, required=True)
    ap.add_argument("--ignore-eos", action="store_true", default=True)
    ap.add_argument("--disable-tqdm", action="store_true")
    ap.add_argument("--result-dir", type=Path, default=None)
    ap.add_argument("--result-filename", default=None)
    args = ap.parse_args()

    if args.base_url is None:
        args.base_url = f"http://{args.host}:{args.port}"

    result = asyncio.run(main_async(args))

    if args.result_dir:
        fname = args.result_filename or f"burstgpt_trace_{args.trace_file.stem}.json"
        out = args.result_dir / fname
        out.write_text(json.dumps(result, indent=2))
        print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
