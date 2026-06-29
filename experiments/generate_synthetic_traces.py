#!/usr/bin/env python3
"""
generate_synthetic_traces.py — generate JSON trace files for 3-way benchmark
comparisons (vLLM / LLMStation / Bubble Tea).

Each trace file follows the same schema as burstgpt_trace_replay.py expects:
    {
      "name": "synth_2048_poisson_1.0rps",
      "compress": 1.0,
      "description": "...",
      "requests": [
        {"prompt_len": 2048, "output_len": 256, "arrival_s": 0.0},
        ...
      ]
    }

Three arrival distributions are supported:
  - poisson:    Exponential inter-arrivals (memoryless, standard baseline)
  - gamma:      Gamma(shape=0.3) inter-arrivals (same mean, heavier tail → bursty clusters)
  - burst_lull: Explicit 3-phase pattern: burst → lull → burst

Usage:
    python generate_synthetic_traces.py \
      --input-lens 2048 4096 8192 16384 \
      --output-len 256 \
      --num-requests 200 \
      --rates 0.4 1.0 4.0 \
      --distributions poisson gamma burst_lull \
      --output-dir compare_results/synthetic_traces \
      --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def generate_poisson_arrivals(
    num_requests: int, rate: float, rng: np.random.Generator
) -> np.ndarray:
    inter_arrivals = rng.exponential(scale=1.0 / rate, size=num_requests)
    inter_arrivals[0] = 0.0
    return np.cumsum(inter_arrivals)


def generate_gamma_arrivals(
    num_requests: int, rate: float, rng: np.random.Generator, shape: float = 0.3
) -> np.ndarray:
    scale = 1.0 / (rate * shape)
    inter_arrivals = rng.gamma(shape=shape, scale=scale, size=num_requests)
    inter_arrivals[0] = 0.0
    return np.cumsum(inter_arrivals)


def generate_burst_lull_arrivals(
    num_requests: int, rate: float, rng: np.random.Generator
) -> np.ndarray:
    n_burst1 = num_requests * 2 // 5   # 80
    n_lull = num_requests // 5          # 40
    n_burst2 = num_requests - n_burst1 - n_lull  # 80

    burst_rate = rate * 5.0
    lull_rate = rate * 0.1

    dt_burst1 = rng.exponential(scale=1.0 / burst_rate, size=n_burst1)
    dt_lull = rng.exponential(scale=1.0 / lull_rate, size=n_lull)
    dt_burst2 = rng.exponential(scale=1.0 / burst_rate, size=n_burst2)

    inter_arrivals = np.concatenate([dt_burst1, dt_lull, dt_burst2])
    inter_arrivals[0] = 0.0
    return np.cumsum(inter_arrivals)


GENERATORS = {
    "poisson": generate_poisson_arrivals,
    "gamma": generate_gamma_arrivals,
    "burst_lull": generate_burst_lull_arrivals,
}

DIST_LABELS = {
    "poisson": "Poisson (exponential inter-arrivals)",
    "gamma": "Gamma-bursty (shape=0.3, same mean rate)",
    "burst_lull": "Burst-lull-burst (80/40/80 split, 5×/0.1×/5× rate)",
}


def build_trace(
    input_len: int,
    output_len: int,
    num_requests: int,
    rate: float,
    distribution: str,
    rng: np.random.Generator,
) -> dict:
    arrivals = GENERATORS[distribution](num_requests, rate, rng)
    name = f"synth_{input_len}_out{output_len}_{distribution}_{rate}rps"
    description = (
        f"{num_requests} synthetic requests, prompt_len={input_len}, "
        f"output_len={output_len}, {DIST_LABELS[distribution]}, "
        f"λ={rate} req/s, span={arrivals[-1]:.1f}s"
    )
    requests = [
        {
            "prompt_len": input_len,
            "output_len": output_len,
            "arrival_s": round(float(t), 4),
        }
        for t in arrivals
    ]
    return {"name": name, "compress": 1.0, "description": description, "requests": requests}


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic trace files for 3-way benchmark comparison"
    )
    parser.add_argument(
        "--input-lens", type=int, nargs="+", default=[2048, 4096, 8192, 16384],
        help="Prompt token lengths to generate traces for",
    )
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument(
        "--rates", type=float, nargs="+", default=[0.4, 1.0, 4.0],
        help="Request rates in req/s",
    )
    parser.add_argument(
        "--distributions", nargs="+", default=["poisson", "gamma", "burst_lull"],
        choices=list(GENERATORS.keys()),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "compare_results" / "synthetic_traces",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    count = 0
    for input_len in args.input_lens:
        for rate in args.rates:
            for dist in args.distributions:
                trace = build_trace(
                    input_len, args.output_len, args.num_requests, rate, dist, rng
                )
                path = args.output_dir / f"{trace['name']}.json"
                path.write_text(json.dumps(trace, indent=2) + "\n")
                span = trace["requests"][-1]["arrival_s"]
                print(f"  {path.name:55s}  {args.num_requests} reqs, span={span:>8.1f}s")
                count += 1

    print(f"\nGenerated {count} trace files in {args.output_dir}/")


if __name__ == "__main__":
    main()
