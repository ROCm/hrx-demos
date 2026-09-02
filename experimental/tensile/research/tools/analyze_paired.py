#!/usr/bin/env python3
"""Summarize a loom-blas paired timing record with a reproducible bootstrap CI."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="loom-blas.paired-probe JSON")
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> None:
    args = parse_args()
    if args.resamples <= 0:
        raise ValueError("--resamples must be positive")
    record = json.loads(args.input.read_text())
    loom = record["loom"]["time_us"]
    incumbent = record["incumbent"]["time_us"]
    if len(loom) != len(incumbent) or not loom:
        raise ValueError("timing arrays must be nonempty and have equal lengths")

    rng = random.Random(args.seed)
    ratios: list[float] = []
    # Resample the two marginal distributions independently. Alternating launch
    # order controls drift, but array position does not define a physical pair.
    for _ in range(args.resamples):
        loom_sample = [loom[rng.randrange(len(loom))] for _ in loom]
        incumbent_sample = [
            incumbent[rng.randrange(len(incumbent))] for _ in incumbent
        ]
        ratios.append(
            statistics.median(loom_sample) / statistics.median(incumbent_sample)
        )

    summary = {
        "schema": "loom-blas.timing-summary.v1",
        "source": str(args.input),
        "sample_count": len(loom),
        "loom_median_us": statistics.median(loom),
        "incumbent_median_us": statistics.median(incumbent),
        "median_ratio": statistics.median(loom) / statistics.median(incumbent),
        "bootstrap": {
            "method": "independent resampling of alternating same-process samples",
            "resamples": args.resamples,
            "seed": args.seed,
            "confidence": 0.95,
            "ratio_ci": [percentile(ratios, 0.025), percentile(ratios, 0.975)],
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
