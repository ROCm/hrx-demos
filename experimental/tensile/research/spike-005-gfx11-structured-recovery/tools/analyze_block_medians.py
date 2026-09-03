#!/usr/bin/env python3
"""Recompute the independent block-median bootstrap in performance.json."""

from argparse import ArgumentParser
import json
from pathlib import Path
import random
from statistics import median


def percentile(sorted_values: list[float], probability: float) -> float:
    index = int(probability * len(sorted_values))
    return sorted_values[min(index, len(sorted_values) - 1)]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()

    record = json.loads(args.result.read_text())
    loom = [float(value) for value in record["loom"]["run_p50_us"]]
    vendor = [float(value) for value in record["vendor"]["run_p50_us"]]
    resamples = int(record["ratio"]["resamples"])
    rng = random.Random(int(record["ratio"]["seed"]))
    ratios = [
        median(rng.choices(loom, k=len(loom)))
        / median(rng.choices(vendor, k=len(vendor)))
        for _ in range(resamples)
    ]
    ratios.sort()
    output = {
        "loom_median_us": median(loom),
        "vendor_median_us": median(vendor),
        "ratio": median(loom) / median(vendor),
        "bootstrap_95_percent": [
            percentile(ratios, 0.025),
            percentile(ratios, 0.975),
        ],
        "passed": percentile(ratios, 0.975)
        <= float(record["gate"]["maximum_upper_ratio"]),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
