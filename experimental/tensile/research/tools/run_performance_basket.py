#!/usr/bin/env python3
"""Benchmark a bounded Loom interior basket against exhaustive vendor APIs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def invoke(command: list[str], environment: dict[str, str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, env=environment)
    if result.returncode != 0:
        raise RuntimeError(f"{command[0]} failed ({result.returncode}):\n{result.stderr}")
    return result.stdout


def json_line(output: str) -> dict[str, object]:
    for line in reversed(output.splitlines()):
        if line.startswith("{"):
            value = json.loads(line)
            if isinstance(value, dict):
                return value
    raise ValueError("tool produced no JSON object")


def vendor_result(
    probe: Path,
    backend: str,
    device: int,
    m: int,
    n: int,
    k: int,
    environment: dict[str, str],
) -> dict[str, object]:
    base = [
        str(probe), "--backend", backend, "--device", str(device),
        "--m", str(m), "--n", str(n), "--k", str(k),
    ]
    sweep = json.loads(
        invoke(
            base
            + [
                "--warmup", "1", "--iterations", "3",
                "--reference-samples", "32", "--skip-algorithm-correctness",
            ],
            environment,
        )
    )
    algorithms = sweep["algorithms"]
    if not algorithms:
        raise RuntimeError(f"{backend} has no eligible algorithms")
    winner = min(
        algorithms,
        key=lambda value: statistics.median(float(item) for item in value["time_us"]),
    )
    forced = json.loads(
        invoke(
            base
            + [
                "--warmup", "5", "--iterations", "20",
                "--reference-samples", "1024", "--solution-index", str(winner["index"]),
            ],
            environment,
        )
    )
    selected = forced["algorithms"][0]
    return {
        "backend": backend,
        "enumerated_count": sweep["enumerated_count"],
        "eligible_count": len(algorithms),
        "solution_index": selected["index"],
        "solution_name": selected.get("solution_name"),
        "kernel_name": selected.get("kernel_name"),
        "workspace_bytes": selected.get("workspace_bytes", 0),
        "samples_us": selected["time_us"],
        "median_us": statistics.median(float(item) for item in selected["time_us"]),
        "correctness": selected["correctness"],
    }


def loom_result(
    runner: Path,
    fixture_tool: Path,
    source: Path,
    symbol: str,
    visible_device: int,
    m: int,
    n: int,
    k: int,
    environment: dict[str, str],
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="loom-blas-perf-") as raw_directory:
        fixture = Path(raw_directory) / "fixture.loom"
        invoke(
            [
                sys.executable, str(fixture_tool), str(source), str(fixture), "--symbol", symbol,
                "--m", str(m), "--n", str(n), "--k", str(k),
            ],
            environment,
        )
        loom_environment = environment.copy()
        loom_environment["ROCR_VISIBLE_DEVICES"] = str(visible_device)
        output = invoke(
            [
                str(runner), str(fixture), "--device=amdgpu", "--pipeline=default",
                f"--benchmark=@spike6_{m}x{n}x{k}", "--measure=dispatch_complete",
                "--batch-size=64", "--iterations=20", "--warmup-iterations=20",
                "--min-time-ms=500", "--input-ring-count=1",
                f"--config=gemm.m={m}", f"--config=gemm.n={n}", f"--config=gemm.k={k}",
            ],
            loom_environment,
        )
    snapshot = json_line(output)
    item = snapshot["work_items"][0]
    measurement = item["measurement"]
    median_us = float(measurement["operation_timing_ns"]["p50"]) / 1000.0
    return {
        "runner": "iree-benchmark-loom",
        "pipeline": "default",
        "median_us": median_us,
        "correctness": item["correctness"],
        "artifact_size": item["compile_report"]["artifact_size"],
        "private_memory_bytes": item["compile_report"]["entries"]["rows"][0]["private_memory_bytes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--visible-device", type=int, required=True)
    parser.add_argument("--vendor-backend", action="append", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--fixture-tool", type=Path, required=True)
    parser.add_argument("--rocm-lib", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(args.rocm_lib)
    records = []
    for ordinal, cell in enumerate(corpus["cells"][: args.limit]):
        m, n, k = (int(cell[name]) for name in ("m", "n", "k"))
        record: dict[str, object] = {
            "ordinal": ordinal,
            "request": {"m": m, "n": n, "k": k},
            "reasons": cell["reasons"],
        }
        try:
            loom = loom_result(
                args.runner, args.fixture_tool, args.source, args.symbol,
                args.visible_device, m, n, k, environment,
            )
            vendors = [
                vendor_result(args.probe, backend, args.device, m, n, k, environment)
                for backend in args.vendor_backend
            ]
            best = min(vendors, key=lambda value: float(value["median_us"]))
            ratio = float(loom["median_us"]) / float(best["median_us"])
            record.update(
                {
                    "loom": loom,
                    "vendors": vendors,
                    "best_vendor": best["backend"],
                    "ratio": ratio,
                    "band": "le_1.05" if ratio <= 1.05 else "le_1.10" if ratio <= 1.10 else "le_1.25" if ratio <= 1.25 else "gt_1.25",
                }
            )
        except RuntimeError as error:
            record["error"] = str(error)
        records.append(record)
    successful = [record for record in records if "error" not in record]
    output = {
        "schema": "loom-blas.performance-basket.v1",
        "scope": "FP16 NN, FP32 accumulation, FP16 output, alpha=1, beta=0, no bias; aligned interiors only",
        "vendor_policy": "exhaustively enumerate eligible public-API solutions with a short sweep, then correctness-check and remeasure the winner",
        "loom_policy": "correctness-gated iree-benchmark-loom execution through the default pipeline",
        "request_count": len(records),
        "successful_count": len(successful),
        "failed_count": len(records) - len(successful),
        "bands": {
            name: sum(record.get("band") == name for record in successful)
            for name in ("le_1.05", "le_1.10", "le_1.25", "gt_1.25")
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
