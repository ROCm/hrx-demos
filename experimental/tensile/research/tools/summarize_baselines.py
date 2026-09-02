#!/usr/bin/env python3
"""Reduce raw blas-probe outputs to small, reviewable baseline evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import cast


JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class Options:
    inputs: tuple[Path, ...]
    output: Path | None


def parse_options() -> Options:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    return Options(inputs=tuple(args.inputs), output=args.output)


def require_dict(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return cast(dict[str, object], value)


def require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    return cast(list[object], value)


def require_number(value: object, context: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    return float(value)


def median_us(record: dict[str, object]) -> float:
    samples = [
        require_number(value, "timing sample")
        for value in require_list(record.get("time_us"), "time_us")
    ]
    if not samples:
        raise ValueError("time_us must not be empty")
    return statistics.median(samples)


def mismatch_count(record: dict[str, object]) -> int:
    correctness = require_dict(record.get("correctness"), "correctness")
    value = correctness.get("mismatches")
    if not isinstance(value, int):
        raise TypeError("correctness.mismatches must be an integer")
    return value


def compact_algorithm(record: dict[str, object], operations: float) -> dict[str, JsonValue]:
    timing = median_us(record)
    result: dict[str, JsonValue] = {
        "index": cast(int, record["index"]),
        "median_us": round(timing, 3),
        "nominal_tflops": round(operations / (timing * 1.0e6), 3),
        "workspace_bytes": cast(int, record.get("workspace_bytes", 0)),
        "mismatches": mismatch_count(record),
    }
    for key in ("solution_name", "kernel_name"):
        value = record.get(key)
        if isinstance(value, str):
            result[key] = value
    return result


def summarize(path: Path) -> dict[str, JsonValue]:
    encoded = path.read_bytes()
    root = require_dict(json.loads(encoded), "probe root")
    if root.get("schema") != "loom-blas.probe.v1":
        raise ValueError(f"unexpected schema in {path}")
    request = require_dict(root.get("request"), "request")
    m = require_number(request.get("m"), "request.m")
    n = require_number(request.get("n"), "request.n")
    k = require_number(request.get("k"), "request.k")
    operations = 2.0 * m * n * k

    algorithm_values = require_list(root.get("algorithms"), "algorithms")
    algorithms = [require_dict(value, "algorithm") for value in algorithm_values]
    correct = [record for record in algorithms if mismatch_count(record) == 0]
    if not correct:
        raise ValueError(f"no correct algorithms in {path}")
    ranked = sorted(correct, key=median_us)
    default = require_dict(root.get("default_query"), "default_query")
    default_time = median_us(default)
    best_time = median_us(ranked[0])
    device = require_dict(root.get("device"), "device")

    return {
        "artifact": str(path),
        "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
        "backend": cast(str, root["backend"]),
        "device": cast(dict[str, JsonValue], device),
        "request": cast(dict[str, JsonValue], request),
        "enumerated_count": cast(int, root["enumerated_count"]),
        "tested_count": len(algorithms),
        "incorrect_count": len(algorithms) - len(correct),
        "default": compact_algorithm(default, operations),
        "best": compact_algorithm(ranked[0], operations),
        "default_slowdown_percent": round((default_time / best_time - 1.0) * 100.0, 2),
        "top_five": [compact_algorithm(record, operations) for record in ranked[:5]],
    }


def main() -> None:
    options = parse_options()
    output: dict[str, JsonValue] = {
        "schema": "loom-blas.baseline-summary.v1",
        "experiments": [summarize(path) for path in options.inputs],
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if options.output is None:
        print(rendered, end="")
    else:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(rendered)
        if not options.output.is_file() or options.output.stat().st_size == 0:
            raise RuntimeError(f"failed to write {options.output}")


if __name__ == "__main__":
    main()
