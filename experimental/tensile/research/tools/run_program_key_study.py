#!/usr/bin/env python3
"""Measure program-derived keys and HSACO cardinality for a router corpus."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import resource
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompileSpec:
    ordinal: int
    m: int
    n: int
    k: int


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], env: dict[str, str] | None = None) -> str:
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command[0]}\n{result.stderr}"
        )
    return result.stdout


def compile_one(
    spec: CompileSpec,
    source_bc: Path,
    formatter: Path,
    benchmark: Path,
    target: str,
    symbol: str,
    namespace: dict[str, object],
    rocm_lib: Path,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="loom-blas-key-") as raw_directory:
        directory = Path(raw_directory)
        config_text = directory / "config.loom"
        config_bc = directory / "config.loombc"
        program_bc = directory / "program.loombc"
        launch_config_bc = directory / "launch-config.loombc"
        hsaco = directory / "kernel.hsaco"
        config_text.write_text(
            f"config.def @gemm.m = {spec.m} : index\n\n"
            f"config.def @gemm.n = {spec.n} : index\n\n"
            f"config.def @gemm.k = {spec.k} : index\n"
        )
        run([str(formatter), str(config_text), "--from=text", "--to=bc", f"--output={config_bc}"])
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = str(rocm_lib)
        output = run(
            [
                str(benchmark),
                str(source_bc),
                str(config_bc),
                target,
                symbol,
                "1",
                str(hsaco),
                str(program_bc),
                str(launch_config_bc),
            ],
            environment,
        )
        timing = json.loads(output)
        program_hash = sha256(program_bc)
        namespace_bytes = json.dumps(namespace, sort_keys=True, separators=(",", ":")).encode()
        key = hashlib.sha256(namespace_bytes + bytes.fromhex(program_hash)).hexdigest()
        return {
            "ordinal": spec.ordinal,
            "request": {"m": spec.m, "n": spec.n, "k": spec.k},
            "program_sha256": program_hash,
            "program_bytes": program_bc.stat().st_size,
            "launch_config_sha256": sha256(launch_config_bc),
            "launch_config_bytes": launch_config_bc.stat().st_size,
            "derived_key": key,
            "hsaco_sha256": sha256(hsaco),
            "hsaco_bytes": hsaco.stat().st_size,
            "timing_us": {
                "setup": timing["setup_us"],
                "link": timing["phases"]["link"]["median_us"],
                "compile": timing["phases"]["compile"]["median_us"],
                "key": timing["phases"]["key"]["median_us"],
                "emit": timing["phases"]["emit"]["median_us"],
                "total": timing["phases"]["total"]["median_us"],
            },
        }


def compile_or_error(*args: object) -> dict[str, object]:
    spec = args[0]
    assert isinstance(spec, CompileSpec)
    try:
        return compile_one(*args)  # type: ignore[arg-type]
    except RuntimeError as error:
        return {
            "ordinal": spec.ordinal,
            "request": {"m": spec.m, "n": spec.n, "k": spec.k},
            "error": str(error),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--compiler-id", required=True)
    parser.add_argument("--formatter", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--rocm-lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--determinism-repeats", type=int, default=10)
    args = parser.parse_args()
    if args.workers < 1 or args.determinism_repeats < 2:
        parser.error("workers must be positive and determinism repeats at least two")
    corpus = json.loads(args.corpus.read_text())
    cells = corpus["cells"]
    specs = [
        CompileSpec(ordinal, int(cell["m"]), int(cell["n"]), int(cell["k"]))
        for ordinal, cell in enumerate(cells)
    ]
    if args.limit:
        specs = specs[: args.limit]
    namespace = {
        "schema": "loom-blas.program-key-namespace.v1",
        "target_profile": args.target,
        "compiler": args.compiler_id,
        "pass_program": "source-to-prepared-low/default",
        "device_abi": "amdgpu-hsa-v1",
        "export": args.symbol,
        "emission": {"format": "hsaco", "options": "defaults"},
        "source_sha256": sha256(args.source),
    }
    with tempfile.TemporaryDirectory(prefix="loom-blas-source-") as raw_directory:
        source_bc = Path(raw_directory) / "source.loombc"
        run([str(args.formatter), str(args.source), "--from=text", "--to=bc", f"--output={source_bc}"])
        start = time.perf_counter_ns()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            records = list(
                executor.map(
                    lambda spec: compile_or_error(
                        spec,
                        source_bc,
                        args.formatter,
                        args.benchmark,
                        args.target,
                        args.symbol,
                        namespace,
                        args.rocm_lib,
                    ),
                    specs,
                )
            )
        wall_us = (time.perf_counter_ns() - start) / 1000
        successful_specs = [specs[int(record["ordinal"])] for record in records if "error" not in record]
        if not successful_specs:
            raise RuntimeError("every corpus compilation failed")
        anchor = successful_specs[0]
        repeats = [
            compile_one(
                anchor,
                source_bc,
                args.formatter,
                args.benchmark,
                args.target,
                args.symbol,
                namespace,
                args.rocm_lib,
            )
            for _ in range(args.determinism_repeats)
        ]

    successful_records = [record for record in records if "error" not in record]
    failed_records = [record for record in records if "error" in record]
    key_groups: dict[str, list[dict[str, int]]] = {}
    hsaco_groups: dict[str, list[dict[str, int]]] = {}
    for record in successful_records:
        key_groups.setdefault(str(record["derived_key"]), []).append(record["request"])  # type: ignore[arg-type]
        hsaco_groups.setdefault(str(record["hsaco_sha256"]), []).append(record["request"])  # type: ignore[arg-type]
    equal_key_hsaco_violations = []
    for key in key_groups:
        hashes = {record["hsaco_sha256"] for record in successful_records if record["derived_key"] == key}
        if len(hashes) != 1:
            equal_key_hsaco_violations.append(key)
    repeat_programs = {str(record["program_sha256"]) for record in repeats}
    repeat_hsacos = {str(record["hsaco_sha256"]) for record in repeats}
    output = {
        "schema": "loom-blas.program-key-study.v1",
        "corpus": args.corpus.name,
        "namespace": namespace,
        "raw_request_count": len(records),
        "successful_request_count": len(successful_records),
        "failed_request_count": len(failed_records),
        "distinct_program_count": len({record["program_sha256"] for record in successful_records}),
        "distinct_key_count": len(key_groups),
        "distinct_hsaco_count": len({record["hsaco_sha256"] for record in successful_records}),
        "request_per_hsaco": len(successful_records) / len({record["hsaco_sha256"] for record in successful_records}),
        "key_groups": key_groups,
        "hsaco_groups": hsaco_groups,
        "equal_key_hsaco_violations": equal_key_hsaco_violations,
        "determinism": {
            "repeats": len(repeats),
            "program_hash_count": len(repeat_programs),
            "hsaco_hash_count": len(repeat_hsacos),
            "pass": len(repeat_programs) == 1 and len(repeat_hsacos) == 1,
        },
        "batch": {"workers": args.workers, "wall_us": wall_us},
        "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "latency_us": {
            phase: {
                "median": statistics.median(float(record["timing_us"][phase]) for record in successful_records),  # type: ignore[index]
                "max": max(float(record["timing_us"][phase]) for record in successful_records),  # type: ignore[index]
            }
            for phase in ("setup", "link", "compile", "key", "emit", "total")
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if args.output.stat().st_size == 0:
        raise RuntimeError("empty output")


if __name__ == "__main__":
    main()
