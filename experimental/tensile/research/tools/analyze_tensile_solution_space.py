#!/usr/bin/env python3
"""Census accelerated Tensile logic and normalize physical schedule families."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


SCHEDULE_FIELDS = {
    "instruction": ("MatrixInstruction",),
    "macro_tile_m": ("MacroTile0",),
    "macro_tile_n": ("MacroTile1",),
    "wave_tile": ("MIWaveTile",),
    "wave_group": ("MIWaveGroup",),
    "work_group": ("WorkGroup",),
    "depth_u": ("DepthU",),
    "prefetch_global_read": ("PrefetchGlobalRead",),
    "prefetch_local_read": ("PrefetchLocalRead",),
    "lds_bytes": ("LdsNumElements", "LdsNumElementsAlignedA"),
    "direct_to_vgpr_a": ("DirectToVgprA",),
    "direct_to_vgpr_b": ("DirectToVgprB",),
    "work_group_mapping": ("WorkGroupMapping",),
    "global_split_u": ("GlobalSplitU",),
    "global_split_algorithm": ("GlobalSplitUAlgorithm",),
    "stream_k": ("StreamK",),
    "store_remap": ("StoreRemapVectorWidth",),
    "vector_width_a": ("VectorWidthA", "VectorWidth"),
    "vector_width_b": ("VectorWidthB", "VectorWidth"),
    "global_read_vector_width_a": (
        "GlobalReadVectorWidthA",
        "GlobalLoadVectorWidthA",
    ),
    "global_read_vector_width_b": (
        "GlobalReadVectorWidthB",
        "GlobalLoadVectorWidthB",
    ),
    "store_vector_width": ("StoreVectorWidth",),
    "wavefront_size": ("WavefrontSize",),
    "kernel_language": ("KernelLanguage",),
}


def jsonable(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    raise TypeError(type(value).__name__)


def first(solution: dict[object, object], names: tuple[str, ...]) -> object:
    for name in names:
        if name in solution:
            return solution[name]
    return None


def schedule(solution: dict[object, object]) -> dict[str, object]:
    result = {
        output: jsonable(first(solution, inputs))
        for output, inputs in SCHEDULE_FIELDS.items()
    }
    return {key: value for key, value in result.items() if value is not None}


def family_id(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def mechanism_schedule(value: dict[str, object]) -> dict[str, object]:
    keys = (
        "instruction",
        "prefetch_global_read",
        "prefetch_local_read",
        "direct_to_vgpr_a",
        "direct_to_vgpr_b",
        "global_split_algorithm",
        "stream_k",
        "store_remap",
        "wavefront_size",
        "kernel_language",
    )
    result = {key: value[key] for key in keys if key in value}
    global_split_u = value.get("global_split_u", 1)
    result["split"] = "none" if global_split_u in (0, 1) else "split"
    return result


def problem_signature(problem: dict[object, object]) -> dict[str, object]:
    keys = (
        "DataTypeA",
        "DataTypeB",
        "ComputeDataType",
        "DestDataType",
        "HighPrecisionAccumulate",
        "TransposeA",
        "TransposeB",
        "Sparse",
        "UseBias",
        "ActivationType",
        "UseScaleAB",
        "UseScaleAlphaVec",
        "GroupedGemm",
    )
    return {key: jsonable(problem[key]) for key in keys if key in problem}


def inspect_file(path: Path, display_root: Path) -> dict[str, object] | None:
    document = yaml.load(path.read_bytes(), Loader=yaml.CSafeLoader)
    if not isinstance(document, list) or len(document) < 8:
        raise ValueError(f"unsupported logic document: {path}")
    problem = document[4]
    solutions = document[5]
    rows = document[7]
    if not isinstance(problem, dict) or not isinstance(solutions, list):
        raise TypeError(f"malformed logic document: {path}")
    indexed: dict[int, dict[object, object]] = {}
    for ordinal, value in enumerate(solutions):
        if not isinstance(value, dict):
            raise TypeError(f"solution {ordinal} is not a mapping: {path}")
        index = value.get("SolutionIndex", ordinal)
        if not isinstance(index, int):
            raise TypeError(f"solution index is not integer: {path}")
        indexed[index] = value
    accelerated = {
        index: value
        for index, value in indexed.items()
        if isinstance(value.get("MatrixInstruction"), list)
        and len(value["MatrixInstruction"]) > 0
    }
    if not accelerated:
        return None

    row_counts: Counter[int] = Counter()
    if isinstance(rows, list):
        for row in rows:
            if (
                isinstance(row, list)
                and len(row) >= 2
                and isinstance(row[1], list)
                and row[1]
                and isinstance(row[1][0], int)
            ):
                row_counts[row[1][0]] += 1

    families: dict[str, dict[str, object]] = {}
    family_solutions: defaultdict[str, list[int]] = defaultdict(list)
    family_rows: Counter[str] = Counter()
    mechanism_families: dict[str, dict[str, object]] = {}
    mechanism_solutions: defaultdict[str, list[int]] = defaultdict(list)
    mechanism_rows: Counter[str] = Counter()
    for index, solution in accelerated.items():
        normalized = schedule(solution)
        identifier = family_id(normalized)
        families[identifier] = normalized
        family_solutions[identifier].append(index)
        family_rows[identifier] += row_counts[index]
        mechanism = mechanism_schedule(normalized)
        mechanism_identifier = family_id(mechanism)
        mechanism_families[mechanism_identifier] = mechanism
        mechanism_solutions[mechanism_identifier].append(index)
        mechanism_rows[mechanism_identifier] += row_counts[index]

    return {
        "source": str(path.relative_to(display_root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "architecture": jsonable(document[2]),
        "library_type": str(document[11]) if len(document) > 11 else "classic",
        "problem": problem_signature(problem),
        "solution_count": len(solutions),
        "accelerated_solution_count": len(accelerated),
        "logic_row_count": len(rows) if isinstance(rows, list) else 0,
        "accelerated_logic_row_count": sum(row_counts[index] for index in accelerated),
        "physical_family_count": len(families),
        "mechanism_family_count": len(mechanism_families),
        "mechanism_families": [
            {
                "id": identifier,
                "solution_count": len(mechanism_solutions[identifier]),
                "example_solution_indices": sorted(mechanism_solutions[identifier])[:8],
                "router_rows": mechanism_rows[identifier],
                "mechanism": mechanism_families[identifier],
            }
            for identifier in sorted(
                mechanism_families, key=lambda item: (-mechanism_rows[item], item)
            )
        ],
        "physical_families": [
            {
                "id": identifier,
                "solution_count": len(family_solutions[identifier]),
                "example_solution_indices": sorted(family_solutions[identifier])[:8],
                "router_rows": family_rows[identifier],
                "schedule": families[identifier],
            }
            for identifier in sorted(
                families, key=lambda item: (-family_rows[item], item)
            )[:16]
        ],
        "physical_families_truncated": len(families) > 16,
        "_physical_family_ids": sorted(families),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files: list[tuple[Path, Path]] = []
    for root in args.roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        files.extend((path, root.parent) for path in sorted(root.rglob("*.yaml")))
    records = [record for path, base in files if (record := inspect_file(path, base))]
    type_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for record in records:
        signature = json.dumps(record["problem"], sort_keys=True, separators=(",", ":"))
        type_counts[signature] += 1
        for identifier in record.pop("_physical_family_ids"):  # type: ignore[union-attr]
            family_counts[str(identifier)] += 1
    output = {
        "schema": "loom-blas.tensile-solution-space.v1",
        "scope": "accelerated MatrixInstruction solutions in supplied roots",
        "logic_file_count": len(records),
        "arithmetic_signature_count": len(type_counts),
        "physical_family_count": len(family_counts),
        "arithmetic_signatures": [
            {"problem": json.loads(key), "logic_files": count}
            for key, count in sorted(type_counts.items())
        ],
        "logic_files": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if args.output.stat().st_size == 0:
        raise RuntimeError("empty output")


if __name__ == "__main__":
    main()
