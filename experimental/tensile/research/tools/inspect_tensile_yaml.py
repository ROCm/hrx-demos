#!/usr/bin/env python3
"""Summarize a Tensile logic YAML and compare its schedules with runtime data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml


JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class Options:
    input_path: Path
    output_path: Path | None
    solution_indices: tuple[int, ...]
    reference_json: Path | None
    reference_solution: int | None
    nearest_count: int


def parse_options() -> Options:
    parser = argparse.ArgumentParser(
        description="Summarize source Tensile YAML and optionally match a runtime solution"
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--solution",
        action="append",
        type=int,
        default=[],
        help="include the complete source solution record (repeatable)",
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        help="inspect_tensile_library.py output containing a runtime solution",
    )
    parser.add_argument(
        "--reference-solution",
        type=int,
        help="runtime solution index to compare with all source solutions",
    )
    parser.add_argument(
        "--nearest-count",
        type=int,
        default=5,
        help="number of closest structural matches to retain (default: 5)",
    )
    args = parser.parse_args()
    if (args.reference_json is None) != (args.reference_solution is None):
        parser.error("--reference-json and --reference-solution must be used together")
    if args.nearest_count < 1:
        parser.error("--nearest-count must be positive")
    return Options(
        input_path=args.input_path,
        output_path=args.output,
        solution_indices=tuple(args.solution),
        reference_json=args.reference_json,
        reference_solution=args.reference_solution,
        nearest_count=args.nearest_count,
    )


def require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return value


def require_dict(value: object, context: str) -> dict[object, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping")
    return value


def json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def first(solution: dict[object, object], *keys: str) -> object:
    for key in keys:
        if key in solution:
            return solution[key]
    return None


def source_schedule(solution: dict[object, object]) -> dict[str, JsonValue]:
    """Project source-YAML spelling into a stable schedule vocabulary."""
    language = str(solution.get("KernelLanguage", ""))
    values: dict[str, object] = {
        "macro_tile": [
            solution.get("MacroTile0"),
            solution.get("MacroTile1"),
            1,
        ],
        "work_group": solution.get("WorkGroup"),
        "wave_group": solution.get("MIWaveGroup"),
        "matrix_instruction": solution.get("MatrixInstruction"),
        "depth_u": solution.get("DepthU"),
        "global_read_vector_width_a": first(
            solution, "GlobalReadVectorWidthA", "GlobalLoadVectorWidthA"
        ),
        "global_read_vector_width_b": first(
            solution, "GlobalReadVectorWidthB", "GlobalLoadVectorWidthB"
        ),
        "global_write_vector_width_d": first(
            solution, "StoreVectorWidth", "GlobalWriteVectorWidth"
        ),
        "prefetch_global_read": solution.get("PrefetchGlobalRead"),
        "expert_scheduling_mode": solution.get("ExpertSchedulingMode"),
        "direct_to_vgpr_a": solution.get("DirectToVgprA"),
        "direct_to_vgpr_b": solution.get("DirectToVgprB"),
        "wave_separate_global_read_a": solution.get("WaveSeparateGlobalReadA"),
        "wave_separate_global_read_b": solution.get("WaveSeparateGlobalReadB"),
        "vector_width_a": first(solution, "VectorWidthA", "VectorWidth"),
        "vector_width_b": first(solution, "VectorWidthB", "VectorWidth"),
        "local_split_u": solution.get("LocalSplitU"),
        "work_group_mapping": solution.get("WorkGroupMapping"),
        "wavefront_size": solution.get("WavefrontSize"),
        "global_split_u": solution.get("GlobalSplitU"),
        "source_kernel": language.lower() == "source",
    }
    return {
        key: json_value(value)
        for key, value in values.items()
        if value is not None and value != [None, None, 1]
    }


def runtime_schedule(solution: dict[object, object]) -> dict[str, JsonValue]:
    mapping = require_dict(solution.get("sizeMapping"), "runtime sizeMapping")
    key_map = {
        "macroTile": "macro_tile",
        "workGroup": "work_group",
        "WaveGroup": "wave_group",
        "matrixInstruction": "matrix_instruction",
        "depthU": "depth_u",
        "grvwA": "global_read_vector_width_a",
        "grvwB": "global_read_vector_width_b",
        "gwvwD": "global_write_vector_width_d",
        "PrefetchGlobalRead": "prefetch_global_read",
        "ExpertSchedulingMode": "expert_scheduling_mode",
        "DirectToVgprA": "direct_to_vgpr_a",
        "DirectToVgprB": "direct_to_vgpr_b",
        "WaveSeparateGlobalReadA": "wave_separate_global_read_a",
        "WaveSeparateGlobalReadB": "wave_separate_global_read_b",
        "VectorWidthA": "vector_width_a",
        "VectorWidthB": "vector_width_b",
        "LocalSplitU": "local_split_u",
        "workGroupMapping": "work_group_mapping",
        "globalSplitU": "global_split_u",
        "sourceKernel": "source_kernel",
    }
    return {
        output_key: json_value(mapping[input_key])
        for input_key, output_key in key_map.items()
        if input_key in mapping
    }


def distribution(solutions: list[dict[object, object]], key: str) -> dict[str, int]:
    counts = Counter(json.dumps(json_value(item.get(key)), sort_keys=True) for item in solutions)
    return dict(sorted(counts.items()))


def compare_schedules(
    reference: dict[str, JsonValue], candidate: dict[str, JsonValue]
) -> tuple[list[str], list[str]]:
    compared = sorted(reference.keys() & candidate.keys())
    differences = [key for key in compared if reference[key] != candidate[key]]
    return compared, differences


def load_runtime_solution(path: Path, index: int) -> dict[object, object]:
    document = require_dict(json.loads(path.read_text()), "runtime summary")
    summary = require_dict(document.get("summary"), "runtime summary.summary")
    selected = require_dict(
        summary.get("selected_solutions"), "runtime selected_solutions"
    )
    return require_dict(selected.get(str(index)), f"runtime solution {index}")


def summarize(options: Options) -> dict[str, JsonValue]:
    encoded = options.input_path.read_bytes()
    document = require_list(yaml.safe_load(encoded), "Tensile YAML root")
    if len(document) < 8:
        raise ValueError("Tensile logic YAML has fewer than eight top-level fields")
    problem_type = require_dict(document[4], "problem type")
    raw_solutions = require_list(document[5], "solutions")
    solutions = [require_dict(item, "solution") for item in raw_solutions]
    logic_rows = require_list(document[7], "logic rows")
    library_type = str(document[11]) if len(document) > 11 else "Equality"

    by_index: dict[int, dict[object, object]] = {}
    for ordinal, solution in enumerate(solutions):
        index = solution.get("SolutionIndex", ordinal)
        if not isinstance(index, int):
            raise TypeError(f"solution {ordinal} has a non-integer SolutionIndex")
        by_index[index] = solution

    missing = sorted(set(options.solution_indices) - by_index.keys())
    if missing:
        raise KeyError(f"source solution indices are absent: {missing}")

    output: dict[str, JsonValue] = {
        "schema": "loom-blas.tensile-yaml.v1",
        "source": str(options.input_path.resolve()),
        "encoded_size": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "schedule_name": json_value(document[1]),
        "architecture": json_value(document[2]),
        "devices": json_value(document[3]),
        "minimum_required_version": json_value(document[0]),
        "library_type": library_type,
        "problem_type": json_value(problem_type),
        "solution_count": len(solutions),
        "logic_row_count": len(logic_rows),
        "solution_characteristics": {
            "kernel_language": distribution(solutions, "KernelLanguage"),
            "matrix_instruction": distribution(solutions, "MatrixInstruction"),
            "wavefront_size": distribution(solutions, "WavefrontSize"),
            "macro_tile_0": distribution(solutions, "MacroTile0"),
            "macro_tile_1": distribution(solutions, "MacroTile1"),
            "depth_u": distribution(solutions, "DepthU"),
            "expert_scheduling_mode": distribution(
                solutions, "ExpertSchedulingMode"
            ),
        },
        "selected_solutions": {
            str(index): json_value(by_index[index]) for index in options.solution_indices
        },
    }

    if options.reference_json is not None and options.reference_solution is not None:
        runtime = load_runtime_solution(
            options.reference_json, options.reference_solution
        )
        reference = runtime_schedule(runtime)
        matches: list[tuple[int, int, dict[str, JsonValue], list[str], list[str]]] = []
        for index, solution in by_index.items():
            candidate = source_schedule(solution)
            compared, differences = compare_schedules(reference, candidate)
            matches.append((len(differences), index, candidate, compared, differences))
        matches.sort(key=lambda item: (item[0], -len(item[3]), item[1]))
        nearest = []
        for mismatch_count, index, candidate, compared, differences in matches[
            : options.nearest_count
        ]:
            nearest.append(
                {
                    "source_solution_index": index,
                    "source_solution_name": json_value(
                        by_index[index].get("SolutionNameMin")
                    ),
                    "mismatch_count": mismatch_count,
                    "compared_field_count": len(compared),
                    "different_fields": differences,
                    "source_schedule": candidate,
                }
            )
        output["runtime_reference"] = {
            "source": str(options.reference_json.resolve()),
            "solution_index": options.reference_solution,
            "schedule": reference,
            "nearest_source_solutions": nearest,
        }
    return output


def main() -> None:
    options = parse_options()
    rendered = json.dumps(summarize(options), indent=2, sort_keys=True) + "\n"
    if options.output_path is None:
        print(rendered, end="")
        return
    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    options.output_path.write_text(rendered)
    if not options.output_path.is_file() or options.output_path.stat().st_size == 0:
        raise RuntimeError(f"failed to write {options.output_path}")


if __name__ == "__main__":
    main()
