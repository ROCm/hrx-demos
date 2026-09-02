#!/usr/bin/env python3
"""Extract auditable summaries from hipBLASLt/TensileLite library files."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import msgpack


JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class Options:
    input_path: Path
    output_path: Path | None
    operation: str | None
    solution_indices: tuple[int, ...]
    include_table: bool


def parse_options() -> Options:
    parser = argparse.ArgumentParser(
        description="Summarize a .dat or .dat.zlib TensileLite library"
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--operation", help="limit a lazy master summary to one operation identifier"
    )
    parser.add_argument(
        "--solution",
        action="append",
        type=int,
        default=[],
        help="include the complete record for this solution index (repeatable)",
    )
    parser.add_argument(
        "--include-table",
        action="store_true",
        help="include every matching-table row instead of only its size",
    )
    args = parser.parse_args()
    return Options(
        input_path=args.input_path,
        output_path=args.output,
        operation=args.operation,
        solution_indices=tuple(args.solution),
        include_table=args.include_table,
    )


def decode(path: Path) -> tuple[bytes, object]:
    encoded = path.read_bytes()
    payload = zlib.decompress(encoded) if path.suffix == ".zlib" else encoded
    return encoded, msgpack.unpackb(payload, raw=False, strict_map_key=False)


def as_dict(value: object, context: str) -> dict[object, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a map, got {type(value).__name__}")
    return cast(dict[object, object], value)


def as_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list, got {type(value).__name__}")
    return cast(list[object], value)


def json_value(value: object) -> JsonValue:
    """Convert decoded MessagePack to JSON while preserving integer map keys."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [json_value(item) for item in cast(list[object], value)]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in cast(dict[object, object], value).items():
            result[str(key)] = json_value(item)
        return result
    raise TypeError(f"unsupported MessagePack value: {type(value).__name__}")


def summarize_matching_library(
    library: dict[object, object], include_table: bool
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {"type": str(library.get("type", ""))}
    properties = library.get("properties")
    if properties is not None:
        result["properties"] = json_value(properties)
    table = library.get("table")
    if isinstance(table, list):
        rows = cast(list[object], table)
        result["table_row_count"] = len(rows)
        if include_table:
            result["table"] = json_value(rows)
    distance = library.get("distance")
    if distance is not None:
        result["distance"] = json_value(distance)
    return result


def summarize_problem_library(
    library: dict[object, object], include_table: bool
) -> dict[str, JsonValue]:
    rows = as_list(library.get("rows"), "problem library rows")
    output_rows: list[JsonValue] = []
    for ordinal, row_value in enumerate(rows):
        row = as_dict(row_value, f"problem row {ordinal}")
        child = as_dict(row.get("library"), f"problem row {ordinal} library")
        child_type = str(child.get("type", ""))
        row_output: dict[str, JsonValue] = {
            "ordinal": ordinal,
            "predicate": json_value(row.get("predicate")),
            "library_type": child_type,
        }
        if child_type == "Placeholder":
            row_output["placeholder"] = json_value(child.get("value"))
        elif child_type == "Matching":
            row_output["matching"] = summarize_matching_library(child, include_table)
        elif child_type == "Problem":
            row_output["problem"] = summarize_problem_library(child, include_table)
        else:
            row_output["library"] = json_value(child)
        output_rows.append(row_output)
    return {"type": str(library.get("type", "")), "rows": output_rows}


def summarize_master(
    library: dict[object, object], operation_filter: str | None
) -> dict[str, JsonValue]:
    hardware_rows = as_list(library.get("rows"), "hardware rows")
    output_rows: list[JsonValue] = []
    for hardware_ordinal, row_value in enumerate(hardware_rows):
        row = as_dict(row_value, f"hardware row {hardware_ordinal}")
        problem_map = as_dict(row.get("library"), "problem map")
        operations = as_dict(problem_map.get("map"), "operation map")
        output_operations: dict[str, JsonValue] = {}
        for operation, problem_value in operations.items():
            operation_name = str(operation)
            if operation_filter is not None and operation_name != operation_filter:
                continue
            output_operations[operation_name] = summarize_problem_library(
                as_dict(problem_value, f"operation {operation_name}"), False
            )
        output_rows.append(
            {
                "ordinal": hardware_ordinal,
                "predicate": json_value(row.get("predicate")),
                "operation_count": len(operations),
                "operations": output_operations,
            }
        )
    return {"type": "Hardware", "rows": output_rows}


def summarize_mapping(
    root: dict[object, object], selected: tuple[int, ...]
) -> dict[str, JsonValue]:
    integer_keys = sorted(key for key in root if isinstance(key, int))
    entries: dict[str, JsonValue]
    if selected:
        entries = {}
        for index in selected:
            position = bisect.bisect_right(integer_keys, index) - 1
            if position < 0:
                raise KeyError(f"solution index precedes first mapping range: {index}")
            base = integer_keys[position]
            entries[str(index)] = {
                "range_base": base,
                "range_offset": index - base,
                "library": json_value(root[base]),
            }
    else:
        entries = {str(key): json_value(value) for key, value in root.items()}
    return {"kind": "solution_mapping", "entry_count": len(root), "entries": entries}


def summarize_library(root: dict[object, object], options: Options) -> dict[str, JsonValue]:
    solutions_value = root.get("solutions")
    library_value = root.get("library")
    if not isinstance(solutions_value, list) or not isinstance(library_value, dict):
        return summarize_mapping(root, options.solution_indices)

    solutions = cast(list[object], solutions_value)
    library = cast(dict[object, object], library_value)
    selected: dict[str, JsonValue] = {}
    wanted = set(options.solution_indices)
    available_indices: list[int] = []
    for solution_value in solutions:
        solution = as_dict(solution_value, "solution")
        index = solution.get("index")
        if isinstance(index, int):
            available_indices.append(index)
            if index in wanted:
                selected[str(index)] = json_value(solution)
    missing = sorted(wanted - {int(index) for index in selected})
    if missing:
        raise KeyError(f"solution indices absent from library: {missing}")

    library_type = str(library.get("type", ""))
    if library_type == "Hardware":
        route_summary = summarize_master(library, options.operation)
        kind = "lazy_master"
    elif library_type == "Problem":
        route_summary = summarize_problem_library(library, options.include_table)
        kind = "solution_shard"
    else:
        route_summary = json_value(library)
        kind = "unknown_library"

    result: dict[str, JsonValue] = {
        "kind": kind,
        "solution_count": len(solutions),
        "route": route_summary,
        "selected_solutions": selected,
    }
    if available_indices:
        result["solution_index_min"] = min(available_indices)
        result["solution_index_max"] = max(available_indices)
    return result


def main() -> None:
    options = parse_options()
    encoded, decoded = decode(options.input_path)
    root = as_dict(decoded, "library root")
    output: dict[str, JsonValue] = {
        "schema": "loom-blas.tensile-library.v1",
        "source": str(options.input_path.resolve()),
        "encoded_size": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "summary": summarize_library(root, options),
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if options.output_path is None:
        print(rendered, end="")
    else:
        options.output_path.parent.mkdir(parents=True, exist_ok=True)
        options.output_path.write_text(rendered)
        if not options.output_path.is_file() or options.output_path.stat().st_size == 0:
            raise RuntimeError(f"failed to write {options.output_path}")


if __name__ == "__main__":
    main()
