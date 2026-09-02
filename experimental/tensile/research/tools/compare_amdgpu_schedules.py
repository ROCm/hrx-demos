#!/usr/bin/env python3
"""Compare two normalized AMDGPU instruction intervals structurally."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def signature(row: dict[str, object], include_operands: bool) -> tuple[object, ...]:
    fields: tuple[object, ...] = (
        row["byte_offset"],
        row["category"],
        row["mnemonic"],
    )
    if include_operands:
        fields += (row["operands"],)
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-operands", action="store_true")
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    reference_rows = reference["significant_schedule"]
    candidate_rows = candidate["significant_schedule"]

    prefix = 0
    for reference_row, candidate_row in zip(reference_rows, candidate_rows):
        if signature(reference_row, args.include_operands) != signature(
            candidate_row, args.include_operands
        ):
            break
        prefix += 1

    categories = sorted(
        set(reference["category_counts"]) | set(candidate["category_counts"])
    )
    category_delta = {
        category: candidate["category_counts"].get(category, 0)
        - reference["category_counts"].get(category, 0)
        for category in categories
    }
    category_delta = {key: value for key, value in category_delta.items() if value}

    unmatched_reference = Counter(
        signature(row, args.include_operands) for row in reference_rows[prefix:]
    )
    unmatched_candidate = Counter(
        signature(row, args.include_operands) for row in candidate_rows[prefix:]
    )
    result = {
        "schema": "loom-blas.amdgpu-schedule-comparison.v1",
        "reference": reference["label"],
        "candidate": candidate["label"],
        "include_operands": args.include_operands,
        "instruction_count": {
            "reference": reference["instruction_count"],
            "candidate": candidate["instruction_count"],
            "delta": candidate["instruction_count"] - reference["instruction_count"],
        },
        "significant_count": {
            "reference": len(reference_rows),
            "candidate": len(candidate_rows),
            "exact_prefix": prefix,
        },
        "category_count_delta": category_delta,
        "first_difference": {
            "reference": reference_rows[prefix] if prefix < len(reference_rows) else None,
            "candidate": candidate_rows[prefix] if prefix < len(candidate_rows) else None,
        },
        "unmatched_multiset_count": {
            "reference": sum((unmatched_reference - unmatched_candidate).values()),
            "candidate": sum((unmatched_candidate - unmatched_reference).values()),
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
