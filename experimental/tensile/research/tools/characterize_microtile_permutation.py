#!/usr/bin/env python3
"""Test whether one raw FP16 matrix is a microtile transpose of another."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def read_f16(path: Path, count: int) -> tuple[float, ...]:
    data = path.read_bytes()
    expected = count * 2
    if len(data) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {len(data)}")
    return struct.unpack(f"<{count}e", data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument(
        "--layout",
        choices=("row-major", "column-major"),
        default="column-major",
        help="storage order of both matrices (default: column-major BLAS output)",
    )
    args = parser.parse_args()
    if args.rows <= 0 or args.cols <= 0 or args.tile <= 0:
        parser.error("matrix and tile dimensions must be positive")
    if args.rows % args.tile or args.cols % args.tile:
        parser.error("matrix dimensions must be divisible by tile size")

    count = args.rows * args.cols
    reference = read_f16(args.reference, count)
    candidate = read_f16(args.candidate, count)
    mismatches = 0
    max_abs_error = 0.0

    def index(row: int, col: int) -> int:
        if args.layout == "row-major":
            return row * args.cols + col
        return col * args.rows + row

    for row in range(args.rows):
        tile_row, inner_row = divmod(row, args.tile)
        for col in range(args.cols):
            tile_col, inner_col = divmod(col, args.tile)
            transposed_row = tile_row * args.tile + inner_col
            transposed_col = tile_col * args.tile + inner_row
            lhs = candidate[index(row, col)]
            rhs = reference[index(transposed_row, transposed_col)]
            error = abs(lhs - rhs)
            if error != 0.0:
                mismatches += 1
                max_abs_error = max(max_abs_error, error)

    print(
        json.dumps(
            {
                "schema": "loom-blas.microtile-permutation.v1",
                "reference": str(args.reference),
                "candidate": str(args.candidate),
                "shape": [args.rows, args.cols],
                "layout": args.layout,
                "microtile": [args.tile, args.tile],
                "relation": "transpose-each-microtile",
                "mismatches": mismatches,
                "max_abs_error": max_abs_error,
                "exact": mismatches == 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
