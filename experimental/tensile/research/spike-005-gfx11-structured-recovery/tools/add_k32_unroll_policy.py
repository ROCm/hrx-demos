#!/usr/bin/env python3
"""Add a full-K32 unroll policy to the structured gfx11 GEMM loop."""

from __future__ import annotations

import argparse
from pathlib import Path


ANCHOR = ") -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<16xf16>, vector<16xf16>, vector<16xf16>, vector<16xf16>, vector<16xf16>) {\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule", choices=("linear", "interleaved", "recurrence"), required=True
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    if source.count(ANCHOR) != 1:
        raise ValueError("expected exactly one K32 loop result signature")
    schedule = "" if args.schedule == "linear" else f" schedule({args.schedule})"
    replacement = ANCHOR[:-3] + f" unroll(%thirty_two){schedule} {{\n"
    args.output.write_text(source.replace(ANCHOR, replacement, 1))


if __name__ == "__main__":
    main()
