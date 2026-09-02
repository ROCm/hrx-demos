#!/usr/bin/env python3
"""Derive the compact-K-loop gfx12 candidate from the fully unrolled source."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove only the outer K-loop unroll marker from a Loom GEMM"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    text = args.input.read_text()
    loop_start = text.index(" = scf.for %k_base = ")
    body_start = text.index(" unroll {", loop_start)
    if text.count(" unroll {") != 1:
        raise ValueError("expected exactly one unrolled loop in the input")
    text = text[:body_start] + " {" + text[body_start + len(" unroll {") :]
    text = text.replace(
        "and fully linearly\n// unroll the shape-specialized outer K loop.",
        "with a compact\n// shape-specialized outer K loop.",
        1,
    )
    text = text.replace(
        "Full linear\n// unrolling removes the loop-carried allocation failure; flattening the load\n// groups then removes 309 redundant instructions from the generated body.",
        "This variant retains the loop so its prepared Low control flow can be\n"
        "// used as the starting point for the locked native-style schedule.",
        1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise ValueError("output was not created")


if __name__ == "__main__":
    main()
