#!/usr/bin/env python3
"""Fence second-half LHS B64 reads before RHS B128 reads, as Tensile does."""

from __future__ import annotations

import argparse
from pathlib import Path


def fence_order(lines: list[str], prefix: str) -> None:
    lhs_end = next(
        i for i, line in enumerate(lines) if f"%{prefix}_lhs_h1_raw7_hi =" in line
    ) + 1
    if f"%{prefix}_q4 =" not in lines[lhs_end]:
        raise RuntimeError(f"unexpected {prefix} second-half LDS order")
    lines.insert(lhs_end, "  low.schedule.fence\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lines = args.input.read_text().splitlines(keepends=True)
    fence_order(lines, "even_compute")
    fence_order(lines, "odd_compute")
    args.output.write_text("".join(lines))
    print("fenced two second-half LDS load blocks")


if __name__ == "__main__":
    main()
