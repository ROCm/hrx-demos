#!/usr/bin/env python3
"""Interleave second-K16 LDS reads through first-K16 gfx11 WMMAs.

This is the first manual PLR1 schedule step.  It preserves the exact prepared
Low operations and their order within the second operand-load bank, but spreads
that bank across the six independent first-bank WMMA updates.  The result is
locked so the emitted schedule remains the authored experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def find_line(lines: list[str], text: str, start: int = 0) -> int:
    for index in range(start, len(lines)):
        if text in lines[index]:
            return index
    raise ValueError(f"missing marker: {text}")


def split_evenly(lines: list[str], count: int) -> list[list[str]]:
    return [
        lines[(len(lines) * i) // count : (len(lines) * (i + 1)) // count]
        for i in range(count)
    ]


def interleave_one(lines: list[str], prefix: str) -> list[str]:
    first_copy = find_line(lines, f"%{prefix}_acc0_0 = low.op") - 1
    second_load = find_line(lines, f"%{prefix}_lhs1_0 = low.concat")
    # Include the definitions preceding lhs1_0 by starting immediately after
    # the sixth first-bank WMMA. The source generator emits no unrelated op in
    # this interval.
    first_wmma_end = find_line(lines, f"%{prefix}_acc0_5 = low.op", first_copy) + 1
    second_copy = find_line(lines, f"%{prefix}_acc1_0 = low.op", second_load) - 1

    first_wmmas = lines[first_copy:first_wmma_end]
    if len(first_wmmas) != 12:
        raise ValueError(f"expected six copy/WMMA pairs for {prefix}")
    second_loads = lines[first_wmma_end:second_copy]
    chunks = split_evenly(second_loads, 6)
    woven: list[str] = []
    for i, chunk in enumerate(chunks):
        woven.extend(first_wmmas[i * 2 : i * 2 + 2])
        woven.extend(chunk)
    return lines[:first_copy] + woven + lines[second_copy:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lines = args.input.read_text().splitlines()
    lines = interleave_one(lines, "even")
    lines = interleave_one(lines, "oddc")
    text = "\n".join(lines) + "\n"
    old = "low.kernel.def retain target<"
    if text.count(old) != 1:
        raise ValueError("expected one unlocked Low kernel")
    text = text.replace(old, "low.kernel.def retain schedule(locked) target<", 1)
    args.output.write_text(text)


if __name__ == "__main__":
    main()
