#!/usr/bin/env python3
"""Reorder each six-WMMA High group to the recovered gfx11 wavefront order.

Fragment loads retain their source positions. Only the independent accumulator
updates move from 0,1,2,3,4,5 to the native 0,3,1,4,2,5 order.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ORDER = (0, 3, 1, 4, 2, 5)


def reorder_group(lines: list[str], start: int, prefix: str) -> int:
    window = lines[start : start + 11]
    wmma_lines = [line for line in window if " = vector.mma " in line]
    load_lines = [line for line in window if " = vector.fragment.load<" in line]
    if len(wmma_lines) != 6 or len(load_lines) != 5:
        raise ValueError(f"malformed {prefix} group at line {start + 1}")
    for index, line in enumerate(wmma_lines):
        if f"%{prefix}{index} =" not in line:
            raise ValueError(f"unexpected {prefix} WMMA order at line {start + 1}")
    replacement: list[str] = []
    for position, accumulator_index in enumerate(ORDER):
        replacement.append(wmma_lines[accumulator_index])
        if position < len(load_lines):
            replacement.append(load_lines[position])
    lines[start : start + 11] = replacement
    return start + 11


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines(keepends=True)
    groups = 0
    index = 0
    while index < len(lines):
        prefix = None
        for candidate in ("first_acc", "second_acc", "tail_first_acc"):
            if f"%{candidate}0 = vector.mma " in lines[index]:
                prefix = candidate
                break
        if prefix is None:
            index += 1
            continue
        index = reorder_group(lines, index, prefix)
        groups += 1
    if groups not in (2, 3):
        raise ValueError(f"expected two loop groups and optional peeled tail; found {groups}")
    args.output.write_text("".join(lines))


if __name__ == "__main__":
    main()
