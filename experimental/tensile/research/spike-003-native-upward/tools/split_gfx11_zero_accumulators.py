#!/usr/bin/env python3
"""Give fixed gfx11 accumulator banks distinct Low SSA entry values.

High CSE intentionally shares the all-zero initializer among all six loop
arguments. Fixed physical loop arguments cannot all tie to that one incoming
value at six different locations, so the allocation witness needs six
equivalent but distinct Low values at the prologue edge.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.read_text()
    line = next(
        candidate
        for candidate in source.splitlines()
        if candidate.startswith("  %zero_acc = low.concat(")
    )
    copies = [line.replace("%zero_acc =", f"%zero_acc{i} =", 1) for i in range(6)]
    source = source.replace(line, "\n".join(copies), 1)
    branch_start = source.find("  low.br ^_bb1(")
    if branch_start < 0:
        raise ValueError("missing shared zero-accumulator loop edge")
    branch_end = source.index("\n", branch_start)
    branch = source[branch_start:branch_end]
    for i in range(6):
        branch, count = re.subn(
            r"%zero_acc(?=:)", f"%zero_acc{i}", branch, count=1
        )
        if count != 1:
            raise ValueError("shared zero-accumulator edge has fewer than six values")
    source = source[:branch_start] + branch + source[branch_end:]
    args.output.write_text(source)


if __name__ == "__main__":
    main()
