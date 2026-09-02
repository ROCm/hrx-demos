#!/usr/bin/env python3
"""Place each gfx11 K32 payload load group in its consuming PLR stage.

The K64 source originally computes and issues both five-load groups before the
first barrier. TensileLite issues one group per K32 interval. Move the second
group after the first loop-body barrier; all of its LDS stores already occur in
the second half, so this changes scheduling freedom without changing dataflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path


START = "  %candidate_next = low.op<amdgpu.s_add_u32.rhs_inline>"
BARRIER = "  low.op<amdgpu.s_barrier>() : ()"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lines = args.input.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(START))
    barrier = next(
        i for i in range(start, len(lines)) if lines[i] == BARRIER
    )
    if barrier <= start:
        raise ValueError("second payload group does not precede its barrier")
    payload_group = lines[start:barrier]
    lines[start : barrier + 1] = [BARRIER, *payload_group]
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
