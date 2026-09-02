#!/usr/bin/env python3
"""Consume the most recently packed LHS fragment first in each WMMA band.

The baseline emits four packed LHS fragments in l0..l3 order and then consumes
them in the same order for each RHS fragment.  The wait-state planner therefore
inserts a second S_DELAY_ALU before l3, after three WMMAs have already issued.
This controlled experiment changes only the order of independent accumulator
updates to l3,l0,l1,l2.  It should consolidate the v_perm-to-WMMA dependency at
the beginning of the band while preserving every accumulator's SSA value.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DELAY = "  low.op<amdgpu.s_delay_alu>() {delay = 9} : ()\n"
FENCE = "  low.schedule.fence\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--remove-authored-delay",
        action="store_true",
        help="Remove the preexisting SALU_CYCLE_1 packet at each pack boundary.",
    )
    parser.add_argument(
        "--keep-wmma-order",
        action="store_true",
        help="Keep l0,l1,l2,l3 issue order while applying the delay option.",
    )
    args = parser.parse_args()

    lines = args.input.read_text().splitlines(keepends=True)
    output: list[str] = []
    band_count = 0
    cursor = 0
    while cursor < len(lines):
        if lines[cursor] != DELAY:
            output.append(lines[cursor])
            cursor += 1
            continue

        band_count += 1
        if not args.remove_authored_delay:
            output.append(lines[cursor])
        cursor += 1
        if lines[cursor] != FENCE:
            raise ValueError(f"band {band_count}: expected fence after delay")
        output.append(lines[cursor])
        cursor += 1

        updates: list[list[str]] = []
        for update_index in range(16):
            chunk = lines[cursor : cursor + 3]
            if (
                len(chunk) != 3
                or " = low.copy " not in chunk[0]
                or "v_wmma_f32_16x16x16_f16" not in chunk[1]
                or chunk[2] != FENCE
            ):
                raise ValueError(
                    f"band {band_count}, update {update_index}: "
                    "expected copy/WMMA/fence triple"
                )
            updates.append(chunk)
            cursor += 3

        # The source groups four LHS fragments under each RHS fragment.
        for group_start in range(0, 16, 4):
            relative_order = (0, 1, 2, 3) if args.keep_wmma_order else (3, 0, 1, 2)
            for relative_index in relative_order:
                output.extend(updates[group_start + relative_index])

    if band_count != 8:
        raise ValueError(f"expected eight K16 WMMA bands, found {band_count}")
    args.output.write_text("".join(output))


if __name__ == "__main__":
    main()
