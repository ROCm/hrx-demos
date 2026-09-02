#!/usr/bin/env python3
"""Replace the accidental row-padded gfx11 A LDS layout with solution 1675's.

The High motif used a 65-f16 row stride and then inherited the incumbent's
32-byte pad at each 2048-byte block boundary.  The selected Tensile kernel
uses only the block padding (LPA16/LBSPPA2048): rows are 128 bytes apart and
the second 2 KiB block begins at byte 2080.  B already has the matching
LPB16/LBSPPB128 layout and is intentionally untouched.

This is a deliberately narrow prepared-Low transform.  It also changes the
cooperative A-store address multiplier from 65 to 64 elements.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


OFFSET = re.compile(r"\{offset = (\d+)\}")


def map_a_offset(old: int) -> int:
    stage = 16384 if old >= 16384 else 0
    relative = old - stage
    row, within_row = divmod(relative, 130)
    if row >= 32 or within_row not in (0, 32):
        raise ValueError(
            f"unexpected row-padded A offset {old}: "
            f"row={row}, within_row={within_row}"
        )
    return stage + row * 128 + (row // 16) * 32 + within_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    old_multiplier = (
        "  %661 = low.const<amdgpu.v_mov_b32> {imm32 = 65} "
        ": reg<amdgpu.vgpr>"
    )
    new_multiplier = old_multiplier.replace("imm32 = 65", "imm32 = 64")
    if source.count(old_multiplier) != 1:
        raise ValueError("expected one hoisted A-store stride constant")
    source = source.replace(old_multiplier, new_multiplier, 1)

    output: list[str] = []
    changed_offsets = 0
    for line in source.splitlines():
        is_a_read = "low.op<amdgpu.ds_" in line and "%684" in line
        is_a_write = "low.op<amdgpu.ds_" in line and line.lstrip().startswith(
            "low.op<amdgpu.ds_write_b128>(%664,"
        )
        if is_a_read or is_a_write:
            match = OFFSET.search(line)
            if match is None:
                raise ValueError(f"A LDS operation has no immediate offset: {line}")
            old = int(match.group(1))
            new = map_a_offset(old)
            line = line[: match.start(1)] + str(new) + line[match.end(1) :]
            changed_offsets += 1
        output.append(line)

    # 32 prologue fragment packets + 128 packets in the authored K64 body,
    # plus six cooperative A stores (two in each publication group).
    if changed_offsets != 166:
        raise ValueError(f"expected 166 A LDS operations, changed {changed_offsets}")
    args.output.write_text("\n".join(output) + "\n")


if __name__ == "__main__":
    main()
