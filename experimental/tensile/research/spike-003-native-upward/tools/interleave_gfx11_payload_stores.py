#!/usr/bin/env python3
"""Move gfx11 prefetched-payload LDS publication to the native issue point.

The prepared cross-iteration source originally keeps five x4 VMEM payloads
live until all six first-half WMMAs and all second-half fragments have been
formed. Solution 1675 publishes the payload after the first three WMMAs. This
transform moves only the address/store block; its memory accesses and values
are unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MOVES = (
    (
        "  %896 = low.const<amdgpu.v_mov_b32> {imm32 = 65}",
        "  %candidate_next = low.op<amdgpu.s_add_u32.rhs_inline>",
        "  %even_first_acc2 = "
        "low.op<amdgpu.v_wmma_f32_16x16x16_f16>",
    ),
    (
        "  %1110 = low.const<amdgpu.v_mov_b32> {imm32 = 65}",
        "  low.op<amdgpu.s_barrier>() : ()",
        "  %odd_first_acc2 = "
        "low.op<amdgpu.v_wmma_f32_16x16x16_f16>",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    for block_start, block_end, insert_after in MOVES:
        start = source.find(block_start)
        end = source.find(block_end, start)
        if start < 0 or end < 0:
            raise ValueError(f"payload publication block not found: {block_start}")
        block = source[start:end]
        source = source[:start] + source[end:]

        marker = source.find(insert_after)
        if marker < 0:
            raise ValueError(f"third first-half WMMA not found: {insert_after}")
        line_end = source.find("\n", marker) + 1
        source = source[:line_end] + block + source[line_end:]
    args.output.write_text(source)


if __name__ == "__main__":
    main()
