#!/usr/bin/env python3
"""Restore the scalar issue slots present in the gfx11 Tensile K32 loop.

The incrementing-SRD experiment removes fourteen one-word SALU instructions
from the incumbent schedule.  This controlled transform restores their issue
slots with dependent add-zero operations while leaving addresses unchanged.
It is intentionally a diagnostic witness, not the final address recurrence.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def filler(prefix: str, source: str, count: int) -> tuple[list[str], str]:
    lines: list[str] = []
    previous = source
    for ordinal in range(count):
        result = f"%{prefix}_{ordinal}"
        lines.append(
            f"  {result} = low.op<amdgpu.s_lshr_b32.rhs_inline>({previous}) "
            "{imm32 = 0} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>"
        )
        lines.append("  low.schedule.fence")
        previous = result
    return lines, previous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--a-pre", type=int, default=1)
    parser.add_argument("--a-post", type=int, default=1)
    parser.add_argument("--b-band", type=int, default=12)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("^_bb1("))
    end = next(i for i in range(header, len(lines)) if lines[i] == "^_bb_backedge_dispatch:")

    # 1675 has three SALU packets between the first eight A reads and WMMA 1;
    # the SRD-ring form has two. Insert one packet before its A advance.
    first = next(
        i for i in range(header, end)
        if " = low.slice %ring_a[0]" in lines[i]
    )
    addition, chain = filler("native_salu_a_pre", "%k_base", args.a_pre)
    lines[first:first] = addition
    end += len(addition)

    # The incumbent has another three packets between the second eight A reads
    # and lgkmcnt(16); the SRD-ring form has its two B-SRD advances there.
    wait16 = next(
        i for i in range(header, end)
        if "{lgkmcnt = 16, vmcnt = 63}" in lines[i]
    )
    addition, chain = filler("native_salu_a_post", chain, args.a_post)
    lines[wait16:wait16] = addition
    end += len(addition)

    # 1675 performs twelve bounds/descriptor SALU packets after the next six A
    # reads and before the first vmcnt(4). The SRD ring has none in this slot.
    vmwait4 = next(
        i for i in range(wait16, end)
        if "{lgkmcnt = 63, vmcnt = 4}" in lines[i]
    )
    addition, chain = filler("native_salu_b_band", chain, args.b_band)
    lines[vmwait4:vmwait4] = addition

    # Keep the diagnostic SALU chain live and semantically transparent. An
    # unused Low result currently receives a zero-length allocation that can
    # alias live SGPR state, so emitting an unused filler is not a legal test.
    k_next = next(
        i for i in range(vmwait4 + len(addition), len(lines))
        if "%k_next = low.op<amdgpu.s_add_u32.rhs_inline>(%k_base)" in lines[i]
    )
    lines[k_next] = lines[k_next].replace("(%k_base)", f"({chain})")

    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
