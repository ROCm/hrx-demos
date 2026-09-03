#!/usr/bin/env python3
"""Insert zero-input locked scheduling fences into the gfx11 High K32 loop.

The fence emits one scalar move with a dead result. It exists only to test
whether source-order barriers can retain deliberate High-level LDS/WMMA
interleaving without carrying wide register values through ``low.invoke``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


TARGET = "amdgpu.target<gfx11-generic> @gfx11 {subgroup_size = 32}\n"
LOOP_START = " = scf.for %k_base = "
LOOP_END = "    scf.yield %second_acc0"
HELPER = """

// Experimental source-order fence. The dead scalar result avoids adding a
// register carrier to the High/Low boundary while schedule(locked) retains
// the call position through the default compiler pipeline.
low.func.def schedule(locked) target<amdgpu.gfx11.generic.core>(@gfx11) @gfx11_schedule_fence() asm {
  %unused = s_mov_b32 0
  return
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=("first", "second", "both"), required=True)
    parser.add_argument("--after", choices=("wmma", "load", "both"), required=True)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    if source.count(TARGET) != 1:
        raise ValueError("expected exactly one gfx11 target declaration")
    source = source.replace(TARGET, TARGET + HELPER, 1)

    in_loop = False
    in_second_half = False
    mma_count = 0
    load_count = 0
    fence_count = 0
    output: list[str] = []
    for line in source.splitlines(keepends=True):
        if LOOP_START in line:
            in_loop = True
        if in_loop and "kernel.barrier<workgroup>" in line:
            in_second_half = True
        output.append(line)
        if in_loop and LOOP_END in line:
            in_loop = False
            continue
        if not in_loop:
            continue

        is_wmma = " = vector.mma " in line
        is_load = " = vector.fragment.load<" in line
        mma_count += int(is_wmma)
        load_count += int(is_load)
        selected_region = (
            args.region == "both"
            or (args.region == "first" and not in_second_half)
            or (args.region == "second" and in_second_half)
        )
        selected_kind = (
            args.after == "both"
            or (args.after == "wmma" and is_wmma)
            or (args.after == "load" and is_load)
        )
        if selected_region and selected_kind and (is_wmma or is_load):
            output.append("    low.invoke @gfx11_schedule_fence() : () -> ()\n")
            fence_count += 1

    if mma_count != 12 or load_count != 10:
        raise ValueError(
            f"expected 12 loop WMMAs and 10 loop fragment loads; "
            f"found {mma_count} and {load_count}"
        )
    if fence_count == 0:
        raise ValueError("selection inserted no fences")
    args.output.write_text("".join(output))


if __name__ == "__main__":
    main()
