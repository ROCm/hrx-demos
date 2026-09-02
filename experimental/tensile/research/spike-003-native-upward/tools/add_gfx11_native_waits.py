#!/usr/bin/env python3
"""Add solution-1675 wait packets to the locked gfx11 K32 payload ring.

The resulting prepared Low is intended for the gated HRX experiment
``LOOM_AMDGPU_EXPERIMENTAL_MANUAL_WAITS=block-2``.  Other blocks retain Loom's
planned waits; only the authored inner-loop waits replace the conservative
memory-effect waits in the hot loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def wait(lgkmcnt: int = 63, vmcnt: int = 63) -> list[str]:
    return [
        f"  low.op<amdgpu.s_waitcnt>() {{lgkmcnt = {lgkmcnt}, vmcnt = {vmcnt}}} : ()",
        "  low.schedule.fence",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines()
    body_start = lines.index("^_bb2:")
    exit_start = lines.index("^_bb3:")
    body = lines[body_start + 1 : exit_start]
    if any("amdgpu.s_waitcnt" in line for line in body):
        raise ValueError("input loop already contains explicit waits")

    result: list[str] = []
    first_wmma_index = 0
    store_count = 0
    barrier_count = 0
    for line in body:
        if "low.copy %out" in line:
            if first_wmma_index == 0:
                result.extend(wait(lgkmcnt=4))
            elif first_wmma_index == 2:
                result.extend(wait(lgkmcnt=16))
            first_wmma_index += 1
        if (
            "low.op<amdgpu.ds_write_b128>(%" in line
            and "_store_base," in line
        ):
            result.extend(wait(vmcnt=4))
            store_count += 1
        if "low.op<amdgpu.s_barrier>" in line:
            # Preserve the incumbent's deliberately redundant drains around
            # the barrier.  They are part of the schedule witness and can be
            # reduced in a separate A/B experiment.
            result.extend(wait(lgkmcnt=0))
            result.extend(wait(lgkmcnt=0))
            result.append(line)
            result.extend(wait(lgkmcnt=0))
            barrier_count += 1
            continue
        result.append(line)

    if first_wmma_index != 6:
        raise ValueError(f"expected six first-K16 accumulator copies, found {first_wmma_index}")
    if store_count != 5:
        raise ValueError(f"expected five payload stores, found {store_count}")
    if barrier_count != 1:
        raise ValueError(f"expected one loop barrier, found {barrier_count}")

    output = lines[: body_start + 1] + result + lines[exit_start:]
    args.output.write_text("\n".join(output) + "\n")


if __name__ == "__main__":
    main()
