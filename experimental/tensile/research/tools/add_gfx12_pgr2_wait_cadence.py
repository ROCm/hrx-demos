#!/usr/bin/env python3
"""Lock the gfx12 PGR2 store/load replacement cadence to loadcnt(7).

The input motif carries eight b128 load results across an iteration and then
replaces each packet immediately after storing it to LDS. Tensile keeps eight
requests in flight with `wait_loadcnt 7; ds_write; global_load` repeated eight
times. This Low experiment makes that cadence explicit and fences each packet
so scheduling cannot collect sixteen loads into one band.
"""

from __future__ import annotations

import argparse
from pathlib import Path


STORE_MARKERS = (
    ", %p_av",
    ", %p_bv",
    ", %even_load_av",
    ", %even_load_bv",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines(keepends=True)
    output: list[str] = []
    modified_stores = 0
    in_loop_body = False
    for line in lines:
        if line.startswith("^_bb2:"):
            in_loop_body = True
        elif in_loop_body and line.startswith("^_bb3:"):
            in_loop_body = False
        is_ring_store = (
            in_loop_body
            and
            "low.op<amdgpu.ds_write_b128>" in line
            and any(marker in line for marker in STORE_MARKERS)
        )
        if is_ring_store:
            indent = line[: len(line) - len(line.lstrip())]
            output.append(f"{indent}low.schedule.fence\n")
            output.append(
                f"{indent}low.op<amdgpu.s_wait_loadcnt>() {{loadcnt = 7}} : ()\n"
            )
            output.append(f"{indent}low.schedule.fence\n")
            output.append(line)
            output.append(f"{indent}low.schedule.fence\n")
            modified_stores += 1
        else:
            output.append(line)
        if (
            in_loop_body
            and "low.op<amdgpu.global_load_b128_saddr>" in line
            and modified_stores
        ):
            indent = line[: len(line) - len(line.lstrip())]
            output.append(f"{indent}low.schedule.fence\n")

    if modified_stores != 16:
        raise RuntimeError(f"expected 16 ring stores, found {modified_stores}")
    args.output.write_text("".join(output))
    print(f"locked {modified_stores} store/load replacements")


if __name__ == "__main__":
    main()
