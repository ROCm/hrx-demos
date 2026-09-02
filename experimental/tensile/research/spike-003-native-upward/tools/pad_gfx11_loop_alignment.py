#!/usr/bin/env python3
"""Insert explicit pre-loop SOPP padding for gfx11 I-cache experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nops", type=int, required=True)
    args = parser.parse_args()
    if args.nops < 0:
        raise ValueError("--nops must be non-negative")

    lines = args.input.read_text().splitlines()
    hits = [i for i, line in enumerate(lines) if line.startswith("^_bb1(")]
    if len(hits) != 1:
        raise ValueError(f"expected one loop header, found {len(hits)}")
    body_header = hits[0]
    branch = max(
        i for i in range(body_header) if "low.br ^_bb1(" in lines[i]
    )
    # s_nop is an emitter wait-state action, not a source-level descriptor.
    # Use a dependency chain of semantically inert scalar adds instead. Feed
    # the final zero to the loop so prepared-Low DCE cannot discard the chain.
    # This executes once, before the hot loop, and changes its placement by one
    # four-byte SOP2 instruction per requested slot.
    padding: list[str] = []
    previous = "%zero"
    for i in range(args.nops):
        result = f"%alignment_pad_{i}"
        padding.append(
            f"  {result} = low.op<amdgpu.s_add_u32>({previous}, %zero) "
            ": (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>"
        )
        previous = result
    if args.nops:
        lines[branch] = lines[branch].replace(
            "low.br ^_bb1(%zero:", f"low.br ^_bb1({previous}:"
        )
    lines[branch:branch] = padding
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
