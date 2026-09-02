#!/usr/bin/env python3
"""Set native-like partial LDS waits for the two loop-resident K32 motifs."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--second", type=int, default=7)
    parser.add_argument(
        "--fence-after",
        action="store_true",
        help="keep the explicit wait ahead of all dependent fragment packing",
    )
    args = parser.parse_args()
    lines = args.input.read_text().splitlines(keepends=True)
    in_loop_body = False
    targets = iter((args.first, args.second, args.first, args.second))
    changed = 0
    output: list[str] = []
    for line in lines:
        if line.startswith("^_bb2:"):
            in_loop_body = True
        elif in_loop_body and line.startswith("^_bb3:"):
            in_loop_body = False
        if in_loop_body and "low.op<amdgpu.s_wait_dscnt>() {dscnt = 0}" in line:
            target = next(targets)
            line = line.replace("{dscnt = 0}", f"{{dscnt = {target}}}")
            changed += 1
            output.append(line)
            if args.fence_after:
                output.append("  low.schedule.fence\n")
            continue
        output.append(line)
    if changed != 4:
        raise RuntimeError(f"expected four loop LDS waits, found {changed}")
    args.output.write_text("".join(output))
    print(
        "set loop LDS waits to "
        f"dscnt({args.first},{args.second},{args.first},{args.second})"
    )


if __name__ == "__main__":
    main()
