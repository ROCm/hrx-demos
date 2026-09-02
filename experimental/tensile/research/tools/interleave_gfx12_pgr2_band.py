#!/usr/bin/env python3
"""Move each PGR2 replacement band ahead of the tile's second WMMA half.

The initial ring motif emitted both WMMA halves and only then performed the
eight LDS-store/global-load replacements. Tensile issues one replacement before
the first WMMA half and seven before the second. This intermediate experiment
moves the whole replacement band immediately before the second LDS wait, which
lets the second sixteen WMMAs cover VMEM latency while preserving a simple,
reviewable transformation.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def move_band(lines: list[str], address_name: str) -> None:
    start = next(i for i, line in enumerate(lines) if f"%{address_name} =" in line)
    end = next(
        i
        for i in range(start + 1, len(lines))
        if "low.op<amdgpu.s_barrier_signal_all>" in lines[i]
    )
    band = lines[start:end]
    del lines[start:end]

    barrier = next(
        i
        for i in range(start, len(lines))
        if "low.op<amdgpu.s_barrier_signal_all>" in lines[i]
    )
    previous_barrier = max(
        (
            i
            for i in range(barrier)
            if "low.op<amdgpu.s_barrier_wait_all>" in lines[i]
        ),
        default=-1,
    )
    waits = [
        i
        for i in range(previous_barrier + 1, barrier)
        if "low.op<amdgpu.s_wait_dscnt>" in lines[i]
    ]
    if len(waits) != 2:
        raise RuntimeError(
            f"expected two LDS waits before {address_name} barrier, found {len(waits)}"
        )
    lines[waits[-1]:waits[-1]] = band


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lines = args.input.read_text().splitlines(keepends=True)
    move_band(lines, "2057")
    move_band(lines, "2372")
    args.output.write_text("".join(lines))
    print("interleaved two eight-packet PGR2 bands")


if __name__ == "__main__":
    main()
