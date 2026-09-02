#!/usr/bin/env python3
"""Split each PGR2 band 1+7 across the two WMMA halves, like Tensile."""

from __future__ import annotations

import argparse
from pathlib import Path


def move_split_band(lines: list[str], address_name: str) -> None:
    start = next(i for i, line in enumerate(lines) if f"%{address_name} =" in line)
    end = next(
        i
        for i in range(start + 1, len(lines))
        if "low.op<amdgpu.s_barrier_signal_all>" in lines[i]
    )
    band = lines[start:end]
    del lines[start:end]

    first_load = next(
        i for i, line in enumerate(band) if "global_load_b128_saddr" in line
    )
    cut = first_load + 1
    while cut < len(band) and "low.schedule.fence" in band[cut]:
        cut += 1
    first_packet = band[:cut]
    remaining_packets = band[cut:]

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
    lines[waits[0]:waits[0]] = first_packet

    barrier = next(
        i
        for i in range(waits[0] + len(first_packet), len(lines))
        if "low.op<amdgpu.s_barrier_signal_all>" in lines[i]
    )
    waits = [
        i
        for i in range(previous_barrier + 1, barrier)
        if "low.op<amdgpu.s_wait_dscnt>" in lines[i]
    ]
    lines[waits[-1]:waits[-1]] = remaining_packets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lines = args.input.read_text().splitlines(keepends=True)
    move_split_band(lines, "2057")
    move_split_band(lines, "2372")
    args.output.write_text("".join(lines))
    print("interleaved two native-like 1+7 PGR2 bands")


if __name__ == "__main__":
    main()
