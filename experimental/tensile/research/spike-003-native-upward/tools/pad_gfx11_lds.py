#!/usr/bin/env python3
"""Apply solution-42's exact gfx11 LDS geometry to prepared Low.

The source recipe uses 28,288 bytes, a 16 KiB stage toggle, A padding of
16 f16 elements per 2,048-byte block, and B padding of 16 f16 elements per
128-byte block.  B fragment reads also use the recipe's doubled logical N
stride.  This transform changes immediate offsets only; the vector address is
the same lane/wave component produced by High lowering.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


OFFSET = re.compile(r"\{offset = (\d+)\}")


def padded_a(raw: int) -> int:
    return raw + (raw // 2048) * 32


def padded_b_store(raw: int) -> int:
    return raw + (raw // 128) * 32


def padded_b_read(raw: int) -> int:
    doubled_n_raw = (raw // 1024) * 2048 + (raw % 1024)
    return doubled_n_raw + (doubled_n_raw // 128) * 32


def map_offset(line: str, old: int) -> int:
    is_write = "ds_write" in line
    if is_write:
        if old < 4096:
            return padded_a(old)
        if old < 10240:
            return 4224 + padded_b_store(old - 4096)
        if old < 14336:
            return 16384 + padded_a(old - 10240)
        return 20608 + padded_b_store(old - 14336)

    if old < 4096:
        return padded_a(old)
    if old < 10240:
        return 4224 + padded_b_read(old - 4096)
    if old < 14336:
        return 16384 + padded_a(old - 10240)
    return 20608 + padded_b_read(old - 14336)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.read_text().replace(
        "byte_length = 20480", "byte_length = 28288", 1
    )
    output: list[str] = []
    changed = 0
    for line in source.splitlines():
        if any(
            f"low.op<amdgpu.ds_{kind}" in line
            for kind in ("load", "read", "write", "store")
        ):
            match = OFFSET.search(line)
            if match is None:
                raise ValueError(f"LDS operation has no immediate offset: {line}")
            old = int(match.group(1))
            new = map_offset(line, old)
            line = line[: match.start(1)] + str(new) + line[match.end(1) :]
            changed += 1
        output.append(line)
    if changed != 167:
        raise ValueError(f"expected 167 LDS operations, transformed {changed}")
    args.output.write_text("\n".join(output) + "\n")


if __name__ == "__main__":
    main()
