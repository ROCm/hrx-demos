#!/usr/bin/env python3
"""Apply the incumbent gfx12 B-tile LDS padding to a prepared-Low motif.

The selected TensileLite kernel encodes LBSPPB128.  In the disassembly this
changes the four cooperative B-write stripes from a 2048-byte pitch to 2560
bytes and the four WMMA B-fragment reads from a 1024-byte pitch to 2560 bytes.
This transformer makes precisely that change, including the attached access
intervals on cooperative writes, so the unpadded source remains a controlled
baseline.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


WRITE_OFFSETS = {
    8192: 8192,
    10240: 10752,
    12288: 13312,
    14336: 15872,
    40960: 40960,
    43008: 43520,
    45056: 46080,
    47104: 48640,
}
READ_OFFSETS = {
    8192: 8192,
    8224: 8224,
    9216: 10752,
    9248: 10784,
    10240: 13312,
    10272: 13344,
    11264: 15872,
    11296: 15904,
}


def rewrite_write(match: re.Match[str]) -> str:
    old = int(match.group("offset"))
    new = WRITE_OFFSETS.get(old)
    if new is None:
        return match.group(0)
    delta = new - old
    fields = [int(value) for value in match.group("access").split(", ")]
    fields[-4:] = [value + delta for value in fields[-4:]]
    return (
        match.group("prefix")
        + str(new)
        + match.group("middle")
        + ", ".join(str(value) for value in fields)
        + match.group("suffix")
    )


def rewrite_read(match: re.Match[str]) -> str:
    old = int(match.group("offset"))
    new = READ_OFFSETS.get(old)
    if new is None:
        raise ValueError(f"unexpected B-fragment read offset {old}")
    return match.group("prefix") + str(new) + match.group("suffix")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    write_pattern = re.compile(
        r"(?P<prefix>low\.op<amdgpu\.ds_write_b128>\([^\n]+?\) \{offset = )"
        r"(?P<offset>\d+)"
        r"(?P<middle>\} memory_access\(\[)"
        r"(?P<access>[^\]]+)"
        r"(?P<suffix>\]\))"
    )
    read_pattern = re.compile(
        r"(?P<prefix>low\.op<amdgpu\.ds_read_b128>\([^\n]+?\) \{offset = )"
        r"(?P<offset>\d+)"
        r"(?P<suffix>\})"
    )
    output, write_count = write_pattern.subn(rewrite_write, source)
    output, read_count = read_pattern.subn(rewrite_read, output)
    if write_count != 32:
        raise ValueError(f"expected 32 cooperative writes, visited {write_count}")
    # There are 32 B reads: four fragments for each of eight K16 phases.
    if read_count != 32:
        raise ValueError(f"expected 32 B-fragment reads, visited {read_count}")
    if output == source:
        raise ValueError("no offsets changed")
    args.output.write_text(output)


if __name__ == "__main__":
    main()
