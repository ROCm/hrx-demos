#!/usr/bin/env python3
"""Collapse gfx11 LDS writes to Tensile's two persistent address bases."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


WRITE_RE = re.compile(
    r"^(?P<indent>\s*)low\.op<amdgpu\.ds_write_b128>\("
    r"(?P<addr>%[A-Za-z0-9_]+), (?P<data>%[A-Za-z0-9_]+)\) "
    r"\{offset = (?P<offset>[0-9]+)\}(?P<tail>.*)$"
)

# Address -> (persistent base, extra immediate offset).
BASES = {
    "%664": ("%664", 0),
    "%668": ("%664", 2080),
    "%670": ("%670", 0),
    "%672": ("%670", 2560),
    "%674": ("%670", 5120),
    "%899": ("%664", 0),
    "%903": ("%664", 2080),
    "%905": ("%670", 0),
    "%907": ("%670", 2560),
    "%909": ("%670", 5120),
    "%1113": ("%664", 0),
    "%1117": ("%664", 2080),
    "%1119": ("%670", 0),
    "%1121": ("%670", 2560),
    "%1123": ("%670", 5120),
}


def rewrite(text: str) -> str:
    output: list[str] = []
    rewritten: set[str] = set()
    for line in text.splitlines():
        match = WRITE_RE.match(line)
        if match is None or match.group("addr") not in BASES:
            output.append(line)
            continue
        old_addr = match.group("addr")
        base, extra = BASES[old_addr]
        offset = int(match.group("offset")) + extra
        output.append(
            f"{match.group('indent')}low.op<amdgpu.ds_write_b128>("
            f"{base}, {match.group('data')}) {{offset = {offset}}}"
            f"{match.group('tail')}"
        )
        rewritten.add(old_addr)
    missing = set(BASES) - rewritten
    if missing:
        raise ValueError(f"missing LDS write addresses: {sorted(missing)}")
    return "\n".join(output) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(rewrite(args.input.read_text()))


if __name__ == "__main__":
    main()
