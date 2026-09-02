#!/usr/bin/env python3
"""Reuse the five prologue buffer vaddrs in every gfx11 K stage.

After global loads become raw-buffer loads, K and workgroup movement lives in
the scalar offset. The lane-local vector offset is invariant. Tensile carries
exactly five such VGPRs; retaining the old per-stage global-address arithmetic
needlessly fragments the constrained WMMA/D16 register layout.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


LOAD_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[A-Za-z0-9_]+) = "
    r"low\.op<amdgpu\.buffer_load_b128>\("
    r"(?P<srd>%[A-Za-z0-9_]+), (?P<vaddr>%[A-Za-z0-9_]+), "
    r"(?P<soffset>%[A-Za-z0-9_]+)\)(?P<tail>.*)$"
)

VADDR_BY_RESULT = {
    "%odd_av0": "%590",
    "%odd_av1": "%621",
    "%odd_bv0": "%604",
    "%odd_bv1": "%635",
    "%odd_bv2": "%651",
    "%next_av0": "%590",
    "%next_av1": "%621",
    "%next_bv0": "%604",
    "%next_bv1": "%635",
    "%next_bv2": "%651",
}


def rewrite(text: str) -> str:
    output: list[str] = []
    rewritten: set[str] = set()
    for line in text.splitlines():
        match = LOAD_RE.match(line)
        if match is None or match.group("result") not in VADDR_BY_RESULT:
            output.append(line)
            continue
        result = match.group("result")
        output.append(
            f"{match.group('indent')}{result} = "
            f"low.op<amdgpu.buffer_load_b128>("
            f"{match.group('srd')}, {VADDR_BY_RESULT[result]}, "
            f"{match.group('soffset')}){match.group('tail')}"
        )
        rewritten.add(result)
    missing = set(VADDR_BY_RESULT) - rewritten
    if missing:
        raise ValueError(f"missing buffer loads: {sorted(missing)}")
    return "\n".join(output) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(rewrite(args.input.read_text()))


if __name__ == "__main__":
    main()
