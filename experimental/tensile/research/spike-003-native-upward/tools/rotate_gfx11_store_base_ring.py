#!/usr/bin/env python3
"""Rotate gfx11 LDS store bases in the same phase as solution 1675.

The original K32 transform carries the currently readable LDS stage and XORs
it before every publication. TensileLite instead enters the loop with the
next writable stage, publishes through that base, and XORs only after all five
stores. This changes phase convention, not the addressed LDS bytes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    branch = "low.br ^_bb1(%zero:"
    branch_pos = source.find(branch)
    if branch_pos < 0:
        raise ValueError("initial loop branch not found")

    insertion = (
        "  %initial_a_store_base = low.op<amdgpu.v_xor_b32.lit>(%664) "
        "{imm32 = 16384} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %initial_b_store_base = low.op<amdgpu.v_xor_b32.lit>(%670) "
        "{imm32 = 16384} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
    )
    line_start = source.rfind("\n", 0, branch_pos) + 1
    source = source[:line_start] + insertion + source[line_start:]

    old_initial = (
        "%684: reg<amdgpu.vgpr>, %722: reg<amdgpu.vgpr>, "
        "%664: reg<amdgpu.vgpr>, %670: reg<amdgpu.vgpr>,"
    )
    new_initial = (
        "%684: reg<amdgpu.vgpr>, %722: reg<amdgpu.vgpr>, "
        "%initial_a_store_base: reg<amdgpu.vgpr>, "
        "%initial_b_store_base: reg<amdgpu.vgpr>,"
    )
    if source.count(old_initial) != 1:
        raise ValueError("unexpected initial store-base branch arguments")
    source = source.replace(old_initial, new_initial, 1)

    replacements = {
        "ds_write_b128>(%next_a_store_base,": "ds_write_b128>(%a_store_base,",
        "ds_write_b128>(%next_b_store_base,": "ds_write_b128>(%b_store_base,",
    }
    for old, new in replacements.items():
        count = source.count(old)
        expected = 2 if "next_a" in old else 3
        if count != expected:
            raise ValueError(f"expected {expected} occurrences of {old!r}, found {count}")
        source = source.replace(old, new)

    args.output.write_text(source)


if __name__ == "__main__":
    main()
