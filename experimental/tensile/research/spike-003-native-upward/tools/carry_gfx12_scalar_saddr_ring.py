#!/usr/bin/env python3
"""Carry and advance scalar PGR2 bases instead of recomputing them in-loop."""

from __future__ import annotations

import argparse
from pathlib import Path


SETUP_START = "  %scalar_ring_a_k = low.op<amdgpu.s_lshl_b32.rhs_inline>"
SETUP_STOP = "  %scalar_ring_b = low.concat(%scalar_ring_b_lo, %scalar_ring_b_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>\n"


def change_line(source: str, offset: int, operation) -> str:
    start = source.rfind("\n", 0, offset) + 1
    end = source.index("\n", offset)
    return source[:start] + operation(source[start:end]) + source[end:]


def append_arguments(line: str, arguments: str, ending: str) -> str:
    if not line.endswith(ending):
        raise ValueError(f"unexpected line ending: {line[-80:]}")
    return line[: -len(ending)] + ", " + arguments + ending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.read_text()

    # Materialize step constants outside the loop. The initial scalar bases
    # are the already-computed tile bases used by the prologue.
    setup_start = source.index(SETUP_START)
    setup_end = source.index(SETUP_STOP, setup_start) + len(SETUP_STOP)
    constants = (
        "  %scalar_ring_a_step = low.const<amdgpu.s_mov_b32> {imm32 = 131072} : reg<amdgpu.sgpr>\n"
        "  %scalar_ring_b_step = low.const<amdgpu.s_mov_b32> {imm32 = 128} : reg<amdgpu.sgpr>\n"
    )
    source = source[:setup_start] + source[setup_end:]
    first_branch = source.index("  low.br ^_bb1(")
    source = source[:first_branch] + constants + source[first_branch:]

    source = change_line(
        source,
        source.index("  low.br ^_bb1("),
        lambda line: append_arguments(
            line,
            "%1628: reg<amdgpu.sgpr x2>, %1644: reg<amdgpu.sgpr x2>",
            ")",
        ),
    )
    header = source.index("\n^_bb1(") + 1
    source = change_line(
        source,
        header,
        lambda line: append_arguments(
            line,
            "%scalar_ring_a: reg<amdgpu.sgpr x2>, %scalar_ring_b: reg<amdgpu.sgpr x2>",
            "):",
        ),
    )

    increment = """\
  %scalar_ring_a_lo_current = low.slice %scalar_ring_a[0] : reg<amdgpu.sgpr x2> -> reg<amdgpu.sgpr>
  %scalar_ring_a_hi_current = low.slice %scalar_ring_a[1] : reg<amdgpu.sgpr x2> -> reg<amdgpu.sgpr>
  %scalar_ring_a_lo_next = low.op<amdgpu.s_add_u32>(%scalar_ring_a_lo_current, %scalar_ring_a_step) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %scalar_ring_a_hi_next = low.op<amdgpu.s_addc_u32>(%scalar_ring_a_hi_current, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %scalar_ring_a_next = low.concat(%scalar_ring_a_lo_next, %scalar_ring_a_hi_next) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>
  %scalar_ring_b_lo_current = low.slice %scalar_ring_b[0] : reg<amdgpu.sgpr x2> -> reg<amdgpu.sgpr>
  %scalar_ring_b_hi_current = low.slice %scalar_ring_b[1] : reg<amdgpu.sgpr x2> -> reg<amdgpu.sgpr>
  %scalar_ring_b_lo_next = low.op<amdgpu.s_add_u32>(%scalar_ring_b_lo_current, %scalar_ring_b_step) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %scalar_ring_b_hi_next = low.op<amdgpu.s_addc_u32>(%scalar_ring_b_hi_current, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %scalar_ring_b_next = low.concat(%scalar_ring_b_lo_next, %scalar_ring_b_hi_next) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>
"""
    back_branch = source.rindex("  low.br ^_bb1(")
    source = source[:back_branch] + increment + source[back_branch:]
    back_branch = source.rindex("  low.br ^_bb1(")
    source = change_line(
        source,
        back_branch,
        lambda line: append_arguments(
            line,
            "%scalar_ring_a_next: reg<amdgpu.sgpr x2>, %scalar_ring_b_next: reg<amdgpu.sgpr x2>",
            ")",
        ),
    )

    args.output.write_text(source)


if __name__ == "__main__":
    main()
