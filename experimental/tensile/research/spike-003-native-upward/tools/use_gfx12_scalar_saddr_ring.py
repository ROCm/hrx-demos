#!/usr/bin/env python3
"""Move the PGR2 global-address ring from VGPRs to scalar saddr values.

The incumbent advances its global base pointers on the scalar pipe while LDS
reads are outstanding.  The evolved Loom motif instead carried two vector byte
deltas and formed eight per-thread addresses with VALU adds per K64 loop.  This
transform preserves all load addresses but computes one scalar A base and one
scalar B base from the loop induction value, leaving each thread's vaddr fixed.
"""

from __future__ import annotations

import argparse
from pathlib import Path


RING_SETUP = """\
  %scalar_ring_a_k = low.op<amdgpu.s_lshl_b32.rhs_inline>(%k_base) {imm32 = 11} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %scalar_ring_a_lo = low.op<amdgpu.s_add_u32>(%1626, %scalar_ring_a_k) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %scalar_ring_a_hi = low.op<amdgpu.s_addc_u32>(%1627, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %scalar_ring_a = low.concat(%scalar_ring_a_lo, %scalar_ring_a_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>
  %scalar_ring_b_k = low.op<amdgpu.s_lshl_b32.rhs_inline>(%k_base) {imm32 = 1} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %scalar_ring_b_lo = low.op<amdgpu.s_add_u32>(%1642, %scalar_ring_b_k) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %scalar_ring_b_hi = low.op<amdgpu.s_addc_u32>(%1643, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %scalar_ring_b = low.concat(%scalar_ring_b_lo, %scalar_ring_b_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>
"""


def replace_once(source: str, old: str, new: str, description: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected one {description}, found {source.count(old)}")
    return source.replace(old, new)


def remove_trailing_arguments(line: str, arguments: str, ending: str) -> str:
    suffix = ", " + arguments + ending
    if not line.endswith(suffix):
        raise ValueError(f"branch/header does not end in expected arguments: {line[-160:]}")
    return line[: -len(suffix)] + ending


def map_line(source: str, marker: str, operation) -> str:
    marker_at = source.index(marker)
    start = source.rfind("\n", 0, marker_at) + 1
    end = source.index("\n", marker_at)
    return source[:start] + operation(source[start:end]) + source[end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.read_text()

    init_arguments = "%1606: reg<amdgpu.vgpr>, %1606: reg<amdgpu.vgpr>"
    source = map_line(
        source,
        "  low.br ^_bb1(",
        lambda line: remove_trailing_arguments(line, init_arguments, ")"),
    )
    header_arguments = (
        "%ring_a_delta: reg<amdgpu.vgpr>, "
        "%ring_b_delta: reg<amdgpu.vgpr>"
    )
    marker_at = source.index("\n^_bb1(") + 1
    end = source.index("\n", marker_at)
    source = (
        source[:marker_at]
        + remove_trailing_arguments(
            source[marker_at:end], header_arguments, "):"
        )
        + source[end:]
    )

    definitions = "\n".join(
        f"  %ring_{kind}{index} = low.op<amdgpu.v_add_u32>"
        for kind in ("av", "bv")
        for index in range(4)
    )
    # The definitions are interleaved A0..A3 then B0..B3 in the source, so
    # locate the complete span instead of matching the synthetic string above.
    del definitions
    start = source.index("  %ring_av0 = low.op<amdgpu.v_add_u32>")
    end = source.index("\n", source.index("  %ring_bv3 = low.op<amdgpu.v_add_u32>", start)) + 1
    source = source[:start] + RING_SETUP + source[end:]

    vectors_a = ("1621", "1655", "1688", "1721")
    vectors_b = ("1637", "1671", "1704", "1737")
    for index, vector in enumerate(vectors_a):
        source = source.replace(
            f"(%ring_av{index}, %1628)", f"(%{vector}, %scalar_ring_a)"
        )
    for index, vector in enumerate(vectors_b):
        source = source.replace(
            f"(%ring_bv{index}, %1644)", f"(%{vector}, %scalar_ring_b)"
        )
    for name in ("ring_av", "ring_bv"):
        if f"%{name}" in source:
            raise ValueError(f"unconverted vector ring reference %{name}")

    for marker in (
        "  %ring_a_delta_next = low.op<amdgpu.v_add_u32.lit>",
        "  %ring_b_delta_next = low.op<amdgpu.v_add_u32.lit>",
    ):
        marker_at = source.index(marker)
        start = source.rfind("\n", 0, marker_at) + 1
        end = source.index("\n", marker_at) + 1
        source = source[:start] + source[end:]

    next_arguments = (
        "%ring_a_delta_next: reg<amdgpu.vgpr>, "
        "%ring_b_delta_next: reg<amdgpu.vgpr>"
    )
    marker_at = source.rindex("  low.br ^_bb1(")
    start = source.rfind("\n", 0, marker_at) + 1
    end = source.index("\n", marker_at)
    source = (
        source[:start]
        + remove_trailing_arguments(source[start:end], next_arguments, ")")
        + source[end:]
    )

    args.output.write_text(source)


if __name__ == "__main__":
    main()
