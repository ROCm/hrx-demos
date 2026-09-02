#!/usr/bin/env python3
"""Hoist invariant addresses from the gfx12 double-buffer Low experiment."""

from __future__ import annotations

import argparse
from pathlib import Path


VECTOR_OFFSETS = (
    ("1621", "1637"),
    ("1655", "1671"),
    ("1688", "1704"),
    ("1721", "1737"),
)


COMMON = """\
  %db_selector_even = low.const<amdgpu.s_mov_b32> {imm32 = 84148480} : reg<amdgpu.sgpr>
  %db_selector_odd = low.const<amdgpu.s_mov_b32> {imm32 = 117834498} : reg<amdgpu.sgpr>
  %db_lane_low = low.op<amdgpu.v_and_b32.src0_inline>(%lane) {imm32 = 15} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %db_lane_low_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%db_lane_low) {imm32 = 3} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %db_lane_high = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%lane) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %db_lane_high_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%db_lane_high) {imm32 = 11} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %db_lhs_wave = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%wave_m_id) {imm32 = 7} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %db_lhs_addr0 = low.op<amdgpu.v_add_u32>(%db_lhs_wave, %db_lane_low_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %db_lhs_addr = low.op<amdgpu.v_add_u32>(%db_lhs_addr0, %db_lane_high_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %db_rhs_wave_n = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%wave_n_id) {imm32 = 12} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %db_rhs_lane_n = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%db_lane_low) {imm32 = 6} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %db_rhs_addr0 = low.op<amdgpu.v_add_u32>(%db_rhs_wave_n, %db_rhs_lane_n) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %db_rhs_lane_k = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%db_lane_high) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %db_rhs_addr = low.op<amdgpu.v_add_u32>(%db_rhs_addr0, %db_rhs_lane_k) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
"""


def dynamic_loads(prefix: str, a_offset: int, b_offset: int) -> str:
    tag = prefix.rstrip("_")
    lines = [
        f"  %{tag}_a_k = low.op<amdgpu.s_lshl_b32.rhs_inline>(%k_base) "
        "{imm32 = 11} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        f"  %{tag}_a_lo = low.op<amdgpu.s_add_u32>(%1626, %{tag}_a_k) : "
        "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        f"  %{tag}_a_hi = low.op<amdgpu.s_addc_u32>(%1627, %zero) : "
        "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        f"  %{tag}_a = low.concat(%{tag}_a_lo, %{tag}_a_hi) : "
        "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>",
        f"  %{tag}_b_k = low.op<amdgpu.s_lshl_b32.rhs_inline>(%k_base) "
        "{imm32 = 1} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        f"  %{tag}_b_lo = low.op<amdgpu.s_add_u32>(%1642, %{tag}_b_k) : "
        "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        f"  %{tag}_b_hi = low.op<amdgpu.s_addc_u32>(%1643, %zero) : "
        "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        f"  %{tag}_b = low.concat(%{tag}_b_lo, %{tag}_b_hi) : "
        "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>",
    ]
    for index, (a_vector, b_vector) in enumerate(VECTOR_OFFSETS):
        lines.append(
            f"  %{prefix}av{index} = low.op<amdgpu.global_load_b128_saddr>"
            f"(%{a_vector}, %{tag}_a) {{offset = {a_offset}}} : "
            "(reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>"
        )
        lines.append(
            f"  %{prefix}bv{index} = low.op<amdgpu.global_load_b128_saddr>"
            f"(%{b_vector}, %{tag}_b) {{offset = {b_offset}}} : "
            "(reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>"
        )
    return "\n".join(lines) + "\n"


def fixed_loads(prefix: str, a_offset: int, b_offset: int) -> str:
    lines: list[str] = []
    for index, (a_vector, b_vector) in enumerate(VECTOR_OFFSETS):
        lines.append(
            f"  %{prefix}av{index} = low.op<amdgpu.global_load_b128_saddr>"
            f"(%{a_vector}, %1628) {{offset = {a_offset}}} : "
            "(reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>"
        )
        lines.append(
            f"  %{prefix}bv{index} = low.op<amdgpu.global_load_b128_saddr>"
            f"(%{b_vector}, %1644) {{offset = {b_offset}}} : "
            "(reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>"
        )
    return "\n".join(lines) + "\n"


def replace_between(text: str, start: int, stop_marker: str, replacement: str) -> str:
    stop = text.index(stop_marker, start)
    return text[:start] + replacement + text[stop:]


def replace_motif_addresses(text: str, prefix: str, stage: int) -> str:
    start = text.index(f"  %{prefix}lhs_selector_even = ")
    final = text.index(f"  %{prefix}rhs_addr = ", start)
    stop = text.index("\n", final) + 1
    if stage:
        addresses = (
            f"  %{prefix}lhs_addr = low.op<amdgpu.v_xor_b32.lit>(%db_lhs_addr) "
            "{imm32 = 32768} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            f"  %{prefix}rhs_addr = low.op<amdgpu.v_xor_b32.lit>(%db_rhs_addr) "
            "{imm32 = 32768} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        )
    else:
        addresses = (
            f"  %{prefix}lhs_addr = low.copy %db_lhs_addr : reg<amdgpu.vgpr> -> reg<amdgpu.vgpr>\n"
            f"  %{prefix}rhs_addr = low.copy %db_rhs_addr : reg<amdgpu.vgpr> -> reg<amdgpu.vgpr>\n"
        )
    text = text[:start] + addresses + text[stop:]
    return text.replace(f"%{prefix}lhs_selector_even", "%db_selector_even").replace(
        f"%{prefix}lhs_selector_odd", "%db_selector_odd"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.input.read_text()

    first_branch = text.index("  low.br ^_bb1(")
    text = text[:first_branch] + COMMON + text[first_branch:]

    bb2 = text.index("^_bb2:\n") + len("^_bb2:\n")
    text = replace_between(
        text,
        bb2,
        "  %even_compute_lhs_selector_even = ",
        dynamic_loads("odd_load_", 65536, 64),
    )

    odd_barrier = text.index(
        "  low.op<amdgpu.s_barrier_wait_all>() : ()\n",
        text.index("  %even_compute_y15 = "),
    )
    even_load_start = text.index("\n", odd_barrier) + 1
    text = replace_between(
        text,
        even_load_start,
        "  %odd_compute_lhs_selector_even = ",
        dynamic_loads("even_load_", 131072, 128),
    )

    bb3 = text.index("^_bb3:\n") + len("^_bb3:\n")
    text = replace_between(
        text,
        bb3,
        "  %tail30_compute_lhs_selector_even = ",
        fixed_loads("tail_load_", 2031616, 1984),
    )

    for prefix, stage in (
        ("even_compute_", 0),
        ("odd_compute_", 1),
        ("tail30_compute_", 0),
        ("tail31_compute_", 1),
    ):
        text = replace_motif_addresses(text, prefix, stage)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise ValueError("output was not created")


if __name__ == "__main__":
    main()
