#!/usr/bin/env python3
"""Rewrite all four matrix motifs in the fixed-K gfx12 double-buffer Low form."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from rewrite_gfx12_compact_b64_permute import result_stores
from rewrite_gfx12_lhs_b64_permute import (
    load_block,
    pack_block,
    reorder_matrix_groups,
    rhs_load_block,
)


MOTIFS = (
    ("even_compute_", 0),
    ("odd_compute_", 1),
    ("tail30_compute_", 0),
    ("tail31_compute_", 1),
)


def replace_fragment_region(
    text: str, *, search_start: int, final_fragment: str, replacement: str
) -> str:
    first_load = text.index("low.op<amdgpu.ds_load_u16_d16>", search_start)
    region_start = text.rfind("\n", 0, first_load) + 1
    final_value = text.index(f"  %{final_fragment} = ", first_load)
    region_stop = text.index("\n", final_value) + 1
    return text[:region_start] + replacement + text[region_stop:]


def address_and_selectors(prefix: str, stage: int) -> str:
    lines = [
        f"  %{prefix}lhs_selector_even = low.const<amdgpu.s_mov_b32> "
        "{imm32 = 84148480} : reg<amdgpu.sgpr>",
        f"  %{prefix}lhs_selector_odd = low.const<amdgpu.s_mov_b32> "
        "{imm32 = 117834498} : reg<amdgpu.sgpr>",
        f"  %{prefix}lane_low = low.op<amdgpu.v_and_b32.src0_inline>(%lane) "
        "{imm32 = 15} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        f"  %{prefix}lane_low_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
        f"(%{prefix}lane_low) {{imm32 = 3}} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        f"  %{prefix}lane_high = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%lane) "
        "{imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        f"  %{prefix}lane_high_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
        f"(%{prefix}lane_high) {{imm32 = 11}} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        f"  %{prefix}lhs_wave = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
        f"(%wave_m_id) {{imm32 = 7}} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        f"  %{prefix}lhs_addr0 = low.op<amdgpu.v_add_u32>(%{prefix}lhs_wave, "
        f"%{prefix}lane_low_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        f"  %{prefix}lhs_addr_base = low.op<amdgpu.v_add_u32>(%{prefix}lhs_addr0, "
        f"%{prefix}lane_high_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        f"  %{prefix}rhs_wave_n = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
        f"(%wave_n_id) {{imm32 = 12}} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        f"  %{prefix}rhs_lane_n = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
        f"(%{prefix}lane_low) {{imm32 = 6}} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        f"  %{prefix}rhs_addr0 = low.op<amdgpu.v_add_u32>(%{prefix}rhs_wave_n, "
        f"%{prefix}rhs_lane_n) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        f"  %{prefix}rhs_lane_k = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
        f"(%{prefix}lane_high) {{imm32 = 4}} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        f"  %{prefix}rhs_addr_base = low.op<amdgpu.v_add_u32>(%{prefix}rhs_addr0, "
        f"%{prefix}rhs_lane_k) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
    ]
    if stage:
        lines.extend(
            [
                f"  %{prefix}lhs_addr = low.op<amdgpu.v_xor_b32.lit>"
                f"(%{prefix}lhs_addr_base) {{imm32 = 32768}} : "
                "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
                f"  %{prefix}rhs_addr = low.op<amdgpu.v_xor_b32.lit>"
                f"(%{prefix}rhs_addr_base) {{imm32 = 32768}} : "
                "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            ]
        )
    else:
        lines.extend(
            [
                f"  %{prefix}lhs_addr = low.copy %{prefix}lhs_addr_base : "
                "reg<amdgpu.vgpr> -> reg<amdgpu.vgpr>",
                f"  %{prefix}rhs_addr = low.copy %{prefix}rhs_addr_base : "
                "reg<amdgpu.vgpr> -> reg<amdgpu.vgpr>",
            ]
        )
    return "\n".join(lines) + "\n"


def motif_half(prefix: str, half: int, stage: int) -> str:
    result_base = half * 4
    return (
        (address_and_selectors(prefix, stage) if half == 0 else "")
        + load_block(f"{prefix}lhs_addr", half, namespace=prefix)
        + rhs_load_block(
            half,
            include_address=False,
            namespace=prefix,
            result_prefix=prefix,
        )
        + "  low.schedule.fence\n"
        + "  low.op<amdgpu.s_wait_dscnt>() {dscnt = 0} : ()\n"
        + pack_block(
            half,
            result_base,
            namespace=prefix,
            result_prefix=prefix,
        )
        + "  low.op<amdgpu.s_delay_alu>() {delay = 9} : ()\n"
        + "  low.schedule.fence\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.input.read_text()
    text = text.replace(
        "low.kernel.def retain target<amdgpu.rdna4.core>",
        "low.kernel.def retain schedule(locked) target<amdgpu.rdna4.core>",
        1,
    )

    swapped = 0
    for prefix, stage in MOTIFS:
        motif_at = text.index(f"  %{prefix}l0 = ")
        text = replace_fragment_region(
            text,
            search_start=text.rfind("s_barrier_wait_all", 0, motif_at),
            final_fragment=f"{prefix}q3",
            replacement=motif_half(prefix, 0, stage),
        )
        text = replace_fragment_region(
            text,
            search_start=text.index(f"  %{prefix}x15 = "),
            final_fragment=f"{prefix}q7",
            replacement=motif_half(prefix, 1, stage),
        )
        text, count = re.subn(
            rf"(low\.op<amdgpu\.v_wmma_f32_16x16x16_f16>\()"
            rf"(%{prefix}l[0-7]), (%{prefix}q[0-7])",
            r"\1\3, \2",
            text,
        )
        swapped += count
        text = reorder_matrix_groups(text, f"{prefix}x")
        text = reorder_matrix_groups(text, f"{prefix}y")

    final_y = "  %tail31_compute_y15 = "
    final_y_at = text.index(final_y)
    output_start = text.index("\n", final_y_at) + 1
    fence = "  low.schedule.fence\n"
    if text.startswith(fence, output_start):
        output_start += len(fence)
    return_at = text.rindex("  low.return")
    output_stop = text.index("\n", return_at) + 1
    text = (
        text[:output_start]
        + result_stores(
            has_workgroups=True,
            m=1024,
            accumulator_prefix="tail31_compute_y",
        )
        + text[output_stop:]
    )

    if "ds_load_u16_d16" in text:
        raise ValueError("unexpected D16 fragment load remains")
    if text.count("low.op<amdgpu.ds_read_b64>") != 64:
        raise ValueError("expected 64 B64 LHS loads")
    if text.count("low.op<amdgpu.v_perm_b32>") != 128:
        raise ValueError("expected 128 LHS permutes")
    if swapped != 128:
        raise ValueError(f"expected 128 swapped WMMA packets, got {swapped}")
    if text.count("low.op<amdgpu.global_store_b16_saddr>") != 128:
        raise ValueError("expected 128 scalar result stores")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise ValueError("output was not created")


if __name__ == "__main__":
    main()
