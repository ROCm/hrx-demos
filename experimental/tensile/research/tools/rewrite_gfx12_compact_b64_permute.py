#!/usr/bin/env python3
"""Rewrite compact prepared Low to the native gfx12 WMMA fragment contract."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from rewrite_gfx12_lhs_b64_permute import (
    load_block,
    pack_block,
    reorder_matrix_groups,
    rhs_load_block,
)


def result_stores(
    *, has_workgroups: bool, m: int, accumulator_prefix: str = "a"
) -> str:
    if m <= 0 or m & (m - 1):
        raise ValueError("the current Low output proof requires power-of-two M")
    log_m = int(math.log2(m))
    lines = [
        "  %native_out_lane_low = low.op<amdgpu.v_and_b32.src0_inline>(%lane) "
        "{imm32 = 15} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        "  %native_out_lane_low_bytes = "
        "low.op<amdgpu.v_lshlrev_b32.src0_inline>(%native_out_lane_low) "
        "{imm32 = 3} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        "  %native_out_lane_high = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%lane) "
        "{imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        "  %native_out_lane_high_bytes = "
        "low.op<amdgpu.v_lshlrev_b32.src0_inline>(%native_out_lane_high) "
        f"{{imm32 = {log_m + 4}}} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        "  %native_wave_m_bytes = "
        "low.op<amdgpu.v_lshlrev_b32.src0_inline>(%wave_m_id) "
        "{imm32 = 7} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        "  %native_in_tile0 = low.op<amdgpu.v_add_u32>(%native_out_lane_low_bytes, "
        "%native_out_lane_high_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> "
        "reg<amdgpu.vgpr>",
        "  %native_in_tile1 = low.op<amdgpu.v_add_u32>(%native_in_tile0, "
        "%native_wave_m_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> "
        "reg<amdgpu.vgpr>",
        "  %native_wave_n_bytes = "
        "low.op<amdgpu.v_lshlrev_b32.src0_inline>(%wave_n_id) "
        f"{{imm32 = {log_m + 7}}} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        "  %native_in_tile = low.op<amdgpu.v_add_u32>(%native_in_tile1, "
        "%native_wave_n_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> "
        "reg<amdgpu.vgpr>",
    ]
    if has_workgroups:
        lines.extend(
            [
                "  %native_wg_m_bytes = low.op<amdgpu.s_lshl_b32.rhs_inline>"
                "(%workgroup_m) {imm32 = 8} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
                "  %native_wg_n_bytes = low.op<amdgpu.s_lshl_b32.rhs_inline>"
                f"(%workgroup_n) {{imm32 = {log_m + 8}}} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
                "  %native_grid_base = low.op<amdgpu.s_add_u32>(%native_wg_m_bytes, "
                "%native_wg_n_bytes) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> "
                "reg<amdgpu.sgpr>",
                "  %native_d_base = low.op<amdgpu.v_add_u32>(%native_grid_base, "
                "%native_in_tile) : (reg<amdgpu.sgpr>, reg<amdgpu.vgpr>) -> "
                "reg<amdgpu.vgpr>",
            ]
        )
    else:
        lines.append(
            "  %native_d_base = low.copy %native_in_tile : "
            "reg<amdgpu.vgpr> -> reg<amdgpu.vgpr>"
        )
    for m_interleave in range(4):
        for n_tile in range(4):
            accumulator = m_interleave * 4 + n_tile
            for element in range(8):
                value = f"native_d_a{accumulator}_e{element}"
                converted = f"{value}_f16"
                offset = m_interleave * 2 + n_tile * 32 * m + element * 2 * m
                lines.append(
                    f"  %{value} = low.slice %{accumulator_prefix}{accumulator}[{element}] : "
                    "reg<amdgpu.vgpr x8> -> reg<amdgpu.vgpr>"
                )
                lines.append(
                    f"  %{converted} = low.op<amdgpu.v_cvt_f16_f32>(%{value}) "
                    ": (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>"
                )
                lines.append(
                    "  low.op<amdgpu.global_store_b16_saddr>"
                    f"(%native_d_base, %{converted}, %d) {{offset = {offset}}} : "
                    "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>)"
                )
    lines.append("  low.return")
    return "\n".join(lines) + "\n"


def rhs_address_block(half: int) -> str:
    lines: list[str] = []
    if half == 0:
        lines.extend(
            [
                "  %rhs_wave_n = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
                "(%wave_n_id) {imm32 = 12} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
                "  %rhs_lane_n = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
                "(%native_lane_low) {imm32 = 6} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
                "  %rhs_addr0 = low.op<amdgpu.v_add_u32>(%rhs_wave_n, %rhs_lane_n) "
                ": (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
                "  %rhs_lane_k = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
                "(%native_lane_high) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
                "  %rhs_addr = low.op<amdgpu.v_add_u32>(%rhs_addr0, %rhs_lane_k) "
                ": (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            ]
        )
    return "\n".join(lines) + ("\n" if lines else "") + rhs_load_block(
        half, include_address=False
    )


def replace_fragment_region(
    text: str, *, search_start: int, final_fragment: str, replacement: str
) -> str:
    first_load = text.index("low.op<amdgpu.ds_load_u16_d16>", search_start)
    region_start = text.rfind("\n", 0, first_load) + 1
    final_value = text.index(f"  %{final_fragment} = ", first_load)
    region_stop = text.index("\n", final_value) + 1
    return text[:region_start] + replacement + text[region_stop:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the native gfx12 fragment contract to a compact Low loop"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument(
        "--omit-output",
        action="store_true",
        help="Replace the epilog with low.return for timing decomposition only",
    )
    args = parser.parse_args()

    text = args.input.read_text()
    text = text.replace(
        "low.kernel.def retain target<amdgpu.rdna4.core>",
        "low.kernel.def retain schedule(locked) target<amdgpu.rdna4.core>",
        1,
    )
    selectors = (
        "  %lhs_selector_even = low.const<amdgpu.s_mov_b32> "
        "{imm32 = 84148480} : reg<amdgpu.sgpr>\n"
        "  %lhs_selector_odd = low.const<amdgpu.s_mov_b32> "
        "{imm32 = 117834498} : reg<amdgpu.sgpr>\n"
        "  %native_lane_low = low.op<amdgpu.v_and_b32.src0_inline>(%lane) "
        "{imm32 = 15} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %native_lane_low_bytes = "
        "low.op<amdgpu.v_lshlrev_b32.src0_inline>(%native_lane_low) "
        "{imm32 = 3} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %native_lane_high = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%lane) "
        "{imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %native_lane_high_bytes = "
        "low.op<amdgpu.v_lshlrev_b32.src0_inline>(%native_lane_high) "
        "{imm32 = 11} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %native_lhs_wave = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
        "(%wave_m_id) {imm32 = 7} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %native_lhs_addr0 = low.op<amdgpu.v_add_u32>(%native_lhs_wave, "
        "%native_lane_low_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> "
        "reg<amdgpu.vgpr>\n"
        "  %native_lhs_addr = low.op<amdgpu.v_add_u32>(%native_lhs_addr0, "
        "%native_lane_high_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> "
        "reg<amdgpu.vgpr>\n"
    )
    first = (
        selectors
        + load_block("native_lhs_addr", half=0)
        + rhs_address_block(half=0)
        + "  low.schedule.fence\n"
        + "  low.op<amdgpu.s_wait_dscnt>() {dscnt = 0} : ()\n"
        + pack_block(half=0, result_base=0)
        + "  low.op<amdgpu.s_delay_alu>() {delay = 9} : ()\n"
        + "  low.schedule.fence\n"
    )
    text = replace_fragment_region(
        text, search_start=0, final_fragment="q3", replacement=first
    )
    second = (
        load_block("native_lhs_addr", half=1)
        + rhs_address_block(half=1)
        + "  low.schedule.fence\n"
        + "  low.op<amdgpu.s_wait_dscnt>() {dscnt = 0} : ()\n"
        + pack_block(half=1, result_base=4)
        + "  low.op<amdgpu.s_delay_alu>() {delay = 9} : ()\n"
        + "  low.schedule.fence\n"
    )
    text = replace_fragment_region(
        text,
        search_start=text.index("  %x15 = "),
        final_fragment="q7",
        replacement=second,
    )
    text, swapped_count = re.subn(
        r"(low\.op<amdgpu\.v_wmma_f32_16x16x16_f16>\()"
        r"(%l[0-7]), (%q[0-7])",
        r"\1\3, \2",
        text,
    )
    text = reorder_matrix_groups(text, "x")
    text = reorder_matrix_groups(text, "y")

    peeled = "%peel_l0" in text
    if peeled:
        peel_selectors = (
            "  %peel_lhs_selector_even = low.const<amdgpu.s_mov_b32> "
            "{imm32 = 84148480} : reg<amdgpu.sgpr>\n"
            "  %peel_lhs_selector_odd = low.const<amdgpu.s_mov_b32> "
            "{imm32 = 117834498} : reg<amdgpu.sgpr>\n"
            "  %peel_native_lane_low = low.op<amdgpu.v_and_b32.src0_inline>(%lane) "
            "{imm32 = 15} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            "  %peel_native_lane_low_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
            "(%peel_native_lane_low) {imm32 = 3} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            "  %peel_native_lane_high = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%lane) "
            "{imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            "  %peel_native_lane_high_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
            "(%peel_native_lane_high) {imm32 = 11} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            "  %peel_native_lhs_wave = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
            "(%wave_m_id) {imm32 = 7} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            "  %peel_native_lhs_addr0 = low.op<amdgpu.v_add_u32>"
            "(%peel_native_lhs_wave, %peel_native_lane_low_bytes) : "
            "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            "  %peel_native_lhs_addr = low.op<amdgpu.v_add_u32>"
            "(%peel_native_lhs_addr0, %peel_native_lane_high_bytes) : "
            "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            "  %peel_rhs_wave_n = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
            "(%wave_n_id) {imm32 = 12} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            "  %peel_rhs_lane_n = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
            "(%peel_native_lane_low) {imm32 = 6} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            "  %peel_rhs_addr0 = low.op<amdgpu.v_add_u32>"
            "(%peel_rhs_wave_n, %peel_rhs_lane_n) : "
            "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            "  %peel_rhs_lane_k = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
            "(%peel_native_lane_high) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            "  %peel_rhs_addr = low.op<amdgpu.v_add_u32>"
            "(%peel_rhs_addr0, %peel_rhs_lane_k) : "
            "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        )
        peel_first = (
            peel_selectors
            + load_block("peel_native_lhs_addr", 0, namespace="peel_")
            + rhs_load_block(
                0,
                include_address=False,
                namespace="peel_",
                result_prefix="peel_",
            )
            + "  low.schedule.fence\n"
            + "  low.op<amdgpu.s_wait_dscnt>() {dscnt = 0} : ()\n"
            + pack_block(
                0, 0, namespace="peel_", result_prefix="peel_"
            )
            + "  low.op<amdgpu.s_delay_alu>() {delay = 9} : ()\n"
            + "  low.schedule.fence\n"
        )
        peel_at = text.index("  %peel_l0 = ")
        peel_compute_start = text.rfind(
            "s_barrier_wait_all", 0, peel_at
        )
        text = replace_fragment_region(
            text,
            search_start=peel_compute_start,
            final_fragment="peel_q3",
            replacement=peel_first,
        )
        peel_second = (
            load_block("peel_native_lhs_addr", 1, namespace="peel_")
            + rhs_load_block(
                1,
                include_address=False,
                namespace="peel_",
                result_prefix="peel_",
            )
            + "  low.schedule.fence\n"
            + "  low.op<amdgpu.s_wait_dscnt>() {dscnt = 0} : ()\n"
            + pack_block(
                1, 4, namespace="peel_", result_prefix="peel_"
            )
            + "  low.op<amdgpu.s_delay_alu>() {delay = 9} : ()\n"
            + "  low.schedule.fence\n"
        )
        text = replace_fragment_region(
            text,
            search_start=text.index("  %peel_x15 = "),
            final_fragment="peel_q7",
            replacement=peel_second,
        )
        text, peel_swapped = re.subn(
            r"(low\.op<amdgpu\.v_wmma_f32_16x16x16_f16>\()"
            r"(%peel_l[0-7]), (%peel_q[0-7])",
            r"\1\3, \2",
            text,
        )
        swapped_count += peel_swapped
        text = reorder_matrix_groups(text, "peel_x")
        text = reorder_matrix_groups(text, "peel_y")

    # Control-flow experiments add prefetch blocks, so the result block's
    # numeric name is not stable. Locate the last block containing the return
    # instead of assuming the baseline loop's ^_bb3 spelling.
    return_at = text.rindex("  low.return")
    if peeled:
        output_start = text.rindex(
            "  low.op<amdgpu.s_barrier_signal_all>() : ()", 0, return_at
        )
    else:
        block_marker = text.rfind("\n^_bb", 0, return_at) + 1
        output_start = text.index("\n", block_marker) + 1
    output_stop = text.index("\n", return_at) + 1
    has_workgroups = "%workgroup_m =" in text and "%workgroup_n =" in text
    replacement_epilog = (
        "  low.return\n"
        if args.omit_output
        else result_stores(
            has_workgroups=has_workgroups,
            m=args.m,
            accumulator_prefix="peel_y" if peeled else "a",
        )
    )
    text = text[:output_start] + replacement_epilog + text[output_stop:]

    motif_count = 2 if peeled else 1
    actual_b64 = text.count("low.op<amdgpu.ds_read_b64>")
    if actual_b64 != 16 * motif_count:
        raise ValueError(
            f"expected exactly {16 * motif_count} B64 LHS loads, got {actual_b64}"
        )
    actual_perm = text.count("low.op<amdgpu.v_perm_b32>")
    if actual_perm != 32 * motif_count:
        raise ValueError(
            f"expected exactly {32 * motif_count} LHS permutes, got {actual_perm}"
        )
    if swapped_count != 32 * motif_count:
        raise ValueError(
            f"expected to swap {32 * motif_count} WMMA operand pairs, got {swapped_count}"
        )
    expected_stores = 0 if args.omit_output else 128
    if text.count("low.op<amdgpu.global_store_b16_saddr>") != expected_stores:
        raise ValueError(f"expected exactly {expected_stores} result stores")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise ValueError("output was not created")


if __name__ == "__main__":
    main()
