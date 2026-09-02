#!/usr/bin/env python3
"""Rewrite the spike-003 prepared Low LHS loads to the TensileLite form."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def replace_region(text: str, start: str, stop: str, replacement: str) -> str:
    start_offset = text.index(start)
    stop_offset = text.index(stop, start_offset)
    stop_offset = text.index("\n", stop_offset) + 1
    return text[:start_offset] + replacement + text[stop_offset:]


def load_block(address: str, half: int, *, namespace: str = "") -> str:
    base_offset = half * 4096
    prefix = f"{namespace}lhs_h{half}"
    lines: list[str] = []
    for k_index in range(8):
        offset = base_offset + k_index * 256
        lines.append(
            f"  %{prefix}_raw{k_index} = low.op<amdgpu.ds_read_b64>"
            f"(%{address}) {{offset = {offset}}} : (reg<amdgpu.vgpr>) -> "
            "reg<amdgpu.vgpr x2>"
        )
        lines.append(
            f"  %{prefix}_raw{k_index}_lo = low.slice %{prefix}_raw{k_index}[0] "
            ": reg<amdgpu.vgpr x2> -> reg<amdgpu.vgpr>"
        )
        lines.append(
            f"  %{prefix}_raw{k_index}_hi = low.slice %{prefix}_raw{k_index}[1] "
            ": reg<amdgpu.vgpr x2> -> reg<amdgpu.vgpr>"
        )

    return "\n".join(lines) + "\n"


def pack_block(
    half: int,
    result_base: int,
    *,
    namespace: str = "",
    result_prefix: str = "",
) -> str:
    prefix = f"{namespace}lhs_h{half}"
    lines: list[str] = []
    for fragment in range(4):
        selector = (
            f"{namespace}lhs_selector_even"
            if fragment % 2 == 0
            else f"{namespace}lhs_selector_odd"
        )
        lane = "lo" if fragment < 2 else "hi"
        packed_names: list[str] = []
        for pair in range(4):
            even = pair * 2
            odd = even + 1
            packed = f"{prefix}_f{fragment}_p{pair}"
            packed_names.append(packed)
            lines.append(
                f"  %{packed} = low.op<amdgpu.v_perm_b32>("
                f"%{prefix}_raw{odd}_{lane}, %{prefix}_raw{even}_{lane}, "
                f"%{selector}) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, "
                "reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr>"
            )
        result = f"{result_prefix}l{result_base + fragment}"
        args = ", ".join(f"%{name}" for name in packed_names)
        types = ", ".join("reg<amdgpu.vgpr>" for _ in packed_names)
        lines.append(
            f"  %{result} = low.concat({args}) : ({types}) -> "
            "reg<amdgpu.vgpr x4>"
        )
        lines.append("  low.schedule.fence")
    return "\n".join(lines) + "\n"


def rhs_load_block(
    half: int,
    *,
    include_address: bool,
    namespace: str = "",
    result_prefix: str = "",
) -> str:
    lines: list[str] = []
    if include_address:
        lines.extend(
            [
                "  %rhs_wave_n = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
                "(%wave_n_id) {imm32 = 12} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
                "  %rhs_lane_n = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
                "(%485) {imm32 = 6} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
                "  %rhs_addr0 = low.op<amdgpu.v_add_u32>(%rhs_wave_n, %rhs_lane_n) "
                ": (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
                "  %rhs_lane_k = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
                "(%488) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
                "  %rhs_addr = low.op<amdgpu.v_add_u32>(%rhs_addr0, %rhs_lane_k) "
                ": (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            ]
        )
    for n_tile in range(4):
        q_index = half * 4 + n_tile
        offset = 8192 + half * 32 + n_tile * 1024
        lines.append(
            f"  %{result_prefix}q{q_index} = low.op<amdgpu.ds_read_b128>"
            f"(%{namespace}rhs_addr) "
            f"{{offset = {offset}}} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x4>"
        )
    return "\n".join(lines) + "\n"


def native_result_stores(base_address: str = "490", accumulator_prefix: str = "y") -> str:
    lines = [
        "  %native_d_wave_n = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
        "(%wave_n_id) {imm32 = 14} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        "  %native_d_base = low.op<amdgpu.v_add_u32>(%490, %native_d_wave_n) "
        ": (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
    ]
    for m_interleave in range(4):
        for n_tile in range(4):
            accumulator = m_interleave * 4 + n_tile
            for element in range(8):
                value = f"native_d_y{accumulator}_e{element}"
                converted = f"{value}_f16"
                offset = m_interleave * 2 + n_tile * 4096 + element * 256
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
    text = "\n".join(lines) + "\n"
    return text.replace("%490, %native_d_wave_n", f"%{base_address}, %native_d_wave_n", 1)


def reorder_matrix_groups(text: str, prefix: str) -> str:
    marker = f"  %{prefix}0 = "
    first_value_start = text.index(marker)
    start = first_value_start
    previous_line_start = text.rfind("\n", 0, first_value_start - 1) + 1
    if " low.copy " in text[previous_line_start:first_value_start]:
        start = previous_line_start
    groups: dict[int, str] = {}
    cursor = start
    for index in range(16):
        value_marker = f"  %{prefix}{index} = "
        value_start = text.index(value_marker, cursor)
        group_start = value_start
        previous_line_start = text.rfind("\n", 0, value_start - 1) + 1
        if " low.copy " in text[previous_line_start:value_start]:
            group_start = previous_line_start
        value_end = text.index("\n", value_start) + 1
        groups[index] = text[group_start:value_end]
        cursor = value_end
    stop = cursor
    order = [m_interleave * 4 + n_tile for n_tile in range(4) for m_interleave in range(4)]
    scheduled_groups = "".join(
        groups[index] + "  low.schedule.fence\n" for index in order
    )
    return text[:start] + scheduled_groups + text[stop:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace gfx12 D16 LHS fragment loads with B64 plus v_perm"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    text = args.input.read_text()
    original = text
    text = text.replace(
        "low.kernel.def retain target<amdgpu.rdna4.core>",
        "low.kernel.def retain schedule(locked) target<amdgpu.rdna4.core>",
        1,
    )
    text = text.replace(
        "%486 = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%485) {imm32 = 1}",
        "%486 = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%485) {imm32 = 3}",
        1,
    )
    selectors = (
        "  %lhs_selector_even = low.const<amdgpu.s_mov_b32> "
        "{imm32 = 84148480} : reg<amdgpu.sgpr>\n"
        "  %lhs_selector_odd = low.const<amdgpu.s_mov_b32> "
        "{imm32 = 117834498} : reg<amdgpu.sgpr>\n"
    )
    first = (
        selectors
        + load_block("490", half=0)
        + rhs_load_block(half=0, include_address=True)
        + "  low.schedule.fence\n"
        + "  low.op<amdgpu.s_wait_dscnt>() {dscnt = 1} : ()\n"
        + pack_block(half=0, result_base=0)
        + "  low.op<amdgpu.s_delay_alu>() {delay = 9} : ()\n"
        + "  low.schedule.fence\n"
    )
    text = replace_region(text, "  %491 = ", "  %q3 = ", first)
    second = (
        load_block("490", half=1)
        + rhs_load_block(half=1, include_address=False)
        + "  low.schedule.fence\n"
        + "  low.op<amdgpu.s_wait_dscnt>() {dscnt = 1} : ()\n"
        + pack_block(half=1, result_base=4)
        + "  low.op<amdgpu.s_delay_alu>() {delay = 9} : ()\n"
        + "  low.schedule.fence\n"
    )
    text = replace_region(text, "  %573 = ", "  %q7 = ", second)
    text, swapped_count = re.subn(
        r"(low\.op<amdgpu\.v_wmma_f32_16x16x16_f16(?:\.acc_zero)?>\()"
        r"(%l[0-7]), (%q[0-7])",
        r"\1\3, \2",
        text,
    )
    text = reorder_matrix_groups(text, "x")
    text = reorder_matrix_groups(text, "y")
    output_start = text.rindex("  low.op<amdgpu.s_barrier_signal_all>() : ()")
    output_stop = text.index("\n", text.index("  low.return", output_start)) + 1
    text = text[:output_start] + native_result_stores() + text[output_stop:]

    if text == original:
        raise ValueError("input was not rewritten")
    if "ds_load_u16_d16" in text:
        raise ValueError("unexpected D16 LHS load remains")
    if text.count("low.op<amdgpu.ds_read_b64>") != 16:
        raise ValueError("expected exactly 16 B64 LHS loads")
    if text.count("low.op<amdgpu.v_perm_b32>") != 32:
        raise ValueError("expected exactly 32 LHS permutes")
    if swapped_count != 32:
        raise ValueError(f"expected to swap 32 WMMA operand pairs, got {swapped_count}")
    if text.count("low.op<amdgpu.global_store_b16_saddr>") != 128:
        raise ValueError("expected exactly 128 native-layout result stores")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise ValueError("output was not created")


if __name__ == "__main__":
    main()
