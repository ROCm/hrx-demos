#!/usr/bin/env python3
"""Map a flat launch through solution 1675's positive WGM=8 blocking order."""

from __future__ import annotations

import argparse
from pathlib import Path


def map_workgroup(linear: int) -> tuple[int, int]:
    """Reference the emitted scalar mapping as (M tile, N tile)."""
    raw_n, raw_m = divmod(linear, 16)
    block, row = divmod(raw_n, 8)
    serial = row * 16 + raw_m
    block_height = 8 if block == 0 else 2
    workgroup_m, n_inner = divmod(serial, block_height)
    return workgroup_m, block * 8 + n_inner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    mapped = [map_workgroup(linear) for linear in range(160)]
    expected = [(m, n) for n in range(10) for m in range(16)]
    if sorted(mapped) != sorted(expected):
        raise ValueError("WGM=8 mapping is not a bijection over the 16x10 grid")

    lines = args.input.read_text().splitlines()
    x = next(i for i, line in enumerate(lines) if "%workgroup_n = low.live_in<amdgpu.workgroup_id.x>" in line)
    y = next(i for i, line in enumerate(lines) if "%workgroup_m = low.live_in<amdgpu.workgroup_id.y>" in line)
    lines[x] = lines[x].replace("%workgroup_n", "%linear_workgroup")
    lines[y] = lines[y].replace("%workgroup_m", "%unused_workgroup_y")

    # The native logical grid is WorkGroup0=M (16 tiles), WorkGroup1=N
    # (10 tiles). Tensile first decomposes the flat serial id row-major, then
    # remaps positive WGM=8 blocks. The last N block has height two.
    mapping = [
        "  %wgm_one = low.const<amdgpu.s_mov_b32> {imm32 = 1} : reg<amdgpu.sgpr>",
        "  %raw_n = low.op<amdgpu.s_lshr_b32.rhs_inline>(%linear_workgroup) {imm32 = 4} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %raw_n_x16 = low.op<amdgpu.s_lshl_b32.rhs_inline>(%raw_n) {imm32 = 4} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %raw_m = low.op<amdgpu.s_sub_u32>(%linear_workgroup, %raw_n_x16) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %wgm_block = low.op<amdgpu.s_lshr_b32.rhs_inline>(%raw_n) {imm32 = 3} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %wgm_block_x8 = low.op<amdgpu.s_lshl_b32.rhs_inline>(%wgm_block) {imm32 = 3} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %wgm_row = low.op<amdgpu.s_sub_u32>(%raw_n, %wgm_block_x8) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %wgm_row_x16 = low.op<amdgpu.s_lshl_b32.rhs_inline>(%wgm_row) {imm32 = 4} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %wgm_serial = low.op<amdgpu.s_add_u32>(%wgm_row_x16, %raw_m) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %wgm_m_full = low.op<amdgpu.s_lshr_b32.rhs_inline>(%wgm_serial) {imm32 = 3} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %wgm_m_tail = low.op<amdgpu.s_lshr_b32.rhs_inline>(%wgm_serial) {imm32 = 1} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %wgm_m_full_x8 = low.op<amdgpu.s_lshl_b32.rhs_inline>(%wgm_m_full) {imm32 = 3} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %wgm_m_tail_x2 = low.op<amdgpu.s_lshl_b32.rhs_inline>(%wgm_m_tail) {imm32 = 1} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %wgm_n_full = low.op<amdgpu.s_sub_u32>(%wgm_serial, %wgm_m_full_x8) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %wgm_n_tail = low.op<amdgpu.s_sub_u32>(%wgm_serial, %wgm_m_tail_x2) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %wgm_is_full = low.op<amdgpu.s_cmp_lt_u32>(%wgm_block, %wgm_one) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.scc>",
        "  %workgroup_m = low.op<amdgpu.s_cselect_b32>(%wgm_m_full, %wgm_m_tail, %wgm_is_full) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>, reg<amdgpu.scc>) -> reg<amdgpu.sgpr>",
        "  %wgm_n_inner = low.op<amdgpu.s_cselect_b32>(%wgm_n_full, %wgm_n_tail, %wgm_is_full) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>, reg<amdgpu.scc>) -> reg<amdgpu.sgpr>",
        "  %workgroup_n = low.op<amdgpu.s_add_u32>(%wgm_n_inner, %wgm_block_x8) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
    ]
    insertion = max(i for i, line in enumerate(lines) if " = low.live_in<" in line) + 1
    lines[insertion:insertion] = mapping
    lines[0] = lines[0].replace("workgroup_count(10, 16, 1)", "workgroup_count(160, 1, 1)")
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
