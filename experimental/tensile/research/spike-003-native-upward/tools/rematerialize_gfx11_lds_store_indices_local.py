#!/usr/bin/env python3
"""Locally rematerialize each gfx11 LDS-store index just before its store."""

from __future__ import annotations

import argparse
from pathlib import Path


EDITS = (
    (
        "  %1110 = low.const<amdgpu.v_mov_b32>",
        """  %store_a0_mask = low.const<amdgpu.v_mov_b32> {imm32 = 7} : reg<amdgpu.vgpr>
  %store_a0_row = low.op<amdgpu.v_and_b32>(%thread, %store_a0_mask) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_a0_k = low.op<amdgpu.v_lshrrev_b32.lit>(%thread) {imm32 = 3} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
""",
        (("%odd_a_kl0", "%store_a0_k"), ("%odd_a_row_vec0", "%store_a0_row")),
    ),
    (
        "  %1115 = low.op<amdgpu.v_mul_lo_u32>",
        """  %store_a1_delta = low.const<amdgpu.v_mov_b32> {imm32 = 128} : reg<amdgpu.vgpr>
  %store_a1_linear = low.op<amdgpu.v_add_u32>(%thread, %store_a1_delta) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_a1_mask = low.const<amdgpu.v_mov_b32> {imm32 = 7} : reg<amdgpu.vgpr>
  %store_a1_row = low.op<amdgpu.v_and_b32>(%store_a1_linear, %store_a1_mask) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_a1_k = low.op<amdgpu.v_lshrrev_b32.lit>(%store_a1_linear) {imm32 = 3} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
""",
        (("%odd_a_kl1", "%store_a1_k"), ("%odd_a_row_vec1", "%store_a1_row")),
    ),
    (
        "  %1118 = low.op<amdgpu.v_mad_u32_u24.src1_lit>",
        """  %store_b0_mask = low.const<amdgpu.v_mov_b32> {imm32 = 3} : reg<amdgpu.vgpr>
  %store_b0_k = low.op<amdgpu.v_and_b32>(%thread, %store_b0_mask) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_b0_n = low.op<amdgpu.v_lshrrev_b32.lit>(%thread) {imm32 = 2} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
""",
        (("%odd_b_nl0", "%store_b0_n"), ("%odd_b_k_vec0", "%store_b0_k")),
    ),
    (
        "  %1120 = low.op<amdgpu.v_mad_u32_u24.src1_lit>",
        """  %store_b1_delta = low.const<amdgpu.v_mov_b32> {imm32 = 128} : reg<amdgpu.vgpr>
  %store_b1_linear = low.op<amdgpu.v_add_u32>(%thread, %store_b1_delta) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_b1_mask = low.const<amdgpu.v_mov_b32> {imm32 = 3} : reg<amdgpu.vgpr>
  %store_b1_k = low.op<amdgpu.v_and_b32>(%store_b1_linear, %store_b1_mask) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_b1_n = low.op<amdgpu.v_lshrrev_b32.lit>(%store_b1_linear) {imm32 = 2} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
""",
        (("%odd_b_nl1", "%store_b1_n"), ("%odd_b_k_vec1", "%store_b1_k")),
    ),
    (
        "  %1122 = low.op<amdgpu.v_mad_u32_u24.src1_lit>",
        """  %store_b2_delta = low.const<amdgpu.v_mov_b32> {imm32 = 256} : reg<amdgpu.vgpr>
  %store_b2_linear = low.op<amdgpu.v_add_u32>(%thread, %store_b2_delta) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_b2_mask = low.const<amdgpu.v_mov_b32> {imm32 = 3} : reg<amdgpu.vgpr>
  %store_b2_k = low.op<amdgpu.v_and_b32>(%store_b2_linear, %store_b2_mask) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_b2_n = low.op<amdgpu.v_lshrrev_b32.lit>(%store_b2_linear) {imm32 = 2} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
""",
        (("%odd_b_nl2", "%store_b2_n"), ("%odd_b_k_vec2", "%store_b2_k")),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.read_text()
    cursor = 0
    for marker, insertion, replacements in EDITS:
        position = source.find(marker, cursor)
        if position < 0:
            raise ValueError(f"marker not found: {marker}")
        source = source[:position] + insertion + source[position:]
        edit_start = position + len(insertion)
        for old, new in replacements:
            operand_position = source.find(old, edit_start)
            if operand_position < 0:
                raise ValueError(f"operand {old} not found after {marker}")
            source = (source[:operand_position] + new +
                      source[operand_position + len(old):])
        cursor = edit_start
    args.output.write_text(source)


if __name__ == "__main__":
    main()
