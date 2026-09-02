#!/usr/bin/env python3
"""Replace the gfx11 cross-lane packed epilogue with native scalar stores.

This form is intended for the source-operand-swapped WMMA control.  Swapping
the WMMA inputs transposes each 16x16 accumulator microtile; changing the store
mapping at the same time transposes it back while making adjacent lanes write
adjacent rows, as in hipBLASLt solution 1675.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EPILOGUE_LABEL = "\n^_bb3("
TARGET_MARKER = "\namdgpu.target<"


def make_epilogue(label_line: str) -> str:
    lines = [
        label_line,
        "  %native_lane_raw = low.op<amdgpu.v_and_b32.lit>(%packed_tid_saved) {imm32 = 1023} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_lane = low.op<amdgpu.v_and_b32.src0_inline>(%native_lane_raw) {imm32 = 31} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_wave_raw = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%packed_tid_saved) {imm32 = 10} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_wave = low.op<amdgpu.v_and_b32.lit>(%native_wave_raw) {imm32 = 1023} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_lane_low = low.op<amdgpu.v_and_b32.src0_inline>(%native_lane) {imm32 = 15} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_lane_low_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%native_lane_low) {imm32 = 1} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_lane_high = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%native_lane) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_lane_high_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%native_lane_high) {imm32 = 11} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_wave_m = low.op<amdgpu.v_and_b32.src0_inline>(%native_wave) {imm32 = 1} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_wave_m_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%native_wave_m) {imm32 = 5} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_wave_n = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%native_wave) {imm32 = 1} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_wave_n_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%native_wave_n) {imm32 = 15} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_lane_bytes = low.op<amdgpu.v_add_u32>(%native_lane_low_bytes, %native_lane_high_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_wave_bytes0 = low.op<amdgpu.v_add_u32>(%native_wave_m_bytes, %native_wave_n_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_in_tile = low.op<amdgpu.v_add_u32>(%native_lane_bytes, %native_wave_bytes0) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
        "  %native_wg_m_bytes = low.op<amdgpu.s_lshl_b32.rhs_inline>(%workgroup_m) {imm32 = 7} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n",
        "  %native_wg_n_scale = low.const<amdgpu.s_mov_b32> {imm32 = 196608} : reg<amdgpu.sgpr>\n",
        "  %native_wg_n_bytes = low.op<amdgpu.s_mul_i32>(%workgroup_n, %native_wg_n_scale) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n",
        "  %native_grid_base = low.op<amdgpu.s_add_u32>(%native_wg_m_bytes, %native_wg_n_bytes) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n",
        "  %native_d_base = low.op<amdgpu.v_add_u32>(%native_grid_base, %native_in_tile) : (reg<amdgpu.sgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
    ]
    for n_group in range(3):
        for element in range(8):
            address_offset = n_group * 65536 + element * 4096
            address = "%native_d_base"
            if address_offset:
                address = f"%native_d_base_n{n_group}_e{element}"
                lines.append(
                    f"  {address} = low.op<amdgpu.v_add_u32.lit>(%native_d_base) "
                    f"{{imm32 = {address_offset}}} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
                )
            for m_group in (0, 1):
                accumulator = n_group + 3 * m_group
                offset = m_group * 64
                stem = f"native_d_a{accumulator}_e{element}"
                lines.extend(
                    [
                        f"  %{stem} = low.slice %final_out{accumulator}[{element}] : reg<amdgpu.vgpr x8> -> reg<amdgpu.vgpr>\n",
                        f"  %{stem}_f16 = low.op<amdgpu.v_cvt_f16_f32>(%{stem}) : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n",
                        f"  low.op<amdgpu.global_store_b16_saddr>({address}, %{stem}_f16, %d) {{offset = {offset}}} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>)\n",
                    ]
                )
    lines.extend(["  low.return\n", "}\n"])
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    label = source.find(EPILOGUE_LABEL)
    if label < 0:
        raise ValueError(f"missing epilogue label {EPILOGUE_LABEL!r}")
    label += 1
    label_end = source.find("\n", label) + 1
    target = source.find(TARGET_MARKER, label_end)
    if label_end == 0 or target < 0:
        raise ValueError("malformed epilogue or missing target declaration")
    transformed = source[:label] + make_epilogue(source[label:label_end]) + source[target:]
    if transformed.count("global_store_b16_saddr") != 48:
        raise ValueError("native epilogue must contain exactly 48 scalar stores")
    args.output.write_text(transformed)


if __name__ == "__main__":
    main()
