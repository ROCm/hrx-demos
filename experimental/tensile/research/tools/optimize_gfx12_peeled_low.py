#!/usr/bin/env python3
"""Hoist repeated addresses in the fixed-loop/peeled-tail gfx12 Low form."""

from __future__ import annotations

import argparse
from pathlib import Path


NEXT = """\
  %fast_next_a_k = low.op<amdgpu.s_lshl_b32.rhs_inline>(%k_base) {imm32 = 11} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_a_grid = low.op<amdgpu.s_add_u32>(%1013, %fast_next_a_k) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_a_lo = low.op<amdgpu.s_add_u32>(%1952, %fast_next_a_grid) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_a_hi = low.op<amdgpu.s_addc_u32>(%1951, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_a = low.concat(%fast_next_a_lo, %fast_next_a_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>
  %fast_next_b_k = low.op<amdgpu.s_lshl_b32.rhs_inline>(%k_base) {imm32 = 1} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_b_grid = low.op<amdgpu.s_add_u32>(%1027, %fast_next_b_k) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_b_lo = low.op<amdgpu.s_add_u32>(%1950, %fast_next_b_grid) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_b_hi = low.op<amdgpu.s_addc_u32>(%1949, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_b = low.concat(%fast_next_b_lo, %fast_next_b_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>
  %next_av0 = low.op<amdgpu.global_load_b128_saddr>(%1012, %fast_next_a) {offset = 65536} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %next_bv0 = low.op<amdgpu.global_load_b128_saddr>(%1026, %fast_next_b) {offset = 64} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %next_av1 = low.op<amdgpu.global_load_b128_saddr>(%1042, %fast_next_a) {offset = 65536} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %next_bv1 = low.op<amdgpu.global_load_b128_saddr>(%1056, %fast_next_b) {offset = 64} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %next_av2 = low.op<amdgpu.global_load_b128_saddr>(%1071, %fast_next_a) {offset = 65536} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %next_bv2 = low.op<amdgpu.global_load_b128_saddr>(%1085, %fast_next_b) {offset = 64} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %next_av3 = low.op<amdgpu.global_load_b128_saddr>(%1100, %fast_next_a) {offset = 65536} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %next_bv3 = low.op<amdgpu.global_load_b128_saddr>(%1114, %fast_next_b) {offset = 64} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.input.read_text()

    begin = text.index("  %1142 = ", text.index("^_bb2:\n"))
    end = text.index("\n", text.index("  %next_bv3 = ", begin)) + 1
    text = text[:begin] + NEXT + text[end:]

    write_def = "  %1125 = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%thread) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
    if text.count(write_def) != 1:
        raise ValueError("expected canonical main-loop LDS write address")
    text = text.replace(write_def, "", 1)

    native_start = text.index("  %lhs_selector_even = ")
    native_stop = text.index("  %lhs_h0_raw0 = ", native_start)
    native = text[native_start:native_stop]
    text = text[:native_start] + text[native_stop:]
    rhs_start = text.index("  %rhs_wave_n = ")
    rhs_stop = text.index("  %q0 = ", rhs_start)
    rhs = text[rhs_start:rhs_stop]
    text = text[:rhs_start] + text[rhs_stop:]

    first_branch = text.index("  low.br ^_bb1(")
    text = text[:first_branch] + write_def + native + rhs + text[first_branch:]

    if text.count("global_load_b128_saddr") != 16:
        raise ValueError("expected eight prologue and eight dynamic loads")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise ValueError("output was not created")


if __name__ == "__main__":
    main()
