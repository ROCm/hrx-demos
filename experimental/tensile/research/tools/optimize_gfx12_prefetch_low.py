#!/usr/bin/env python3
"""Specialize and hoist the prepared-Low gfx12 prefetch-loop addresses."""

from __future__ import annotations

import argparse
from pathlib import Path


FAST_NEXT_LOADS = """\
  %fast_next_a_k = low.op<amdgpu.s_lshl_b32.rhs_inline>(%k_base) {imm32 = 11} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_a_grid = low.op<amdgpu.s_add_u32>(%824, %fast_next_a_k) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_a_lo = low.op<amdgpu.s_add_u32>(%1601, %fast_next_a_grid) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_a_hi = low.op<amdgpu.s_addc_u32>(%1600, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_a = low.concat(%fast_next_a_lo, %fast_next_a_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>
  %fast_next_b_k = low.op<amdgpu.s_lshl_b32.rhs_inline>(%k_base) {imm32 = 1} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_b_grid = low.op<amdgpu.s_add_u32>(%838, %fast_next_b_k) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_b_lo = low.op<amdgpu.s_add_u32>(%1599, %fast_next_b_grid) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_b_hi = low.op<amdgpu.s_addc_u32>(%1598, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_next_b = low.concat(%fast_next_b_lo, %fast_next_b_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>
  %next_av0 = low.op<amdgpu.global_load_b128_saddr>(%823, %fast_next_a) {offset = 65536} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %next_bv0 = low.op<amdgpu.global_load_b128_saddr>(%837, %fast_next_b) {offset = 64} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %next_av1 = low.op<amdgpu.global_load_b128_saddr>(%853, %fast_next_a) {offset = 65536} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %next_bv1 = low.op<amdgpu.global_load_b128_saddr>(%867, %fast_next_b) {offset = 64} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %next_av2 = low.op<amdgpu.global_load_b128_saddr>(%882, %fast_next_a) {offset = 65536} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %next_bv2 = low.op<amdgpu.global_load_b128_saddr>(%896, %fast_next_b) {offset = 64} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %next_av3 = low.op<amdgpu.global_load_b128_saddr>(%911, %fast_next_a) {offset = 65536} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %next_bv3 = low.op<amdgpu.global_load_b128_saddr>(%925, %fast_next_b) {offset = 64} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.input.read_text()

    # All four initial A packets share %824 and all B packets share %838.
    # Those are the already computed workgroup byte offsets used below.
    block3 = text.index("^_bb3:\n") + len("^_bb3:\n")
    block3_end = text.index("  low.br ^_bb5", block3)
    text = text[:block3] + FAST_NEXT_LOADS + text[block3_end:]

    # LDS write address and native read/packing addresses are invariant across
    # K. Move them ahead of the initial branch rather than rebuilding them in
    # every dynamic iteration.
    write_def = "  %936 = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%thread) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
    if text.count(write_def) != 1:
        raise ValueError("expected the canonical LDS write address definition")
    text = text.replace(write_def, "", 1)

    native_start = text.index("  %lhs_selector_even = ")
    native_stop = text.index("  %lhs_h0_raw0 = ", native_start)
    native_prelude = text[native_start:native_stop]
    text = text[:native_start] + text[native_stop:]

    rhs_start = text.index("  %rhs_wave_n = ")
    rhs_stop = text.index("  %q0 = ", rhs_start)
    rhs_prelude = text[rhs_start:rhs_stop]
    text = text[:rhs_start] + text[rhs_stop:]

    first_branch = text.index("  low.br ^_bb1(")
    prelude = write_def + native_prelude + rhs_prelude
    text = text[:first_branch] + prelude + text[first_branch:]

    if text.count("s_lshl_b32.rhs_inline>(%workgroup_m)") != 5:
        # Three redundant initial-load constructions and the output-grid
        # address remain; the dynamic path itself must not add four copies.
        raise ValueError("unexpected workgroup-M address construction count")
    if text.count("global_load_b128_saddr") != 16:
        raise ValueError("expected eight prologue and eight loop loads")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise ValueError("output was not created")


if __name__ == "__main__":
    main()
