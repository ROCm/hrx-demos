#!/usr/bin/env python3
"""Rematerialize gfx11 LDS-store indices immediately before their use.

The High lowering shares the vector indices used by the five global loads with
the corresponding later LDS stores.  That keeps seven scalar VGPR values live
across the braided WMMA/LDS body. Tensile instead recomputes/maintains a small
store-address state. This prepared-Low experiment duplicates the cheap index
arithmetic at the store site, allowing the global-load copies to die early.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "  %1110 = low.const<amdgpu.v_mov_b32> {imm32 = 65}"
INSERT = """  %store_mask7 = low.const<amdgpu.v_mov_b32> {imm32 = 7} : reg<amdgpu.vgpr>
  %store_mask3 = low.const<amdgpu.v_mov_b32> {imm32 = 3} : reg<amdgpu.vgpr>
  %store_plus128 = low.const<amdgpu.v_mov_b32> {imm32 = 128} : reg<amdgpu.vgpr>
  %store_plus256 = low.const<amdgpu.v_mov_b32> {imm32 = 256} : reg<amdgpu.vgpr>
  %store_linear1 = low.op<amdgpu.v_add_u32>(%thread, %store_plus128) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_linear2 = low.op<amdgpu.v_add_u32>(%thread, %store_plus256) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_a_row0 = low.op<amdgpu.v_and_b32>(%thread, %store_mask7) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_a_kl0 = low.op<amdgpu.v_lshrrev_b32.lit>(%thread) {imm32 = 3} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_a_row1 = low.op<amdgpu.v_and_b32>(%store_linear1, %store_mask7) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_a_kl1 = low.op<amdgpu.v_lshrrev_b32.lit>(%store_linear1) {imm32 = 3} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_b_k0 = low.op<amdgpu.v_and_b32>(%thread, %store_mask3) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_b_n0 = low.op<amdgpu.v_lshrrev_b32.lit>(%thread) {imm32 = 2} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_b_k1 = low.op<amdgpu.v_and_b32>(%store_linear1, %store_mask3) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_b_n1 = low.op<amdgpu.v_lshrrev_b32.lit>(%store_linear1) {imm32 = 2} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_b_k2 = low.op<amdgpu.v_and_b32>(%store_linear2, %store_mask3) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %store_b_n2 = low.op<amdgpu.v_lshrrev_b32.lit>(%store_linear2) {imm32 = 2} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
"""

REPLACEMENTS = {
    "%odd_a_kl0": "%store_a_kl0",
    "%odd_a_row_vec0": "%store_a_row0",
    "%odd_a_kl1": "%store_a_kl1",
    "%odd_a_row_vec1": "%store_a_row1",
    "%odd_b_nl0": "%store_b_n0",
    "%odd_b_k_vec0": "%store_b_k0",
    "%odd_b_nl1": "%store_b_n1",
    "%odd_b_k_vec1": "%store_b_k1",
    "%odd_b_nl2": "%store_b_n2",
    "%odd_b_k_vec2": "%store_b_k2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.read_text()
    marker_pos = source.find(MARKER)
    if marker_pos < 0:
        raise ValueError("store-address marker not found")
    source = source[:marker_pos] + INSERT + source[marker_pos:]
    store_region = source[marker_pos + len(INSERT) :]
    for old, new in REPLACEMENTS.items():
        if old not in store_region:
            raise ValueError(f"store-region operand {old} not found")
        store_region = store_region.replace(old, new, 1)
    source = source[: marker_pos + len(INSERT)] + store_region
    args.output.write_text(source)


if __name__ == "__main__":
    main()
