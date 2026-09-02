#!/usr/bin/env python3
"""Hoist invariant gfx12 compact-loop global/LDS address arithmetic.

This is deliberately a second transformation after the fragment-contract
rewrite.  Keeping the two passes separate preserves the schedule progression:
first recover the WMMA packet contract, then remove High-origin address
reconstruction from the dynamic K loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PRELUDE = """\
  %fast_c15 = low.const<amdgpu.v_mov_b32> {imm32 = 15} : reg<amdgpu.vgpr>
  %fast_c3 = low.const<amdgpu.v_mov_b32> {imm32 = 3} : reg<amdgpu.vgpr>
  %fast_c16k = low.const<amdgpu.v_mov_b32> {imm32 = 16384} : reg<amdgpu.vgpr>
  %fast_c32k = low.const<amdgpu.v_mov_b32> {imm32 = 32768} : reg<amdgpu.vgpr>
  %fast_c48k = low.const<amdgpu.v_mov_b32> {imm32 = 49152} : reg<amdgpu.vgpr>
  %fast_c64k = low.const<amdgpu.v_mov_b32> {imm32 = 65536} : reg<amdgpu.vgpr>
  %fast_c128k = low.const<amdgpu.v_mov_b32> {imm32 = 131072} : reg<amdgpu.vgpr>
  %fast_c192k = low.const<amdgpu.v_mov_b32> {imm32 = 196608} : reg<amdgpu.vgpr>
  %fast_a_row = low.op<amdgpu.v_and_b32>(%thread, %fast_c15) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_a_row_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%fast_a_row) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_a_k = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%thread) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_a_k_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%fast_a_k) {imm32 = 11} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_a0 = low.op<amdgpu.v_add_u32>(%fast_a_row_bytes, %fast_a_k_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_a1 = low.op<amdgpu.v_add_u32>(%fast_a0, %fast_c16k) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_a2 = low.op<amdgpu.v_add_u32>(%fast_a0, %fast_c32k) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_a3 = low.op<amdgpu.v_add_u32>(%fast_a0, %fast_c48k) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_b_k = low.op<amdgpu.v_and_b32>(%thread, %fast_c3) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_b_k_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%fast_b_k) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_b_n = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%thread) {imm32 = 2} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_b_n_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%fast_b_n) {imm32 = 11} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_b0 = low.op<amdgpu.v_add_u32>(%fast_b_k_bytes, %fast_b_n_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_b1 = low.op<amdgpu.v_add_u32>(%fast_b0, %fast_c64k) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_b2 = low.op<amdgpu.v_add_u32>(%fast_b0, %fast_c128k) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_b3 = low.op<amdgpu.v_add_u32>(%fast_b0, %fast_c192k) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_lds_write = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%thread) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %fast_a_ptr_lo = low.slice %1153[0] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>
  %fast_a_ptr_hi = low.slice %1153[1] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>
  %fast_b_ptr_lo = low.slice %1153[2] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>
  %fast_b_ptr_hi = low.slice %1153[3] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>
  %fast_a_wg = low.op<amdgpu.s_lshl_b32.rhs_inline>(%workgroup_m) {imm32 = 8} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_b_wg = low.op<amdgpu.s_lshl_b32.rhs_inline>(%workgroup_n) {imm32 = 18} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
"""


LOADER = """\
  %fast_a_k_base = low.op<amdgpu.s_lshl_b32.rhs_inline>(%k_base) {imm32 = 11} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_a_grid_k = low.op<amdgpu.s_add_u32>(%fast_a_wg, %fast_a_k_base) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_a_base_lo = low.op<amdgpu.s_add_u32>(%fast_a_ptr_lo, %fast_a_grid_k) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_a_base_hi = low.op<amdgpu.s_addc_u32>(%fast_a_ptr_hi, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_a_base = low.concat(%fast_a_base_lo, %fast_a_base_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>
  %fast_b_k_base = low.op<amdgpu.s_lshl_b32.rhs_inline>(%k_base) {imm32 = 1} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_b_grid_k = low.op<amdgpu.s_add_u32>(%fast_b_wg, %fast_b_k_base) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_b_base_lo = low.op<amdgpu.s_add_u32>(%fast_b_ptr_lo, %fast_b_grid_k) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_b_base_hi = low.op<amdgpu.s_addc_u32>(%fast_b_ptr_hi, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %fast_b_base = low.concat(%fast_b_base_lo, %fast_b_base_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>
  %av0 = low.op<amdgpu.global_load_b128_saddr>(%fast_a0, %fast_a_base) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %bv0 = low.op<amdgpu.global_load_b128_saddr>(%fast_b0, %fast_b_base) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %av1 = low.op<amdgpu.global_load_b128_saddr>(%fast_a1, %fast_a_base) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %bv1 = low.op<amdgpu.global_load_b128_saddr>(%fast_b1, %fast_b_base) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %av2 = low.op<amdgpu.global_load_b128_saddr>(%fast_a2, %fast_a_base) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %bv2 = low.op<amdgpu.global_load_b128_saddr>(%fast_b2, %fast_b_base) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %av3 = low.op<amdgpu.global_load_b128_saddr>(%fast_a3, %fast_a_base) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  %bv3 = low.op<amdgpu.global_load_b128_saddr>(%fast_b3, %fast_b_base) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> reg<amdgpu.vgpr x4>
  low.op<amdgpu.ds_write_b128>(%fast_lds_write, %av0) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)
  low.op<amdgpu.ds_write_b128>(%fast_lds_write, %bv0) {offset = 8192} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)
  low.op<amdgpu.ds_write_b128>(%fast_lds_write, %av1) {offset = 2048} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)
  low.op<amdgpu.ds_write_b128>(%fast_lds_write, %bv1) {offset = 10240} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)
  low.op<amdgpu.ds_write_b128>(%fast_lds_write, %av2) {offset = 4096} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)
  low.op<amdgpu.ds_write_b128>(%fast_lds_write, %bv2) {offset = 12288} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)
  low.op<amdgpu.ds_write_b128>(%fast_lds_write, %av3) {offset = 6144} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)
  low.op<amdgpu.ds_write_b128>(%fast_lds_write, %bv3) {offset = 14336} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)
  low.op<amdgpu.s_barrier_signal_all>() : ()
  low.op<amdgpu.s_barrier_wait_all>() : ()
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hoist invariant global addresses out of a gfx12 compact K loop"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    text = args.input.read_text()
    branch = "  low.br ^_bb1("
    branch_at = text.index(branch)
    text = text[:branch_at] + PRELUDE + text[branch_at:]

    body_at = text.index("^_bb2:\n") + len("^_bb2:\n")
    loader_start = text.index("  %476 = ", body_at)
    compute_start = text.index("  %624 = ", loader_start)
    text = text[:loader_start] + LOADER + text[compute_start:]

    # The fragment rewrite happens after prepared-Low DCE.  Its replacement
    # leaves the old High-derived fragment-address calculation dead, and
    # --pipeline=none intentionally does not clean it up.  Delete that region
    # and move the replacement's invariant A/B LDS addresses and selectors to
    # the entry prelude.
    dead_start = text.index("  %624 = ")
    native_start = text.index("  %lhs_selector_even = ", dead_start)
    text = text[:dead_start] + text[native_start:]

    native_start = text.index("  %lhs_selector_even = ")
    native_stop = text.index("  %lhs_h0_raw0 = ", native_start)
    native_prelude = text[native_start:native_stop]
    text = text[:native_start] + text[native_stop:]

    rhs_start = text.index("  %rhs_wave_n = ")
    rhs_stop = text.index("  %q0 = ", rhs_start)
    rhs_prelude = text[rhs_start:rhs_stop]
    text = text[:rhs_start] + text[rhs_stop:]

    branch_at = text.index(branch)
    text = text[:branch_at] + native_prelude + rhs_prelude + text[branch_at:]

    if text.count("global_load_b128_saddr") != 8:
        raise ValueError("expected exactly eight compact-loop global loads")
    if text.count("ds_write_b128") != 8:
        raise ValueError("expected exactly eight compact-loop LDS writes")
    if any(name in text for name in ("%linear1", "%linear2", "%linear3")):
        raise ValueError("old per-packet address reconstruction survived")
    if "%624" in text or "%636" in text:
        raise ValueError("dead High-derived fragment addressing survived")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise ValueError("output was not created")


if __name__ == "__main__":
    main()
