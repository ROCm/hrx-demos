#!/usr/bin/env python3
"""Port the selected TensileLite kernel's dynamic B LDS swizzle.

This consumes a motif to which ``pad_gfx12_b_lds.py`` has already been
applied.  The incumbent does two things in addition to the 128-byte static
padding:

* cooperative B stores transpose the linear workitem id in groups of four;
* B fragment reads fold wave-N and lane coordinates into an unpadded byte
  address, then add 32 bytes after each 128-byte block.

Both mappings were reconstructed from the exact solution-133309 HSACO.  The
existing global-load mapping is retained; it already produces the logical
``[n_local, k_local:k_local+8]`` vector consumed by the transposed store.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    anchor = (
        "  %wave_n_id = low.op<amdgpu.v_lshrrev_b32.lit>(%wave) {imm32 = 1} : "
        "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>"
    )
    if source.count(anchor) != 1:
        raise ValueError("expected one wave-N anchor")
    setup = "\n".join(
        (
            anchor,
            "  %native_b_lds_3 = low.const<amdgpu.v_mov_b32> {imm32 = 3} : reg<amdgpu.vgpr>",
            "  %native_b_lds_8192 = low.const<amdgpu.v_mov_b32> {imm32 = 8192} : reg<amdgpu.vgpr>",
            "  %native_b_store_group = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%thread) {imm32 = 2} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_store_group_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%native_b_store_group) {imm32 = 6} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_store_lane = low.op<amdgpu.v_and_b32>(%thread, %native_b_lds_3) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_store_lane_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%native_b_store_lane) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_store_raw = low.op<amdgpu.v_add_u32>(%native_b_store_group_bytes, %native_b_store_lane_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_store_pad_block = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%native_b_store_raw) {imm32 = 7} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_store_pad_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%native_b_store_pad_block) {imm32 = 5} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_store_padded = low.op<amdgpu.v_add_u32>(%native_b_store_raw, %native_b_store_pad_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_store_addr = low.op<amdgpu.v_add_u32>(%native_b_store_padded, %native_b_lds_8192) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_store_stage1 = low.op<amdgpu.v_xor_b32.lit>(%native_b_store_addr) {imm32 = 32768} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_read_lane_low = low.op<amdgpu.v_and_b32.src0_inline>(%lane) {imm32 = 15} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_read_lane_low_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%native_b_read_lane_low) {imm32 = 6} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_read_lane_high = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%lane) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_read_lane_high_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%native_b_read_lane_high) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_read_wave_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%wave_n_id) {imm32 = 10} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_read_raw0 = low.op<amdgpu.v_add_u32>(%native_b_read_lane_low_bytes, %native_b_read_lane_high_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_read_raw = low.op<amdgpu.v_add_u32>(%native_b_read_raw0, %native_b_read_wave_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_read_pad_block = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%native_b_read_raw) {imm32 = 7} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_read_pad_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%native_b_read_pad_block) {imm32 = 5} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_read_padded = low.op<amdgpu.v_add_u32>(%native_b_read_raw, %native_b_read_pad_bytes) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %native_b_read_addr = low.op<amdgpu.v_add_u32>(%native_b_read_padded, %native_b_lds_8192) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        )
    )
    source = source.replace(anchor, setup)

    rhs_definition = re.compile(
        r"  %db_rhs_addr = low\.op<amdgpu\.v_add_u32>\([^\n]+\) : "
        r"\(reg<amdgpu\.vgpr>, reg<amdgpu\.vgpr>\) -> reg<amdgpu\.vgpr>"
    )
    source, rhs_count = rhs_definition.subn(
        "  %db_rhs_addr = low.copy %native_b_read_addr : reg<amdgpu.vgpr> -> reg<amdgpu.vgpr>",
        source,
    )
    if rhs_count != 1:
        raise ValueError(f"expected one B read base, replaced {rhs_count}")

    write_pattern = re.compile(
        r"(?P<prefix>low\.op<amdgpu\.ds_write_b128>\()"
        r"(?P<address>%[A-Za-z0-9_]+), "
        r"(?P<value>%[A-Za-z0-9_]*bv[0-9]+)\) \{offset = "
        r"(?P<offset>\d+)(?P<suffix>\} memory_access\([^\n]+)"
    )

    low_offsets = {8192: 0, 10752: 2560, 13312: 5120, 15872: 7680}
    high_offsets = {40960: 0, 43520: 2560, 46080: 5120, 48640: 7680}

    def rewrite_write(match: re.Match[str]) -> str:
        old = int(match.group("offset"))
        if old in low_offsets:
            address, new = "%native_b_store_addr", low_offsets[old]
        elif old in high_offsets:
            address, new = "%native_b_store_stage1", high_offsets[old]
        else:
            raise ValueError(f"unexpected padded B-write offset {old}")
        return (
            match.group("prefix")
            + address
            + ", "
            + match.group("value")
            + ") {offset = "
            + str(new)
            + match.group("suffix")
        )

    source, write_count = write_pattern.subn(rewrite_write, source)
    if write_count != 16:
        raise ValueError(f"expected 16 B cooperative writes, replaced {write_count}")

    read_offsets = {
        8192: 0,
        8224: 32,
        10752: 2560,
        10784: 2592,
        13312: 5120,
        13344: 5152,
        15872: 7680,
        15904: 7712,
    }
    read_pattern = re.compile(
        r"(?P<prefix>low\.op<amdgpu\.ds_read_b128>\([^\n]+?\) \{offset = )"
        r"(?P<offset>\d+)(?P<suffix>\})"
    )

    def rewrite_read(match: re.Match[str]) -> str:
        old = int(match.group("offset"))
        return match.group("prefix") + str(read_offsets[old]) + match.group("suffix")

    source, read_count = read_pattern.subn(rewrite_read, source)
    if read_count != 32:
        raise ValueError(f"expected 32 B fragment reads, replaced {read_count}")
    args.output.write_text(source)


if __name__ == "__main__":
    main()
