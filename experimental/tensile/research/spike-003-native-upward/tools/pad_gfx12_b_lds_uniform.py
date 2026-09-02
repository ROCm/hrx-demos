#!/usr/bin/env python3
"""Apply 128-byte B-LDS padding without changing the motif's data mapping.

For a byte address ``x`` relative to the B tile, this applies the bijection
``x -> x + (x >> 7) * 32`` to both cooperative writes and fragment reads.
Unlike the earlier static-offset experiment, the dynamic wave/lane part of
the read address is transformed too.  This makes the experiment semantic
preserving by construction while testing the incumbent's 128B+32B bank
layout independently of its different wave-to-output mapping.
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
    source, reserve_count = re.subn(
        r"byte_length = 49152|byte_length = 51200",
        "byte_length = 51200",
        source,
        count=1,
    )
    if reserve_count != 1:
        raise ValueError("expected one 49152- or 51200-byte LDS reservation")

    thread_anchor = (
        "  %thread = low.op<amdgpu.v_lshl_add_u32.shift_imm>(%wave, %lane) "
        "{shift = 5} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>"
    )
    if source.count(thread_anchor) != 1:
        raise ValueError("expected one linear-thread anchor")
    write_setup = "\n".join(
        (
            thread_anchor,
            "  %uniform_b_base = low.const<amdgpu.v_mov_b32> {imm32 = 8192} : reg<amdgpu.vgpr>",
            "  %uniform_b_store_raw = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%thread) {imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %uniform_b_store_block = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%uniform_b_store_raw) {imm32 = 7} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %uniform_b_store_pad = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%uniform_b_store_block) {imm32 = 5} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %uniform_b_store_rel = low.op<amdgpu.v_add_u32>(%uniform_b_store_raw, %uniform_b_store_pad) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %uniform_b_store = low.op<amdgpu.v_add_u32>(%uniform_b_store_rel, %uniform_b_base) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %uniform_b_store_stage1 = low.op<amdgpu.v_xor_b32.lit>(%uniform_b_store) {imm32 = 32768} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        )
    )
    source = source.replace(thread_anchor, write_setup)

    read_anchor_re = re.compile(
        r"(?P<anchor>  %db_rhs_addr = low\.op<amdgpu\.v_add_u32>\([^\n]+\) : "
        r"\(reg<amdgpu\.vgpr>, reg<amdgpu\.vgpr>\) -> reg<amdgpu\.vgpr>)"
    )
    match = read_anchor_re.search(source)
    if match is None:
        raise ValueError("expected one B fragment-read address anchor")
    read_setup = "\n".join(
        (
            match.group("anchor"),
            "  %uniform_b_read_block = low.op<amdgpu.v_lshrrev_b32.src0_inline>(%db_rhs_addr) {imm32 = 7} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %uniform_b_read_pad = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%uniform_b_read_block) {imm32 = 5} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %uniform_b_read_rel = low.op<amdgpu.v_add_u32>(%db_rhs_addr, %uniform_b_read_pad) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %uniform_b_read = low.op<amdgpu.v_add_u32>(%uniform_b_read_rel, %uniform_b_base) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        )
    )
    source = read_anchor_re.sub(read_setup, source, count=1)
    # All later copies/xors use this name.  Keep the original definition as
    # the input to the transform and redirect only its consumers.
    suffix = source[match.start() + len(read_setup) :]
    suffix = suffix.replace("%db_rhs_addr", "%uniform_b_read")
    source = source[: match.start() + len(read_setup)] + suffix

    low_offsets = {8192: 0, 10240: 2560, 12288: 5120, 14336: 7680}
    high_offsets = {40960: 0, 43008: 2560, 45056: 5120, 47104: 7680}
    write_re = re.compile(
        r"(?P<prefix>low\.op<amdgpu\.ds_write_b128>\()"
        r"(?P<address>%[A-Za-z0-9_]+), (?P<value>%[A-Za-z0-9_]*bv[0-9]+)\) "
        r"\{offset = (?P<offset>\d+)\} memory_access\(\["
        r"(?P<access>[^\]]+)\]\)(?P<suffix>[^\n]*)"
    )

    def write_repl(m: re.Match[str]) -> str:
        old = int(m.group("offset"))
        if old in low_offsets:
            address, new = "%uniform_b_store", low_offsets[old]
        elif old in high_offsets:
            address, new = "%uniform_b_store_stage1", high_offsets[old]
        else:
            raise ValueError(f"unexpected B-write offset {old}")
        # The last four memory_access fields are the minimum/maximum byte
        # address and the corresponding one-past-end bounds for the wave.
        # Preserve the descriptor facts ahead of them, but make the envelope
        # agree with the nonlinear F(x) address authored above.  Each lane
        # writes 16 bytes and thread 127 starts at F(127 * 16) = 2512.
        access = [int(field.strip()) for field in m.group("access").split(",")]
        if len(access) < 4:
            raise ValueError("malformed memory_access annotation")
        stage_base = 8192 if address == "%uniform_b_store" else 40960
        wave_min = stage_base + new
        wave_max = wave_min + 2512
        access[-4:] = [wave_min, wave_max, wave_min + 16, wave_max + 16]
        access_text = ", ".join(str(field) for field in access)
        return (
            f'{m.group("prefix")}{address}, {m.group("value")}) '
            f'{{offset = {new}}} memory_access([{access_text}]){m.group("suffix")}'
        )

    source, write_count = write_re.subn(write_repl, source)
    if write_count != 16:
        raise ValueError(f"expected 16 B writes, replaced {write_count}")

    read_offsets = {
        8192: 0,
        8224: 32,
        9216: 1280,
        9248: 1312,
        10240: 2560,
        10272: 2592,
        11264: 3840,
        11296: 3872,
    }
    # Static offsets are transformed relative to the dynamic read base.  The
    # dynamic term is a multiple of 128 after lane decomposition, so this
    # decomposition is equivalent to transforming the complete address.
    read_re = re.compile(
        r"(?P<prefix>low\.op<amdgpu\.ds_read_b128>\([^\n]+?\) \{offset = )"
        r"(?P<offset>\d+)(?P<suffix>\})"
    )

    def read_repl(m: re.Match[str]) -> str:
        old = int(m.group("offset"))
        return m.group("prefix") + str(read_offsets[old]) + m.group("suffix")

    source, read_count = read_re.subn(read_repl, source)
    if read_count != 32:
        raise ValueError(f"expected 32 B reads, replaced {read_count}")

    args.output.write_text(source)


if __name__ == "__main__":
    main()
