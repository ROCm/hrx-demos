#!/usr/bin/env python3
"""Adopt solution 133309's wave/q ownership for the padded B tile.

The semantic-preserving uniform-padding experiment retains the original Loom
ownership ``physical_n_block = 4 * wave_n + q``.  The incumbent instead uses
``physical_n_block = wave_n + 2 * q``.  Its cooperative B writes are otherwise
identical.  This transform changes the fragment reads and the matching output
addresses together, so every accumulator is still stored to the columns from
which its RHS fragment was loaded.

This script intentionally operates on the retained prepared-Low reproducer:
the ownership choice is precisely the schedule/IP detail under experiment.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def replace_once(source: str, old: str, new: str, description: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected one {description}, found {source.count(old)}")
    return source.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()

    # Before F(x)=x+(x>>7)*32, move wave 1 by 1024 bytes rather than 4096.
    # F(1024)=1280, which is one incumbent physical N block.
    source = replace_once(
        source,
        "  %db_rhs_wave_n = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%wave_n_id) {imm32 = 12} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        "  %db_rhs_wave_n = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%wave_n_id) {imm32 = 10} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        "RHS wave-stride instruction",
    )

    # q0..q3 select physical blocks 0,2,4,6.  q4..q7 are their +16-byte
    # K-half partners.  Adding the wave base selects blocks 1,3,5,7.
    q_offsets = {
        0: 0,
        1: 2560,
        2: 5120,
        3: 7680,
        4: 32,
        5: 2592,
        6: 5152,
        7: 7712,
    }
    read_re = re.compile(
        r"(?P<prefix>%[A-Za-z0-9_]+_compute_q(?P<q>[0-7]) = "
        r"low\.op<amdgpu\.ds_read_b128>\([^\n]+?\) \{offset = )"
        r"(?P<offset>\d+)(?P<suffix>\})"
    )

    def read_repl(match: re.Match[str]) -> str:
        q = int(match.group("q"))
        return match.group("prefix") + str(q_offsets[q]) + match.group("suffix")

    source, read_count = read_re.subn(read_repl, source)
    if read_count != 32:
        raise ValueError(f"expected 32 B fragment reads, changed {read_count}")

    # D is column-major with a 1024-element leading dimension and fp16
    # elements.  A 16-column N block is therefore 32768 bytes.  Make wave_n
    # select one block rather than four blocks.
    source = replace_once(
        source,
        "  %native_wave_n_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%wave_n_id) {imm32 = 17} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        "  %native_wave_n_bytes = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%wave_n_id) {imm32 = 15} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        "output wave-stride instruction",
    )

    # Each q group now advances two physical N blocks, not one.  The low
    # 15-bit per-element offsets are unchanged; only the q-group component is
    # doubled.  This also keeps all accesses inside the 256-column tile.
    store_re = re.compile(
        r"(?P<prefix>low\.op<amdgpu\.global_store_b64_saddr>\([^\n]+?\) "
        r"\{offset = )(?P<offset>\d+)(?P<suffix>\})"
    )

    def store_repl(match: re.Match[str]) -> str:
        old = int(match.group("offset"))
        q, within_q = divmod(old, 32768)
        if q > 3 or within_q % 2048 != 0 or within_q > 14336:
            raise ValueError(f"unexpected output offset {old}")
        new = q * 65536 + within_q
        return match.group("prefix") + str(new) + match.group("suffix")

    source, store_count = store_re.subn(store_repl, source)
    if store_count != 32:
        raise ValueError(f"expected 32 vector output stores, changed {store_count}")

    args.output.write_text(source)


if __name__ == "__main__":
    main()
