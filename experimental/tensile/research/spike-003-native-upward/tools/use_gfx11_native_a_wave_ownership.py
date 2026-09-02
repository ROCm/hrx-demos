#!/usr/bin/env python3
"""Match solution 1675's interleaved A-wave ownership on gfx11.

The source motif gives each wave a contiguous 32-row A fragment.  Tensile's
selected 64x96 macro-tile gives wave 0 rows [0:16) and [32:48), and wave 1
rows [16:32) and [48:64).  This narrow prepared-Low transform changes the A
read base from ``wave_m * 64`` bytes to ``wave_m * 32`` bytes and moves each
second M fragment from +32 to +64 bytes.  It makes the corresponding output
mapping change so the mathematical result is unchanged.

Run this after ``use_gfx11_native_a_block_padding.py``.  The assertions are
intentional: this is an auditable experiment, not a general source rewriter.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


OFFSET = re.compile(r"\{offset = (\d+)\}")
OUT_SLICE = re.compile(r"low\.slice %out([0-5])\[")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    replacements = (
        (
            "  %681 = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%wave_m_id) "
            "{imm32 = 6}",
            "  %681 = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%wave_m_id) "
            "{imm32 = 5}",
        ),
        (
            "  %1194 = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
            "(%epilogue_wave_m_id) {imm32 = 6}",
            "  %1194 = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
            "(%epilogue_wave_m_id) {imm32 = 5}",
        ),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"expected exactly one occurrence of {old!r}")
        source = source.replace(old, new, 1)

    output: list[str] = []
    changed_a_reads = 0
    changed_stores = 0
    lhs_groups = 0
    pending_a_reads: list[int] = []
    current_out: int | None = None
    for line in source.splitlines():
        slice_match = OUT_SLICE.search(line)
        if slice_match is not None:
            current_out = int(slice_match.group(1))

        if "low.op<amdgpu.ds_load_u16_d16" in line and "%684" in line:
            pending_a_reads.append(len(output))

        lhs_concat = re.search(r"%[A-Za-z0-9_]*lhs([01]) = low\.concat", line)
        if lhs_concat is not None:
            if len(pending_a_reads) != 16:
                raise ValueError(
                    f"expected 16 A reads before lhs concat, got {len(pending_a_reads)}"
                )
            if lhs_concat.group(1) == "1":
                for index in pending_a_reads:
                    read_line = output[index]
                    match = OFFSET.search(read_line)
                    if match is None:
                        raise ValueError(
                            f"A LDS read has no immediate offset: {read_line}"
                        )
                    old = int(match.group(1))
                    new = old + 32
                    output[index] = (
                        read_line[: match.start(1)]
                        + str(new)
                        + read_line[match.end(1) :]
                    )
                    changed_a_reads += 1
            pending_a_reads.clear()
            lhs_groups += 1

        if (
            current_out is not None
            and current_out >= 3
            and "low.op<amdgpu.global_store_b32_saddr>" in line
        ):
            match = OFFSET.search(line)
            if match is None:
                raise ValueError(f"output store has no immediate offset: {line}")
            old = int(match.group(1))
            if old not in range(32, 61, 4):
                raise ValueError(
                    f"unexpected second-M output offset {old} for out{current_out}"
                )
            new = old + 32
            line = line[: match.start(1)] + str(new) + line[match.end(1) :]
            changed_stores += 1

        output.append(line)

    if pending_a_reads:
        raise ValueError(f"unclaimed trailing A reads: {len(pending_a_reads)}")
    if lhs_groups != 10:
        raise ValueError(f"expected 10 A fragment groups, found {lhs_groups}")
    if changed_a_reads != 80:
        raise ValueError(f"expected 80 second-M A reads, changed {changed_a_reads}")
    if changed_stores != 24:
        raise ValueError(f"expected 24 second-M output stores, changed {changed_stores}")
    args.output.write_text("\n".join(output) + "\n")


if __name__ == "__main__":
    main()
