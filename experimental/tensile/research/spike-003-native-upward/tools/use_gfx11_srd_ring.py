#!/usr/bin/env python3
"""Replace gfx11 steady-loop dynamic soffsets with incrementing SRDs.

This transform is intentionally specialized to the fixed 1024x960x1024
schedule-recovery witness. It keeps the existing prologue payload addresses,
seeds A/B descriptors at K=32, advances them to the refill tile at the start
of each steady trip, and emits the five buffer loads with zero soffset. The
peeled tail guarantees that no K>=1024 refill is issued.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ADDRESS_DEFS = {
    "payload_k_base", "742", "743", "744", "745", "746", "760", "761",
    "762", "763", "780", "781", "782", "784", "800", "801", "818",
    "819",
}


def result_name(line: str) -> str | None:
    stripped = line.lstrip()
    if not stripped.startswith("%") or " = " not in stripped:
        return None
    return stripped[1:].split(" = ", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines()

    # Seed the descriptors from the same scalar addresses used for the K=32
    # payloads already loaded by the prologue. Keep the static packet offsets
    # (4080 for A and 64 for B) on the refill loads.
    entry_branch = next(i for i, line in enumerate(lines) if "low.br ^_bb1(" in line)
    seed = [
        "  %ring_a_seed_lo = low.op<amdgpu.s_add_u32>(%buffer_a_lo_raw, %init_746) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %ring_a_seed_hi = low.op<amdgpu.s_addc_u32>(%buffer_a_hi, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %ring_a_seed = low.concat(%ring_a_seed_lo, %ring_a_seed_hi, %buffer_extent, %buffer_flags) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x4>",
        "  %ring_b_seed_lo = low.op<amdgpu.s_add_u32>(%buffer_b_lo_raw, %init_763) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %ring_b_seed_hi = low.op<amdgpu.s_addc_u32>(%buffer_b_hi, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %ring_b_seed = low.concat(%ring_b_seed_lo, %ring_b_seed_hi, %buffer_extent, %buffer_flags) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x4>",
        "  %ring_a_stride = low.const<amdgpu.s_mov_b32> {imm32 = 65536} : reg<amdgpu.sgpr>",
    ]
    lines[entry_branch:entry_branch] = seed
    entry_branch += len(seed)
    lines[entry_branch] = lines[entry_branch].replace(
        ")", ", %ring_a_seed: reg<amdgpu.sgpr x4>, %ring_b_seed: reg<amdgpu.sgpr x4>)", 1
    )

    header = next(i for i, line in enumerate(lines) if line.startswith("^_bb1("))
    lines[header] = lines[header].replace(
        "):",
        ", %ring_a: reg<amdgpu.sgpr x4>, %ring_b: reg<amdgpu.sgpr x4>):",
    )
    main_end = next(i for i in range(header, len(lines)) if lines[i] == "^_bb_backedge_dispatch:")

    # Replace the first dynamic-address band with SRD slices and A advance.
    first_address = next(
        i for i in range(header, main_end) if result_name(lines[i]) == "payload_k_base"
    )
    a_advance = [
        "  %ring_a_lo = low.slice %ring_a[0] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>",
        "  %ring_a_hi = low.slice %ring_a[1] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>",
        "  %ring_a_extent = low.slice %ring_a[2] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>",
        "  %ring_a_flags = low.slice %ring_a[3] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>",
        "  %ring_a_next_lo = low.op<amdgpu.s_add_u32>(%ring_a_lo, %ring_a_stride) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  low.schedule.fence",
        "  %ring_a_next_hi = low.op<amdgpu.s_addc_u32>(%ring_a_hi, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  low.schedule.fence",
        "  %ring_a_next = low.concat(%ring_a_next_lo, %ring_a_next_hi, %ring_a_extent, %ring_a_flags) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x4>",
    ]
    lines[first_address:first_address] = a_advance
    main_end += len(a_advance)

    b_address = next(i for i in range(first_address, main_end) if result_name(lines[i]) == "760")
    b_advance = [
        "  %ring_b_lo = low.slice %ring_b[0] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>",
        "  %ring_b_hi = low.slice %ring_b[1] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>",
        "  %ring_b_extent = low.slice %ring_b[2] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>",
        "  %ring_b_flags = low.slice %ring_b[3] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>",
        "  %ring_b_next_lo = low.op<amdgpu.s_add_u32.rhs_inline>(%ring_b_lo) {imm32 = 64} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  low.schedule.fence",
        "  %ring_b_next_hi = low.op<amdgpu.s_addc_u32>(%ring_b_hi, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  low.schedule.fence",
        "  %ring_b_next = low.concat(%ring_b_next_lo, %ring_b_next_hi, %ring_b_extent, %ring_b_flags) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x4>",
    ]
    lines[b_address:b_address] = b_advance
    main_end += len(b_advance)

    # Delete all original scalar address definitions and their immediately
    # following fences. Slices/concats above are zero-packet Low structure.
    i = header
    while i < main_end:
        if result_name(lines[i]) in ADDRESS_DEFS:
            del lines[i]
            main_end -= 1
            if i < main_end and lines[i].strip() == "low.schedule.fence":
                del lines[i]
                main_end -= 1
            continue
        i += 1

    for i in range(header, main_end):
        line = lines[i]
        if "low.op<amdgpu.buffer_load_b128>" not in line:
            continue
        if "%buffer_a," in line:
            line = line.replace("%buffer_a,", "%ring_a_next,")
            # The third operand is the old dynamic soffset.
            for old in ("%746", "%784"):
                line = line.replace(old, "%zero")
        elif "%buffer_b," in line:
            line = line.replace("%buffer_b,", "%ring_b_next,")
            for old in ("%763", "%801", "%819"):
                line = line.replace(old, "%zero")
        line = line.replace(
            "low.op<amdgpu.buffer_load_b128>",
            "low.op<amdgpu.buffer_load_b128_vaddr_offset>",
        )
        line = line.replace(", %zero) {", ") {")
        line = line.replace(
            ", reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr x4>",
            ") -> reg<amdgpu.vgpr x4>",
        )
        lines[i] = line

    backedge = next(
        i for i in range(main_end, len(lines))
        if lines[i].startswith("  low.br ^_bb1(")
    )
    lines[backedge] = lines[backedge].replace(
        ")", ", %ring_a_next: reg<amdgpu.sgpr x4>, %ring_b_next: reg<amdgpu.sgpr x4>)", 1
    )

    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
