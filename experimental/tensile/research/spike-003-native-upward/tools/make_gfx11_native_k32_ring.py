#!/usr/bin/env python3
"""Reduce the working gfx11 K64 motif to solution 1675's K32 LDS ring.

The input contains two inlined K32 publications per CFG trip.  This transform
keeps one: it carries the next tile's first-K16 fragments across the backedge,
XOR-swaps all four LDS read/store addresses by 0x4000, and increments K by 32.
The resulting loop has one global-load group, one five-packet LDS publication,
one barrier, twelve WMMAs, and the next first-K16 LDS reads per trip.

This is intentionally tied to the named prepared-Low values emitted by the
recorded motif lineage.  Every surgery point is asserted.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


OFFSET = re.compile(r"\{offset = (\d+)\}")


def rewrite_offset(line: str, delta: int) -> str:
    match = OFFSET.search(line)
    if match is None:
        raise ValueError(f"expected immediate offset: {line}")
    value = int(match.group(1)) + delta
    if value < 0:
        raise ValueError(f"negative rewritten offset {value}: {line}")
    return line[: match.start(1)] + str(value) + line[match.end(1) :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    if source.count("{imm32 = 961}") != 1:
        raise ValueError("expected the specialized K64 loop-end constant")
    source = source.replace("{imm32 = 961}", "{imm32 = 993}", 1)

    old_header = (
        "^_bb1(%k_base: reg<amdgpu.sgpr>, %out0: reg<amdgpu.vgpr x8>, "
        "%out1: reg<amdgpu.vgpr x8>, %out2: reg<amdgpu.vgpr x8>, "
        "%out3: reg<amdgpu.vgpr x8>, %out4: reg<amdgpu.vgpr x8>, "
        "%out5: reg<amdgpu.vgpr x8>, %cur_lhs0: reg<amdgpu.vgpr x8>, "
        "%cur_lhs1: reg<amdgpu.vgpr x8>, %cur_rhs0: reg<amdgpu.vgpr x8>, "
        "%cur_rhs1: reg<amdgpu.vgpr x8>, %cur_rhs2: reg<amdgpu.vgpr x8>):"
    )
    new_header = old_header[:-2] + (
        ", %a_read_base: reg<amdgpu.vgpr>, "
        "%b_read_base: reg<amdgpu.vgpr>, "
        "%a_store_base: reg<amdgpu.vgpr>, "
        "%b_store_base: reg<amdgpu.vgpr>):"
    )
    if source.count(old_header) != 1:
        raise ValueError("expected one K64 loop header")
    source = source.replace(old_header, new_header, 1)

    prologue = re.search(r"  low\.br \^_bb1\([^\n]+\)\n\^_bb1", source)
    if prologue is None:
        raise ValueError("could not locate prologue backedge invocation")
    prologue_line = prologue.group(0).splitlines()[0]
    if not prologue_line.endswith(")"):
        raise ValueError("unexpected prologue branch spelling")
    prologue_new = prologue_line[:-1] + (
        ", %684: reg<amdgpu.vgpr>, %722: reg<amdgpu.vgpr>, "
        "%664: reg<amdgpu.vgpr>, %670: reg<amdgpu.vgpr>)"
    )
    source = source.replace(prologue_line, prologue_new, 1)

    lines = source.splitlines()
    body_start = lines.index("^_bb2:")
    exit_start = lines.index("^_bb3:")
    candidate = next(
        i for i in range(body_start, exit_start)
        if "%candidate_next =" in lines[i]
    )
    barrier = next(
        i for i in range(candidate, exit_start)
        if "low.op<amdgpu.s_barrier>" in lines[i]
    )
    old_backedge = next(
        i for i in range(barrier, exit_start)
        if lines[i].lstrip().startswith("low.br ^_bb1(")
    )

    prefix = lines[: body_start + 1]

    first_half = lines[body_start + 1 : candidate]
    current_a_reads = 0
    current_b_reads = 0
    stage_stores = 0
    for line in first_half:
        if "low.op<amdgpu.ds_" in line and "%684" in line:
            line = line.replace("%684", "%a_read_base")
            current_a_reads += 1
        if "low.op<amdgpu.ds_read_b128>" in line and "%722" in line:
            line = line.replace("%722", "%b_read_base")
            current_b_reads += 1

        if "low.op<amdgpu.ds_write_b128>(%664, %odd_av" in line:
            line = line.replace("(%664,", "(%next_a_store_base,")
            line = rewrite_offset(line, -16384)
            stage_stores += 1
        elif "low.op<amdgpu.ds_write_b128>(%670, %odd_bv" in line:
            line = line.replace("(%670,", "(%next_b_store_base,")
            line = rewrite_offset(line, -16384)
            stage_stores += 1

        # Define next-stage bases immediately before their first use.
        if "low.op<amdgpu.ds_write_b128>" in line and stage_stores == 1:
            prefix.extend(
                [
                    "  %next_a_store_base = low.op<amdgpu.v_xor_b32.lit>"
                    "(%a_store_base) {imm32 = 16384} : "
                    "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
                    "  %next_b_store_base = low.op<amdgpu.v_xor_b32.lit>"
                    "(%b_store_base) {imm32 = 16384} : "
                    "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
                ]
            )
        prefix.append(line)

    if (current_a_reads, current_b_reads, stage_stores) != (32, 6, 5):
        raise ValueError(
            "unexpected first-half census: "
            f"A={current_a_reads}, B={current_b_reads}, stores={stage_stores}"
        )

    prefix.append(lines[barrier])
    prefix.extend(
        [
            "  %next_a_read_base = low.op<amdgpu.v_xor_b32.lit>"
            "(%a_read_base) {imm32 = 16384} : "
            "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
            "  %next_b_read_base = low.op<amdgpu.v_xor_b32.lit>"
            "(%b_read_base) {imm32 = 16384} : "
            "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
        ]
    )

    even_acc_lines = 0
    next_a_reads = 0
    next_b_reads = 0
    next_concats = 0
    for line in lines[barrier + 1 : old_backedge]:
        keep = False
        if "low.copy %even_first_acc" in line or re.match(
            r"\s+%even_second_acc[0-5] =", line
        ):
            keep = True
            even_acc_lines += 1
        elif "low.op<amdgpu.ds_" in line and "%684" in line:
            match = OFFSET.search(line)
            if match is not None and 16384 <= int(match.group(1)) <= 18368:
                line = line.replace("%684", "%next_a_read_base")
                line = rewrite_offset(line, -16384)
                keep = True
                next_a_reads += 1
        elif "low.op<amdgpu.ds_read_b128>" in line and "%722" in line:
            match = OFFSET.search(line)
            if match is not None and int(match.group(1)) in {
                20608,
                20624,
                23168,
                23184,
                25728,
                25744,
            }:
                line = line.replace("%722", "%next_b_read_base")
                line = rewrite_offset(line, -16384)
                keep = True
                next_b_reads += 1
        elif re.search(r"%odd_first_(lhs[01]|rhs[012]) = low\.concat", line):
            keep = True
            next_concats += 1
        if keep:
            prefix.append(line)

    if (even_acc_lines, next_a_reads, next_b_reads, next_concats) != (12, 32, 6, 5):
        raise ValueError(
            "unexpected carried-half census: "
            f"acc={even_acc_lines}, A={next_a_reads}, "
            f"B={next_b_reads}, concat={next_concats}"
        )

    prefix.extend(
        [
            "  %k_next = low.op<amdgpu.s_add_u32.rhs_inline>(%k_base) "
            "{imm32 = 32} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
            "  low.br ^_bb1(%k_next: reg<amdgpu.sgpr>, "
            "%even_second_acc0: reg<amdgpu.vgpr x8>, "
            "%even_second_acc1: reg<amdgpu.vgpr x8>, "
            "%even_second_acc2: reg<amdgpu.vgpr x8>, "
            "%even_second_acc3: reg<amdgpu.vgpr x8>, "
            "%even_second_acc4: reg<amdgpu.vgpr x8>, "
            "%even_second_acc5: reg<amdgpu.vgpr x8>, "
            "%odd_first_lhs0: reg<amdgpu.vgpr x8>, "
            "%odd_first_lhs1: reg<amdgpu.vgpr x8>, "
            "%odd_first_rhs0: reg<amdgpu.vgpr x8>, "
            "%odd_first_rhs1: reg<amdgpu.vgpr x8>, "
            "%odd_first_rhs2: reg<amdgpu.vgpr x8>, "
            "%next_a_read_base: reg<amdgpu.vgpr>, "
            "%next_b_read_base: reg<amdgpu.vgpr>, "
            "%next_a_store_base: reg<amdgpu.vgpr>, "
            "%next_b_store_base: reg<amdgpu.vgpr>)",
        ]
    )
    prefix.extend(lines[exit_start:])
    args.output.write_text("\n".join(prefix) + "\n")


if __name__ == "__main__":
    main()
