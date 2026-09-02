#!/usr/bin/env python3
"""Fold the two-tile K64 PGR2 body into a native-shaped K32 ring.

Each iteration computes one K32 tile, writes the carried next tile to the
opposite LDS stage, and issues the tile after that into the freed payload
registers. Four independent LDS addresses are toggled because cooperative
writes and fragment reads use different lane mappings.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def line_bounds(source: str, offset: int) -> tuple[int, int]:
    return source.rfind("\n", 0, offset) + 1, source.index("\n", offset)


def append_arguments(line: str, arguments: str, ending: str) -> str:
    if not line.endswith(ending):
        raise ValueError(f"unexpected line ending: {line[-100:]}")
    return line[: -len(ending)] + ", " + arguments + ending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.read_text()

    # The prologue's cooperative A-write address is the stage-0 write base.
    # Materialize its other-stage partner before entering the loop.
    a0_marker = "  %1631 = low.op<amdgpu.v_lshlrev_b32.src0_inline>"
    marker_at = source.index(a0_marker)
    _, marker_end = line_bounds(source, marker_at)
    a1_definition = (
        "\n  %k32_a_store_stage1 = low.op<amdgpu.v_xor_b32.lit>(%1631) "
        "{imm32 = 32768} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>"
    )
    source = source[:marker_end] + a1_definition + source[marker_end:]

    initial_stage_args = (
        "%db_lhs_addr: reg<amdgpu.vgpr>, "
        "%uniform_b_read: reg<amdgpu.vgpr>, "
        "%k32_a_store_stage1: reg<amdgpu.vgpr>, "
        "%uniform_b_store_stage1: reg<amdgpu.vgpr>"
    )
    first_branch = source.index("  low.br ^_bb1(")
    start, end = line_bounds(source, first_branch)
    source = (
        source[:start]
        + append_arguments(source[start:end], initial_stage_args, ")")
        + source[end:]
    )

    header_at = source.index("\n^_bb1(") + 1
    start, end = line_bounds(source, header_at)
    stage_header = (
        "%k32_lhs_read: reg<amdgpu.vgpr>, "
        "%k32_rhs_read: reg<amdgpu.vgpr>, "
        "%k32_a_write: reg<amdgpu.vgpr>, "
        "%k32_b_write: reg<amdgpu.vgpr>"
    )
    source = (
        source[:start]
        + append_arguments(source[start:end], stage_header, "):")
        + source[end:]
    )

    source = source.replace(
        "  %even_compute_lhs_addr = low.copy %db_lhs_addr",
        "  %even_compute_lhs_addr = low.copy %k32_lhs_read",
        1,
    ).replace(
        "  %even_compute_rhs_addr = low.copy %uniform_b_read",
        "  %even_compute_rhs_addr = low.copy %k32_rhs_read",
        1,
    )

    # Delete the redundant in-loop reconstruction of thread*16.
    setup_start = source.index(
        "  %2057 = low.op<amdgpu.v_lshl_add_u32.shift_imm>"
    )
    setup_end = source.index(
        "\n", source.index("  %2058 = low.op<amdgpu.v_lshlrev_b32", setup_start)
    ) + 1
    source = source[:setup_start] + source[setup_end:]

    odd_start = source.index("  %odd_compute_lhs_addr =")
    prefix = source[:odd_start]
    suffix = source[odd_start:]

    a_store_re = re.compile(
        r"low\.op<amdgpu\.ds_write_b128>\(%2058, (?P<value>%p_av[0-3])\) "
        r"\{offset = (?P<offset>\d+)\} memory_access\(\[[^\]]+\]\)"
    )

    def a_store_repl(match: re.Match[str]) -> str:
        offset = int(match.group("offset"))
        if offset < 32768:
            raise ValueError(f"unexpected stage-1 A offset {offset}")
        local_offset = offset - 32768
        # Conservative two-stage envelope for this one dynamically toggled op.
        minimum = local_offset
        maximum = 32768 + local_offset + 2032
        return (
            f"low.op<amdgpu.ds_write_b128>(%k32_a_write, {match.group('value')}) "
            f"{{offset = {local_offset}}} "
            f"memory_access([0, 3, 61, -1, 43, 16, 0, 16, 3, "
            f"{minimum}, {maximum}, {minimum + 16}, {maximum + 16}])"
        )

    prefix, a_count = a_store_re.subn(a_store_repl, prefix)
    if a_count != 4:
        raise ValueError(f"expected four even-stage A stores, changed {a_count}")
    prefix = prefix.replace(
        "low.op<amdgpu.ds_write_b128>(%uniform_b_store_stage1, %p_bv",
        "low.op<amdgpu.ds_write_b128>(%k32_b_write, %p_bv",
    )
    if prefix.count("low.op<amdgpu.ds_write_b128>(%k32_b_write, %p_bv") != 4:
        raise ValueError("expected four even-stage B stores")
    b_store_re = re.compile(
        r"(?P<head>low\.op<amdgpu\.ds_write_b128>\(%k32_b_write, %p_bv[0-3]\) "
        r"\{offset = (?P<offset>\d+)\} )memory_access\(\[[^\]]+\]\)"
    )

    def b_store_repl(match: re.Match[str]) -> str:
        offset = int(match.group("offset"))
        minimum = 8192 + offset
        maximum = 40960 + offset + 2512
        return (
            match.group("head")
            + "memory_access([0, 3, 61, -1, 43, 16, 0, 16, 3, "
            + f"{minimum}, {maximum}, {minimum + 16}, {maximum + 16}])"
        )

    prefix, b_count = b_store_re.subn(b_store_repl, prefix)
    if b_count != 4:
        raise ValueError(f"expected four B envelopes, changed {b_count}")

    # The old odd half ends at the loop backedge. Replace all of it with a
    # stage toggle and a branch carrying the even half's products.
    back_at = suffix.index("  low.br ^_bb1(")
    _, back_end = line_bounds(suffix, back_at)
    old_back = suffix[suffix.rfind("\n", 0, back_at) + 1 : back_end]
    branch = old_back.replace("%odd_compute_y", "%even_compute_y").replace(
        "%p_next_", "%even_load_"
    )
    # Drop any existing induction/address updates immediately before the old
    # branch by replacing from odd_start through the branch as one unit.
    stage_update = """\
  %k32_lhs_read_next = low.op<amdgpu.v_xor_b32.lit>(%k32_lhs_read) {imm32 = 32768} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %k32_rhs_read_next = low.op<amdgpu.v_xor_b32.lit>(%k32_rhs_read) {imm32 = 32768} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %k32_a_write_next = low.op<amdgpu.v_xor_b32.lit>(%k32_a_write) {imm32 = 32768} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %k32_b_write_next = low.op<amdgpu.v_xor_b32.lit>(%k32_b_write) {imm32 = 32768} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  %k32_base_next = low.op<amdgpu.s_add_u32.rhs_inline>(%k_base) {imm32 = 32} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
"""
    branch = re.sub(r"^  low\.br \^_bb1\(%[A-Za-z0-9_]+:",
                    "  low.br ^_bb1(%k32_base_next:", branch)
    branch = append_arguments(
        branch,
        "%k32_lhs_read_next: reg<amdgpu.vgpr>, "
        "%k32_rhs_read_next: reg<amdgpu.vgpr>, "
        "%k32_a_write_next: reg<amdgpu.vgpr>, "
        "%k32_b_write_next: reg<amdgpu.vgpr>",
        ")",
    )
    suffix = stage_update + branch + suffix[back_end:]
    source = prefix + suffix

    args.output.write_text(source)


if __name__ == "__main__":
    main()
