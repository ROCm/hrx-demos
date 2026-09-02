#!/usr/bin/env python3
"""Convert the two-tile gfx12 Low loop from PGR1 to a PGR2 load ring."""

from __future__ import annotations

import argparse
from pathlib import Path


VECTOR_OFFSETS = (
    ("1621", "1637"),
    ("1655", "1671"),
    ("1688", "1704"),
    ("1721", "1737"),
)


RING_ADDR = """\
  %ring_a_k = low.op<amdgpu.s_lshl_b32.rhs_inline>(%k_base) {imm32 = 11} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %ring_a_lo = low.op<amdgpu.s_add_u32>(%1626, %ring_a_k) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %ring_a_hi = low.op<amdgpu.s_addc_u32>(%1627, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %ring_a = low.concat(%ring_a_lo, %ring_a_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>
  %ring_b_k = low.op<amdgpu.s_lshl_b32.rhs_inline>(%k_base) {imm32 = 1} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %ring_b_lo = low.op<amdgpu.s_add_u32>(%1642, %ring_b_k) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %ring_b_hi = low.op<amdgpu.s_addc_u32>(%1643, %zero) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %ring_b = low.concat(%ring_b_lo, %ring_b_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>
"""


def load_line(name: str, vector: str, saddr: str, offset: int) -> str:
    return (
        f"  %{name} = low.op<amdgpu.global_load_b128_saddr>(%{vector}, %{saddr}) "
        f"{{offset = {offset}}} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> "
        "reg<amdgpu.vgpr x4>"
    )


def replace_block_prefix(text: str, block: str, stop_marker: str, replacement: str) -> str:
    start = text.index(block) + len(block)
    stop = text.index(stop_marker, start)
    return text[:start] + replacement + text[stop:]


def append_branch_args(line: str, args: list[str], terminator: str) -> str:
    if not line.endswith(terminator):
        raise ValueError(f"unexpected branch/header terminator: {line[-20:]}")
    return line[: -len(terminator)] + ", " + ", ".join(args) + terminator


def replace_line(text: str, marker: str, replacement: str) -> str:
    start = text.index(marker)
    line_start = text.rfind("\n", 0, start) + 1
    line_stop = text.index("\n", start)
    return text[:line_start] + replacement + text[line_stop:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.input.read_text()

    # Seed the carried bank with tile 1. Tile 0 is already materialized in LDS
    # by the original prologue.
    init_lines: list[str] = []
    for index, (a_vector, b_vector) in enumerate(VECTOR_OFFSETS):
        init_lines.append(load_line(f"p_init_av{index}", a_vector, "1628", 65536))
        init_lines.append(load_line(f"p_init_bv{index}", b_vector, "1644", 64))
    init = "\n".join(init_lines) + "\n"
    first_branch_at = text.index("  low.br ^_bb1(")
    text = text[:first_branch_at] + init + text[first_branch_at:]

    carried_init = [
        f"%p_init_{kind}{index}: reg<amdgpu.vgpr x4>"
        for index in range(4)
        for kind in ("av", "bv")
    ]
    carried_args = [
        f"%p_{kind}{index}: reg<amdgpu.vgpr x4>"
        for index in range(4)
        for kind in ("av", "bv")
    ]

    first_branch_stop = text.index("\n", text.index("  low.br ^_bb1("))
    line = text[text.rfind("\n", 0, first_branch_stop) + 1 : first_branch_stop]
    text = text[: text.rfind("\n", 0, first_branch_stop) + 1] + append_branch_args(
        line, carried_init, ")"
    ) + text[first_branch_stop:]

    header_start = text.index("\n^_bb1(") + 1
    header_stop = text.index("\n", header_start)
    header = text[header_start:header_stop]
    text = text[:header_start] + append_branch_args(header, carried_args, "):") + text[header_stop:]

    # The carried tile-1 payload replaces the old in-loop odd load band.
    text = replace_block_prefix(
        text,
        "^_bb2:\n",
        "  %even_compute_lhs_addr = ",
        RING_ADDR,
    )
    for index in range(4):
        text = text.replace(f"%odd_load_av{index}", f"%p_av{index}")
        text = text.replace(f"%odd_load_bv{index}", f"%p_bv{index}")

    # Delete the old contiguous tile-2 load band. Reinsert each request after
    # the matching tile-1 LDS store, which frees the carried register bank.
    even_load_start = text.index("  %even_load_a_k = ")
    even_load_stop = text.index("\n", text.index("  %even_load_bv3 = ", even_load_start)) + 1
    text = text[:even_load_start] + text[even_load_stop:]
    for index, (a_vector, b_vector) in enumerate(VECTOR_OFFSETS):
        a_store = f"  low.op<amdgpu.ds_write_b128>(%2058, %p_av{index})"
        a_load = load_line(f"even_load_av{index}", a_vector, "ring_a", 131072)
        text = replace_line(text, a_store, text[text.rfind("\n", 0, text.index(a_store)) + 1 : text.index("\n", text.index(a_store))] + "\n" + a_load)
        b_store = f"  low.op<amdgpu.ds_write_b128>(%2058, %p_bv{index})"
        b_load = load_line(f"even_load_bv{index}", b_vector, "ring_b", 128)
        text = replace_line(text, b_store, text[text.rfind("\n", 0, text.index(b_store)) + 1 : text.index("\n", text.index(b_store))] + "\n" + b_load)

    # After tile 2 has been consumed, store it to stage 0 and issue tile 3
    # into the same logical ring slots for the next pair iteration.
    for index, (a_vector, b_vector) in enumerate(VECTOR_OFFSETS):
        a_store = f"  low.op<amdgpu.ds_write_b128>(%2373, %even_load_av{index})"
        a_load = load_line(f"p_next_av{index}", a_vector, "ring_a", 196608)
        text = replace_line(text, a_store, text[text.rfind("\n", 0, text.index(a_store)) + 1 : text.index("\n", text.index(a_store))] + "\n" + a_load)
        b_store = f"  low.op<amdgpu.ds_write_b128>(%2373, %even_load_bv{index})"
        b_load = load_line(f"p_next_bv{index}", b_vector, "ring_b", 192)
        text = replace_line(text, b_store, text[text.rfind("\n", 0, text.index(b_store)) + 1 : text.index("\n", text.index(b_store))] + "\n" + b_load)

    back_branch_at = text.index("  low.br ^_bb1(", text.index("^_bb2:\n"))
    back_branch_stop = text.index("\n", back_branch_at)
    back_line = text[back_branch_at:back_branch_stop]
    carried_next = [
        f"%p_next_{kind}{index}: reg<amdgpu.vgpr x4>"
        for index in range(4)
        for kind in ("av", "bv")
    ]
    text = text[:back_branch_at] + append_branch_args(back_line, carried_next, ")") + text[back_branch_stop:]

    # The exit carries tile 31. Remove the redundant fixed tail load band and
    # use those values when materializing stage 1.
    text = replace_block_prefix(
        text,
        "^_bb3:\n",
        "  %tail30_compute_lhs_addr = ",
        "",
    )
    for index in range(4):
        text = text.replace(f"%tail_load_av{index}", f"%p_av{index}")
        text = text.replace(f"%tail_load_bv{index}", f"%p_bv{index}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise ValueError("output was not created")


if __name__ == "__main__":
    main()
