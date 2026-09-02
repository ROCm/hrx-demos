#!/usr/bin/env python3
"""Rewrite the fixed gfx11 cross-iteration motif to use raw-buffer loads.

This intentionally preserves the existing scalar offset calculations. It only
removes the repeated 64-bit base additions/concats around each global load so
the address-mode delta can be measured independently from schedule changes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


LOAD_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[A-Za-z0-9_]+) = "
    r"low\.op<amdgpu\.global_load_b128_saddr>\("
    r"(?P<vaddr>%[A-Za-z0-9_]+), %[A-Za-z0-9_]+\) "
    r"\{offset = (?P<offset>[0-9]+)\} : "
    r"\(reg<amdgpu\.vgpr>, reg<amdgpu\.sgpr x2>\) -> "
    r"reg<amdgpu\.vgpr x4>$"
)

A_SOFFSETS = {
    "%pro_av0": "%591",
    "%pro_av1": "%622",
    "%odd_av0": "%746",
    "%odd_av1": "%784",
    "%next_av0": "%919",
    "%next_av1": "%948",
}

B_SOFFSETS = {
    "%pro_bv0": "%606",
    "%pro_bv1": "%606",
    "%pro_bv2": "%606",
    "%odd_bv0": "%763",
    "%odd_bv1": "%801",
    "%odd_bv2": "%819",
    "%next_bv0": "%934",
    "%next_bv1": "%963",
    "%next_bv2": "%978",
}


def rewrite(text: str) -> str:
    anchor = "  %zero = low.const<amdgpu.s_mov_b32> {imm32 = 0} : reg<amdgpu.sgpr>"
    setup = "\n".join(
        [
            anchor,
            "  %buffer_a_lo_raw = low.slice %1516[0] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>",
            "  %buffer_a_hi_raw = low.slice %1516[1] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>",
            "  %buffer_b_lo_raw = low.slice %1516[2] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>",
            "  %buffer_b_hi_raw = low.slice %1516[3] : reg<amdgpu.sgpr x4> -> reg<amdgpu.sgpr>",
            "  %buffer_hi_mask = low.const<amdgpu.s_mov_b32> {imm32 = 65535} : reg<amdgpu.sgpr>",
            "  %buffer_extent = low.const<amdgpu.s_mov_b32> {imm32 = 2097152} : reg<amdgpu.sgpr>",
            "  %buffer_flags = low.const<amdgpu.s_mov_b32> {imm32 = 822173696} : reg<amdgpu.sgpr>",
            "  %buffer_a_hi = low.op<amdgpu.s_and_b32>(%buffer_a_hi_raw, %buffer_hi_mask) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
            "  %buffer_a_ptr = low.concat(%buffer_a_lo_raw, %buffer_a_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>",
            "  %buffer_a = low.concat(%buffer_a_ptr, %buffer_extent, %buffer_flags) : (reg<amdgpu.sgpr x2>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x4>",
            "  %buffer_b_hi = low.op<amdgpu.s_and_b32>(%buffer_b_hi_raw, %buffer_hi_mask) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
            "  %buffer_b_ptr = low.concat(%buffer_b_lo_raw, %buffer_b_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>",
            "  %buffer_b = low.concat(%buffer_b_ptr, %buffer_extent, %buffer_flags) : (reg<amdgpu.sgpr x2>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x4>",
        ]
    )
    if text.count(anchor) != 1:
        raise ValueError("expected exactly one SRD insertion anchor")
    text = text.replace(anchor, setup)

    output: list[str] = []
    rewritten = 0
    for line in text.splitlines():
        match = LOAD_RE.match(line)
        if match is None:
            output.append(line)
            continue
        result = match.group("result")
        if result in A_SOFFSETS:
            buffer = "%buffer_a"
            soffset = A_SOFFSETS[result]
        elif result in B_SOFFSETS:
            buffer = "%buffer_b"
            soffset = B_SOFFSETS[result]
        else:
            raise ValueError(f"unclassified global load result: {result}")
        output.append(
            f"{match.group('indent')}{result} = low.op<amdgpu.buffer_load_b128>("
            f"{buffer}, {match.group('vaddr')}, {soffset}) "
            f"{{offset = {match.group('offset')}}} : "
            "(reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>) -> "
            "reg<amdgpu.vgpr x4>"
        )
        rewritten += 1

    if rewritten != 15:
        raise ValueError(f"expected 15 global loads, rewrote {rewritten}")
    return "\n".join(output) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(rewrite(args.input.read_text()))


if __name__ == "__main__":
    main()
