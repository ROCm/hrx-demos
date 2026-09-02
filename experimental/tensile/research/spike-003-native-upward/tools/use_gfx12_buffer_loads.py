#!/usr/bin/env python3
"""Rewrite the fixed gfx12 GEMM Low motif to use raw-buffer loads.

This is intentionally a narrow experiment, not a general Low rewriter.  The
input motif has one stable A and B scalar base and four fixed K-stage offsets.
The native TensileLite kernel uses BUFFER_LOAD_B128 through two SRDs, whereas
the raised Loom motif uses GLOBAL_LOAD_B128_SADDR.  Buffer immediates are only
12 bits, so the large A stage offsets are supplied through a stable scalar
soffset; B stage offsets remain instruction immediates.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


A_BASES = {"%1628", "%1662", "%1695", "%1728"}
B_BASES = {"%1644", "%1678", "%1711", "%1744"}
A_SOFFSETS = {
    0: "%zero",
    65536: "%buffer_a_k32",
    131072: "%buffer_a_k64",
    196608: "%buffer_a_k96",
}

LOAD_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[A-Za-z0-9_]+) = "
    r"low\.op<amdgpu\.global_load_b128_saddr>\("
    r"(?P<vaddr>%[A-Za-z0-9_]+), (?P<saddr>%[A-Za-z0-9_]+)\) "
    r"\{offset = (?P<offset>[0-9]+)\} : "
    r"\(reg<amdgpu\.vgpr>, reg<amdgpu\.sgpr x2>\) -> "
    r"reg<amdgpu\.vgpr x4>$"
)


def rewrite(text: str) -> str:
    a_anchor = (
        "  %1628 = low.concat(%1626, %1627) : "
        "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>"
    )
    a_setup = "\n".join(
        [
            a_anchor,
            "  %buffer_hi_mask = low.const<amdgpu.s_mov_b32> {imm32 = 65535} : reg<amdgpu.sgpr>",
            "  %buffer_extent = low.const<amdgpu.s_mov_b32> {imm32 = 2097152} : reg<amdgpu.sgpr>",
            "  %buffer_flags = low.const<amdgpu.s_mov_b32> {imm32 = 822173696} : reg<amdgpu.sgpr>",
            "  %buffer_a_k32 = low.const<amdgpu.s_mov_b32> {imm32 = 65536} : reg<amdgpu.sgpr>",
            "  %buffer_a_k64 = low.const<amdgpu.s_mov_b32> {imm32 = 131072} : reg<amdgpu.sgpr>",
            "  %buffer_a_k96 = low.const<amdgpu.s_mov_b32> {imm32 = 196608} : reg<amdgpu.sgpr>",
            "  %buffer_a_hi = low.op<amdgpu.s_and_b32>(%1627, %buffer_hi_mask) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
            "  %buffer_a_ptr = low.concat(%1626, %buffer_a_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>",
            "  %buffer_a = low.concat(%buffer_a_ptr, %buffer_extent, %buffer_flags) : (reg<amdgpu.sgpr x2>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x4>",
        ]
    )
    if text.count(a_anchor) != 1:
        raise ValueError("expected exactly one A-base anchor")
    text = text.replace(a_anchor, a_setup)

    b_anchor = (
        "  %1644 = low.concat(%1642, %1643) : "
        "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>"
    )
    b_setup = "\n".join(
        [
            b_anchor,
            "  %buffer_b_hi = low.op<amdgpu.s_and_b32>(%1643, %buffer_hi_mask) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
            "  %buffer_b_ptr = low.concat(%1642, %buffer_b_hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>",
            "  %buffer_b = low.concat(%buffer_b_ptr, %buffer_extent, %buffer_flags) : (reg<amdgpu.sgpr x2>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x4>",
        ]
    )
    if text.count(b_anchor) != 1:
        raise ValueError("expected exactly one B-base anchor")
    text = text.replace(b_anchor, b_setup)

    rewritten = 0
    output: list[str] = []
    for line in text.splitlines():
        match = LOAD_RE.match(line)
        if match is None:
            output.append(line)
            continue
        saddr = match.group("saddr")
        offset = int(match.group("offset"))
        result = match.group("result")
        vaddr = match.group("vaddr")
        indent = match.group("indent")
        if saddr in A_BASES:
            try:
                soffset = A_SOFFSETS[offset]
            except KeyError as exc:
                raise ValueError(f"unexpected A offset: {offset}") from exc
            line = (
                f"{indent}{result} = low.op<amdgpu.buffer_load_b128>("
                f"%buffer_a, {vaddr}, {soffset}) {{offset = 0}} : "
                "(reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>) "
                "-> reg<amdgpu.vgpr x4>"
            )
        elif saddr in B_BASES:
            if offset > 4095:
                raise ValueError(f"B offset does not fit buffer immediate: {offset}")
            line = (
                f"{indent}{result} = "
                f"low.op<amdgpu.buffer_load_b128_vaddr_offset>(%buffer_b, {vaddr}) "
                f"{{offset = {offset}}} : "
                "(reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>) -> "
                "reg<amdgpu.vgpr x4>"
            )
        else:
            raise ValueError(f"unknown scalar base in load: {saddr}")
        output.append(line)
        rewritten += 1

    if rewritten != 32:
        raise ValueError(f"expected 32 loads, rewrote {rewritten}")
    return "\n".join(output) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(rewrite(args.input.read_text()))


if __name__ == "__main__":
    main()
