#!/usr/bin/env python3
"""Retarget the terminal gfx11 FP16 Low motif to BF16 inputs/output.

The gfx11 FP16 and BF16 WMMA instructions have the same physical operand and
accumulator widths. gfx11 has no V_CVT_PK_BF16_F32, so the scalar epilogue uses
the standard integer round-to-nearest-even sequence before storing the low
16-bit result. This keeps the proven wave-coalesced address map unchanged.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CONVERSION = re.compile(
    r"(?P<prefix>\s+%(?P<result>[A-Za-z0-9_]+)_f16 = )"
    r"low\.op<amdgpu\.v_cvt_f16_f32>\(%(?P<input>[A-Za-z0-9_]+)\) "
    r": \(reg<amdgpu\.vgpr>\) -> reg<amdgpu\.vgpr>"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    source, wmma_count = re.subn(
        r"amdgpu\.v_wmma_f32_16x16x16_f16",
        "amdgpu.v_wmma_f32_16x16x16_bf16",
        source,
    )

    def convert(match: re.Match[str]) -> str:
        result = match.group("result")
        input_value = match.group("input")
        return (
            f"\n  %{result}_lsb = low.op<amdgpu.v_bfe_u32.offset_width_inline>"
            f"(%{input_value}) {{offset = 16, width = 1}} : "
            "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            f"  %{result}_bias = low.op<amdgpu.v_add_u32.lit>(%{result}_lsb) "
            "{imm32 = 32767} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            f"  %{result}_rounded = low.op<amdgpu.v_add_u32>"
            f"(%{input_value}, %{result}_bias) : "
            "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            f"  %{result}_f16 = low.op<amdgpu.v_lshrrev_b32.src0_inline>"
            f"(%{result}_rounded) {{imm32 = 16}} : "
            "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>"
        )

    source, conversion_count = CONVERSION.subn(convert, source)
    source = source.replace(
        "gemm_f16_f32_mt64x96x32_gfx11_pgr2",
        "gemm_bf16_f32_mt64x96x32_gfx11_pgr2",
    )

    if wmma_count != 36:
        raise ValueError(f"expected 36 WMMA sites, retargeted {wmma_count}")
    if conversion_count != 48:
        raise ValueError(
            f"expected 48 scalar output conversions, retargeted {conversion_count}"
        )
    args.output.write_text(source)


if __name__ == "__main__":
    main()
