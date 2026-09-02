#!/usr/bin/env python3
"""Recompute gfx11 wave coordinates after the K loop from the saved TID."""

from __future__ import annotations

import argparse
from pathlib import Path


ANCHOR = (
    "  %1191 = low.op<amdgpu.v_add_u32>(%1187, %1190) : "
    "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
)
INSERTION = (
    "  %epilogue_wave_m_id = "
    "low.op<amdgpu.v_and_b32.src0_inline>(%1189) {imm32 = 1} : "
    "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
    "  %epilogue_wave_n_id = "
    "low.op<amdgpu.v_lshrrev_b32.lit>(%1189) {imm32 = 1} : "
    "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.read_text()
    if source.count(ANCHOR) != 1:
        raise ValueError("expected one epilogue coordinate anchor")
    epilogue_start = source.index(ANCHOR) + len(ANCHOR)
    source = source[:epilogue_start] + INSERTION + source[epilogue_start:]
    prefix = source[:epilogue_start + len(INSERTION)]
    suffix = source[epilogue_start + len(INSERTION):]
    if suffix.count("%wave_m_id") != 1 or suffix.count("%wave_n_id") != 1:
        raise ValueError("unexpected epilogue wave-id use count")
    suffix = suffix.replace("%wave_m_id", "%epilogue_wave_m_id")
    suffix = suffix.replace("%wave_n_id", "%epilogue_wave_n_id")
    args.output.write_text(prefix + suffix)


if __name__ == "__main__":
    main()
