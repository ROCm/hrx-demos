#!/usr/bin/env python3
"""Map a flat 160-workgroup launch to the gfx11 10x16 tile grid."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines()
    x = next(i for i, line in enumerate(lines) if "%workgroup_n = low.live_in<amdgpu.workgroup_id.x>" in line)
    y = next(i for i, line in enumerate(lines) if "%workgroup_m = low.live_in<amdgpu.workgroup_id.y>" in line)
    lines[x] = lines[x].replace("%workgroup_n", "%linear_workgroup")
    lines[y] = lines[y].replace("%workgroup_m", "%unused_workgroup_y")
    mapping = [
        "  %wg_div10_magic = low.const<amdgpu.s_mov_b32> {imm32 = 3435973837} : reg<amdgpu.sgpr>",
        "  %wg_div10_hi = low.op<amdgpu.s_mul_hi_u32>(%linear_workgroup, %wg_div10_magic) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %workgroup_m = low.op<amdgpu.s_lshr_b32.rhs_inline>(%wg_div10_hi) {imm32 = 3} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %wg_ten = low.const<amdgpu.s_mov_b32> {imm32 = 10} : reg<amdgpu.sgpr>",
        "  %wg_m_times_ten = low.op<amdgpu.s_mul_i32>(%workgroup_m, %wg_ten) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
        "  %workgroup_n = low.op<amdgpu.s_sub_u32>(%linear_workgroup, %wg_m_times_ten) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
    ]
    insertion = max(i for i, line in enumerate(lines) if " = low.live_in<" in line) + 1
    lines[insertion:insertion] = mapping
    # Authored launch facts must agree with the flattened launch.
    lines[0] = lines[0].replace("workgroup_count(10, 16, 1)", "workgroup_count(160, 1, 1)")
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
