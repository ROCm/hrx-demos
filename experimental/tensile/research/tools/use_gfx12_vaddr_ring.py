#!/usr/bin/env python3
"""Keep GEMM scalar base addresses stable and advance the PGR2 ring in VGPRs.

RDNA VMEM scalar address operands retain a source-read lease until the load
counter advances. Rebuilding the scalar address pair at the next iteration
therefore forces a loadcnt drain. Tensile keeps scalar bases stable and advances
vector offsets. This transformer carries two vector deltas, derives the eight
packet addresses from them, and advances only those deltas on the backedge.
"""

from __future__ import annotations

import argparse
from pathlib import Path


A_VADDRS = ("1621", "1655", "1688", "1721")
B_VADDRS = ("1637", "1671", "1704", "1737")


def append_branch_args(line: str, args: str) -> str:
    for marker in (")\n", "):\n"):
        if line.endswith(marker):
            return line[: -len(marker)] + args + marker
    raise RuntimeError("unexpected branch spelling")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines(keepends=True)
    output: list[str] = []
    in_loop_body = False
    skipping_scalar_ring = False
    inserted_vaddrs = False
    replaced_loads = 0

    for line in lines:
        if line.startswith("^_bb2:"):
            in_loop_body = True
        elif in_loop_body and line.startswith("^_bb3:"):
            in_loop_body = False

        if line.startswith("  low.br ^_bb1(") and "%p_init_av0" in line:
            line = append_branch_args(
                line,
                ", %1606: reg<amdgpu.vgpr>, %1606: reg<amdgpu.vgpr>",
            )
        elif line.startswith("^_bb1("):
            line = append_branch_args(
                line,
                ", %ring_a_delta: reg<amdgpu.vgpr>, %ring_b_delta: reg<amdgpu.vgpr>",
            )

        if in_loop_body and "%ring_a_k =" in line:
            skipping_scalar_ring = True
            indent = "  "
            for index, value in enumerate(A_VADDRS):
                output.append(
                    f"{indent}%ring_av{index} = low.op<amdgpu.v_add_u32>(%{value}, %ring_a_delta) : "
                    "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
                )
            for index, value in enumerate(B_VADDRS):
                output.append(
                    f"{indent}%ring_bv{index} = low.op<amdgpu.v_add_u32>(%{value}, %ring_b_delta) : "
                    "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
                )
            inserted_vaddrs = True
        if skipping_scalar_ring:
            if "%ring_b = low.concat" in line:
                skipping_scalar_ring = False
            continue

        if in_loop_body and "low.op<amdgpu.global_load_b128_saddr>" in line:
            for index, value in enumerate(A_VADDRS):
                old = f"(%{value}, %ring_a)"
                if old in line:
                    line = line.replace(old, f"(%ring_av{index}, %1628)")
                    replaced_loads += 1
            for index, value in enumerate(B_VADDRS):
                old = f"(%{value}, %ring_b)"
                if old in line:
                    line = line.replace(old, f"(%ring_bv{index}, %1644)")
                    replaced_loads += 1

        if in_loop_body and line.startswith("  low.br ^_bb1(") and "%p_next_av0" in line:
            output.append(
                "  %ring_a_delta_next = low.op<amdgpu.v_add_u32.lit>(%ring_a_delta) "
                "{imm32 = 131072} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            )
            output.append(
                "  %ring_b_delta_next = low.op<amdgpu.v_add_u32.lit>(%ring_b_delta) "
                "{imm32 = 128} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
            )
            line = append_branch_args(
                line,
                ", %ring_a_delta_next: reg<amdgpu.vgpr>, %ring_b_delta_next: reg<amdgpu.vgpr>",
            )

        output.append(line)

    if not inserted_vaddrs or replaced_loads != 16:
        raise RuntimeError(
            f"expected vector ring and 16 loads, got ring={inserted_vaddrs} loads={replaced_loads}"
        )
    args.output.write_text("".join(output))
    print("replaced dynamic scalar addresses with a two-VGPR offset ring")


if __name__ == "__main__":
    main()
