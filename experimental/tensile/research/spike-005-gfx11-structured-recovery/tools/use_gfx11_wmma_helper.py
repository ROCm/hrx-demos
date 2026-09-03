#!/usr/bin/env python3
"""Replace High gfx11 FP16 MMA operations with a register-only Low helper.

This isolates the effect of a schedule-free or schedule-locked microkernel
boundary. All global and LDS memory operations remain in High Loom.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TARGET = "amdgpu.target<gfx11-generic> @gfx11 {subgroup_size = 32}\n"
MMA_PATTERN = re.compile(
    r"(?P<indent>\s*)(?P<result>%[A-Za-z0-9_]+) = vector\.mma "
    r"(?P<lhs>%[A-Za-z0-9_]+), (?P<rhs>%[A-Za-z0-9_]+), "
    r"(?P<acc>%[A-Za-z0-9_]+) : vector<16xf16>, vector<16xf16>, "
    r"vector<8xf32>"
)


def helper(name: str, schedule: str) -> str:
    schedule_attr = " schedule(locked)" if schedule == "locked" else ""
    return f"""

// Register-only gfx11 WMMA microkernel. The schedule-free form measures the
// ordinary lowering boundary; schedule(locked) is an experimental oracle for
// whether preserving each authored WMMA position recovers LDS/WMMA overlap.
low.func.def{schedule_attr} target<amdgpu.gfx11.generic.core>(@gfx11) @{name}(%lhs: reg<amdgpu.vgpr x8>, %rhs: reg<amdgpu.vgpr x8>, %acc: reg<amdgpu.vgpr x8>) -> (reg<amdgpu.vgpr x8>) asm {{
  %result = v_wmma_f32_16x16x16_f16 %lhs, %rhs, %acc
  return %result
}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", choices=("free", "locked"), required=True)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=12,
        help="expected number of FP16 WMMA operations in the input",
    )
    parser.add_argument(
        "--locked-indices",
        default="",
        help="comma-separated zero-based WMMA ordinals to lock; overrides the "
        "uniform --schedule choice for those calls",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    if source.count(TARGET) != 1:
        raise ValueError("expected exactly one gfx11 target declaration")
    locked_indices = {
        int(value) for value in args.locked_indices.split(",") if value
    }
    if any(index < 0 or index >= args.expected_count for index in locked_indices):
        raise ValueError(
            f"locked WMMA indices must be in [0, {args.expected_count - 1}]"
        )
    use_mixed_helpers = bool(locked_indices) and args.schedule == "free"
    helper_source = helper("gfx11_wmma_f16_f32", args.schedule)
    if use_mixed_helpers:
        helper_source += helper("gfx11_wmma_f16_f32_locked", "locked")
    source = source.replace(TARGET, TARGET + helper_source, 1)

    mma_ordinal = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal mma_ordinal
        groups = match.groupdict()
        helper_name = (
            "gfx11_wmma_f16_f32_locked"
            if mma_ordinal in locked_indices
            else "gfx11_wmma_f16_f32"
        )
        mma_ordinal += 1
        return (
            f"{groups['indent']}{groups['result']} = low.invoke "
            f"@{helper_name}({groups['lhs']}, {groups['rhs']}, "
            f"{groups['acc']}) : (vector<16xf16>, vector<16xf16>, "
            "vector<8xf32>) -> (vector<8xf32>)"
        )

    source, count = MMA_PATTERN.subn(replace, source)
    if count != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} FP16 WMMA operations, found {count}"
        )
    args.output.write_text(source)


if __name__ == "__main__":
    main()
