#!/usr/bin/env python3
"""Prepare the retained gfx11 High source for exact-device runner evidence."""

from __future__ import annotations

import argparse
from pathlib import Path


FIXTURE = r'''

// Uniform inputs exercise every production address without depending on the
// host tensor-major convention. 1024 * (1/32) * (1/32) = 1 exactly.
check.case public @gemm_f16_f32_mt64x96x32_gfx11_pgr2_case {
  %a = check.generate.fill value(0.03125) : tensor<1024x1024xf16>
  %b = check.generate.fill value(0.03125) : tensor<960x1024xf16>
  %c = check.generate.fill value(0.0) : tensor<1024x960xf16>
  %d = check.generate.fill value(0.0) : tensor<1024x960xf16>
  %expected = check.generate.fill value(1.0) : tensor<1024x960xf16>
  kernel.launch @gemm_f16_f32_mt64x96x32_gfx11_pgr2(%a, %b, %c, %d) : (tensor<1024x1024xf16>, tensor<960x1024xf16>, tensor<1024x960xf16>, tensor<1024x960xf16>)
  check.expect.close actual(%d) expected(%expected) atol(0.0) rtol(0.0) nan(same) : tensor<1024x960xf16>
  check.return
}

check.benchmark<@gemm_f16_f32_mt64x96x32_gfx11_pgr2_case> @gemm_f16_f32_mt64x96x32_gfx11_pgr2_1024x960x1024
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    old = "amdgpu.target<gfx11-generic> @gfx11 {subgroup_size = 32}"
    new = "amdgpu.target<gfx1100> @gfx11 {subgroup_size = 32}"
    if source.count(old) != 1:
        raise ValueError(f"expected exactly one generic target declaration: {old}")
    if "check.case" in source:
        raise ValueError("input unexpectedly already contains a check fixture")
    transformed = source.replace(old, new) + FIXTURE
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(transformed)
    if not args.output.exists() or args.output.stat().st_size == 0:
        raise RuntimeError(f"failed to create {args.output}")


if __name__ == "__main__":
    main()
