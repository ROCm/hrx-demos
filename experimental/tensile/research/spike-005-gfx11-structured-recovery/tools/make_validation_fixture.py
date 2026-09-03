#!/usr/bin/env python3
"""Derive a nonuniform correctness fixture from the maintained gfx11 motif."""

from argparse import ArgumentParser
from pathlib import Path


CHECK_MARKER = "// Uniform inputs exercise every production address"


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    kernel, marker, _ = source.partition(CHECK_MARKER)
    if not marker:
        raise ValueError(f"check marker not found in {args.input}")

    checks = r'''// The buffers use the physical matrix ordering consumed by the kernel:
// A is KxM, B is NxK, and D is NxM. The CPU oracle therefore computes B*A,
// which is the physical transpose of the logical column-major D=A*B result.
// Distinct prime periods ensure both output axes and K positions vary.
check.case public @gemm_f16_f32_mt64x96x32_gfx11_nonuniform_case {
  %a = check.generate.iota offset(-0.34375) step(0.03125) period(23) : tensor<1024x1024xf16>
  %b = check.generate.iota offset(-0.28125) step(0.03125) period(19) : tensor<960x1024xf16>
  %c = check.generate.fill value(0.0) : tensor<960x1024xf16>
  %d = check.generate.fill value(-7.0) : tensor<960x1024xf16>
  kernel.launch @gemm_f16_f32_mt64x96x32_gfx11_pgr2(%a, %b, %c, %d) : (tensor<1024x1024xf16>, tensor<960x1024xf16>, tensor<960x1024xf16>, tensor<960x1024xf16>)
  %expected = check.oracle.call<reference.matmul> {accumulator = "f32", lhs = "f16", result = "f16", rhs = "f16"} callee(@gemm_f16_f32_mt64x96x32_gfx11_pgr2) inputs(%b, %a, %c) : (tensor<960x1024xf16>, tensor<1024x1024xf16>, tensor<960x1024xf16>) -> (tensor<960x1024xf16>)
  check.expect.close actual(%d) expected(%expected) atol(0.02) rtol(0.01) nan(different) : tensor<960x1024xf16>
  check.return
}

check.case public @gemm_f16_f32_mt64x96x32_gfx11_min_nonuniform_case {
  %a = check.generate.iota offset(-0.34375) step(0.03125) period(23) : tensor<64x64xf16>
  %b = check.generate.iota offset(-0.28125) step(0.03125) period(19) : tensor<96x64xf16>
  %c = check.generate.fill value(0.0) : tensor<96x64xf16>
  %d = check.generate.fill value(-7.0) : tensor<96x64xf16>
  kernel.launch @gemm_f16_f32_mt64x96x32_gfx11_pgr2(%a, %b, %c, %d) : (tensor<64x64xf16>, tensor<96x64xf16>, tensor<96x64xf16>, tensor<96x64xf16>)
  %expected = check.oracle.call<reference.matmul> {accumulator = "f32", lhs = "f16", result = "f16", rhs = "f16"} callee(@gemm_f16_f32_mt64x96x32_gfx11_pgr2) inputs(%b, %a, %c) : (tensor<96x64xf16>, tensor<64x64xf16>, tensor<96x64xf16>) -> (tensor<96x64xf16>)
  check.expect.close actual(%d) expected(%expected) atol(0.02) rtol(0.01) nan(different) : tensor<96x64xf16>
  check.return
}

// Exact row-address witness: A[k,m]=m/64 and B[n,k]=1/64, so D[n,m]=m/64.
// It makes a publication permutation immediately visible in the first error.
check.case public @gemm_f16_f32_mt64x96x32_gfx11_min_row_witness_case {
  %a = check.generate.iota offset(0.0) step(0.015625) period(64) : tensor<64x64xf16>
  %b = check.generate.fill value(0.015625) : tensor<96x64xf16>
  %c = check.generate.fill value(0.0) : tensor<96x64xf16>
  %d = check.generate.fill value(-7.0) : tensor<96x64xf16>
  %expected = check.generate.iota offset(0.0) step(0.015625) period(64) : tensor<96x64xf16>
  kernel.launch @gemm_f16_f32_mt64x96x32_gfx11_pgr2(%a, %b, %c, %d) : (tensor<64x64xf16>, tensor<96x64xf16>, tensor<96x64xf16>, tensor<96x64xf16>)
  check.expect.close actual(%d) expected(%expected) atol(0.0) rtol(0.0) nan(different) : tensor<96x64xf16>
  check.return
}
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(kernel + checks)


if __name__ == "__main__":
    main()
