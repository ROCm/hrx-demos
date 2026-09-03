#!/usr/bin/env python3
"""Append a correctness-gated benchmark for one exact interior GEMM shape."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0:
        parser.error("dimensions must be positive")
    expected = args.k / 1024.0
    suffix = f"{args.m}x{args.n}x{args.k}"
    fixture = f"""

// Generated Spike 006 interior-cell witness. Uniform nonzero operands make
// the tensor-major convention immaterial; the checked value is K / 1024.
check.case public @spike6_{suffix}_case {{
  %a = check.generate.fill value(0.03125) : tensor<{args.k}x{args.m}xf16>
  %b = check.generate.fill value(0.03125) : tensor<{args.n}x{args.k}xf16>
  %c = check.generate.fill value(0.0) : tensor<{args.m}x{args.n}xf16>
  %d = check.generate.fill value(0.0) : tensor<{args.m}x{args.n}xf16>
  %expected = check.generate.fill value({expected}) : tensor<{args.m}x{args.n}xf16>
  kernel.launch @{args.symbol}(%a, %b, %c, %d) : (tensor<{args.k}x{args.m}xf16>, tensor<{args.n}x{args.k}xf16>, tensor<{args.m}x{args.n}xf16>, tensor<{args.m}x{args.n}xf16>)
  check.expect.close actual(%d) expected(%expected) atol(0.0) rtol(0.0) nan(same) : tensor<{args.m}x{args.n}xf16>
  check.return
}}

check.benchmark<@spike6_{suffix}_case> @spike6_{suffix}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(args.source.read_text() + fixture)
    if args.output.stat().st_size == 0:
        raise RuntimeError("empty output")


if __name__ == "__main__":
    main()
