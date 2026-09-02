#!/usr/bin/env python3
"""Make the deliberately invalid gfx11 WMMA operand-swap negative control.

Matching the incumbent's printed ISA operand order this way transposes every
16x16 output microtile.  Keep this transformer only as evidence that mnemonic
and register-order congruence is not a semantic proof.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MMA = re.compile(r"(= vector\.mma )([^,]+), ([^,]+),")
LOW_MMA = re.compile(
    r"(= low\.op<amdgpu\.v_wmma_f32_16x16x16_f16>\()([^,]+), ([^,]+),"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="fail unless exactly this many WMMA sites are swapped",
    )
    args = parser.parse_args()
    source = args.input.read_text()
    source, high_count = MMA.subn(r"\1\3, \2,", source)
    source, low_count = LOW_MMA.subn(r"\1\3, \2,", source)
    count = high_count + low_count
    if count == 0:
        raise ValueError("no WMMA sites found")
    if args.expected_count is not None and count != args.expected_count:
        raise ValueError(f"expected {args.expected_count} WMMA sites, swapped {count}")
    args.output.write_text(source)


if __name__ == "__main__":
    main()
