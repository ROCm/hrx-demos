#!/usr/bin/env python3
"""Remove only gfx11 output stores as an epilogue timing control.

The transformed kernel retains address generation, cross-lane permutation,
FP32-to-FP16 conversion, packing, and EXEC manipulation.  It intentionally
does not publish output and therefore is not a correctness candidate.
"""

from __future__ import annotations

import argparse
from pathlib import Path


STORE = "low.op<amdgpu.global_store_b32_saddr>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-count", type=int, default=48)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines(keepends=True)
    kept = [line for line in lines if STORE not in line]
    removed = len(lines) - len(kept)
    if removed != args.expected_count:
        raise ValueError(f"expected {args.expected_count} stores, removed {removed}")
    args.output.write_text("".join(kept))


if __name__ == "__main__":
    main()
