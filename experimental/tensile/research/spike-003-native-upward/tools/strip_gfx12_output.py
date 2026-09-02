#!/usr/bin/env python3
"""Remove the fixed motif's output epilogue for timing attribution only."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.input.read_text()
    markers = ("  %native_d_wave_n = ", "  %native_d_base = ")
    starts = [text.find(marker) for marker in markers]
    start = min(position for position in starts if position >= 0)
    stop = text.index("  low.return", start)
    removed = text[start:stop]
    store_count = removed.count("low.op<amdgpu.global_store_b16_saddr>")
    if store_count != 128:
        raise ValueError(f"expected 128 output stores, found {store_count}")
    args.output.write_text(text[:start] + text[stop:])
    print("removed 128 scalar output stores; result is timing-only")


if __name__ == "__main__":
    main()
