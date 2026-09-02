#!/usr/bin/env python3
"""Reuse only invariant gfx11 LDS store addresses in prepared Low.

This is deliberately narrower than ``hoist_gfx11_cross_iteration_addresses``:
the latter also lengthens global-load and LDS-read address live ranges and is a
known spill regression.  Tensile keeps the five padded-layout store addresses
live and changes the LDS stage with instruction offsets, so this transform
isolates that one schedule property.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPLACEMENTS = {
    "%899": "%664",
    "%903": "%668",
    "%905": "%670",
    "%907": "%672",
    "%909": "%674",
    "%1113": "%664",
    "%1117": "%668",
    "%1119": "%670",
    "%1121": "%672",
    "%1123": "%674",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.read_text()
    for old, new in REPLACEMENTS.items():
        definition = source.find(old)
        if definition < 0:
            raise ValueError(f"missing prepared-Low value {old}")
        split = definition + len(old)
        source = source[:split] + source[split:].replace(old, new)
    args.output.write_text(source)


if __name__ == "__main__":
    main()
