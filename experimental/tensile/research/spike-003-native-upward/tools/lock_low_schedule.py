#!/usr/bin/env python3
"""Add a locked source-schedule contract to one prepared-Low kernel."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.read_text()
    old = "low.kernel.def retain target<"
    if source.count(old) != 1:
        raise ValueError(f"expected one unlocked retained kernel, found {source.count(old)}")
    args.output.write_text(source.replace(old, "low.kernel.def retain schedule(locked) target<", 1))


if __name__ == "__main__":
    main()
