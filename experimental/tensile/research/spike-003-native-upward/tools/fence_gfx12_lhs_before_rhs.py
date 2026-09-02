#!/usr/bin/env python3
"""Force each K16 band's LHS LDS reads to issue before its RHS reads."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.read_text()
    pattern = re.compile(
        r"(?=  %(?:even|odd|tail30|tail31)_compute_q[04] = "
        r"low\.op<amdgpu\.ds_read_b128>)"
    )
    source, count = pattern.subn("  low.schedule.fence\n", source)
    if count != 8:
        raise ValueError(f"expected eight K16 RHS bands, fenced {count}")
    args.output.write_text(source)


if __name__ == "__main__":
    main()
