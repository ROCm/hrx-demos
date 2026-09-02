#!/usr/bin/env python3
"""Replace Loom's generic gfx11 buffer descriptor flags with solution 1675's."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    text = args.input.read_text()
    generic = "822173696"  # 0x31016000
    native = "822099968"   # 0x31004000
    if text.count(generic) != 1:
        raise ValueError("expected exactly one generic buffer flag constant")
    args.output.write_text(text.replace(generic, native))


if __name__ == "__main__":
    main()
