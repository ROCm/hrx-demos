#!/usr/bin/env python3
"""Remove the gfx11 epilogue to isolate main-loop runtime.

The result intentionally does not publish output and is only a timing/ATT
control.  It replaces the final tail-to-epilogue branch with ``low.return`` and
removes the epilogue block while preserving the target declaration.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EPILOGUE_LABEL = "\n^_bb3("
TARGET_MARKER = "\namdgpu.target<"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    label = source.find(EPILOGUE_LABEL)
    if label < 0:
        raise ValueError(f"missing epilogue label {EPILOGUE_LABEL!r}")
    branch_start = source.rfind("  low.br ^_bb3(", 0, label)
    if branch_start < 0:
        raise ValueError("missing final branch to epilogue")
    target = source.find(TARGET_MARKER, label)
    if target < 0:
        raise ValueError("missing target declaration after epilogue")

    transformed = source[:branch_start] + "  low.return\n}\n" + source[target:]
    if transformed.count("low.return") != 1:
        raise ValueError("expected exactly one return in transformed kernel")
    args.output.write_text(transformed)


if __name__ == "__main__":
    main()
