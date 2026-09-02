#!/usr/bin/env python3
"""Mark a prepared Low kernel fixed/locked to probe authored allocation support."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.input.read_text()
    marker = "low.kernel.def retain "
    if text.count(marker) != 1:
        raise ValueError("expected one retained Low kernel")
    args.output.write_text(
        text.replace(
            marker,
            "low.kernel.def retain allocation(fixed) schedule(locked) ",
            1,
        )
    )


if __name__ == "__main__":
    main()
