#!/usr/bin/env python3
"""Wrap prepared Low as an updateable loom-check allocation fixture."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--function", required=True)
    parser.add_argument("--diagnostics", default="none")
    parser.add_argument(
        "--fixed",
        action="append",
        default=[],
        metavar="NAME:KIND:BASE:COUNT",
        help="Append a loom-check fixed allocation request.",
    )
    args = parser.parse_args()
    fixed = "".join(f" fixed=%{spec}" for spec in args.fixed)
    directive = (
        f"// RUN: emit low-allocation-json @{args.function} "
        f"diagnostics={args.diagnostics} output=json{fixed}\n"
    )
    args.output.write_text(directive + args.input.read_text())


if __name__ == "__main__":
    main()
