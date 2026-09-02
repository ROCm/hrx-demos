#!/usr/bin/env python3
"""Reuse the two invariant gfx11 LDS read bases across both PLR stages.

The bank is already represented by each LDS operation's immediate offset.
Recomputing the lane-local base before every half-iteration only extends the
wave-id lifetimes and differs from the TensileLite witness, which retains one
LHS and one RHS vector address. Definitions are left for ``low-dce`` so the
rewrite remains mechanically auditable.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPLACEMENTS = {
    # LHS: bank selection is in the 0/16384 immediate range.
    "%990": "%684",
    "%1128": "%684",
    # RHS: bank selection is in the 4224/20608 immediate range.
    "%1032": "%722",
    "%1170": "%722",
}


def replace_uses_after_definition(source: str, old: str, new: str) -> str:
    definition = source.find(old)
    if definition < 0:
        raise ValueError(f"missing expected prepared-Low value {old}")
    split = definition + len(old)
    return source[:split] + source[split:].replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.read_text()
    for old, new in REPLACEMENTS.items():
        source = replace_uses_after_definition(source, old, new)
    args.output.write_text(source)


if __name__ == "__main__":
    main()
