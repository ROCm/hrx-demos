#!/usr/bin/env python3
"""Hoist invariant addresses from the gfx11 cross-iteration PLR Low form."""

from __future__ import annotations

import argparse
from pathlib import Path


REPLACEMENTS = {
    # Per-thread global vector addresses; K advances in scalar saddr.
    "%741": "%590",
    "%779": "%621",
    "%759": "%604",
    "%797": "%635",
    "%815": "%651",
    # One A and one B scalar base per dynamic K half.
    "%790": "%752",
    "%807": "%769",
    "%825": "%769",
    "%954": "%925",
    "%969": "%940",
    "%984": "%940",
    # LDS stage selection lives in immediate offsets.
    "%990": "%684",
    "%1032": "%722",
    "%1128": "%684",
    "%1170": "%722",
    # The padded-layout store addresses are identical across stages.
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
