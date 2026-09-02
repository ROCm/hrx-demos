#!/usr/bin/env python3
"""Hoist loop-invariant gfx11 vector addresses in prepared Low.

The High lowering reconstructed identical per-thread VMEM and LDS addresses in
each half of the K64 loop.  K movement already lives in the scalar saddr, and
the LDS stage is encoded in immediate offsets, so those vector expressions are
loop invariant.  Reusing the prologue/even-half values lets ordinary DCE remove
the redundant VALU address trees without changing an access.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.read_text()

    replacements = {
        # VMEM per-thread vaddrs; K is entirely in saddr.
        "%726": "%636",
        "%764": "%667",
        "%744": "%650",
        "%782": "%681",
        "%800": "%697",
        "%965": "%636",
        "%1005": "%667",
        "%983": "%650",
        "%1023": "%681",
        "%1043": "%697",
        # All payloads in a half share the same scalar A or B base.
        "%775": "%737",
        "%792": "%754",
        "%810": "%754",
        "%1014": "%974",
        "%1033": "%993",
        "%1053": "%993",
        # LDS stage selection is in immediate offsets, not the vaddr.
        "%1057": "%821",
        "%1095": "%859",
        # The five LDS stores always use thread * 16.
        "%944": "%708",
        "%1180": "%708",
    }
    for old, new in replacements.items():
        if old not in source:
            raise ValueError(f"missing expected prepared-Low value {old}")
        # Keep the defining occurrence intact and rewrite later operands only.
        definition = source.index(old)
        tail = source[definition + len(old) :].replace(old, new)
        source = source[: definition + len(old)] + tail

    args.output.write_text(source)


if __name__ == "__main__":
    main()
