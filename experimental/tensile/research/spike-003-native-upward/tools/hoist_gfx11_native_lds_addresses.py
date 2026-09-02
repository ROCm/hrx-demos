#!/usr/bin/env python3
"""Share invariant gfx11 address values in the exact-LDS prepared Low motif.

The stage is carried by LDS immediates and K advances in the scalar address.
Each mapping below therefore reuses an already-computed per-thread vector
address or per-half scalar base without changing the accessed byte address.
The original definitions are deliberately retained for ``low-dce`` to remove;
this keeps the transformation auditable against the prepared input.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPLACEMENTS = {
    # The three vector-load sites use the prologue's per-thread vaddrs.
    "%672": "%578",
    "%710": "%609",
    "%690": "%592",
    "%728": "%623",
    "%746": "%639",
    "%915": "%578",
    "%955": "%609",
    "%933": "%592",
    "%973": "%623",
    "%993": "%639",
    # A and B payloads within each half share their scalar base.
    "%721": "%683",
    "%738": "%700",
    "%756": "%700",
    "%964": "%924",
    "%983": "%943",
    "%1003": "%943",
    # Corresponding LDS operations differ only in their stage immediate.
    "%1007": "%767",
    "%1045": "%805",
    "%892": "%652",
    "%896": "%656",
    "%898": "%658",
    "%900": "%660",
    "%902": "%662",
    "%1132": "%652",
    "%1136": "%656",
    "%1138": "%658",
    "%1140": "%660",
    "%1142": "%662",
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
