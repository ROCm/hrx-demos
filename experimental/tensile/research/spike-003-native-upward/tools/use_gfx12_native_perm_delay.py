#!/usr/bin/env python3
"""Move the B-pack-to-WMMA dependency delay to the incumbent boundary.

The initial Low transcription used S_DELAY_ALU SALU_CYCLE_1 (immediate 9),
which does not advance the relevant VALU dependency window.  Loom consequently
inserts a VALU_DEP_4 packet before the fourth WMMA.  The incumbent uses S_NOP 1
and then issues all 16 WMMAs without an internal delay.  Source Low does not
yet expose S_NOP, so this experiment authors VALU_DEP_4 at that boundary.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    old = "  low.op<amdgpu.s_delay_alu>() {delay = 9} : ()"
    count = source.count(old)
    if count != 8:
        raise ValueError(f"expected eight K16 packing delays, found {count}")
    # Source Low does not currently expose S_NOP as a descriptor.  VALU_DEP_4
    # is the closest directly authorable form: place the required vector
    # dependency delay at the same boundary instead of letting the hazard
    # planner discover it before the fourth WMMA.
    args.output.write_text(
        source.replace(old, "  low.op<amdgpu.s_delay_alu>() {delay = 4} : ()")
    )


if __name__ == "__main__":
    main()
