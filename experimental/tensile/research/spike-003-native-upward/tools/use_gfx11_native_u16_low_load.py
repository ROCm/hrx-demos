#!/usr/bin/env python3
"""Use gfx11's full-register ``ds_load_u16`` for low fragment halves.

Loom's descriptor is named ``amdgpu.ds_read_u16``; LLVM spells the emitted
gfx11 opcode ``ds_load_u16``.  Tensile solution 1675 uses that opcode for the
low half and ``ds_load_u16_d16_hi`` for the tied high half.  The ordinary
fragment lowering instead selects ``ds_load_u16_d16`` for the low half.
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
    old = "low.op<amdgpu.ds_load_u16_d16>(%684)"
    new = "low.op<amdgpu.ds_read_u16>(%684)"
    count = source.count(old)
    if count != 80:
        raise ValueError(f"expected 80 low-half A fragment reads, found {count}")
    source = source.replace(old, new)
    args.output.write_text(source)


if __name__ == "__main__":
    main()
