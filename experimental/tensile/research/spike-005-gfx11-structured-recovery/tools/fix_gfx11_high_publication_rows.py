#!/usr/bin/env python3
"""Adapt the prepared-Low gfx11 publication map to High fragment roles.

The recovered prepared-Low oracle assigns its two accumulator row groups as
``wave_m * 16 + {0, 32}``. High ``vector.fragment.load`` assigns them as
``wave_m * 32 + {0, 16}``. Uniform inputs hide this permutation, so the
transform is deliberately narrow and validated by the row witness.
"""

from argparse import ArgumentParser
from pathlib import Path


REPLACEMENTS = {
    "  %native_wave_m_delta = index.mul %wave_m_id, %native_sixteen : index\n":
        "  %native_wave_m_delta = index.mul %wave_m_id, %thirty_two : index\n",
    "  %native_row1 = index.add %native_row0, %thirty_two : index\n":
        "  %native_row1 = index.add %native_row0, %native_sixteen : index\n",
}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    for old, new in REPLACEMENTS.items():
        if source.count(old) != 1:
            raise ValueError(f"expected exactly one publication anchor: {old.strip()}")
        source = source.replace(old, new, 1)
    args.output.write_text(source)


if __name__ == "__main__":
    main()
