#!/usr/bin/env python3
"""Make the gfx11 High motif use the paired native WMMA/store mapping.

The native scalar epilogue is not merely a store optimization: gfx11 WMMA
payload elements become directly storable only when every MMA's inputs are
swapped.  The two changes transpose the 16x16 microtile twice and therefore
preserve GEMM semantics.  This High form exists both as a readable source
description and as an access-sanitizer witness for the address map used by the
schedule-locked Low motif.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


EPILOGUE_START = "  %out0 = vector.fragment<result>"


def make_epilogue() -> str:
    lines = [
        "  %native_sixteen = index.constant 16 : index\n",
        "  %native_lane_low = index.rem %lane, %native_sixteen : index\n",
        "  %native_lane_high = index.div %lane, %native_sixteen : index\n",
        "  %native_wave_m_delta = index.mul %wave_m_id, %native_sixteen : index\n",
        "  %native_row_base0 = index.add %workgroup_m_base, %native_wave_m_delta : index\n",
        "  %native_row0 = index.add %native_row_base0, %native_lane_low : index\n",
        "  %native_row1 = index.add %native_row0, %thirty_two : index\n",
        "  %native_col0 = index.add %wave_n_base, %native_lane_high : index\n",
        "  %native_col1 = index.add %native_col0, %thirty_two : index\n",
        "  %native_col2 = index.add %native_col1, %thirty_two : index\n",
    ]
    for accumulator in range(6):
        lines.append(
            f"  %native_r{accumulator}_f16 = vector.fptrunc %r{accumulator} "
            ": vector<8xf32> to vector<8xf16>\n"
        )
    for n_group in range(3):
        for element in range(8):
            column = f"%native_col{n_group}"
            if element:
                column = f"%native_col{n_group}_e{element}"
                previous = (
                    f"%native_col{n_group}"
                    if element == 1
                    else f"%native_col{n_group}_e{element - 1}"
                )
                lines.append(
                    f"  {column} = index.add {previous}, %two : index\n"
                )
            for m_group in range(2):
                accumulator = n_group + 3 * m_group
                row = f"%native_row{m_group}"
                value = f"%native_a{accumulator}_e{element}"
                lines.extend(
                    [
                        f"  {value} = vector.extract %native_r{accumulator}_f16[{element}] "
                        ": vector<8xf16> -> f16\n",
                        f"  view.store {value}, %d[{row}, {column}] : f16, "
                        "view<[%m]x[%n]xf16, %d_layout>\n",
                    ]
                )
    lines.append("  kernel.return\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--keep-mma-roles",
        action="store_true",
        help=(
            "retain High lhs/rhs roles and change only the physical store map; "
            "this is an address-sanitizer witness, not a semantic GEMM motif"
        ),
    )
    args = parser.parse_args()

    source = args.input.read_text()
    if not args.keep_mma_roles:
        source, swap_count = re.subn(
            r"(= vector\.mma )(%[A-Za-z0-9_]+), (%[A-Za-z0-9_]+),",
            r"\1\3, \2,",
            source,
        )
        if swap_count != 24:
            raise ValueError(f"expected 24 gfx11 WMMA operations, found {swap_count}")
    start = source.find(EPILOGUE_START)
    stop = source.find("  kernel.return\n", start)
    if start < 0 or stop < 0:
        raise ValueError("could not isolate the original fragment epilogue")
    stop += len("  kernel.return\n")
    transformed = source[:start] + make_epilogue() + source[stop:]
    if transformed.count("view.store %native_") != 48:
        raise ValueError("native High epilogue must contain 48 scalar stores")
    args.output.write_text(transformed)


if __name__ == "__main__":
    main()
