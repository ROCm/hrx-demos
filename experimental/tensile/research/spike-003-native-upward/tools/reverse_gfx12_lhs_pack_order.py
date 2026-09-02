#!/usr/bin/env python3
"""Pack LHS fragments l3..l0 while retaining the incumbent WMMA order.

This is an alternate Low spelling for gfx12's permutation-to-WMMA latency:
the first consumed fragment is packed last, so a single delay before the WMMA
band covers the newest producer without a second delay inside the band.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


START = re.compile(
    r"  %(?P<prefix>(?:even|odd|tail30|tail31)_compute)_lhs_h(?P<half>[01])_f0_p0 = "
)
FENCE = "  low.schedule.fence\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines(keepends=True)
    output: list[str] = []
    cursor = 0
    band_count = 0
    while cursor < len(lines):
        start_match = START.match(lines[cursor])
        if start_match is None:
            output.append(lines[cursor])
            cursor += 1
            continue

        fragments: list[list[str]] = []
        for fragment in range(4):
            chunk = lines[cursor : cursor + 6]
            if len(chunk) != 6:
                raise ValueError(f"band {band_count}: truncated fragment {fragment}")
            if not all(f"_f{fragment}_p{part} = " in chunk[part] for part in range(4)):
                raise ValueError(f"band {band_count}: malformed fragment {fragment}")
            result_index = int(start_match.group("half")) * 4 + fragment
            if f"_l{result_index} = low.concat" not in chunk[4] or chunk[5] != FENCE:
                raise ValueError(f"band {band_count}: malformed fragment tail {fragment}")
            fragments.append(chunk)
            cursor += 6
        for fragment in reversed(fragments):
            output.extend(fragment)
        band_count += 1

    if band_count != 8:
        raise ValueError(f"expected eight K16 pack bands, found {band_count}")
    args.output.write_text("".join(output))


if __name__ == "__main__":
    main()
