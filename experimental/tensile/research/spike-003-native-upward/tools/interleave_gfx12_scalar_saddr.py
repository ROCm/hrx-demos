#!/usr/bin/env python3
"""Place scalar ring-address formation between the sixth and seventh LHS reads.

The gfx1201 incumbent uses the latency window opened by its first six LHS LDS
reads for scalar global-address updates, then issues the final two LHS reads
and four RHS reads. This transformer preserves the otherwise identical Low
motif and makes that ordering an explicit source contract with schedule fences.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SCALAR_START = "  %scalar_ring_a_k = "
SCALAR_END_PREFIX = "  %scalar_ring_b = "
INSERT_AFTER_PREFIX = "  %even_compute_lhs_h0_raw5_hi = "


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.startswith(SCALAR_START)), None)
    if start is None:
        raise ValueError("scalar ring-address block start not found")
    end = next(
        (
            i + 1
            for i in range(start, len(lines))
            if lines[i].startswith(SCALAR_END_PREFIX)
        ),
        None,
    )
    if end is None:
        raise ValueError("scalar ring-address block end not found")
    scalar_lines = lines[start:end]
    del lines[start:end]

    insertion = next(
        (
            i + 1
            for i, line in enumerate(lines)
            if line.startswith(INSERT_AFTER_PREFIX)
        ),
        None,
    )
    if insertion is None:
        raise ValueError("sixth even-half LHS read result not found")
    payload = ["  low.schedule.fence\n", *scalar_lines, "  low.schedule.fence\n"]
    lines[insertion:insertion] = payload

    args.output.write_text("".join(lines))


if __name__ == "__main__":
    main()
