#!/usr/bin/env python3
"""Carry solution 1675's five global-load payloads across the K32 backedge.

The first K32 reduction still loaded and stored each next-stage payload in the
same iteration.  Tensile instead enters the loop with A0/A1/B0/B1/B2 already
resident, stores each packet, and immediately refills the same VGPR bank for
the following trip.  This transform adds that missing PGR stage and advances
the body refill by one additional K32; the cloned prologue prefetch retains
the original next-tile address.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


RESULT = re.compile(r"^  (%[A-Za-z0-9_]+) =")


def find(lines: list[str], needle: str, begin: int = 0, end: int | None = None) -> int:
    limit = len(lines) if end is None else end
    hits = [i for i in range(begin, limit) if needle in lines[i]]
    if len(hits) != 1:
        raise ValueError(f"expected one {needle!r}, found {len(hits)}")
    return hits[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines()
    body_start = lines.index("^_bb2:")
    exit_start = lines.index("^_bb3:")
    prologue_branch = find(lines, "low.br ^_bb1(", 0, body_start)

    load_results = ("odd_av0", "odd_bv0", "odd_av1", "odd_bv1", "odd_bv2")
    first_address = find(lines, "%742 =", body_start, exit_start)
    last_load = find(lines, "%odd_bv2 =", body_start, exit_start)
    prefetch_lines = lines[first_address : last_load + 1]
    definitions = [
        match.group(1)
        for line in prefetch_lines
        if (match := RESULT.match(line)) is not None
    ]
    init_mapping = {name: "%init_" + name[1:] for name in definitions}
    init_lines: list[str] = []
    for line in prefetch_lines:
        line = line.replace("%k_base", "%zero")
        # Longest first avoids replacing a prefix of another SSA name.
        for old in sorted(init_mapping, key=len, reverse=True):
            line = line.replace(old, init_mapping[old])
        init_lines.append(line)
    for name in load_results:
        if "%init_" + name not in "\n".join(init_lines):
            raise ValueError(f"initial payload {name} was not materialized")
    lines[prologue_branch:prologue_branch] = init_lines
    body_start += len(init_lines)
    exit_start += len(init_lines)
    prologue_branch += len(init_lines)

    header = lines[body_start - 3]  # ^_bb1 follows branch and its source line.
    if not header.startswith("^_bb1("):
        header_index = find(lines, "^_bb1(", prologue_branch, body_start)
    else:
        header_index = body_start - 3
    header = lines[header_index]
    if not header.endswith("):"):
        raise ValueError("unexpected loop header spelling")
    payload_parameters = ", ".join(
        f"%cur_{name}: reg<amdgpu.vgpr x4>" for name in load_results
    )
    lines[header_index] = header[:-2] + ", " + payload_parameters + "):"

    branch = lines[prologue_branch]
    if not branch.endswith(")"):
        raise ValueError("unexpected prologue branch spelling")
    init_arguments = ", ".join(
        f"%init_{name}: reg<amdgpu.vgpr x4>" for name in load_results
    )
    lines[prologue_branch] = branch[:-1] + ", " + init_arguments + ")"

    # Re-find the body after header edits and use carried payloads for this
    # trip's publication. The body loads retain their names and become the
    # payloads passed to the next trip.
    body_start = lines.index("^_bb2:")
    exit_start = lines.index("^_bb3:")
    first_address = find(lines, "%742 =", body_start, exit_start)
    last_load = find(lines, "%odd_bv2 =", body_start, exit_start)
    lines.insert(
        first_address,
        "  %payload_k_base = low.op<amdgpu.s_add_u32.rhs_inline>(%k_base) "
        "{imm32 = 32} : (reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>",
    )
    last_load += 1
    for i in range(first_address + 1, last_load + 1):
        lines[i] = lines[i].replace("%k_base", "%payload_k_base")
    exit_start += 1
    changed_stores = 0
    for i in range(body_start, exit_start):
        for name in load_results:
            needle = f", %{name})"
            if "low.op<amdgpu.ds_write_b128>" in lines[i] and needle in lines[i]:
                lines[i] = lines[i].replace(needle, f", %cur_{name})")
                changed_stores += 1
    if changed_stores != 5:
        raise ValueError(f"expected five payload stores, changed {changed_stores}")

    backedge = find(lines, "low.br ^_bb1(", body_start, exit_start)
    branch = lines[backedge]
    if not branch.endswith(")"):
        raise ValueError("unexpected backedge spelling")
    next_arguments = ", ".join(
        f"%{name}: reg<amdgpu.vgpr x4>" for name in load_results
    )
    lines[backedge] = branch[:-1] + ", " + next_arguments + ")"

    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
