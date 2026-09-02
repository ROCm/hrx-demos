#!/usr/bin/env python3
"""Author a solution-1675-shaped schedule for the prepared-Low K32 ring.

The generic scheduler groups all low-half LDS reads and all six WMMAs into
wide bands.  The incumbent instead alternates WMMAs with paired low/high A
reads, global prefetch, LDS publication, and B reads.  This transform rewrites
only the order inside ``^_bb2`` and separates every emitted operation with a
``low.schedule.fence``.  The fences make the desired machine order explicit
while leaving waits to Loom's hazard planner.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def find(lines: list[str], needle: str) -> int:
    hits = [i for i, line in enumerate(lines) if needle in line]
    if len(hits) != 1:
        raise ValueError(f"expected one {needle!r}, found {len(hits)}")
    return hits[0]


def result_pair(lines: list[str], result: str) -> list[str]:
    index = find(lines, f"%{result} = low.op<amdgpu.v_wmma")
    if "low.copy" not in lines[index - 1]:
        raise ValueError(f"WMMA {result} is not preceded by its accumulator copy")
    return lines[index - 1 : index + 1]


def range_through(lines: list[str], first: str, last: str) -> list[str]:
    begin = find(lines, first)
    end = find(lines, last)
    if end < begin:
        raise ValueError(f"inverted range {first!r} through {last!r}")
    return lines[begin : end + 1]


def packet_pairs(group: list[str]) -> tuple[list[list[str]], str]:
    if len(group) != 17 or "low.concat" not in group[-1]:
        raise ValueError(f"expected 16 packet ops and concat, got {len(group)} lines")
    return [group[i : i + 2] for i in range(0, 16, 2)], group[-1]


def fenced(lines: list[str]) -> list[str]:
    result: list[str] = []
    for line in lines:
        result.append(line)
        if "low.op<" in line or "low.const<" in line:
            result.append("  low.schedule.fence")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines()
    body_start = lines.index("^_bb2:")
    exit_start = lines.index("^_bb3:")
    body = lines[body_start + 1 : exit_start]

    # Five independent scalar-address/global-load packets.  In the payload
    # ring, the VMEM results must follow the matching carried-payload store so
    # allocator-visible lifetimes do not overlap: this is the incumbent's
    # store-then-refill register-ring discipline.
    prefetch = [
        range_through(body, "%742 =", "%odd_av0 ="),
        range_through(body, "%760 =", "%odd_bv0 ="),
        range_through(body, "%780 =", "%odd_av1 ="),
        range_through(body, "%800 =", "%odd_bv1 ="),
        range_through(body, "%818 =", "%odd_bv2 ="),
    ]
    payload_k_advance = (
        [body[find(body, "%payload_k_base =")]]
        if any("%payload_k_base =" in line for line in body)
        else []
    )
    carries_payloads = any("%cur_odd_av0" in line for line in body)
    if carries_payloads:
        prefetch_prefix = [group[:-1] for group in prefetch]
        prefetch_load = [[group[-1]] for group in prefetch]
    else:
        prefetch_prefix = prefetch
        prefetch_load = [[] for _ in prefetch]

    # Loom's generic microtile is lhs-major, while solution 1675 issues the
    # same six products rhs-major. The physical accumulator witness below is
    # paired with this [0,3,1,4,2,5] semantic permutation.
    native_accumulator_order = [0, 3, 1, 4, 2, 5]
    first_wmma = [
        result_pair(body, f"even_first_acc{i}") for i in native_accumulator_order
    ]
    second_wmma = [
        result_pair(body, f"even_second_acc{i}") for i in native_accumulator_order
    ]

    current_lhs0, current_lhs0_concat = packet_pairs(
        range_through(body, "%839 =", "%even_second_lhs0 =")
    )
    current_lhs1, current_lhs1_concat = packet_pairs(
        range_through(body, "%858 =", "%even_second_lhs1 =")
    )
    next_lhs0, next_lhs0_concat = packet_pairs(
        range_through(body, "%991 =", "%odd_first_lhs0 =")
    )
    next_lhs1, next_lhs1_concat = packet_pairs(
        range_through(body, "%1010 =", "%odd_first_lhs1 =")
    )

    current_rhs = [
        range_through(body, "%881 =", "%even_second_rhs0 ="),
        range_through(body, "%886 =", "%even_second_rhs1 ="),
        range_through(body, "%891 =", "%even_second_rhs2 ="),
    ]
    next_rhs = [
        range_through(body, "%1033 =", "%odd_first_rhs0 ="),
        range_through(body, "%1038 =", "%odd_first_rhs1 ="),
        range_through(body, "%1043 =", "%odd_first_rhs2 ="),
    ]
    store_xors = range_through(body, "%next_a_store_base =", "%next_b_store_base =")
    stores = [
        [line]
        for line in body
        if "low.op<amdgpu.ds_write_b128>(%" in line
        and "_store_base," in line
    ]
    if len(stores) != 5:
        raise ValueError(f"expected five LDS publication packets, found {len(stores)}")
    read_xors = range_through(body, "%next_a_read_base =", "%next_b_read_base =")
    barrier = [body[find(body, "low.op<amdgpu.s_barrier>")]]
    tail = range_through(body, "%k_next =", "low.br ^_bb1(")

    scheduled: list[str] = []

    def emit(*groups: list[str]) -> None:
        for group in groups:
            scheduled.extend(group)

    # First K16.  This mirrors the incumbent's four-pair/four-pair split for
    # lhs0 and three-pair/five-pair split for lhs1.
    emit(first_wmma[0], *current_lhs0[:4], payload_k_advance, prefetch_prefix[0])
    emit(first_wmma[1], *current_lhs0[4:], [current_lhs0_concat], prefetch_prefix[1])
    emit(first_wmma[2], current_rhs[0], *current_lhs1[:3])
    emit(
        prefetch_prefix[2],
        prefetch_prefix[3],
        prefetch_prefix[4],
        stores[0],
        prefetch_load[0],
        stores[1],
        prefetch_load[2],
        stores[2],
        prefetch_load[1],
        stores[3],
        prefetch_load[3],
        stores[4],
        prefetch_load[4],
        store_xors,
    )
    # TensileLite publishes/refills all five payload banks before resuming
    # matrix issue. Its fifth lhs1 packet straddles the fourth and fifth
    # first-K16 WMMAs, an exact detail that is easy to miss when grouping the
    # low/high d16 pair as one source-level operation.
    emit(first_wmma[3], *current_lhs1[3:6], [current_lhs1[6][0]])
    emit(
        first_wmma[4],
        [current_lhs1[6][1]],
        current_lhs1[7],
        [current_lhs1_concat],
        current_rhs[1],
        current_rhs[2],
    )
    # The native loop XORs read bases before its final first-K16 WMMA.
    emit(read_xors, first_wmma[5], barrier)

    # Second K16, overlapped with the next tile's first-K16 LDS reads.
    emit(second_wmma[0], *next_lhs0[:4])
    emit(second_wmma[1], *next_lhs0[4:], [next_lhs0_concat])
    emit(second_wmma[2], next_rhs[0], *next_lhs1[:3])
    emit(second_wmma[3], *next_lhs1[3:6])
    emit([next_lhs1[6][0]], second_wmma[4], [next_lhs1[6][1]], next_lhs1[7], [next_lhs1_concat])
    emit(next_rhs[1], next_rhs[2], second_wmma[5], tail)

    # Ensure the rewrite neither loses nor duplicates source statements.
    ignored = {"  low.schedule.fence"}
    original_multiset = sorted(line for line in body if line not in ignored)
    scheduled_multiset = sorted(line for line in scheduled if line not in ignored)
    if original_multiset != scheduled_multiset:
        missing = sorted(set(original_multiset) - set(scheduled_multiset))
        extra = sorted(set(scheduled_multiset) - set(original_multiset))
        raise ValueError(f"schedule census mismatch; missing={missing}, extra={extra}")

    output = lines[: body_start + 1] + fenced(scheduled) + lines[exit_start:]
    header = output[0]
    if "schedule(locked)" not in header:
        output[0] = header.replace(
            "low.kernel.def retain ", "low.kernel.def retain schedule(locked) ", 1
        )
    args.output.write_text("\n".join(output) + "\n")


if __name__ == "__main__":
    main()
