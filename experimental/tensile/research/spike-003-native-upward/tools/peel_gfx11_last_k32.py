#!/usr/bin/env python3
"""Peel the final two K32 trips from the locked gfx11 payload-ring loop.

The penultimate trip consumes K=960, stores the already-fetched K=992
payload, and prepares the final first-K16 fragments, but does not refill global
payloads. The final trip consumes K=992 and omits stores, refills, the barrier,
and reads for a nonexistent successor. This mirrors the structural tail in the
solution-1675 Tensile kernel while retaining the recovered steady-loop body.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ARG_TYPES = {
    "k_base": "reg<amdgpu.sgpr>",
    **{f"out{i}": "reg<amdgpu.vgpr x8>" for i in range(6)},
    "cur_lhs0": "reg<amdgpu.vgpr x8>",
    "cur_lhs1": "reg<amdgpu.vgpr x8>",
    "cur_rhs0": "reg<amdgpu.vgpr x8>",
    "cur_rhs1": "reg<amdgpu.vgpr x8>",
    "cur_rhs2": "reg<amdgpu.vgpr x8>",
    "a_read_base": "reg<amdgpu.vgpr>",
    "b_read_base": "reg<amdgpu.vgpr>",
    "a_store_base": "reg<amdgpu.vgpr>",
    "b_store_base": "reg<amdgpu.vgpr>",
    "cur_odd_av0": "reg<amdgpu.vgpr x4>",
    "cur_odd_bv0": "reg<amdgpu.vgpr x4>",
    "cur_odd_av1": "reg<amdgpu.vgpr x4>",
    "cur_odd_bv1": "reg<amdgpu.vgpr x4>",
    "cur_odd_bv2": "reg<amdgpu.vgpr x4>",
}

HEADER_ARGS = list(ARG_TYPES)
ADDRESS_DEFS = {
    "payload_k_base", "742", "743", "744", "745", "746", "760", "761",
    "762", "763", "780", "781", "782", "784", "800", "801", "818",
    "819",
}
REFILL_DEFS = {"odd_av0", "odd_av1", "odd_bv0", "odd_bv1", "odd_bv2"}
NEXT_READ_DEFS = {
    *(str(i) for i in range(991, 1007)),
    *(str(i) for i in range(1010, 1026)),
    "1033", "1034", "1038", "1039", "1043", "1044",
    "odd_first_lhs0", "odd_first_lhs1", "odd_first_rhs0", "odd_first_rhs1",
    "odd_first_rhs2", "next_a_read_base", "next_b_read_base",
}
STORE_BASE_DEFS = {"next_a_store_base", "next_b_store_base"}


def branch_args(names: list[str], prefix: str = "") -> str:
    return ", ".join(f"%{prefix}{name}: {ARG_TYPES[name]}" for name in names)


def defined_names(line: str) -> list[str]:
    if " = " not in line:
        return []
    return re.findall(r"%([A-Za-z0-9_]+)", line.split(" = ", 1)[0])


def rename(line: str, mapping: dict[str, str]) -> str:
    return re.sub(
        r"%([A-Za-z0-9_]+)",
        lambda match: "%" + mapping.get(match.group(1), match.group(1)),
        line,
    )


def clone_body(
    body: list[str], prefix: str, arg_mapping: dict[str, str], tail: int
) -> tuple[list[str], dict[str, str]]:
    local_defs = {name for line in body for name in defined_names(line)}
    mapping = {name: prefix + name for name in local_defs}
    mapping.update(arg_mapping)
    output: list[str] = []
    pending_fence = False
    vm_wait = 4
    for line in body:
        defs = set(defined_names(line))
        skip = bool(defs & (ADDRESS_DEFS | REFILL_DEFS | STORE_BASE_DEFS))
        if "buffer_load_b128" in line:
            skip = True
        if tail == 2:
            if defs & NEXT_READ_DEFS:
                skip = True
            if "ds_write_b128" in line or "s_barrier" in line:
                skip = True
            # The two full drains surround the producer/consumer barrier and
            # are unnecessary when the final trip has no successor reads.
            if "s_waitcnt" in line and "lgkmcnt = 0" in line:
                skip = True
            if "s_waitcnt" in line and "vmcnt = 4" in line:
                skip = True
        elif "s_waitcnt" in line and "vmcnt = 4" in line:
            # No refill follows the store, so drain one additional payload at
            # each penultimate-trip store exactly as the incumbent does.
            line = line.replace("vmcnt = 4", f"vmcnt = {vm_wait}")
            vm_wait -= 1
        if skip:
            pending_fence = True
            continue
        if line.strip() == "low.schedule.fence" and pending_fence:
            pending_fence = False
            continue
        pending_fence = False
        output.append(rename(line, mapping))
    return output, mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines()
    body_start = lines.index("^_bb2:") + 1
    backedge = next(
        i for i in range(body_start, len(lines)) if "low.br ^_bb1(" in lines[i]
    )
    epilogue = lines.index("^_bb3:")
    body = lines[body_start:backedge]
    tail_body = [
        line for line in body if not (set(defined_names(line)) & {"k_next"})
    ]

    # Tensile's steady loop is bottom-tested. Branch directly from the body to
    # the tail with its freshly produced recurrence values; routing those
    # values through the header phi needlessly extends their fixed live ranges
    # across both successors and makes the exact cyclic banks impossible.
    header = lines.index(next(line for line in lines if line.startswith("^_bb1(")))
    old_cond = "  low.cond_br %733, ^_bb2, ^_bb3 : reg<amdgpu.scc>"
    cond_index = lines.index(old_cond)
    del lines[cond_index]
    # The compare is immediately before the conditional branch.
    del lines[cond_index - 1]
    body_label = lines.index("^_bb2:")
    del lines[body_label]
    body_start -= 3
    backedge -= 3
    epilogue -= 3
    entry_branch = next(i for i in range(header) if "low.br ^_bb1(" in lines[i])
    lines.insert(
        entry_branch,
        "  %steady_end = low.const<amdgpu.s_mov_b32> {imm32 = 960} : reg<amdgpu.sgpr>",
    )
    backedge += 1
    epilogue += 1

    t1_arg_map = {name: "t1_" + name for name in HEADER_ARGS}
    t1_body, t1_map = clone_body(tail_body, "t1_", t1_arg_map, tail=1)
    t1_outputs = [t1_map[f"even_second_acc{i}"] for i in range(6)]
    t1_next_fragments = [
        t1_map["odd_first_lhs0"], t1_map["odd_first_lhs1"],
        t1_map["odd_first_rhs0"], t1_map["odd_first_rhs1"],
        t1_map["odd_first_rhs2"], t1_map["next_a_read_base"],
        t1_map["next_b_read_base"],
    ]

    t2_arg_names = [
        *(f"out{i}" for i in range(6)), "cur_lhs0", "cur_lhs1", "cur_rhs0",
        "cur_rhs1", "cur_rhs2", "a_read_base", "b_read_base",
    ]
    t2_types = {name: ARG_TYPES[name] for name in t2_arg_names}
    t2_arg_map = {name: "t2_" + name for name in t2_arg_names}
    t2_body, t2_map = clone_body(tail_body, "t2_", t2_arg_map, tail=2)
    t2_outputs = [t2_map[f"even_second_acc{i}"] for i in range(6)]

    t1_to_t2_values = t1_outputs + t1_next_fragments
    t1_to_t2 = ", ".join(
        f"%{value}: {t2_types[name]}"
        for value, name in zip(t1_to_t2_values, t2_arg_names, strict=True)
    )
    original_backedge = lines[backedge]
    if "low.br ^_bb1(" not in original_backedge:
        raise ValueError("lost steady-loop backedge while rewriting CFG")
    tail_branch = original_backedge.replace("^_bb1(", "^_bb_tail1(")
    lines[backedge:backedge + 1] = [
        "  %steady_continue = low.op<amdgpu.s_cmp_lt_i32>(%k_next, %steady_end) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.scc>",
        "  low.cond_br %steady_continue, ^_bb_backedge_dispatch, ^_bb_tail_dispatch : reg<amdgpu.scc>",
        "^_bb_backedge_dispatch:",
        original_backedge,
        "^_bb_tail_dispatch:",
        tail_branch,
    ]
    epilogue += 5

    t1_block = [
        "^_bb_tail1(" + branch_args(HEADER_ARGS, "t1_") + "):",
        *t1_body,
        "  low.br ^_bb_tail2(" + t1_to_t2 + ")",
    ]
    t2_to_epilogue = ", ".join(
        f"%{value}: reg<amdgpu.vgpr x8>" for value in t2_outputs
    )
    t2_block = [
        "^_bb_tail2(" + ", ".join(
            f"%t2_{name}: {t2_types[name]}" for name in t2_arg_names
        ) + "):",
        *t2_body,
        "  low.br ^_bb3(" + t2_to_epilogue + ")",
    ]

    # Insert peeled blocks before the epilogue and make accumulator ownership
    # explicit at that join.
    lines[backedge + 6:epilogue] = t1_block + t2_block
    epilogue = lines.index("^_bb3:")
    lines[epilogue] = "^_bb3(" + ", ".join(
        f"%final_out{i}: reg<amdgpu.vgpr x8>" for i in range(6)
    ) + "):"
    for i in range(epilogue + 1, len(lines)):
        for acc in range(6):
            lines[i] = re.sub(
                rf"%out{acc}(?![A-Za-z0-9_])", f"%final_out{acc}", lines[i]
            )

    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
