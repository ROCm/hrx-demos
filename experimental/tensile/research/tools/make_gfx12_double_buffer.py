#!/usr/bin/env python3
"""Derive a fixed-K, two-stage LDS gfx12 GEMM experiment.

The generated High Loom form processes two K32 tiles per loop iteration.  It
keeps the phase statically known (stage 0, then stage 1), overlaps each next
global load with the current tile's matrix work, and peels the final two tiles
so that no speculative out-of-bounds load is required.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def definitions(region: str) -> set[str]:
    return set(re.findall(r"^\s*(%[A-Za-z0-9_]+)\s*=", region, re.MULTILINE))


def rename(region: str, mapping: dict[str, str]) -> str:
    for old in sorted(mapping, key=len, reverse=True):
        region = re.sub(rf"{re.escape(old)}(?![A-Za-z0-9_])", mapping[old], region)
    return region


def instantiate(region: str, prefix: str, k_value: str, stage: int) -> str:
    result = rename(region, {name: f"%{prefix}{name[1:]}" for name in definitions(region)})
    result = rename(result, {"%k_base": k_value})
    if stage == 1:
        result = rename(
            result,
            {
                "%a_lds_store": "%a_lds_store_stage1",
                "%b_lds_store": "%b_lds_store_stage1",
                "%a_lds": "%a_lds_stage1",
                "%b_lds": "%b_lds_stage1",
            },
        )
    return result


def without_loads(region: str) -> str:
    return "\n".join(line for line in region.splitlines() if "vector.load" not in line) + "\n"


def without_stores(region: str) -> str:
    return "\n".join(line for line in region.splitlines() if "vector.store" not in line) + "\n"


def compute_instance(
    compute: str, prefix: str, stage: int, accumulators: list[str]
) -> tuple[str, list[str]]:
    result = instantiate(compute, prefix, "%unused_k", stage)
    result = rename(result, {f"%a{i}": accumulators[i] for i in range(16)})
    outputs = [f"%{prefix}y{i}" for i in range(16)]
    return result, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.input.read_text()

    loop_at = text.index("  %r0, %r1, %r2")
    header_end = text.index(" {\n", loop_at) + 3
    first_barrier = text.index(
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n", header_end
    )
    loader = text[header_end:first_barrier]
    compute_start = first_barrier + len(
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
    )
    second_barrier = text.index(
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n", compute_start
    )
    compute = text[compute_start:second_barrier]
    loop_close = text.index("  }\n", second_barrier) + len("  }\n")
    epilog = text[loop_close:]

    before = text[:loop_at]
    before = before.replace(
        "%lds_bytes = index.constant 16384 : offset",
        "%lds_bytes = index.constant 51200 : offset",
    )
    view_anchor = (
        "  %b_lds_store = buffer.view %lds[%b_offset] : buffer -> view<128x32xf16>\n"
    )
    stage1_views = (
        "  %stage1_offset = index.constant 32768 : offset\n"
        "  %a_lds_stage1 = buffer.view %lds[%stage1_offset] : buffer -> view<128x32xf16, %a_lds_layout>\n"
        "  %a_lds_store_stage1 = buffer.view %lds[%stage1_offset] : buffer -> view<32x128xf16>\n"
        "  %b_stage1_offset = index.constant 40960 : offset\n"
        "  %b_lds_stage1 = buffer.view %lds[%b_stage1_offset] : buffer -> view<32x128xf16, %b_lds_layout>\n"
        "  %b_lds_store_stage1 = buffer.view %lds[%b_stage1_offset] : buffer -> view<128x32xf16>\n"
    )
    if before.count(view_anchor) != 1:
        raise ValueError("expected stage-zero LDS view anchor")
    before = before.replace(view_anchor, view_anchor + stage1_views)

    load_only = without_stores(loader)
    store_only = without_loads(loader)

    # Prologue: load and materialize tile 0 in stage 0.
    prologue = instantiate(loader, "pro_", "%zero", 0)
    prologue += "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"

    acc_types = ", ".join("vector<8xf32>" for _ in range(16))
    zero_args = ", ".join(f"%a{i} = %zero_acc : vector<8xf32>" for i in range(16))
    result_names = [f"%pair_a{i}" for i in range(16)]
    header = (
        "  %pair_end = index.sub %k, %sixty_four : index\n"
        f"  {', '.join(result_names)} = scf.for %k_base = [%zero to %pair_end step %sixty_four]"
        f"({zero_args}) -> ({acc_types}) {{\n"
    )

    # Tile 2p+1: load while tile 2p is resident in stage 0, compute, then
    # materialize the payload into stage 1.
    load1 = instantiate(load_only, "odd_load_", "%odd_k", 1)
    comp0, acc0 = compute_instance(compute, "even_compute_", 0, [f"%a{i}" for i in range(16)])
    store1 = instantiate(store_only, "odd_store_", "%odd_k", 1)
    store1 = rename(
        store1,
        {f"%av{i}": f"%odd_load_av{i}" for i in range(4)}
        | {f"%bv{i}": f"%odd_load_bv{i}" for i in range(4)},
    )

    # Tile 2p+2: load while tile 2p+1 is resident in stage 1, compute, then
    # materialize the payload back into stage 0 for the following iteration.
    load2 = instantiate(load_only, "even_load_", "%even_k", 0)
    comp1, acc1 = compute_instance(compute, "odd_compute_", 1, acc0)
    store2 = instantiate(store_only, "even_store_", "%even_k", 0)
    store2 = rename(
        store2,
        {f"%av{i}": f"%even_load_av{i}" for i in range(4)}
        | {f"%bv{i}": f"%even_load_bv{i}" for i in range(4)},
    )
    body = (
        "    %odd_k = index.add %k_base, %depth_u : index\n"
        + load1
        + comp0
        + store1
        + "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
        + "    %even_k = index.add %odd_k, %depth_u : index\n"
        + load2
        + comp1
        + store2
        + "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
        + f"    scf.yield {', '.join(acc1)} : {acc_types}\n"
        + "  }\n\n"
    )

    # The loop exits with tile 30 in stage 0.  Load tile 31, compute tile 30,
    # publish tile 31 to stage 1, then compute the final tile.
    tail_k = "%tail_k"
    tail_load = instantiate(load_only, "tail_load_", tail_k, 1)
    tail_comp0, tail_acc0 = compute_instance(compute, "tail30_compute_", 0, result_names)
    tail_store = instantiate(store_only, "tail_store_", tail_k, 1)
    tail_store = rename(
        tail_store,
        {f"%av{i}": f"%tail_load_av{i}" for i in range(4)}
        | {f"%bv{i}": f"%tail_load_bv{i}" for i in range(4)},
    )
    tail_comp1, tail_acc1 = compute_instance(compute, "tail31_compute_", 1, tail_acc0)
    tail = (
        "  %tail_k = index.sub %k, %depth_u : index\n"
        + tail_load
        + tail_comp0
        + tail_store
        + "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
        + tail_comp1
    )
    epilog = rename(epilog, {f"%r{i}": tail_acc1[i] for i in range(16)})

    # The two-tile loop requires this fixed increment as an SSA constant.
    before = before.replace(
        "  %forty_eight = index.constant 48 : index\n",
        "  %forty_eight = index.constant 48 : index\n  %sixty_four = index.constant 64 : index\n",
    )
    generated = before + prologue + header + body + tail + epilog
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise ValueError("output was not created")


if __name__ == "__main__":
    main()
