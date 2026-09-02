#!/usr/bin/env python3
"""Derive a one-stage global-prefetch High loop from the compact gfx12 GEMM.

The generated experiment carries the eight vector loads for tile K through
the loop, writes them to LDS, launches tile K+1, and then computes tile K.
It is intentionally expressed in High Loom before the separate native
fragment-contract Low rewrite.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def rename_definitions(region: str, prefix: str) -> str:
    definitions = set(re.findall(r"^\s*(%[A-Za-z0-9_]+)\s*=", region, re.MULTILINE))
    for name in sorted(definitions, key=len, reverse=True):
        region = re.sub(rf"{re.escape(name)}(?![A-Za-z0-9_])", f"%{prefix}{name[1:]}", region)
    return region


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a gfx12 High prefetch loop")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.input.read_text()

    loop_result = text.index("  %r0, %r1, %r2")
    header_end = text.index(" {\n", loop_result) + 3
    loader_start = text.index("    %linear0 = ", header_end)
    last_store = text.index("    vector.store %bv3", loader_start)
    loader_stop = text.index("\n", last_store) + 1
    loader = text[loader_start:loader_stop]

    initial = rename_definitions(loader.replace("%k_base", "%zero"), "init_")
    initial = "\n".join(
        line for line in initial.splitlines() if "vector.store" not in line
    ) + "\n\n"

    carried_order = ["av0", "bv0", "av1", "bv1", "av2", "bv2", "av3", "bv3"]
    carried_args = ", ".join(
        f"%p_{name} = %init_{name} : vector<8xf16>" for name in carried_order
    )
    carried_types = ", ".join("vector<8xf16>" for _ in carried_order)

    header = text[loop_result:header_end]
    header = header[:-3]  # strip " {\n"
    close = header.rindex(") -> (")
    header = header[:close] + ", " + carried_args + header[close:]
    header = header[:-1] + ", " + carried_types + ") {\n"

    # Preserve the source's store maps but store the prior iteration's
    # prefetched values. The surrounding index arithmetic is retained here;
    # prepared-Low experiments may subsequently hoist it.
    stores = loader
    stores = "\n".join(
        line for line in stores.splitlines() if "vector.load" not in line
    ) + "\n"
    for name in carried_order:
        stores = re.sub(rf"%{name}(?![A-Za-z0-9_])", f"%p_{name}", stores)

    next_loader = rename_definitions(loader.replace("%k_base", "%next_k"), "next_")
    next_loader = "\n".join(
        line for line in next_loader.splitlines() if "vector.store" not in line
    ) + "\n"
    next_values = ", ".join(f"%next_{name}" for name in carried_order)
    old_values = ", ".join(f"%p_{name}" for name in carried_order)
    yield_types = ", ".join("vector<8xf16>" for _ in carried_order)
    prefetch = (
        "    %next_k = index.add %k_base, %depth_u : index\n"
        "    %has_next = index.cmp ult, %next_k, %k : index\n"
        f"    {', '.join(f'%carry_{name}' for name in carried_order)} = "
        f"scf.if %has_next -> ({yield_types}) {{\n"
        + next_loader
        + f"      scf.yield {next_values} : {yield_types}\n"
        + "    } else {\n"
        + f"      scf.yield {old_values} : {yield_types}\n"
        + "    }\n"
    )

    # Remove the old loader, retain its first barrier and compute body.
    body_tail = text[loader_stop:]
    yield_line_at = body_tail.index("    scf.yield %y0")
    yield_line_end = body_tail.index("\n", yield_line_at)
    old_yield = body_tail[yield_line_at:yield_line_end]
    extra_yields = ", " + ", ".join(f"%carry_{name}" for name in carried_order)
    extra_types = ", " + carried_types
    new_yield = old_yield.replace(" : ", extra_yields + " : ", 1) + extra_types
    body_tail = body_tail[:yield_line_at] + new_yield + body_tail[yield_line_end:]

    # Only the original sixteen accumulators escape the loop.
    result_types_end = header.index(") {\n")
    # The loop result list itself also needs names for carried results, even
    # though DCE will remove them after proving the final iteration.
    result_names_end = text.index(" = scf.for", loop_result)
    original_result_names = text[loop_result:result_names_end]
    extra_result_names = ", " + ", ".join(f"%unused_{name}" for name in carried_order)
    header = header.replace(original_result_names, original_result_names + extra_result_names, 1)

    generated = text[:loop_result] + initial + header + stores + prefetch + body_tail
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise ValueError("output was not created")


if __name__ == "__main__":
    main()
