#!/usr/bin/env python3
"""Derive a High gfx12 prefetch loop with the final K tile peeled."""

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


def prefix_definitions(region: str, prefix: str) -> str:
    return rename(region, {name: f"%{prefix}{name[1:]}" for name in definitions(region)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.input.read_text()

    loop_at = text.index("  %r0, %r1, %r2")
    header_end = text.index(" {\n", loop_at) + 3
    loader_start = text.index("    %linear0 = ", header_end)
    last_store = text.index("    vector.store %bv3", loader_start)
    loader_stop = text.index("\n", last_store) + 1
    loader = text[loader_start:loader_stop]
    compute_start = loader_stop
    yield_at = text.index("    scf.yield %y0", compute_start)
    yield_end = text.index("\n", yield_at) + 1
    loop_close = text.index("  }\n", yield_end) + len("  }\n")
    compute = text[compute_start:yield_at]
    epilog = text[loop_close:]

    order = ["av0", "bv0", "av1", "bv1", "av2", "bv2", "av3", "bv3"]
    vector_types = ", ".join("vector<8xf16>" for _ in order)

    initial = prefix_definitions(loader.replace("%k_base", "%zero"), "init_")
    initial = "\n".join(line for line in initial.splitlines() if "vector.store" not in line) + "\n"

    main_names = [f"%main_a{i}" for i in range(16)]
    tail_payload_names = [f"%tail_{name}" for name in order]
    result_names = ", ".join(main_names + tail_payload_names)

    old_header = text[loop_at:header_end]
    args_open = old_header.index("(", old_header.index("scf.for"))
    result_type_open = old_header.rindex(") -> (")
    iter_args = old_header[args_open + 1 : result_type_open]
    iter_args += ", " + ", ".join(
        f"%p_{name} = %init_{name} : vector<8xf16>" for name in order
    )
    original_result_types = old_header[
        result_type_open + len(") -> (") : old_header.rindex(") {\n")
    ]
    result_types = original_result_types + ", " + vector_types
    header = (
        f"  {result_names} = scf.for %k_base = [%zero to %k_main_end step %depth_u]"
        f"({iter_args}) -> ({result_types}) {{\n"
    )

    stores = "\n".join(line for line in loader.splitlines() if "vector.load" not in line) + "\n"
    stores = rename(stores, {f"%{name}": f"%p_{name}" for name in order})

    next_loader = prefix_definitions(loader.replace("%k_base", "%next_k"), "next_")
    next_loader = "\n".join(line for line in next_loader.splitlines() if "vector.store" not in line) + "\n"
    next_values = [f"%next_{name}" for name in order]
    yield_values = ", ".join([f"%y{i}" for i in range(16)] + next_values)
    yield_types = original_result_types + ", " + vector_types
    main_body = (
        stores
        + "    %next_k = index.add %k_base, %depth_u : index\n"
        + next_loader
        + compute
        + f"    scf.yield {yield_values} : {yield_types}\n"
        + "  }\n\n"
    )

    # Peel tile K-32. It consumes the final carried payload without issuing an
    # out-of-bounds K+32 request.
    tail_stores = prefix_definitions(loader.replace("%k_base", "%k_main_end"), "peel_")
    tail_stores = "\n".join(line for line in tail_stores.splitlines() if "vector.load" not in line) + "\n"
    tail_stores = rename(
        tail_stores,
        {f"%peel_{name}": f"%tail_{name}" for name in order},
    )

    tail_compute = prefix_definitions(compute, "peel_")
    tail_compute = rename(
        tail_compute,
        {f"%a{i}": f"%main_a{i}" for i in range(16)},
    )
    epilog = rename(epilog, {f"%r{i}": f"%peel_y{i}" for i in range(16)})

    before = text[:loop_at]
    before += "  %k_main_end = index.sub %k, %depth_u : index\n"
    generated = before + initial + header + main_body + tail_stores + tail_compute + epilog

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise ValueError("output was not created")


if __name__ == "__main__":
    main()
