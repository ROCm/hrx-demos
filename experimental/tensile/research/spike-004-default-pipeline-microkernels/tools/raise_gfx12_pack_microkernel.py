#!/usr/bin/env python3
"""Replace gfx12 High LHS fragment loads with B64 loads plus a Low pack helper.

This is intentionally strict and anchored to the retained Spike 003 High
double-buffer source. The generated exact-target source is an execution
witness while family-generic Low carrier rebinding remains a compiler contract
gap; it is not the final maintained target spelling.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TARGET = "amdgpu.target<gfx12-generic> @gfx12 {subgroup_size = 32}"
EXACT_TARGET = "amdgpu.target<gfx1201> @gfx12 {subgroup_size = 32}"
GROUP = re.compile(
    r"(?P<indent>    )%(?P<prefix>even_compute|odd_compute|tail30_compute|tail31_compute)_l(?P<first>[04]) = vector\.fragment\.load<lhs>[^\n]+\n"
    r"    %(?P=prefix)_l(?P<second>[15]) = vector\.fragment\.load<lhs>[^\n]+\n"
    r"    %(?P=prefix)_l(?P<third>[26]) = vector\.fragment\.load<lhs>[^\n]+\n"
    r"    %(?P=prefix)_l(?P<fourth>[37]) = vector\.fragment\.load<lhs>[^\n]+"
)


def replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected one {description}, found {count}")
    return source.replace(old, new)


def extract_pack_helper(source: str) -> str:
    start = source.index("low.func.def schedule(locked)")
    end_marker = "\n}\n\nlow.func.def schedule(locked)"
    end = source.index(end_marker, start) + 2
    helper = source[start:end]
    return helper.replace("(@target)", "(@gfx12)")


def replace_group(match: re.Match[str]) -> str:
    prefix = match.group("prefix")
    first = int(match.group("first"))
    expected = [first + i for i in range(4)]
    observed = [int(match.group(name)) for name in ("first", "second", "third", "fourth")]
    if observed != expected:
        raise ValueError(f"malformed fragment group {prefix}: {observed}")
    stage_view = "%a_lds_flat_stage1" if prefix in {"odd_compute", "tail31_compute"} else "%a_lds_flat"
    half_base = 2048 if first == 4 else 0
    offsets = [half_base + 128 * i for i in range(8)]
    lines: list[str] = []
    for i, offset in enumerate(offsets):
        index_name = f"%{prefix}_h{first // 4}_raw_index{i}"
        lines.append(
            f"    {index_name} = index.add %lhs_element_base, %lhs_offset_{offset} : index"
        )
        lines.append(
            f"    %{prefix}_h{first // 4}_raw{i} = vector.load {stage_view}[{index_name}] : view<4096xf16> -> vector<4xf16>"
        )
    operands = ", ".join(f"%{prefix}_h{first // 4}_raw{i}" for i in range(8))
    results = ", ".join(f"%{prefix}_l{i}_values" for i in expected)
    arg_types = ", ".join("vector<4xf16>" for _ in range(8))
    result_types = ", ".join("vector<8xf16>" for _ in range(4))
    lines.append(
        f"    {results} = low.invoke @pack_lhs_fragments({operands}) : ({arg_types}) -> ({result_types})"
    )
    for i in expected:
        lines.append(
            f"    %{prefix}_l{i} = vector.fragment<lhs> %{prefix}_l{i}_values shape [%sixteen, %sixteen] : vector<8xf16>"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Spike 003 High GEMM source")
    parser.add_argument("helper_source", type=Path, help="Spike 004 helper fixture")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    helper = extract_pack_helper(args.helper_source.read_text())
    source = replace_once(
        source, TARGET, f"{EXACT_TARGET}\n\n{helper}", "gfx12 target declaration"
    )
    source = replace_once(
        source,
        "  %lds_bytes = index.constant 51200 : offset",
        "  %lds_bytes = index.constant 51200 : offset\n"
        "  %lhs_stride_1024 = index.constant 1024 : index\n"
        + "\n".join(
            f"  %lhs_offset_{offset} = index.constant {offset} : index"
            for offset in [0, 128, 256, 384, 512, 640, 768, 896, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944]
        ),
        "LDS byte constant",
    )
    source = replace_once(
        source,
        "  %a_lds_store = buffer.view %lds[%base] : buffer -> view<32x128xf16>",
        "  %a_lds_store = buffer.view %lds[%base] : buffer -> view<32x128xf16>\n"
        "  %a_lds_flat = buffer.view %lds[%base] : buffer -> view<4096xf16>",
        "stage-zero A LDS view",
    )
    source = replace_once(
        source,
        "  %a_lds_store_stage1 = buffer.view %lds[%stage1_offset] : buffer -> view<32x128xf16>",
        "  %a_lds_store_stage1 = buffer.view %lds[%stage1_offset] : buffer -> view<32x128xf16>\n"
        "  %a_lds_flat_stage1 = buffer.view %lds[%stage1_offset] : buffer -> view<4096xf16>",
        "stage-one A LDS view",
    )
    source = replace_once(
        source,
        "  %wave_n_base = index.add %workgroup_n_base, %wave_n_delta : index",
        "  %wave_n_base = index.add %workgroup_n_base, %wave_n_delta : index\n"
        "  %lhs_lane_low = index.rem %lane, %sixteen : index\n"
        "  %lhs_lane_row = index.mul %lhs_lane_low, %four : index\n"
        "  %lhs_lane_high = index.div %lane, %sixteen : index\n"
        "  %lhs_lane_k = index.mul %lhs_lane_high, %lhs_stride_1024 : index\n"
        "  %lhs_wave_row = index.mul %wave_m_id, %sixty_four : index\n"
        "  %lhs_element_base0 = index.add %lhs_wave_row, %lhs_lane_row : index\n"
        "  %lhs_element_base = index.add %lhs_element_base0, %lhs_lane_k : index",
        "wave base calculation",
    )
    source, group_count = GROUP.subn(replace_group, source)
    if group_count != 8:
        raise ValueError(f"expected eight LHS K16 fragment groups, replaced {group_count}")
    args.output.write_text(source)


if __name__ == "__main__":
    main()
