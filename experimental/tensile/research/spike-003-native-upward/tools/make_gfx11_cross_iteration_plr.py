#!/usr/bin/env python3
"""Rewrite the exact-LDS gfx11 motif with cross-iteration PLR1 operands.

The incumbent carries the first K16 fragment bank across its K32 backedge.
While those six WMMAs execute it reads the second bank, and while the second
bank executes it reads the first bank for the next iteration.  This source
keeps a K64 outer loop (two static LDS stages) but expresses the same carried
operand lifetime.  Each High fragment read is still atomic; a later Low cut
can split its packets to reproduce the exact instruction-by-instruction braid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from generate_gfx11_double_buffer import emit_payload_load, emit_payload_store


F16_FRAGMENT = "vector<16xf16>"
F32_ACC = "vector<8xf32>"


def emit_fragments(prefix: str, stage: int, depth: int) -> list[str]:
    suffix = "" if stage == 0 else "_stage1"
    k_pos = "%zero" if depth == 0 else "%sixteen"
    return [
        f"    %{prefix}_lhs0 = vector.fragment.load<lhs> %a_lds{suffix}[%wave_m_delta, {k_pos}] shape [%sixteen, %sixteen] : view<64x32xf16, %a_lds_layout> -> {F16_FRAGMENT}",
        f"    %{prefix}_lhs1 = vector.fragment.load<lhs> %a_lds{suffix}[%frag_m1, {k_pos}] shape [%sixteen, %sixteen] : view<64x32xf16, %a_lds_layout> -> {F16_FRAGMENT}",
        f"    %{prefix}_rhs0 = vector.fragment.load<rhs> %b_lds{suffix}[{k_pos}, %wave_n_delta] shape [%sixteen, %sixteen] : view<32x96xf16, %b_lds_layout> -> {F16_FRAGMENT}",
        f"    %{prefix}_rhs1 = vector.fragment.load<rhs> %b_lds{suffix}[{k_pos}, %frag_n1] shape [%sixteen, %sixteen] : view<32x96xf16, %b_lds_layout> -> {F16_FRAGMENT}",
        f"    %{prefix}_rhs2 = vector.fragment.load<rhs> %b_lds{suffix}[{k_pos}, %frag_n2] shape [%sixteen, %sixteen] : view<32x96xf16, %b_lds_layout> -> {F16_FRAGMENT}",
    ]


def emit_braided_compute(
    prefix: str,
    lhs: tuple[str, str],
    rhs: tuple[str, str, str],
    acc: list[str],
    interleaved_fragments: list[str],
) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    outputs: list[str] = []
    for q in range(6):
        m, n = divmod(q, 3)
        output = f"%{prefix}_acc{q}"
        lines.append(
            f"    {output} = vector.mma {lhs[m]}, {rhs[n]}, {acc[q]} : {F16_FRAGMENT}, {F16_FRAGMENT}, {F32_ACC}"
        )
        outputs.append(output)
        if q < len(interleaved_fragments):
            lines.append(interleaved_fragments[q])
    return lines, outputs


def rewrite(source: str, *, unroll: str | None = None) -> str:
    start_marker = "  %last_pair = index.sub %k, %sixty_four : index\n"
    start = source.index(start_marker)
    yield_marker = "    scf.yield %oddc_acc1_0"
    old_yield = source.index(yield_marker, start)
    end = source.index("  }\n", old_yield) + len("  }\n")

    lines = [
        "  %frag_m1 = index.add %wave_m_delta, %sixteen : index",
        "  %frag_n1 = index.add %wave_n_delta, %thirty_two : index",
        "  %frag_n2 = index.add %wave_n_delta, %sixty_four : index",
    ]
    initial = [line[4:] for line in emit_fragments("initial", 0, 0)]
    lines.extend(initial)
    lines.extend(
        [
            "  %last_pair = index.sub %k, %sixty_four : index",
            "  %loop_end = index.add %last_pair, %one : index",
        ]
    )

    result_names = [f"%r{i}" for i in range(6)] + [
        "%final_lhs0",
        "%final_lhs1",
        "%final_rhs0",
        "%final_rhs1",
        "%final_rhs2",
    ]
    iter_args = [f"%a{i} = %zero_acc : {F32_ACC}" for i in range(6)] + [
        f"%cur_lhs0 = %initial_lhs0 : {F16_FRAGMENT}",
        f"%cur_lhs1 = %initial_lhs1 : {F16_FRAGMENT}",
        f"%cur_rhs0 = %initial_rhs0 : {F16_FRAGMENT}",
        f"%cur_rhs1 = %initial_rhs1 : {F16_FRAGMENT}",
        f"%cur_rhs2 = %initial_rhs2 : {F16_FRAGMENT}",
    ]
    result_types = [F32_ACC] * 6 + [F16_FRAGMENT] * 5
    lines.append(
        "  "
        + ", ".join(result_names)
        + " = scf.for %k_base = [%zero to %loop_end step %sixty_four]("
        + ", ".join(iter_args)
        + ") -> ("
        + ", ".join(result_types)
        + (") {" if unroll is None else f") unroll{unroll} {{")
    )
    lines.append("    %odd_k = index.add %k_base, %thirty_two : index")
    lines.extend(emit_payload_load("odd", "%odd_k"))

    even_second = emit_fragments("even_second", 0, 1)
    first_lines, first_acc = emit_braided_compute(
        "even_first",
        ("%cur_lhs0", "%cur_lhs1"),
        ("%cur_rhs0", "%cur_rhs1", "%cur_rhs2"),
        [f"%a{i}" for i in range(6)],
        even_second,
    )
    lines.extend(first_lines)
    lines.extend(emit_payload_store("odd", 1))
    lines.extend(
        [
            "    %candidate_next = index.add %k_base, %sixty_four : index",
            "    %has_next = index.cmp ult, %candidate_next, %k : index",
            "    %next_k = scf.select %has_next, %candidate_next, %k_base : index",
        ]
    )
    lines.extend(emit_payload_load("next", "%next_k"))
    lines.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")

    odd_first = emit_fragments("odd_first", 1, 0)
    second_lines, second_acc = emit_braided_compute(
        "even_second",
        ("%even_second_lhs0", "%even_second_lhs1"),
        ("%even_second_rhs0", "%even_second_rhs1", "%even_second_rhs2"),
        first_acc,
        odd_first,
    )
    lines.extend(second_lines)

    odd_second = emit_fragments("odd_second", 1, 1)
    third_lines, third_acc = emit_braided_compute(
        "odd_first",
        ("%odd_first_lhs0", "%odd_first_lhs1"),
        ("%odd_first_rhs0", "%odd_first_rhs1", "%odd_first_rhs2"),
        second_acc,
        odd_second,
    )
    lines.extend(third_lines)
    lines.extend(emit_payload_store("next", 0))
    lines.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")

    next_first = emit_fragments("next_first", 0, 0)
    fourth_lines, fourth_acc = emit_braided_compute(
        "odd_second",
        ("%odd_second_lhs0", "%odd_second_lhs1"),
        ("%odd_second_rhs0", "%odd_second_rhs1", "%odd_second_rhs2"),
        third_acc,
        next_first,
    )
    lines.extend(fourth_lines)
    yields = fourth_acc + [
        "%next_first_lhs0",
        "%next_first_lhs1",
        "%next_first_rhs0",
        "%next_first_rhs1",
        "%next_first_rhs2",
    ]
    lines.append("    scf.yield " + ", ".join(yields) + " : " + ", ".join(result_types))
    lines.append("  }")
    return source[:start] + "\n".join(lines) + "\n" + source[end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    unroll_group = parser.add_mutually_exclusive_group()
    unroll_group.add_argument(
        "--unroll",
        action="store_const",
        const="",
        dest="unroll",
        help="request full compile-time unrolling of the K64 outer loop",
    )
    unroll_group.add_argument(
        "--unroll-factor",
        choices=("two", "four", "eight"),
        dest="unroll_factor",
        help="strip-mine the K64 loop by the named in-scope index constant",
    )
    args = parser.parse_args()
    unroll = args.unroll
    if args.unroll_factor is not None:
        unroll = f"(%{args.unroll_factor})"
    args.output.write_text(rewrite(args.input.read_text(), unroll=unroll))


if __name__ == "__main__":
    main()
