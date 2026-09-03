#!/usr/bin/env python3
"""Peel the final K32 tile from the structured gfx11 High loop.

The unpeeled source speculatively reloads and republishes its current tile on
the final iteration. Peeling removes those five global loads, five LDS stores,
one barrier, and the unused next-fragment reads while retaining High memory.
"""

from __future__ import annotations

from pathlib import Path
import argparse


LOOP_HEADER = "  %r0, %r1, %r2, %r3, %r4, %r5, %final_lhs0, %final_lhs1, %final_rhs0, %final_rhs1, %final_rhs2 = scf.for %k_base = [%zero to %k step %thirty_two]"
OUTPUT_ANCHOR = "  %out0 = vector.fragment<result> %r0"

TAIL = """  %final_stage_quotient = index.div %last_k, %thirty_two : index
  %final_stage = index.rem %final_stage_quotient, %two : index
  %final_stage_offset = index.scale %final_stage, %stage_stride : index, offset -> offset
  %final_b_stage_offset = index.add %final_stage_offset, %b0_offset : offset
  %final_a_lds = buffer.view %lds[%final_stage_offset] : buffer -> view<64x32xf16, %a_lds_layout>
  %final_b_lds = buffer.view %lds[%final_b_stage_offset] : buffer -> view<32x96xf16, %b_lds_layout>
  %tail_first_acc0 = vector.mma %final_lhs0, %final_rhs0, %loop_r0 : vector<16xf16>, vector<16xf16>, vector<8xf32>
  %tail_second_lhs0 = vector.fragment.load<lhs> %final_a_lds[%wave_m_delta, %sixteen] shape [%sixteen, %sixteen] : view<64x32xf16, %a_lds_layout> -> vector<16xf16>
  %tail_first_acc1 = vector.mma %final_lhs0, %final_rhs1, %loop_r1 : vector<16xf16>, vector<16xf16>, vector<8xf32>
  %tail_second_lhs1 = vector.fragment.load<lhs> %final_a_lds[%frag_m1, %sixteen] shape [%sixteen, %sixteen] : view<64x32xf16, %a_lds_layout> -> vector<16xf16>
  %tail_first_acc2 = vector.mma %final_lhs0, %final_rhs2, %loop_r2 : vector<16xf16>, vector<16xf16>, vector<8xf32>
  %tail_second_rhs0 = vector.fragment.load<rhs> %final_b_lds[%sixteen, %wave_n_delta] shape [%sixteen, %sixteen] : view<32x96xf16, %b_lds_layout> -> vector<16xf16>
  %tail_first_acc3 = vector.mma %final_lhs1, %final_rhs0, %loop_r3 : vector<16xf16>, vector<16xf16>, vector<8xf32>
  %tail_second_rhs1 = vector.fragment.load<rhs> %final_b_lds[%sixteen, %frag_n1] shape [%sixteen, %sixteen] : view<32x96xf16, %b_lds_layout> -> vector<16xf16>
  %tail_first_acc4 = vector.mma %final_lhs1, %final_rhs1, %loop_r4 : vector<16xf16>, vector<16xf16>, vector<8xf32>
  %tail_second_rhs2 = vector.fragment.load<rhs> %final_b_lds[%sixteen, %frag_n2] shape [%sixteen, %sixteen] : view<32x96xf16, %b_lds_layout> -> vector<16xf16>
  %tail_first_acc5 = vector.mma %final_lhs1, %final_rhs2, %loop_r5 : vector<16xf16>, vector<16xf16>, vector<8xf32>
  %r0 = vector.mma %tail_second_lhs0, %tail_second_rhs0, %tail_first_acc0 : vector<16xf16>, vector<16xf16>, vector<8xf32>
  %r1 = vector.mma %tail_second_lhs0, %tail_second_rhs1, %tail_first_acc1 : vector<16xf16>, vector<16xf16>, vector<8xf32>
  %r2 = vector.mma %tail_second_lhs0, %tail_second_rhs2, %tail_first_acc2 : vector<16xf16>, vector<16xf16>, vector<8xf32>
  %r3 = vector.mma %tail_second_lhs1, %tail_second_rhs0, %tail_first_acc3 : vector<16xf16>, vector<16xf16>, vector<8xf32>
  %r4 = vector.mma %tail_second_lhs1, %tail_second_rhs1, %tail_first_acc4 : vector<16xf16>, vector<16xf16>, vector<8xf32>
  %r5 = vector.mma %tail_second_lhs1, %tail_second_rhs2, %tail_first_acc5 : vector<16xf16>, vector<16xf16>, vector<8xf32>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    if source.count(LOOP_HEADER) != 1 or source.count(OUTPUT_ANCHOR) != 1:
        raise ValueError("expected one unpeeled K32 loop and one output anchor")
    replacement = (
        "  %last_k = index.sub %k, %thirty_two : index\n"
        + LOOP_HEADER.replace(
            "%r0, %r1, %r2, %r3, %r4, %r5,",
            "%loop_r0, %loop_r1, %loop_r2, %loop_r3, %loop_r4, %loop_r5,",
        ).replace("to %k step", "to %last_k step")
    )
    source = source.replace(LOOP_HEADER, replacement, 1)
    source = source.replace(OUTPUT_ANCHOR, TAIL + OUTPUT_ANCHOR, 1)
    args.output.write_text(source)


if __name__ == "__main__":
    main()
