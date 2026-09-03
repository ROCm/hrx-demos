#!/usr/bin/env python3
"""Rewrite the Spike 4 gfx11 K64 source loop as a High Loom K32 LDS ring.

The resulting source retains all global and LDS memory operations in High
Loom. Two explicitly-strided LDS stages are selected by a scalar ring index;
the loop carries only the six accumulators and five first-K16 fragments.
Every replacement is anchored to the checked Spike 4 source so drift fails
instead of silently producing a different experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path


OLD_LDS = """  %a_lds_layout = encoding.layout.strided [1, 65] : encoding<layout>
  %b_lds_layout = encoding.layout.strided [1, 40] : encoding<layout>
  %lds = buffer.alloca<workgroup> align(16) %lds_bytes : buffer
  %a_lds = buffer.view %lds[%base] : buffer -> view<64x32xf16, %a_lds_layout>
  %a_lds_store = buffer.view %lds[%base] : buffer -> view<32x65xf16>
  %b0_offset = index.constant 4224 : offset
  %b_lds = buffer.view %lds[%b0_offset] : buffer -> view<32x96xf16, %b_lds_layout>
  %b_lds_store = buffer.view %lds[%b0_offset] : buffer -> view<96x40xf16>
  %a1_offset = index.constant 16384 : offset
  %a_lds_stage1 = buffer.view %lds[%a1_offset] : buffer -> view<64x32xf16, %a_lds_layout>
  %a_lds_store_stage1 = buffer.view %lds[%a1_offset] : buffer -> view<32x65xf16>
  %b1_offset = index.constant 20608 : offset
  %b_lds_stage1 = buffer.view %lds[%b1_offset] : buffer -> view<32x96xf16, %b_lds_layout>
  %b_lds_store_stage1 = buffer.view %lds[%b1_offset] : buffer -> view<96x40xf16>
"""

NEW_LDS = """  %a_lds_layout = encoding.layout.strided [1, 65] : encoding<layout>
  %b_lds_layout = encoding.layout.strided [1, 40] : encoding<layout>
  %lds = buffer.alloca<workgroup> align(16) %lds_bytes : buffer
  %a_lds = buffer.view %lds[%base] : buffer -> view<64x32xf16, %a_lds_layout>
  %a_lds_store = buffer.view %lds[%base] : buffer -> view<32x65xf16>
  %b0_offset = index.constant 4224 : offset
  %b_lds = buffer.view %lds[%b0_offset] : buffer -> view<32x96xf16, %b_lds_layout>
  %b_lds_store = buffer.view %lds[%b0_offset] : buffer -> view<96x40xf16>
  %stage_stride = index.constant 16384 : offset
"""

LOOP_START = "  %last_pair = index.sub %k, %sixty_four : index\n"
LOOP_STOP = "  %out0 = vector.fragment<result> %r0"


K32_LOOP = """  %r0, %r1, %r2, %r3, %r4, %r5, %final_lhs0, %final_lhs1, %final_rhs0, %final_rhs1, %final_rhs2 = scf.for %k_base = [%zero to %k step %thirty_two](%a0 = %zero_acc : vector<8xf32>, %a1 = %zero_acc : vector<8xf32>, %a2 = %zero_acc : vector<8xf32>, %a3 = %zero_acc : vector<8xf32>, %a4 = %zero_acc : vector<8xf32>, %a5 = %zero_acc : vector<8xf32>, %cur_lhs0 = %initial_lhs0 : vector<16xf16>, %cur_lhs1 = %initial_lhs1 : vector<16xf16>, %cur_rhs0 = %initial_rhs0 : vector<16xf16>, %cur_rhs1 = %initial_rhs1 : vector<16xf16>, %cur_rhs2 = %initial_rhs2 : vector<16xf16>) -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<16xf16>, vector<16xf16>, vector<16xf16>, vector<16xf16>, vector<16xf16>) {
    %current_stage_quotient = index.div %k_base, %thirty_two : index
    %current_stage = index.rem %current_stage_quotient, %two : index
    %current_stage_offset = index.scale %current_stage, %stage_stride : index, offset -> offset
    %current_b_stage_offset = index.add %current_stage_offset, %b0_offset : offset
    %current_a_lds = buffer.view %lds[%current_stage_offset] : buffer -> view<64x32xf16, %a_lds_layout>
    %current_b_lds = buffer.view %lds[%current_b_stage_offset] : buffer -> view<32x96xf16, %b_lds_layout>
    %candidate_next = index.add %k_base, %thirty_two : index
    %has_next = index.cmp ult, %candidate_next, %k : index
    %next_k = scf.select %has_next, %candidate_next, %k_base : index
    %next_stage_quotient = index.div %candidate_next, %thirty_two : index
    %next_stage = index.rem %next_stage_quotient, %two : index
    %next_stage_offset = index.scale %next_stage, %stage_stride : index, offset -> offset
    %next_b_stage_offset = index.add %next_stage_offset, %b0_offset : offset
    %next_a_lds = buffer.view %lds[%next_stage_offset] : buffer -> view<64x32xf16, %a_lds_layout>
    %next_a_lds_store = buffer.view %lds[%next_stage_offset] : buffer -> view<32x65xf16>
    %next_b_lds = buffer.view %lds[%next_b_stage_offset] : buffer -> view<32x96xf16, %b_lds_layout>
    %next_b_lds_store = buffer.view %lds[%next_b_stage_offset] : buffer -> view<96x40xf16>
    %next_a_gk0 = index.add %next_k, %pro_a_kl0 : index
    %next_av0 = vector.load %a_load[%next_a_gk0, %pro_a_gr0] : view<[%k]x[%m]xf16> -> vector<8xf16>
    %next_b_gk0 = index.add %next_k, %pro_b_kl0 : index
    %next_bv0 = vector.load %b_load[%pro_b_gn0, %next_b_gk0] : view<[%n]x[%k]xf16> -> vector<8xf16>
    %next_a_gk1 = index.add %next_k, %pro_a_kl1 : index
    %next_av1 = vector.load %a_load[%next_a_gk1, %pro_a_gr1] : view<[%k]x[%m]xf16> -> vector<8xf16>
    %next_b_gk1 = index.add %next_k, %pro_b_kl1 : index
    %next_bv1 = vector.load %b_load[%pro_b_gn1, %next_b_gk1] : view<[%n]x[%k]xf16> -> vector<8xf16>
    %next_b_gk2 = index.add %next_k, %pro_b_kl2 : index
    %next_bv2 = vector.load %b_load[%pro_b_gn2, %next_b_gk2] : view<[%n]x[%k]xf16> -> vector<8xf16>
    %first_acc0 = vector.mma %cur_lhs0, %cur_rhs0, %a0 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %second_lhs0 = vector.fragment.load<lhs> %current_a_lds[%wave_m_delta, %sixteen] shape [%sixteen, %sixteen] : view<64x32xf16, %a_lds_layout> -> vector<16xf16>
    %first_acc1 = vector.mma %cur_lhs0, %cur_rhs1, %a1 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %second_lhs1 = vector.fragment.load<lhs> %current_a_lds[%frag_m1, %sixteen] shape [%sixteen, %sixteen] : view<64x32xf16, %a_lds_layout> -> vector<16xf16>
    %first_acc2 = vector.mma %cur_lhs0, %cur_rhs2, %a2 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %second_rhs0 = vector.fragment.load<rhs> %current_b_lds[%sixteen, %wave_n_delta] shape [%sixteen, %sixteen] : view<32x96xf16, %b_lds_layout> -> vector<16xf16>
    %first_acc3 = vector.mma %cur_lhs1, %cur_rhs0, %a3 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %second_rhs1 = vector.fragment.load<rhs> %current_b_lds[%sixteen, %frag_n1] shape [%sixteen, %sixteen] : view<32x96xf16, %b_lds_layout> -> vector<16xf16>
    %first_acc4 = vector.mma %cur_lhs1, %cur_rhs1, %a4 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %second_rhs2 = vector.fragment.load<rhs> %current_b_lds[%sixteen, %frag_n2] shape [%sixteen, %sixteen] : view<32x96xf16, %b_lds_layout> -> vector<16xf16>
    %first_acc5 = vector.mma %cur_lhs1, %cur_rhs2, %a5 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    vector.store %next_av0, %next_a_lds_store[%pro_a_kl0, %pro_a_row0] : vector<8xf16>, view<32x65xf16>
    vector.store %next_av1, %next_a_lds_store[%pro_a_kl1, %pro_a_row1] : vector<8xf16>, view<32x65xf16>
    vector.store %next_bv0, %next_b_lds_store[%pro_b_nl0, %pro_b_kl0] : vector<8xf16>, view<96x40xf16>
    vector.store %next_bv1, %next_b_lds_store[%pro_b_nl1, %pro_b_kl1] : vector<8xf16>, view<96x40xf16>
    vector.store %next_bv2, %next_b_lds_store[%pro_b_nl2, %pro_b_kl2] : vector<8xf16>, view<96x40xf16>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %second_acc0 = vector.mma %second_lhs0, %second_rhs0, %first_acc0 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %next_first_lhs0 = vector.fragment.load<lhs> %next_a_lds[%wave_m_delta, %zero] shape [%sixteen, %sixteen] : view<64x32xf16, %a_lds_layout> -> vector<16xf16>
    %second_acc1 = vector.mma %second_lhs0, %second_rhs1, %first_acc1 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %next_first_lhs1 = vector.fragment.load<lhs> %next_a_lds[%frag_m1, %zero] shape [%sixteen, %sixteen] : view<64x32xf16, %a_lds_layout> -> vector<16xf16>
    %second_acc2 = vector.mma %second_lhs0, %second_rhs2, %first_acc2 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %next_first_rhs0 = vector.fragment.load<rhs> %next_b_lds[%zero, %wave_n_delta] shape [%sixteen, %sixteen] : view<32x96xf16, %b_lds_layout> -> vector<16xf16>
    %second_acc3 = vector.mma %second_lhs1, %second_rhs0, %first_acc3 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %next_first_rhs1 = vector.fragment.load<rhs> %next_b_lds[%zero, %frag_n1] shape [%sixteen, %sixteen] : view<32x96xf16, %b_lds_layout> -> vector<16xf16>
    %second_acc4 = vector.mma %second_lhs1, %second_rhs1, %first_acc4 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %next_first_rhs2 = vector.fragment.load<rhs> %next_b_lds[%zero, %frag_n2] shape [%sixteen, %sixteen] : view<32x96xf16, %b_lds_layout> -> vector<16xf16>
    %second_acc5 = vector.mma %second_lhs1, %second_rhs2, %first_acc5 : vector<16xf16>, vector<16xf16>, vector<8xf32>
    scf.yield %second_acc0, %second_acc1, %second_acc2, %second_acc3, %second_acc4, %second_acc5, %next_first_lhs0, %next_first_lhs1, %next_first_rhs0, %next_first_rhs1, %next_first_rhs2 : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<16xf16>, vector<16xf16>, vector<16xf16>, vector<16xf16>, vector<16xf16>
  }
"""


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected one {label}, found {count}")
    return source.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    source = replace_once(
        source,
        "amdgpu.target<gfx1100> @gfx11",
        "amdgpu.target<gfx11-generic> @gfx11",
        "gfx11 family target",
    )
    source = replace_once(source, OLD_LDS, NEW_LDS, "LDS declaration block")
    for old, new in (
    ):
        source = source.replace(old, new)

    start = source.find(LOOP_START)
    stop = source.find(LOOP_STOP, start)
    if start < 0 or stop < 0:
        raise ValueError("could not isolate the K64 loop")
    source = source[:start] + K32_LOOP + source[stop:]
    args.output.write_text(source)


if __name__ == "__main__":
    main()
