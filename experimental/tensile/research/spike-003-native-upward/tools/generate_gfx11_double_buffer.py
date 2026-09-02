#!/usr/bin/env python3
"""Generate the full-tile gfx11 PGR2/PLR1 schedule anchor.

This is intentionally a generator instead of a hand-expanded 1,000-line Loom
file: the repeated load, LDS-transfer, fragment-load, and accumulator sites
must have distinct SSA names so their relative order remains reviewable in the
prepared Low form.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def emit_payload_load(prefix: str, k_base: str) -> list[str]:
    lines: list[str] = []
    for i in range(3):
        linear = "%thread" if i == 0 else f"%{prefix}_linear{i}"
        if i:
            lines.extend(
                [
                    f"    %{prefix}_delta{i} = index.mul %{['zero', 'one', 'two'][i]}, %load_stride : index",
                    f"    {linear} = index.add %thread, %{prefix}_delta{i} : index",
                ]
            )
        if i < 2:
            lines.extend(
                [
                    f"    %{prefix}_a_row_vec{i} = index.rem {linear}, %eight : index",
                    f"    %{prefix}_a_row0_{i} = index.mul %{prefix}_a_row_vec{i}, %eight : index",
                    f"    %{prefix}_a_row{i} = index.assume %{prefix}_a_row0_{i} [range(%{prefix}_a_row0_{i}, 0, 56)] : index",
                    f"    %{prefix}_a_k0_{i} = index.div {linear}, %eight : index",
                    f"    %{prefix}_a_kl{i} = index.assume %{prefix}_a_k0_{i} [range(%{prefix}_a_k0_{i}, 0, 31)] : index",
                    f"    %{prefix}_a_gr{i} = index.add %workgroup_m_base, %{prefix}_a_row{i} : index",
                    f"    %{prefix}_a_gk{i} = index.add {k_base}, %{prefix}_a_kl{i} : index",
                    f"    %{prefix}_av{i} = vector.load %a_load[%{prefix}_a_gk{i}, %{prefix}_a_gr{i}] : view<[%k]x[%m]xf16> -> vector<8xf16>",
                ]
            )
        lines.extend(
            [
                f"    %{prefix}_b_k_vec{i} = index.rem {linear}, %four : index",
                f"    %{prefix}_b_k0_{i} = index.mul %{prefix}_b_k_vec{i}, %eight : index",
                f"    %{prefix}_b_kl{i} = index.assume %{prefix}_b_k0_{i} [range(%{prefix}_b_k0_{i}, 0, 24)] : index",
                f"    %{prefix}_b_n0_{i} = index.div {linear}, %four : index",
                f"    %{prefix}_b_nl{i} = index.assume %{prefix}_b_n0_{i} [range(%{prefix}_b_n0_{i}, 0, 95)] : index",
                f"    %{prefix}_b_gn{i} = index.add %workgroup_n_base, %{prefix}_b_nl{i} : index",
                f"    %{prefix}_b_gk{i} = index.add {k_base}, %{prefix}_b_kl{i} : index",
                f"    %{prefix}_bv{i} = vector.load %b_load[%{prefix}_b_gn{i}, %{prefix}_b_gk{i}] : view<[%n]x[%k]xf16> -> vector<8xf16>",
            ]
        )
    return lines


def emit_payload_store(prefix: str, stage: int) -> list[str]:
    suffix = "" if stage == 0 else "_stage1"
    lines: list[str] = []
    for i in range(2):
        lines.append(
            f"    vector.store %{prefix}_av{i}, %a_lds_store{suffix}[%{prefix}_a_kl{i}, %{prefix}_a_row{i}] : vector<8xf16>, view<32x65xf16>"
        )
    for i in range(3):
        lines.append(
            f"    vector.store %{prefix}_bv{i}, %b_lds_store{suffix}[%{prefix}_b_nl{i}, %{prefix}_b_kl{i}] : vector<8xf16>, view<96x40xf16>"
        )
    return lines


def emit_compute(prefix: str, stage: int, acc_in: list[str]) -> tuple[list[str], list[str]]:
    suffix = "" if stage == 0 else "_stage1"
    lines: list[str] = []
    current = acc_in
    for depth, k_name in enumerate(("%zero", "%sixteen")):
        lhs: list[str] = []
        rhs: list[str] = []
        for m in range(2):
            mpos = "%wave_m_delta" if m == 0 else f"%{prefix}_m1_{depth}"
            if m:
                lines.append(
                    f"    {mpos} = index.add %wave_m_delta, %sixteen : index"
                )
            name = f"%{prefix}_lhs{depth}_{m}"
            lines.append(
                f"    {name} = vector.fragment.load<lhs> %a_lds{suffix}[{mpos}, {k_name}] shape [%sixteen, %sixteen] : view<64x32xf16, %a_lds_layout> -> vector<16xf16>"
            )
            lhs.append(name)
        for n in range(3):
            if n == 0:
                npos = "%wave_n_delta"
            else:
                npos = f"%{prefix}_n{n}_{depth}"
                delta = "%thirty_two" if n == 1 else "%sixty_four"
                lines.append(f"    {npos} = index.add %wave_n_delta, {delta} : index")
            name = f"%{prefix}_rhs{depth}_{n}"
            lines.append(
                f"    {name} = vector.fragment.load<rhs> %b_lds{suffix}[{k_name}, {npos}] shape [%sixteen, %sixteen] : view<32x96xf16, %b_lds_layout> -> vector<16xf16>"
            )
            rhs.append(name)
        next_acc: list[str] = []
        for m in range(2):
            for n in range(3):
                q = m * 3 + n
                name = f"%{prefix}_acc{depth}_{q}"
                lines.append(
                    f"    {name} = vector.mma {lhs[m]}, {rhs[n]}, {current[q]} : vector<16xf16>, vector<16xf16>, vector<8xf32>"
                )
                next_acc.append(name)
        current = next_acc
    return lines, current


def generate() -> str:
    lines = [
        "// gfx11-family FP16 full-tile schedule anchor.",
        "// Two LDS stages implement PGR2/PLR1: global loads for the next K32",
        "// tile overlap the current stage's twelve WMMAs, followed by five LDS",
        "// writes and one barrier. N must be a multiple of 96; tails are routed",
        "// to a separate JIT specialization so this hot loop has no edge paths.",
        "amdgpu.target<gfx11-generic> @gfx11 {subgroup_size = 32}",
        "",
        "config.decl @gemm.m : %value: index where [range(%value, 64, 65536), mul(%value, 64)]",
        "config.decl @gemm.n : %value: index where [range(%value, 96, 65472), mul(%value, 96)]",
        "config.decl @gemm.k : %value: index where [range(%value, 64, 65536), mul(%value, 64)]",
        "",
        "kernel.def target(@gfx11) @gemm_f16_f32_mt64x96x32_gfx11_pgr2() {",
        "  %m = config.get @gemm.m : index",
        "  %n = config.get @gemm.n : index",
        "  %macro_m = index.constant 64 : index",
        "  %macro_n = index.constant 96 : index",
        "  %wg_m = index.div %m, %macro_m : index",
        "  %wg_n = index.div %n, %macro_n : index",
        "  %wave_size = index.constant 32 : index",
        "  %waves = index.constant 4 : index",
        "  %one = index.constant 1 : index",
        "  kernel.launch.config workgroups(%wg_n, %wg_m, %one) workgroup_size(%wave_size, %waves, %one) : index",
        "} launch(%a_buffer: buffer, %b_buffer: buffer, %c_buffer: buffer, %d_buffer: buffer) {",
        "  %m = config.get @gemm.m : index",
        "  %n = config.get @gemm.n : index",
        "  %k = config.get @gemm.k : index",
        "  %base = index.constant 0 : offset",
        "  %zero = index.constant 0 : index",
        "  %one = index.constant 1 : index",
        "  %two = index.constant 2 : index",
        "  %four = index.constant 4 : index",
        "  %eight = index.constant 8 : index",
        "  %sixteen = index.constant 16 : index",
        "  %thirty_two = index.constant 32 : index",
        "  %sixty_four = index.constant 64 : index",
        "  %load_stride = index.constant 128 : index",
        "  %macro_m = index.constant 64 : index",
        "  %macro_n = index.constant 96 : index",
        "  %wave_m_extent = index.constant 32 : index",
        "  %wave_n_extent = index.constant 16 : index",
        "  %lds_bytes = index.constant 28288 : offset",
        "  %workgroup_n = kernel.workgroup.id<x> : index",
        "  %workgroup_m = kernel.workgroup.id<y> : index",
        "  %lane = kernel.workitem.id<x> : index",
        "  %wave = kernel.workitem.id<y> : index",
        "  %wave_thread = index.mul %wave, %thirty_two : index",
        "  %thread = index.add %wave_thread, %lane : index",
        "  %wave_m_id = index.rem %wave, %two : index",
        "  %wave_n_id = index.div %wave, %two : index",
        "  %workgroup_m_base = index.mul %workgroup_m, %macro_m : index",
        "  %workgroup_n_base = index.mul %workgroup_n, %macro_n : index",
        "  %wave_m_delta = index.mul %wave_m_id, %wave_m_extent : index",
        "  %wave_n_delta = index.mul %wave_n_id, %wave_n_extent : index",
        "  %wave_m_base = index.add %workgroup_m_base, %wave_m_delta : index",
        "  %wave_n_base = index.add %workgroup_n_base, %wave_n_delta : index",
        "  %a_global = buffer.assume.memory_space<global> %a_buffer : buffer",
        "  %b_global = buffer.assume.memory_space<global> %b_buffer : buffer",
        "  %d_global = buffer.assume.memory_space<global> %d_buffer : buffer",
        "  %a_aligned = buffer.assume.alignment %a_global {minimum_alignment = 16} : buffer",
        "  %b_aligned = buffer.assume.alignment %b_global {minimum_alignment = 16} : buffer",
        "  %d_aligned = buffer.assume.alignment %d_global {minimum_alignment = 16} : buffer",
        "  %a_load = buffer.view %a_aligned[%base] : buffer -> view<[%k]x[%m]xf16>",
        "  %b_load = buffer.view %b_aligned[%base] : buffer -> view<[%n]x[%k]xf16>",
        "  %d_layout = encoding.layout.strided [1, %m] : encoding<layout>",
        "  %d = buffer.view %d_aligned[%base] : buffer -> view<[%m]x[%n]xf16, %d_layout>",
        "  %a_lds_layout = encoding.layout.strided [1, 65] : encoding<layout>",
        "  %b_lds_layout = encoding.layout.strided [1, 40] : encoding<layout>",
        "  %lds = buffer.alloca<workgroup> align(16) %lds_bytes : buffer",
        "  %a_lds = buffer.view %lds[%base] : buffer -> view<64x32xf16, %a_lds_layout>",
        "  %a_lds_store = buffer.view %lds[%base] : buffer -> view<32x65xf16>",
        "  %b0_offset = index.constant 4224 : offset",
        "  %b_lds = buffer.view %lds[%b0_offset] : buffer -> view<32x96xf16, %b_lds_layout>",
        "  %b_lds_store = buffer.view %lds[%b0_offset] : buffer -> view<96x40xf16>",
        "  %a1_offset = index.constant 16384 : offset",
        "  %a_lds_stage1 = buffer.view %lds[%a1_offset] : buffer -> view<64x32xf16, %a_lds_layout>",
        "  %a_lds_store_stage1 = buffer.view %lds[%a1_offset] : buffer -> view<32x65xf16>",
        "  %b1_offset = index.constant 20608 : offset",
        "  %b_lds_stage1 = buffer.view %lds[%b1_offset] : buffer -> view<32x96xf16, %b_lds_layout>",
        "  %b_lds_store_stage1 = buffer.view %lds[%b1_offset] : buffer -> view<96x40xf16>",
        "  %zero_payload = vector.constant 0.0 : vector<8xf32>",
        "  %zero_acc = vector.fragment<init> %zero_payload shape [%sixteen, %sixteen] : vector<8xf32>",
    ]
    lines.extend(emit_payload_load("pro", "%zero"))
    lines.extend(emit_payload_store("pro", 0))
    lines.extend(
        [
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "  %last_pair = index.sub %k, %sixty_four : index",
            "  %loop_end = index.add %last_pair, %one : index",
            "  %r0, %r1, %r2, %r3, %r4, %r5 = scf.for %k_base = [%zero to %loop_end step %sixty_four](%a0 = %zero_acc : vector<8xf32>, %a1 = %zero_acc : vector<8xf32>, %a2 = %zero_acc : vector<8xf32>, %a3 = %zero_acc : vector<8xf32>, %a4 = %zero_acc : vector<8xf32>, %a5 = %zero_acc : vector<8xf32>) -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {",
            "    %odd_k = index.add %k_base, %thirty_two : index",
        ]
    )
    lines.extend(emit_payload_load("odd", "%odd_k"))
    even_lines, even_acc = emit_compute(
        "even", 0, [f"%a{i}" for i in range(6)]
    )
    lines.extend(even_lines)
    lines.extend(emit_payload_store("odd", 1))
    lines.extend(
        [
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    %candidate_next = index.add %k_base, %sixty_four : index",
            "    %has_next = index.cmp ult, %candidate_next, %k : index",
            "    %next_k = scf.select %has_next, %candidate_next, %k_base : index",
        ]
    )
    lines.extend(emit_payload_load("next", "%next_k"))
    odd_lines, odd_acc = emit_compute("oddc", 1, even_acc)
    lines.extend(odd_lines)
    lines.extend(emit_payload_store("next", 0))
    lines.extend(
        [
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    scf.yield " + ", ".join(odd_acc) + " : " + ", ".join(["vector<8xf32>"] * 6),
            "  }",
        ]
    )
    for m in range(2):
        mpos = "%wave_m_base" if m == 0 else "%store_m1"
        if m:
            lines.append("  %store_m1 = index.add %wave_m_base, %sixteen : index")
        for n in range(3):
            if n == 0:
                npos = "%wave_n_base"
            else:
                npos = f"%store_n{m}_{n}"
                delta = "%thirty_two" if n == 1 else "%sixty_four"
                lines.append(f"  {npos} = index.add %wave_n_base, {delta} : index")
            q = m * 3 + n
            lines.extend(
                [
                    f"  %out{q} = vector.fragment<result> %r{q} shape [%sixteen, %sixteen] : vector<8xf32>",
                    f"  vector.fragment.store<result> %out{q}, %d[{mpos}, {npos}] shape [%sixteen, %sixteen] : vector<8xf32>, view<[%m]x[%n]xf16, %d_layout>",
                ]
            )
    lines.extend(["  kernel.return", "}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)


if __name__ == "__main__":
    main()
