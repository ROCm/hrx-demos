# Compiler usability: gfx12 WMMA LHS fragment packing

## Summary

The original gfx12 FP16 High-level schedule gap is narrow enough to assign as a
compiler feature. The High Loom source asks for four `16x16` LHS fragments
from a strided A tile. Loom selects
`strided_d16_packed_b16_fragment_load` and emits 32
`ds_load_u16_d16`/`ds_load_u16_d16_hi` packets per K=16 half. The exact
TensileLite winner instead reads the same four fragments with eight
`ds_load_b64` packets and transposes/packs them with sixteen `v_perm_b32`
instructions.

The RHS path already has the desired shape: four `ds_load_b128` packets per
K=16 half. This is therefore not a general matrix-fragment, LDS, or WMMA
representability failure. It is one missing cross-packet LHS load plan for the
gfx12 FP16 WMMA fragment ABI.

A prepared-Low implementation of the exact alternate plan now exists and is
performance-congruent with the incumbent. This changes the issue from a
feasibility blocker into a raising/usability item: the compiler need not invent
the packetization, but High Loom or an inlined `low.invoke` microkernel must be
able to request and preserve it.

## Exact source request and actual lowering

The candidate uses the family target `gfx12-generic` and these four High Loom
operations per half:

```text
vector.fragment.load<lhs> %a_lds[%m0, %k] shape [16, 16]
vector.fragment.load<lhs> %a_lds[%m1, %k] shape [16, 16]
vector.fragment.load<lhs> %a_lds[%m2, %k] shape [16, 16]
vector.fragment.load<lhs> %a_lds[%m3, %k] shape [16, 16]
```

The complete source is
[`loom/gemm-f16-f32-mt128x128x32-gfx12-scalar-acc.loom`](loom/gemm-f16-f32-mt128x128x32-gfx12-scalar-acc.loom).
The retained K=32 lowering is
[`final-prepared-low.loom`](artifacts/gfx12-scalar-acc-k32-trace/final-prepared-low.loom),
and its compile report is
[`report.json`](artifacts/gfx12-scalar-acc-k32-trace/report.json).

The report records one source selection covering all eight LHS fragment
loads:

```json
{
  "source_op": "vector.fragment.load",
  "selection": "plan",
  "plan_key": "strided_d16_packed_b16_fragment_load",
  "selected_op_count": 8,
  "emitted_low_op_count": 90
}
```

For the full K=32 tile this becomes 64 low-half and 64 high-half D16 LDS
instructions. Per K=16 half that is 64 LDS instructions for A, versus eight
LDS instructions plus sixteen permutes in the incumbent.

## Native schedule to reproduce

Runtime solution 133309 and the regenerated source recipe agree on
MT128x128x32, WG32x4, MI16x16x16, wave tile 4x4, PGR2, TLDS1, and 16-element B
LDS padding. The normalized native interval is
[`gfx1201-incumbent-133309-loop.json`](results/gfx1201-incumbent-133309-loop.json).

For each K=16 half the relevant native pattern is:

```text
8 x ds_load_b64     A depth slices at offsets 0, 256, ..., 1792
4 x ds_load_b128    B fragments at offsets 0, 2560, 5120, 7680
16 x v_perm_b32     transpose/pack A into four WMMA LHS fragments
16 x v_wmma_f32_16x16x16_f16
```

The full steady-state iteration contains 24 LDS loads, 32 WMMAs, 11 waits,
two barrier packets, and 147 instructions. The original High lowering contains
72 LDS loads, 32 WMMAs, 82 waits, four barrier packets, and 609 instructions
per dynamic iteration. Full linear unrolling removes allocator spills and
some redundant scheduling work, but it deliberately preserves the same 72 LDS
loads per tile; its normalized whole-function schedule is
[`gfx1201-loom-high-scalar-acc-k-unroll-full-linear-flat.json`](schedules/gfx1201-loom-high-scalar-acc-k-unroll-full-linear-flat.json).

## Alternatives exercised

Changing the LDS write layout is not an adequate source-only workaround:

- A vectorized transpose/scatter store was rejected because
  `memory_access.vector_axis_stride` was not satisfied. The raw diagnostics
  are in
  [`gfx12-scalar-acc-a-transpose-scatter`](artifacts/gfx12-scalar-acc-a-transpose-scatter/).
- Manually scalarizing the scattered stores compiled with zero spills and
  produced correct results, but moved the inefficiency to the write side and
  measured about 85 microseconds. Its artifacts are in
  [`gfx12-scalar-acc-a-transpose-scalar-store`](artifacts/gfx12-scalar-acc-a-transpose-scalar-store/).
- Retaining contiguous 128-bit A writes and the current D16 fragment reads is
  the best source-level form measured so far.

## Required compiler contract

Add a gfx12-family FP16 WMMA LHS fragment-load plan in
`loom/src/loom/target/arch/amdgpu/lower/matrix_fragment_memory_packet.c` that:

1. recognizes the `16x16` LHS fragment ABI over the `[1, 128]` workgroup
   stride used by this macro-tile;
2. packetizes four adjacent LHS fragment requests jointly as eight B64 depth
   reads rather than packetizing each result register independently;
3. emits the two selector forms and sixteen `v_perm_b32` operations needed to
   build the four legal WMMA operands;
4. carries memory and cross-lane dependencies so normal Low scheduling may
   interleave the B reads, permutes, waits, and WMMAs as in the retained native
   interval; and
5. reports a distinct plan/descriptor key so the choice is visible in compile
   evidence and can remain config controlled.

The capability belongs to `gfx12-generic`, not `gfx1201`: nothing in the
pattern is specific to that exact processor beyond it being the current
physical witness. If the planner cannot safely discover the group of four
requests, expose an explicit High or reusable inlined-Low transpose-pack motif
with the same input/output fragment contract. The desired schedule above is
the acceptance test for either implementation.

## Acceptance test

The compiler work is complete for this case only when all of the following are
true:

- the K=32 prepared Low contains 16 B64 A reads and 32 permutes total, while B
  remains eight B128 reads;
- no private memory or materialized spill traffic is introduced;
- the existing full-shape differential and one-tile native `access` case pass;
- the dynamic K-loop retains the exact load/permute/WMMA dependency mechanism
  without requiring full K unrolling; and
- the paired candidate/oracle timing gate is rerun. A matching mnemonic count
  alone is not schedule or performance congruence.

## Low escape-hatch result

The final prepared-Low source is
[`low/gfx12-gemm-double-buffer-pgr2-ring-native-split-exact-fenced-svw4-uniform-bpad-scalar-saddr-ring-native-rhs-lhs-first.loom`](low/gfx12-gemm-double-buffer-pgr2-ring-native-split-exact-fenced-svw4-uniform-bpad-scalar-saddr-ring-native-rhs-lhs-first.loom).
It implements the desired B64 reads and permutes, remains spill-free, and
measures 30.0605 us against 28.800 us for solution 133309. The ratio is 1.0438
with 95% bootstrap interval `[1.0395, 1.0474]`, which passes this spike's 1.05
timing gate.

The residual compiler work is therefore to expose the recovered code as a
family-generic fragment/microkernel contract without losing its schedule. The
first raising experiment should use `low.func.def` plus `low.invoke` from the
High template. It must explicitly test whether config/value facts propagate
through the call boundary and whether inlining restores specialization when
they do not. The prepared-Low source remains the acceptance oracle.

Access sanitization is not closed for this form: `--pipeline=none` preserves
the Low schedule but does not instrument it. That separate issue is tracked in
[`compiler-usability-access-sanitizer.md`](compiler-usability-access-sanitizer.md).
