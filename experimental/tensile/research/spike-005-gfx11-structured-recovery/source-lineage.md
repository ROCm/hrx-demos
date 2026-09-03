# Source lineage

The final source is derived mechanically from the checked Spike 4 High source.
Run `loom-format --in-place` after each chain when comparing byte identity.

1. `make_gfx11_high_k32_ring.py` transforms
   `../spike-004-default-pipeline-microkernels/loom/gemm-f16-f32-mt64x96x32-gfx11-high-exact.loom`
   into `loom/gemm-f16-f32-mt64x96x32-gfx11-high-k32-ring.loom`.
2. `add_k32_unroll_policy.py --schedule linear` adds exact-shape full unrolling.
3. `reorder_gfx11_mma_wavefront.py` changes each cyclic six-WMMA group from
   `0,1,2,3,4,5` to `0,3,1,4,2,5` while leaving fragment loads in place.
4. `peel_gfx11_high_k32_tail.py` removes the speculative final publication of
   the ring and appends one straight-line K32 tail. These steps produce
   `loom/gemm-f16-f32-mt64x96x32-gfx11-high-k32-unrolled-high-only.loom`.
5. Spike 3's `use_gfx11_native_high_epilogue.py` swaps the symmetric WMMA
   operands and substitutes direct scalar FP16 publication.
6. `use_gfx11_wmma_helper.py --schedule locked --expected-count 24` raises all
   WMMAs through one register-only Low function. It does not move memory into
   Low.
7. `fix_gfx11_high_publication_rows.py` adapts the prepared-Low store mapping
   to High fragment ownership. This produces the maintained final source.
8. `make_validation_fixture.py` replaces the cheap benchmark check with the
   minimum/full nonuniform reference cases and exact row witness.

The ordering of steps 3 and 4 is intentional: only the recurrent groups use
the oracle-derived issue wavefront; the peeled tail retains its straightforward
dependency order. Every transform checks exact anchors and fails on drift.

## High/Low ownership boundary

High Loom owns dimensions, target specialization, launch, buffers, global
loads/stores, LDS allocation/views, barriers, fragments, loop structure, and
the epilogue address formulas. Low owns exactly one legal instruction:
`v_wmma_f32_16x16x16_f16`. This keeps Loom's access sanitizer authoritative
for every memory operation and makes the helper independently replaceable
once High schedule controls can retain the desired issue wavefront.
