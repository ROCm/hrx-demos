# Compiler usability: gfx11 FP16 publication

## Correct maintained source

The final motif converts six `vector<8xf32>` accumulators and performs 48
scalar High `view.store` operations. The address map is fully explicit and all
memory remains visible to Loom's access sanitizer. It passes the complete
nonuniform reference differential.

The direct source was necessary because the generic fragment-result route in
the High-only candidate performs 48 32-bit stores plus fragment repacking and
measures 53.811 us. The direct route reaches 45.341 us.

## Remaining lowering issue

The direct High form lowers each scalar `view.store` through a changing vector
address and emits 46 `s_waitcnt_vscnt vscnt(0)` drains. Attempts to preserve a
stable vector base with algebraically-zero dynamic expressions either
canonicalized away or produced the same address form. The final kernel is fast
enough despite this, so the issue is not a Spike 5 blocker, but it is likely to
matter for fused epilogues and smaller GEMMs.

The should-work source is the publication tail in
[`loom/gemm-f16-f32-mt64x96x32-gfx11-k32-unrolled-locked-native-pub.loom`](loom/gemm-f16-f32-mt64x96x32-gfx11-k32-unrolled-locked-native-pub.loom).
An improved lowering should retain one uniform scalar resource/base and encode
the per-lane/per-element displacement in the vector address or immediate where
legal, coalescing store waits according to true reuse hazards rather than
draining after almost every store.

## Semantic pitfall found by the fixture

The prepared-Low native oracle's accumulator/store map cannot be pasted into a
High fragment kernel. Uniform values falsely passed. The preserved row witness
proved that High fragment groups use `wave_m*32 + {0,16}`, whereas the Low
oracle's fixed registers used `wave_m*16 + {0,32}`. Any publication transform
or report recommendation must reason from the High fragment contract, not raw
register numbers alone.

## Compile-report request

Add a grouped publication diagnostic when several stores share a source root
but repeatedly mutate/recreate the address base. Report:

- number and width of publication stores;
- scalar versus vector base evolution;
- `vscnt(0)` waits attributed to address/base reuse;
- maximum stores between drains; and
- the originating fragment/publication operation when available.

This would have made the epilogue bottleneck visible without manually counting
assembly waits and would accumulate a reusable rule in `loom-compile-report`.
