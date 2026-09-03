# Spike 005: gfx11 structured schedule recovery

## Outcome

The maintained gfx11 FP16 motif now passes the performance gate at the target
`M=1024, N=960, K=1024` cell. Five default-pipeline Loom runs have a median
of run medians of **45.341 us**; five public hipBLASLt solution-1675 runs have
**44.601 us**. The Loom/vendor ratio is 1.0166 with an independent block
bootstrap 95% interval of `[1.0002, 1.0221]`, below the predeclared 1.05 upper
bound.

This is not a prepared-Low or `--pipeline=none` result. The maintained source
targets `gfx11-generic`, owns all global/LDS access in High Loom, uses ordinary
structured control flow, and invokes one locked, register-only Low WMMA
microkernel. It compiles and executes through Loom's default pipeline.

Correctness is established by a nonuniform minimum-tile reference check, an
exact row-address witness, a complete 983,040-element nonuniform reference
differential at the performance shape, and Loom's native access sanitizer.
The stronger checks rejected an initially fast but permuted publication map;
that failure and correction are retained rather than hidden.

## Maintained motif

[`loom/gemm-f16-f32-mt64x96x32-gfx11-k32-unrolled-locked-native-pub.loom`](loom/gemm-f16-f32-mt64x96x32-gfx11-k32-unrolled-locked-native-pub.loom)
contains:

- a `64x96x32` K-ring with PGR2/PLR1-style global/LDS overlap;
- exact-shape full unrolling and a peeled final K32 tile;
- High `vector.load`, `view.store`, LDS views, fragment loads, and barriers;
- a straight-line locked Low helper containing only
  `v_wmma_f32_16x16x16_f16`;
- source WMMA order `0,3,1,4,2,5` in each loop half; and
- direct all-lane FP16 publication with the High-fragment row mapping.

The final artifact has 144 VGPRs, 22 SGPRs, 28,288 bytes LDS, no private
memory or spills, 384 WMMAs, 160 global loads, 48 global stores, 32 barriers,
and one branch. Full unrolling expands it to 39,104 code bytes. Bytecode link,
default compilation, and in-memory HSACO emission take 40.707 ms median over
30 optimized-host samples. That is acceptable only under the design invariant
that many MNK requests collapse to a small artifact set; this spike does not
establish that cardinality.

## What changed the result

The key finding is specific, not “the compiler cannot schedule the loop.” The
wait planner relocates loop-carried SSA dependencies to the loop entry and
emits a target-count-zero wait. For this ring it drains all 38 outstanding LDS
reads at each backedge. Exact-shape unrolling removes that backedge contract
and exposes 95 useful partial waits, moving the High-only candidate from about
61.6 us to 53.8 us. A locked register-only WMMA helper then retains the useful
LDS-read/WMMA wavefront and direct publication removes the expensive generic
fragment epilogue, reaching the final 45.3 us result.

This is a viable maintained form, but not the desired final compiler behavior.
The compiler handoff asks for a structured-loop solution that preserves legal
cross-iteration outstanding work without requiring wholesale unrolling.

## Directory map

- [`experiment-log.md`](experiment-log.md) — ordered hypothesis/result ladder,
  including failed approaches.
- [`source-lineage.md`](source-lineage.md) — exact derivation of each retained
  source and why the High/Low boundary is placed where it is.
- [`reproduce.md`](reproduce.md) — default-pipeline correctness, sanitizer,
  performance, and bytecode JIT commands.
- [`experiments/`](experiments/) — focused interpretation of the loop, helper,
  and publication investigations.
- [`compiler-usability-loop-carried-waits.md`](compiler-usability-loop-carried-waits.md)
  — should-work source and acceptance contract for the wait planner.
- [`compiler-usability-publication.md`](compiler-usability-publication.md) —
  High publication lowering and compile-report opportunities.
- [`loom/`](loom/) — the structured ring, High-only accepted intermediate,
  final motif, and nonuniform validation fixture.
- [`tools/`](tools/) — fail-fast source transforms that reproduce the ladder.
- [`results/`](results/) — compact timing, resource, JIT, and validation facts.

Generated HSACOs, full 16 MiB compile reports, profiles, and raw artifact
bundles remain ignored. The compact records retain the facts needed to detect
regression without pretending that generated binaries are stable source.

## Scope boundary

This spike closes one gfx11 FP16 interior cell. It does not establish edge
handling, alpha/beta, bias, other accelerated types, broad MNK coverage,
artifact-key collapse, provider integration, or gfx906 enablement. No HRX
source change was needed, so the conditional gfx1201 compiler-regression run
was not triggered.

## Alternatives Considered

- Keeping the exact prepared-Low oracle as the maintained kernel was rejected:
  it bypasses most of Loom and does not satisfy the default-pipeline contract.
- Treating full unrolling as proof that loop scheduling is solved was rejected:
  it is a useful exact-shape specialization and a diagnostic workaround, but
  its code-size/JIT cost makes artifact cardinality an explicit follow-up.
- Retaining Loom's generic fragment publication is correct and simpler, but it
  leaves roughly 8.5 us at this cell and misses the performance gate.
- Copying the prepared-Low direct-store map verbatim was rejected by the
  nonuniform row witness. High fragment roles require a different row-group
  assignment.
