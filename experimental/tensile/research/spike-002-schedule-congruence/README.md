# Spike 002: Radeon BLAS schedule congruence

## Decision this spike must support

Determine whether Loom can express and reproduce the schedules that make the
best installed Radeon BLAS kernels fast. The result must distinguish a kernel
problem from a compiler problem: a merely correct GEMM or an unexplained timing
gap is not a successful result.

This spike deliberately emits a fully specialized HSACO for every request.
Provider integration, dispatch-cache design, and pre-emission Loom program-key
collapse are deferred. The eventual design invariant remains that exact shape
and epilogue specialization maps onto a small family of cached machine
programs.

## Completion states

Every row in [`basket.json`](basket.json) ends in exactly one state:

- `congruent`: the critical schedule graph matches the incumbent and the upper
  bootstrap confidence bound for the Loom/incumbent median ratio is at most
  1.05.
- `explained_delta`: every material instruction, memory, occupancy, or launch
  difference has been isolated and its performance effect measured.
- `compiler_blocker`: a minimal reproducer and concrete Loom compiler work
  items explain why schedule construction cannot proceed.

Schedule congruence means equivalent macro-tile/depth/wave ownership, dynamic
accelerated-operation count, global and LDS traffic, staging/barrier/wait
graph, result publication, and occupancy without spills. It does not require
byte-identical assembly.

## Execution order

1. Reconstruct gfx1201 FP16 solution 133222: MT128x128x32, workgroup
   `[32,4,1]`, four wave32 waves arranged 2x2, MI16x16x16, GRVWA/B 8,
   VectorWidthA/B 4, StoreVectorWidth 4, PGR2, WGM8, GSU1.
2. Exercise its square, skinny, and K-shallow structural variants.
3. Reconstruct gfx1100 FP16 solution 1675: MT64x96x32, workgroup `[32,4,1]`,
   four wave32 waves arranged 2x2, MI16x16x16, GRVWA/B 8, PGR2.
4. Reuse the staging and ownership motifs for accelerated BF16, I8, and
   gfx1201 FP8/BF8 contracts, splitting only where fragment or instruction
   contracts require it. Census I4 without inventing an incumbent corpus.
5. Only after gfx11 and gfx12 have conclusions, enable gfx906 far enough to
   reproduce a packed `v_dot2_f32_f16` packet and then the incumbent
   MT128x64x16, WG16x16, TT8x4 GEMM. Compiler changes are experimental evidence
   and work-item input, not proposed patches for immediate landing.

The complete basket is the stop condition. Investigation continues while a
row lacks one of the completion states above.

## Evidence required per row

Incumbent evidence:

- exact API request and exhaustive eligible-solution timings;
- selected runtime solution/index and router path;
- serialized solution, source YAML recipe, code object, and symbol;
- normalized native K-loop schedule and generated assembly when recoverable;
- supporting profiler counters where static evidence is ambiguous.

Loom evidence:

- exact source revision, target, config parameters, compiler invocation,
  HSACO hash, artifact manifest, and compile report;
- correctness against the same deterministic CPU reference;
- normalized native schedule, VGPR/SGPR/LDS/spill and occupancy evidence;
- at least 30 same-process, alternating Loom/incumbent timing pairs after a
  short exhaustive search; bootstrap confidence interval for the median ratio;
- device identity, clocks, temperature, runtime/library provenance, and all raw
  samples.

Generated code objects, disassemblies, and large raw traces live under the
ignored `artifacts/` directory. Compact JSON summaries, Loom motifs, analysis,
and minimal reproducers are tracked.

## Deliverables

- `basket.json`: stable request matrix and row status.
- `evidence-schema.json`: machine-readable minimum evidence contract.
- `loom/`: family-generic, config-specialized motifs.
- `results/`: compact incumbent and Loom evidence and final matrix.
- `gfx906-work-items.md`: target enablement findings and minimal reproducers.
- `findings.md`: conclusions, quantified deltas, and reusable schedule motifs.
- updates to `docs/design/loom-blas.md`, including alternatives considered.
