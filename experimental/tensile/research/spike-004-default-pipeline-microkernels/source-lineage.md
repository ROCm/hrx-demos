# Source lineage and motif map

This map identifies where each Spike 4 structure came from. The native and
Tensile/TensileLite artifacts are schedule oracles, not source to transcribe
verbatim. Spike 3 retains the complete archaeology and the unsuccessful steps;
Spike 4 keeps only the smallest source needed to test default-pipeline
composition.

## gfx12 / gfx1201

The runtime route for FP16 `1024x1024x1024` selects hipBLASLt solution
`133309`, local TensileLite YAML solution `91`. Its serialized name and native
loop establish MT128x128x32, WG32x4, wave tile 4x4, wave32
`v_wmma_f32_16x16x16_f16`, PGR2, TLDS1, store width four, and a 51,200-byte
padded double-buffered LDS allocation. The retained route record and normalized
loop are in Spike 3:

- `../spike-003-native-upward/results/gfx1201-solution-133309-runtime-record.json`;
- `../spike-003-native-upward/results/gfx1201-incumbent-133309-loop.json`;
- the “Gfx12 anchor” section of `../spike-003-native-upward/README.md`.

The important non-obvious source fact is the A-fragment packing: every K16
half uses eight B64 LDS reads and sixteen `v_perm_b32` packets to build four
legal gfx12 LHS fragments. B uses four B128 reads. The Spike 3 experiment
sequence shows that B64+permute packing, the padded LDS layout, scalar address
evolution, native RHS ownership, and LHS-first issue ordering each closed a
measured portion of the performance gap.

Spike 4 maps that information as follows:

| Responsibility | Retained representation | Status |
| --- | --- | --- |
| Public buffers, shape, launch, global/LDS memory, WMMA, output | High Loom in `loom/gemm-f16-f32-mt128x128x32-gfx12-pack-microkernel-exact.loom` | Correct, access-sanitized, performance accepted |
| Eight-register A-fragment permutation | Single-block register-only Low object function invoked from High | Works through the default pipeline on exact gfx1201 |
| Full PGR2/wait/issue graph | Spike 3 prepared-Low whole-kernel oracle | Not raised; requires multi-block Low composition or a better High schedule |
| Family portability | Authored `gfx12.generic.core` helper specialized to gfx1201 | Blocked by physical carrier rebinding |

The current fast candidate is therefore evidence that the semantic kernel can
remain High while a narrow target idiom is Low. It is not yet the intended
family-generic motif and does not reproduce the incumbent wait graph.

## gfx11 / gfx1100

The actual retained FP16 witness is `1024x960x1024`, not a square request. Its
runtime route selects hipBLASLt solution `1675`, solution `42` in the navi33
GridBased HHS+bias table. The recipe and native loop establish MT64x96x32,
WG32x4, wave tile 2x3, PGR2, PLR1, GRVWA/B=8, TLDS1, two padded LDS stages,
and wave32 `v_wmma_f32_16x16x16_f16`. The retained native K32 interval has five
global loads, 76 LDS loads, five LDS stores, 12 WMMAs, ten waits, one barrier,
and one branch. See the “Gfx11 anchor” section of the Spike 3 README and
`../spike-003-native-upward/results/gfx1100-incumbent-1675-loop.json`.

The load-bearing behavior is cross-iteration: first-half fragments cross the
backedge, the two WMMA halves overlap LDS reads and the next VMEM payload, and
the terminal iteration is peeled. Spike 3 also found a distinct publication
choice that matters: swap the physically symmetric WMMA operands and directly
store FP16 from all 32 lanes, instead of using cross-lane exchange and 16
publishing lanes.

Spike 4 maps that information as follows:

| Responsibility | Retained representation | Status |
| --- | --- | --- |
| Full semantic GEMM baseline | High Loom in `loom/gemm-f16-f32-mt64x96x32-gfx11-high-exact.loom` | Correct, but 1.34x incumbent and spills four values |
| Exact payload/fragment/SRD ring and peeled loop | Spike 3 prepared-Low oracle | Exact 134-instruction category schedule and timing oracle only |
| Maintained High+Low composition | A required-inline multi-block Low helper | Blocked by `callee_body_not_single_block` |
| Direct wave-coalesced publication | Stable provider epilogue policy informed by the oracle | Must be composed and revalidated after CFG invocation works |

The blocker is structural rather than a vague request for “better codegen”:
without a CFG-bearing microkernel boundary, High Loom reconstructs the loop
with 107 miscellaneous instructions per normalized K32 instead of 24 and
materializes private spill traffic.

## Shared authored motif

`loom/epilogue-f16-bias-config.loom` demonstrates one stable argument superset
with `epilogue.has_bias` and row/column `epilogue.bias_axis` configuration.
Specialization removes the bias load in the no-bias case. This follows the
semantic split visible in TensileLite routing without copying its need for
separate precompiled adjacency buckets. It is a standalone representation
witness; the real GEMM accumulator ownership and nonuniform row/column address
maps still require a composed differential test.

## Blind alleys retained elsewhere

Spike 3 intentionally retains the complete Low reconstruction ladder and
negative experiments. Its `experiment-log.md`, `low/`, `att/`, compiler
usability reports, and result summaries document why count matching, banded
LDS reads, structural recoloring, and progressively larger fixed-register
experiments did or did not help. Spike 4's `experiments/` directory records
the new composition-specific failures: target carrier rebinding, schedule-lock
scope, CFG inlining, sanitizer coverage, and the bytecode link API ordering.

## Alternatives considered

- Copying the complete incumbent assembly into the maintained source would
  preserve too much routing and ABI policy at Low level.
- Keeping the entire GEMM in High is insufficient for the gfx11 anchor with
  the current compiler: the measured schedule and spill deltas are material.
- Treating one exact ISA as the authored family is not acceptable; exact
  gfx1201 is retained only as a working control for missing generic carrier
  rebinding.
