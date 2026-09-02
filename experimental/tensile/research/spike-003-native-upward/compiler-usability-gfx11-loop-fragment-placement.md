# Gfx11 loop-fragment placement usability report

## User intent

Express the native gfx11 GEMM braid as an ordinary `scf.for` carrying six x8
FP32 accumulators and five x8 WMMA operand fragments. The body deliberately
constructs the next iteration's fragments between current-iteration WMMAs so
LDS latency overlaps matrix issue. The expected lowering is a copy-free
backedge with each next fragment produced directly in its loop-header bank.

The should-work High source is
[`loom/gemm-f16-f32-mt64x96x32-gfx11-cross-iteration-plr.loom`](loom/gemm-f16-f32-mt64x96x32-gfx11-cross-iteration-plr.loom).
Its exact post-lowering fixture is
[`low/gfx11-cross-iteration-plr-split-zero-prepared.loom`](low/gfx11-cross-iteration-plr-split-zero-prepared.loom).

## Observed default result

The source is accepted and numerically correct, but the final object contains
40 register moves on the recurrent fragment edge and four scalar spill/reload
pairs. The report records 151 scheduled live VGPR units, 160 final VGPRs, 16
private bytes, and 71 branch-edge units including entry initialization. The
result measures 1.6303x hipBLASLt solution 1675.

`loom-compile-report suggest` only emits `amdgpu.spill_traffic`; it does not
identify the loop-edge bank rotation or contiguous-run failure.

## Why `scf.for` unroll is not an escape hatch

The same source was compiled with factor-two, factor-four, factor-eight, and
full unroll attributes:

| Mode | Spill objects | Private bytes | Reload bytes | Final VGPRs |
| --- | ---: | ---: | ---: | ---: |
| default | 4 | 16 | 16 | 160 |
| factor 2 | 4 | 16 | 24 | 160 |
| factor 4 | 49 | 512 | 1304 | 168 |
| factor 8 | 73 | 1076 | 2276 | 168 |
| full | 154 | 1024 | 2724 | 152 |

Thus the form the author expected to work is preserved, and its unroll
controls make the allocation pathology larger rather than producing the
native schedule.

## Smoking gun

An instrumented late-allocation experiment sees five fragment sources in
banks 40, 40, 40, 48, and 96 and their distinct loop-header destinations at
16, 24, 32, 40, and 48. Repeated source ranges are locally legal because the
short-lived producers do not overlap until edge semantics require all five
values simultaneously.

Recoloring only the five aggregate results changes no instructions: the same
copies materialize at `low.concat`. Propagating each target color through the
non-edge placement closure identifies a 43-assignment atomic unit. Recoloring
that unit makes the `ds_read` results land directly in VGPRs 16:55 and removes
all 40 recurrent fragment moves. Static dynamic-move count drops from 687 to
303; the remaining 47 branch-edge units are loop-entry initialization.

The experiment is on HRX branch
`loom-blas/gfx11-structural-loop-recolor` at `0b70e4379`, gated by:

```sh
LOOM_LOW_EXPERIMENTAL_WIDE_LOOP_EDGE_RELOCATION=1
```

This patch is a specification and reproducer, not a proposed landing.

## Lease-safety lesson

The first atomic implementation faulted gfx1100. It moved the four-register
`next_bv1` global-load value to VGPRs 112:115 and a later LDS address `%1170`
to VGPR 113. Ordinary SSA liveness said the ranges were disjoint, but
`next_bv1` has an explicit asynchronous storage lease active through the
address use. Valid late recoloring must project assignment-backed lease ranges
to every proposed location and check the whole proposal atomically.

The corrected artifact places `next_bv1` at 160:163 and `%1170` at 113. The
compiler builds, all 20 allocation tests pass, and after resetting gfx1100 the
artifact passes repeated full-shape correctness and a 40-sample paired timing
run. It remains 1.6285x solution 1675, so recoloring resolves the copy
mechanism but not performance. Native access sanitization remains pending
because the prepared-Low `--pipeline=none` route is not instrumented.

## Requested compiler behavior

1. Treat a loop-carried aggregate and its non-edge placement component as one
   recoloring unit when minimizing a recurrent parallel move.
2. Include active storage-lease intervals in both initial coloring and every
   late joint recoloring proposal.
3. Preserve pre-repair allocation snapshots in detailed compile reports.
4. Report structural move round trips: a removed edge move that reappears at a
   concat/copy is not an optimization.
5. Add a verifier check that no relocated lease overlaps any simultaneously
   live assignment, even when their ordinary SSA intervals are disjoint.

## Reproduction

Compile the prepared fixture with the experimental HRX checkout:

```sh
LOOM_LOW_EXPERIMENTAL_WIDE_LOOP_EDGE_RELOCATION=1 \
LOOM_LOW_EXPERIMENTAL_LOOP_EDGE_TRACE=1 \
../hrx-system/bazel-bin/loom/src/loom/tools/loom-compile/loom-compile \
  research/spike-003-native-upward/low/gfx11-cross-iteration-plr-split-zero-prepared.loom \
  --backend=amdgpu-hal --pipeline=none \
  --compile-report=details \
  --compile-report-output=research/spike-003-native-upward/artifacts/gfx11-cross-iteration-plr-wide-relocation-trace/report.json \
  --output=research/spike-003-native-upward/artifacts/gfx11-cross-iteration-plr-wide-relocation-trace/kernel.hsaco
```

The retained `trace.log` names the five bank mappings, 43-value closure, two
evictions, and final application. The compact facts are in
[`results/gfx1100-loop-structural-recolor-summary.json`](results/gfx1100-loop-structural-recolor-summary.json).
