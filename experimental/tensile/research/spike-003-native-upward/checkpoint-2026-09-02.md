# Checkpoint: Radeon FP16 schedule recovery

This checkpoint closes the FP16 feasibility question for both Radeon family
witnesses and deliberately leaves the accelerated-datatype basket open. It is
the stable resume point before BF16/I8/FP8 schedule work.

## Conclusions

| Row | Correctness | Schedule | Runtime |
| --- | --- | --- | --- |
| gfx12 / gfx1201 FP16, 1024 cubed | Independent differential passes | Required B64+permute fragment packing and PGR2 schedule represented in prepared Low | 30.0605 us vs 28.800 us; ratio 1.0438, upper 95% CI 1.0474 |
| gfx11 / gfx1100 FP16, 1024 cubed | Every output passes; max absolute 0.002 | 134/134 steady-state instructions with identical category counts; only equivalent branch polarity differs | 37.721 us vs 44.320 us; ratio 0.8511, 95% CI `[0.8502, 0.8553]` |

The gfx11 terminal result changes the project-level conclusion. The missing
performance was not an irreducible compiler limitation. The exact native loop
can be represented, and the final large delta was an output-publication layout:
the old cross-lane packed store path cost roughly 11.7 us. A paired physical
WMMA operand swap and all-lane scalar store map is correct and faster than the
selected incumbent.

## Canonical sources and generated objects

| Artifact | SHA-256 | Notes |
| --- | --- | --- |
| `low/gfx12-gemm-double-buffer-pgr2-ring-native-split-exact-fenced-svw4-uniform-bpad-scalar-saddr-ring-native-rhs-lhs-first.loom` | `0139b173349a5a780373f933623570e4fecac6463dbe2e329c47baf07e364a47` | Terminal gfx12 FP16 prepared Low |
| `artifacts/gfx12-uniform-bpad-scalar-saddr-native-rhs-lhs-first/gemm.hsaco` | `7ba667f285ca1b16321d2875a6f4996a4f1be3e55d049b8a65484bfe71cabf91` | 13,328 bytes; ignored generated artifact |
| `low/gfx11-native-k32-payload-ring-rotated-1675-schedule-manual-waits-peeled-srd-ring-native-salu-slots-wgm8-swapped-native-epilogue.loom` | `199e26002e7b0ec008085e330c039b95dd089e5cef8de4d1c57e2d2d69769441` | Terminal gfx11 FP16 prepared Low |
| `artifacts/gfx11-native-k32-wgm8-swapped-native-epilogue-exact-branch/kernel.hsaco` | `68f0cdf9fe26b5b48ff47fb083b896be9d3e3cd1585e77e403f20c23dcec7176` | 13,312 bytes; ignored generated artifact |
| `low/gfx11-bf16-native-k32-payload-ring-terminal.loom` | `598c78b74bb6fc8f207fb22d44b3e7a398da067f32a215322a1c22cb763693fd` | First BF16 retarget; compilation only at this checkpoint |
| `artifacts/gfx11-bf16-terminal/kernel.hsaco` | `a03396de678ad0b7a0c8fc4055eb7b4071d189329040199916db9d556cdf8da0` | 13,320 bytes; runtime validation pending |

The generated `artifacts/` tree is intentionally ignored by Git and currently
contains compile reports, HSACOs, disassemblies, traces, and failing logs.
Tracked `results/`, `schedules/`, `att/`, sources, and transformers retain the
compact reproducible record. The small
[`evidence/checkpoint-2026-09-02/`](evidence/checkpoint-2026-09-02/)
directory promotes the terminal compile summaries, artifact manifest, passing
sanitizer records, and High fragment-role rejection needed by peers who do not
have the ignored artifact tree. Preserve both sets when making a full local
archive.

## Compiler checkpoint

HRX branch `loom-blas/gfx11-structural-loop-recolor` is clean at
`b422b5056` (`amdgpu: gate native gfx11 schedule experiments`). The terminal
gfx11 build uses these opt-in controls:

```text
LOOM_LOW_EXPERIMENTAL_ALLOW_UNALIGNED_FIXED_VALUES=1
LOOM_LOW_EXPERIMENTAL_PACK_REGISTER_INTERVALS=wide
LOOM_AMDGPU_EXPERIMENTAL_FIXED_GFX11_LOOP_BANKS=k32-ring-all-blocks
LOOM_AMDGPU_EXPERIMENTAL_MANUAL_WAITS=block-1
LOOM_AMDGPU_EXPERIMENTAL_FOLD_EMPTY_COND_FORWARD=1
LOOM_AMDGPU_EXPERIMENTAL_SCC_BRANCH_FORWARDING=native-gfx11
```

These controls are evidence for compiler work items, not proposed defaults.
The interval-assignment and wait-packet test packages pass on that branch.

## Sanitizer statement

The earlier gfx11 raw-HIP page fault is root-caused: sanitized HSACOs require
Loom runtime initialization for shadow/report globals. The reduced High
exact-LDS case and the exact terminal scalar-address witness both pass under an
AMDGPU-enabled `iree-test-loom`, with their numerical oracle and no access
event. Prepared Low under `--pipeline=none` is not instrumented, so the terminal
Low objects are not claimed as directly access-sanitized.

The complete paired gfx11 High source is preserved even though High
fragment-role validation rejects the physically valid WMMA operand swap. This
is the should-work source for a `low.invoke`/physical-fragment escape hatch.

## Resume state

The gfx11 BF16 incumbent is hipBLASLt solution 601, an MT128x48x32,
WG64x2, MIWT2x3, PGR2/PLR1 schedule measuring about 46 us in the first API
probe. Its exact installed code object has been extracted and its symbol
summarized in `results/gfx1100-bf16-601-native.json`. The terminal gfx11 FP16
motif has been mechanically retargeted to BF16 WMMA; because gfx11 lacks a
native FP32-to-BF16 pack instruction, its epilogue uses an explicit integer
round-to-nearest-even sequence. The object compiles, but no runtime result is
accepted in this checkpoint.

Next, finish BF16 on gfx11/gfx12, then I8 and gfx12 FP8/BF8. Only after those
rows have schedule/timing conclusions should work begin on `gfx9-0-generic`
and the gfx906 compiler enablement witness.
