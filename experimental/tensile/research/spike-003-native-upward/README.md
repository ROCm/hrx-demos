# Spike 003: native-upward schedule recovery

This spike treats the shipped BLAS kernel as an executable specification. It
starts with the exact production object and regenerated Tensile/TensileLite
recipe, reproduces the hot loop in prepared Low, and only then raises reusable
pieces toward family-generic Loom. “Compiler limitation” is not a completion
state: a remaining gap needs a preserved source, a narrow mechanism, and an
acceptance test.

## Prepared-Low oracle result at a glance

| Family witness | Incumbent | Best Loom result | Status |
| --- | --- | --- | --- |
| gfx12 / gfx1201 FP16, 1024 cubed | hipBLASLt 133309, 28.800 us | 30.0605 us, ratio 1.0438, 95% CI `[1.0395, 1.0474]` | Prepared-Low acceptance oracle: performance-congruent, correct, and spill-free under the experiment's direct execution path. |
| gfx11 / gfx1100 FP16, `1024x960x1024` | hipBLASLt 1675, 44.320 us | 37.721 us, ratio 0.8511, 95% CI `[0.8502, 0.8553]` | Prepared-Low acceptance oracle: exact 134-instruction steady-state category congruence and full differential correctness. |

Together these results establish that the schedules can be represented in
Loom Low and can reach the incumbent timing envelope. They are acceptance
oracles, not default-pipeline Loom execution claims: the terminal kernels were
assembled from prepared Low with `--pipeline=none` and timed through the
experiment's direct HIP harness. Spike 004 re-evaluates raised forms through
the default pipeline and sanctioned Loom runners. The gated compiler branch is
a mechanism witness and the retained Low motifs are schedule specifications.

The oracle timings are same-process alternating samples with both outputs
checked against an independent CPU reference. Their statistical gate remains
useful for the oracle, but it does not override the execution-path distinction.

## Gfx12 anchor

### Production source

The runtime winner is hipBLASLt solution 133309 and local YAML solution 91.
It is an MT128x128x32, WG32x4, wave32 WMMA kernel with wave tile 4x4, PGR2,
TLDS1, and a padded B LDS layout. The shipped object and regenerated
TensileLite assembly agree. Its steady-state K=32 interval has 147
instructions: eight global loads, 24 LDS loads, eight LDS stores, 32 WMMAs,
11 waits, two barrier packets, and one branch. See
[`results/gfx1201-incumbent-133309-loop.json`](results/gfx1201-incumbent-133309-loop.json).

The non-obvious fragment contract is on A. Each K=16 half uses eight B64 LDS
reads and sixteen permutes to construct four legal gfx12 LHS fragments. B uses
four B128 reads. High Loom originally emitted 64 D16 reads for A per half.

### Recovered motif

The best retained source is
[`low/gfx12-gemm-double-buffer-pgr2-ring-native-split-exact-fenced-svw4-uniform-bpad-scalar-saddr-ring-native-rhs-lhs-first.loom`](low/gfx12-gemm-double-buffer-pgr2-ring-native-split-exact-fenced-svw4-uniform-bpad-scalar-saddr-ring-native-rhs-lhs-first.loom).
It is deliberately a fully specialized prepared-Low witness, not the final
provider source. It establishes all of the schedule material that must survive
raising:

- four-wave MT128x128 ownership and 16 independent x8 FP32 accumulators;
- PGR2 global-load ring and double-buffered 51,200-byte LDS;
- eight B64 A reads plus sixteen `v_perm_b32` operations per K=16 half;
- four B128 B reads per half using native RHS ownership;
- exact K=32 loop specialization, scalar base-address evolution, and
  LHS-before-RHS fence ordering;
- vector-width-four FP16 result publication;
- no private memory and no materialized spill traffic.

The compile report records 215 peak live VGPR units and a final 224 VGPR
allocation. The loop executes 1,024 WMMAs for K=1024 and has no spill traffic.
The paired result is
[`results/gfx1201-uniform-bpad-scalar-saddr-native-rhs-lhs-first-vs-133309-paired.json`](results/gfx1201-uniform-bpad-scalar-saddr-native-rhs-lhs-first-vs-133309-paired.json),
with bootstrap analysis in the adjacent `-analysis.json` file.

The path from the original High form to parity was not cosmetic:

| Change | Approximate candidate/incumbent ratio | What it established |
| --- | ---: | --- |
| Full-linear High, D16 LHS loads | 1.687 | Removing loop-CFG allocation spills is insufficient. |
| Low B64+permute fragment construction | 1.22 | The TensileLite packing recipe is material. |
| Uniform padded B LDS layout | 1.10 | The authored LDS bank layout is material. |
| Scalar global-address ring | 1.053 | Repeated vector address construction was material. |
| Native RHS ownership and LHS-first fence | 1.0438 | The final ordering closes the declared timing gate. |

### Residual difference

This is performance congruence, not byte-identical assembly. ATT reports the
candidate matrix stall at 9.204 cycles per hit versus 7.514 for the incumbent,
LDS-read stall at 0.149 versus 0.098, and permute stall at 3.303 versus 3.097.
The candidate is therefore still a useful schedule-tuning target even though
it passes the end-to-end timing gate. The retained comparison is
[`results/gfx1201-att-current-vs-incumbent.json`](results/gfx1201-att-current-vs-incumbent.json).

The High-level fragment usability report now records both the original
failure and the successful Low escape hatch:
[`compiler-usability-gfx12-lhs-fragment-load.md`](compiler-usability-gfx12-lhs-fragment-load.md).
The proposed raising bridge is also preserved, but currently fails AMDGPU
target-contract selection with `TARGET/001` before inlining or fact
propagation can be evaluated; see
[`compiler-usability-low-invoke-amdgpu.md`](compiler-usability-low-invoke-amdgpu.md).

## Gfx11 anchor

### Production source

The incumbent is runtime solution 1675, solution 42 in the navi33 GridBased
HHS+bias logic: MT64x96x32, WG32x4, MIWT2x3, PGR2, PLR1, GRVWA/B=8, TLDS1,
and two padded LDS stages. Its exact steady-state interval is `[0xaf000,
0xaf3b0)`; the later interval is the final no-global-load iteration.

The native K=32 loop has 134 instructions: five global loads, 76 LDS loads,
five LDS stores, 12 WMMAs, ten waits, one barrier, and one branch. It carries
the first K=16 fragments across the loop edge. The first six WMMAs overlap
second-half LDS reads and the next VMEM payload publication; after the
barrier, the second six overlap reads for the next iteration's first half.
The operand banks are visible in the disassembly: accumulators v0:v47, A
fragments v56:v87, B fragments v121:v168, global payload v170:v189, and LDS
addresses v53:v55/v190. See
[`results/gfx1100-incumbent-1675-loop.json`](results/gfx1100-incumbent-1675-loop.json).

### Terminal recovered motif

The terminal source is
[`low/gfx11-native-k32-payload-ring-rotated-1675-schedule-manual-waits-peeled-srd-ring-native-salu-slots-wgm8-swapped-native-epilogue.loom`](low/gfx11-native-k32-payload-ring-rotated-1675-schedule-manual-waits-peeled-srd-ring-native-salu-slots-wgm8-swapped-native-epilogue.loom).
It recovers the incumbent mechanism rather than relying on a generic compiler
schedule:

- a K=32 payload/fragment ring with the five buffer loads, five LDS stores,
  and twelve WMMA operations issued in the incumbent order;
- explicit native waits, barrier placement, peeled terminal iteration, two
  carried SRDs, and WGM8 workgroup mapping;
- fixed accumulator, fragment, payload, and address banks behind compiler
  experiment gates;
- direct 16-bit output stores from every lane, using a wave-coalesced address
  map rather than the original cross-lane packed epilogue;
- swapped symmetric WMMA source operands paired with the direct store map, so
  the physical accumulator microtile transpose is reversed at publication.

The normalized steady-state interval contains exactly 134 instructions and
has the same category multiset as solution 1675: five global loads, 76 LDS
loads, five LDS stores, twelve WMMAs, ten waits, one barrier, one branch, and
24 other instructions. The normalized significant sequence shares a
109-instruction prefix and differs only at the terminal branch mnemonic:
Loom uses `s_cbranch_scc1`, while the incumbent's equivalent loop formulation
uses `s_cbranch_scc0`. See
[`results/gfx1100-terminal-exact-branch-schedule-comparison.json`](results/gfx1100-terminal-exact-branch-schedule-comparison.json).

The 80-pair same-process run measures 37.721 us for Loom and 44.320 us for the
incumbent, ratio 0.8511 with 95% bootstrap interval `[0.8502, 0.8553]`.
The complete `1024x960x1024` differential checks every output: Loom has zero
mismatches and maximum absolute error 0.002; the incumbent has zero mismatches
and maximum absolute error 0.004. The compile report records 192 VGPRs, 32
SGPRs, 28,288 bytes LDS, no private memory, and no spill traffic.

The decisive performance mechanism was the epilogue, not another inner-loop
instruction. The former path used 48 `global_store_b32` operations after
`ds_bpermute` and activated only 16 lanes with a 2,048-byte lane stride. The
native path uses 48 scalar `global_store_b16` operations across all 32 lanes,
with adjacent lanes writing adjacent rows. No-output and no-store controls put
the compute body near 35.9 us and isolate roughly 11.7 us in the old store
path. ATT and the exact microtile-permutation witness are retained under
`att/gfx1100/` and `results/gfx1100-swapped-operands-microtile-permutation.json`.

### Historical instruction-budget witness

[`low/gfx11-gemm-double-buffer-native-lds-address-hoisted-dce.loom`](low/gfx11-gemm-double-buffer-native-lds-address-hoisted-dce.loom)
has the same primitive material without cross-iteration PLR. Hoisting the LDS
address arithmetic reduces the report's exact dynamic descriptor count from
6,026 to 4,266, approximately 133 instructions per K=32 versus the native
134. It uses no private memory, peaks at 123 live VGPR units, and allocates
128 VGPRs. This proves that excess instruction count and register capacity are
not intrinsic to the Loom representation.

It still measures 1.3506x the incumbent because its issue graph is different:
LDS reads are emitted in bands, drained, and only then consumed by WMMA. The
wait planner reports 28 full drains, including a maximum drain of 38 LDS
operations. Matching counts without the native overlap does not recover
performance.

### Historical native-order witness and smoking gun

[`loom/gemm-f16-f32-mt64x96x32-gfx11-cross-iteration-plr.loom`](loom/gemm-f16-f32-mt64x96x32-gfx11-cross-iteration-plr.loom)
expresses the native K64 braid in High Loom with atomic fragment loads between
WMMAs. Its prepared form is
[`low/gfx11-cross-iteration-plr-split-zero-prepared.loom`](low/gfx11-cross-iteration-plr-split-zero-prepared.loom).
It carries six accumulators and five fragments across the loop boundary and
therefore captures the semantic PLR schedule that the count-matched form
lacks.

The allocation report is the smoking gun:

- scheduled VGPR pressure peaks at 151, well below the 256-unit architectural
  capacity and below the final 160-register allocation;
- `even_second_rhs2` requires one contiguous x8 run at the high-water point;
- 14 lower VGPR units are free, but they are split into three runs and the
  largest is only six units;
- 50 active assignments cover 234 units and 48 active storage leases cover
  96 units at that point;
- loop-edge materialization moves 71 units;
- four one-register LDS-store address computations are selected for scratch,
  producing four stores and four reloads (16 bytes each direction).

This is contiguous-run fragmentation coupled to loop-carried storage leases,
not aggregate register pressure. The current `loom-compile-report suggest`
output only observes 16 spill bytes and recommends shortening live ranges.
[`loom-compile-report-suggestions.md`](loom-compile-report-suggestions.md)
turns this and the gfx12 evidence into proposed diagnostic rules.

The unmodified native-order form is correct but allocates 16 bytes of private
memory and measures 1.6303x. It is a compiler acceptance test, not a shipping
motif yet.

The Loom author's expected `scf.for` unroll escape hatch does not recover this
case. Factor two retains four spill objects and increases branch-edge moves;
factors four and eight create 49 and 73 spill objects, and full unrolling
creates 154. The generated High sources are retained beside the original, and
the exact matrix is in
[`results/gfx1100-loop-structural-recolor-summary.json`](results/gfx1100-loop-structural-recolor-summary.json).

### Experimental structural recoloring path

The compiler trace finally identifies the placement mechanism precisely. Five
carried x8 fragments arrive in source banks 40, 40, 40, 48, and 96, while the
loop-header banks are the distinct ranges starting at 16, 24, 32, 40, and 48.
The repeated source bank is legal before the branch because the `low.concat`
producers are short lived, but it forces a parallel-move braid at the edge.
Moving only the concat results is a false win: it transfers the same moves to
the concat sites. The coherent recoloring unit is the full 43-assignment
non-edge placement closure containing each aggregate, `ds_read` packet, and
D16 in-place chain.

HRX branch `loom-blas/gfx11-structural-loop-recolor` at `0b70e4379` implements
that joint recoloring behind
`LOOM_LOW_EXPERIMENTAL_WIDE_LOOP_EDGE_RELOCATION=1`. It also moves an
overlapping scalar LDS address and an active four-register global-load lease.
The resulting static artifact has no fragment moves on the backedge: dynamic
register moves fall from 687 to 303 and the only remaining branch-edge group
is the 47-unit loop-entry initialization. The five next-iteration fragment
loads land directly in VGPRs 16:55.

The first lease-aware attempt faulted the device because its two evictions were
legal separately but not together: the moved global packet occupied VGPRs
112:115 through point 573 while the later LDS address occupied VGPR 113. This
is a useful blind alley. The corrected compiler checks projected lease
intervals against the entire joint proposal and instead places the packet in
VGPRs 160:163. All 20 allocation tests pass and disassembly has the intended
copy-free backedge. After resetting gfx1100, the corrected artifact passes the
full `1024x960x1024` numerical check repeatedly. Its paired median is 72.240 us
versus 44.361 us for solution 1675: 1.6285x with 95% bootstrap interval
[1.6187, 1.6433]. Structural recoloring is sound for this run but does not
recover runtime parity. Prepared-Low native-access validation remains blocked
by the sanitizer pipeline gap described below.

At that historical point, the steady-state disassembly explained the remaining performance result. The
recolored Loom loop executes 414 instructions per K=64 (about 207/K=32), while
the incumbent executes 134/K=32. MMA and LDS work agree; the dominant delta is
the `other` category, about 95 versus 24 instructions/K=32. Loom reconstructs
five 64-bit global pointers per stage and issues `global_load_b128`; solution
1675 carries two SRDs and issues `buffer_load_b128`.

[`tools/use_gfx11_buffer_loads.py`](tools/use_gfx11_buffer_loads.py) preserves
the current scalar offsets and substitutes exactly those 15 loads. This first
isolation is a negative result: long-lived SRDs change allocation repair from
4 spill objects/16 bytes to 20/80, wait actions rise from 55 to 124, and the
correct kernel measures about 131 us. The raw-buffer address mode is available,
but its SRDs and offsets must be part of the jointly placed native schedule;
naive substitution is not compositional with the current allocator.

## Blind alleys retained

The tree intentionally keeps failed transformations because they constrain
future compiler work:

- preloading all PLR1 fragments in High measured about 1.386x;
- locked Low without cross-iteration PLR measured about 1.352x;
- a hand depth-interleave measured about 1.355x;
- swapping gfx11 WMMA operands failed High target selection (`TARGET/039`),
  while the forced Low swap alone was numerically wrong; pairing the swap with
  the native scalar store map is the correct terminal transform;
- reusing only the five invariant LDS store addresses increased spills from
  four to ten;
- full and localized rematerialization of store indices produced five spill
  objects plus repeated reloads;
- a wide-first interval-order control made no difference;
- packing only x8 aligned values first increased the result to 27 spills;
- relaxing fixed-register alignment and shifting the native fragment banks by
  1, 4, or 8 registers produced 12 spills; exact accumulator pinning conflicts
  with already allocated ranges;
- relocating wide loop-edge vector intervals, both alone and combined with
  the gfx12 vector-edge experiment, leaves the same four spills and 71 units
  of branch-edge moves;
- moving both five-payload LDS publication groups to the incumbent's position
  after the first three WMMAs leaves four spills and moves the x8 high-water
  failure to `odd_first_rhs2`; branch-edge moves increase from 71 to 95 units,
  so the candidate did not earn timing;
- simply treating every virtual register as alignment-one increased the
  result to 20 spills.
- `scf.for` unroll factors two, four, eight, and full do not recover the
  carried-fragment placement; larger factors amplify spill materialization.
- relocating a lease owner and a later address independently can create a
  physical overlap invisible to ordinary SSA liveness; that first joint
  recoloring attempt faulted gfx1100 and is recorded in the structural summary.
- locally rematerializing seven lane-decomposition inputs at the payload-write
  peak lowers scheduled pressure but expands repair to 19 spill objects;
- naively carrying two SRDs and substituting raw-buffer loads expands repair to
  20 spill objects and regresses runtime to about 131 us.

The corresponding sources, tools, reports, and HRX branches are indexed in
[`experiment-log.md`](experiment-log.md).

## Native access sanitizer status

The gfx12 High address map passes the native `access` sanitizer for one full
macro-tile, but instrumentation of the fully unrolled production shape reaches
the spill-materialization iteration limit.

The earlier gfx11 page fault was caused by launching a sanitized HSACO through
the raw HIP-module harness, which bypasses Loom's sanitizer shadow/report
runtime initialization. It was not kernel-legality evidence. Building
`iree-test-loom` with the AMDGPU driver and running the reduced High case
through that testbench passes the numerical oracle with no sanitizer event in
both ordinary and `report-only` modes. A High address-only witness for the
terminal scalar epilogue also passes the native access sanitizer.

Prepared Low compiled with `--pipeline=none` bypasses sanitizer
instrumentation, while feeding it through the default sanitizer pipeline hits
an internal lowering failure. The complete paired High expression is also
blocked because its physically valid gfx11 WMMA source swap is rejected by
High fragment-role validation (`TARGET/039`). Consequently the terminal Low
objects themselves have not been instrumented. Their High address maps and
independent full numerical differentials pass, but they must not be described
as directly sanitizer-validated. Full details and exact
reproducers are in
[`compiler-usability-access-sanitizer.md`](compiler-usability-access-sanitizer.md).

## Accelerated datatype reuse

The executable witnesses prove the native matrix contracts for gfx11/gfx12
FP16/BF16/I8/I4 and gfx12 FP8/BF8. The public-library census is in
[`../spike-002-schedule-congruence/results/accelerated-types.json`](../spike-002-schedule-congruence/results/accelerated-types.json).
All selected library rows use the same WMMA/LDS ownership vocabulary. The
meaningful family splits are fragment carrier width and packing, DepthU/read
width, and occasionally macro-tile shape; they do not justify independent
router implementations.

Only FP16 has reached a complete GEMM schedule experiment. A gfx11 BF16 port
of the exact Low schedule now compiles with native BF16 WMMA and an explicit
VALU round-to-nearest-even output conversion; runtime comparison is not yet a
checkpoint claim. I8, FP8/BF8,
rectangular shapes, tails, beta, and epilogues remain representability-backed
work items rather than performance claims. The provider design therefore uses
one family motif per structural WMMA contract and specializes datatype,
packing, shape partition, and epilogue configuration at compile time.

## Compiler experiment branches

No compiler experiment is proposed for direct landing. The preserved HRX
branches are:

- `loom-blas/gfx12-wait-state-parity` at `ec40ab4f0`;
- `loom-blas/gfx11-loop-bank-triage` at `5c4d6ef41`;
- `loom-blas/vector-loop-edge-relocation` at `c18abc7b9`;
- `loom-blas/gfx11-vector-loop-edge-relocation-combined` at `60b26addb`;
- `loom-blas/gfx11-structural-loop-recolor` at `b422b5056`.

Each change is config/environment controlled. Their value is the paired
positive or negative result captured here, not the patch in isolation.

## Next acceptance order

1. Complete the BF16, I8, and gfx12 FP8/BF8 schedule/timing basket using the
   recovered family motifs, splitting only where fragment packing or native
   conversion semantics require it.
2. Make prepared-Low kernels participate in the native access sanitizer and
   make the report state explicitly when instrumentation was not applied.
3. Add the missing AMDGPU target-contract/lowering bridge for the preserved
   `low.invoke` should-work fixture, then raise the gfx12 B64+permute inner loop
   and verify specialization facts and sanitizer visibility across it.
4. Raise the terminal direct-store epilogue and provide a deliberate escape
   for the semantically valid gfx11 physical WMMA source-role swap.
5. Only after gfx11/gfx12 datatype rows close, start `gfx9-0-generic` enablement and compare
   the family name and feature set against LLVM. gfx906 should use Loom's FP
   emulation where needed; Instinct-specific gfx942/gfx950 motifs remain a
   separate future effort.
