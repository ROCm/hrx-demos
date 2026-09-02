# Loom BLAS provider: source-led kernel reconstruction

**Date:** 2026-09-02

**Status:** Working design with Spike 4 checkpoint

**Initial physical witnesses:** `gfx906`, `gfx1100`, `gfx1201`

**Initial authored target families:** `gfx9-0-generic` once enabled,
`gfx11-generic`, and `gfx12-generic`

**Initial operation:** GEMM and the fused GEMM forms needed by the public BLAS
provider

## Summary

Build a BLAS provider whose kernels are authored and JIT-compiled with Loom.
Develop it by reconstructing the best applicable rocBLAS and hipBLASLt
implementations from their source logic, not by treating either library as a
black-box benchmark.

The incumbent implementation is two coupled bodies of IP:

1. **Selection IP:** problem normalization, ordered predicates, exact and
   grid-based routing tables, applicability tests, workspace policy, and
   fallback order.
2. **Kernel IP:** the Tensile/TensileLite recipe, the generator decisions that
   expand it, and the resulting native schedule.

Both must be made inspectable. A YAML row does not by itself identify the
kernel that a public API call executes, and a tile shape does not reproduce a
kernel schedule. The first project deliverable is therefore a standalone BLAS
laboratory that can explain and force incumbent selections, extract a
normalized recipe and schedule packet, run a Loom candidate, and retain joined
numerical, compiler, and physical evidence.

The implementation should migrate incrementally. The public provider may use
the Loom provider only for target/problem cells that have passed their gates;
all other cells continue to use an incumbent provider. Provider API design is
assumed and is otherwise out of scope here.

The working kernel-architecture hypothesis is deliberately small: each gfx
family needs one primitive GEMM dataflow, specialized into a small number of
authored arithmetic variants. The first variants cover every provider-visible
input/accumulator combination with native accelerated matrix instructions;
fusion variants follow only after those primitives work. Layout, exact shape,
edge behavior, alpha/beta cases, and most epilogues should be facts or
composable variants rather than separately invented kernels. Tile sizes and
other tuned configurations may produce many cached executables without
multiplying the number of Loom source algorithms. The incumbent corpus is used
first to falsify this hypothesis, not to justify recreating its solution count.

An authored kernel declares the broadest target family on which its mechanism
is legal. `gfx1100` and `gfx1201` are physical witnesses and exact JIT target
profiles, not source-family names. An exact-ISA authored variant is allowed
only when an instruction, fragment ABI, memory rule, or other structural choice
is legal on that narrow target and cannot be expressed by family specialization.
Measured tuning preference alone does not justify an exact-ISA source variant.

Epilogue choice is initially part of the specialization signature, not the
identity of the core GEMM algorithm. One Loom family should express at least
`none` and `bias` as configuration-selected epilogue regions, with unused
arguments and paths erased during specialization. Activation, input/output
scales, alpha-vector scaling, and auxiliary output extend that composition only
when their dataflow remains compatible. They become separate authored kernels
only when evidence shows a materially different ownership, liveness,
synchronization, or launch structure.

Initial due diligence supports the approach, with concrete compiler contracts
still open:

- The Radeon recipes examined are representable in terms of Loom's existing
  structured compute, views, specialization, launch, artifact, and JIT
  facilities. The available llama.cpp Loom corpus is additional evidence that
  these mechanisms can produce strong kernels, but is not BLAS validation.
- Representative kernels compile through Loom for `gfx1100` and `gfx1201` on
  this machine.
- The three sampled incumbent winners have been joined to and disassembled
  from their shipped code objects. The RDNA winners contain native FP16 WMMA;
  the `gfx906` winner contains the expected packed FP16 dot-product loop. Their
  large multi-path symbol bodies also expose concrete specialization work.
- The current Loom checkout recognizes `gfx906` as an exact AMDGPU processor
  but does not provide a complete physical target row for it. A representative
  compile fails with `AMDGPU target 'gfx906' is not supported`. Enabling and
  validating that target is a prerequisite, not a GEMM tuning problem.
- The installed ROCm artifact census does not match the initial assumption
  that `gfx1100` is rocBLAS-only: this installation contains hipBLASLt artifacts
  for `gfx1100` and `gfx1201`, and rocBLAS artifacts for `gfx906`. Source trees
  contain additional logic not necessarily packaged in this installation.
  Runtime and artifact identity must therefore be captured for every result.

### Radeon FP16 schedule-recovery checkpoints

Spike 003 answers the Low representability question. Its terminal objects are
prepared-Low acceptance oracles assembled with `--pipeline=none` and executed
through the research HIP harness; they are not default-pipeline Loom claims:

| Family witness | Incumbent | Loom result | Schedule conclusion |
| --- | --- | --- | --- |
| gfx12 / gfx1201 | hipBLASLt 133309, 28.800 us | 30.0605 us; ratio 1.0438, upper 95% CI 1.0474 | Performance gate passes after recovering B64+permute LHS packing, PGR2, padded LDS, scalar address evolution, and native issue ordering. |
| gfx11 / gfx1100, `1024x960x1024` | hipBLASLt 1675, 44.320 us | 37.721 us; ratio 0.8511, 95% CI `[0.8502, 0.8553]` | The prepared-Low steady state is 134 instructions, exactly matching incumbent category counts. The only mnemonic difference is equivalent loop-branch polarity. |

The gfx11 result is especially important. A count-matched but banded schedule,
then a structurally recolored cross-iteration schedule, both remained slow.
The final recipe carries the native SRD/payload/fragment ring and explicit
waits, but the decisive last delta was the epilogue: the former path used
cross-lane exchange and stores from 16 lanes with a 2,048-byte lane stride.
The terminal path swaps the physically symmetric WMMA operands and directly
publishes 16-bit values from all 32 lanes, with adjacent lanes writing adjacent
rows. The paired transform is fully correct and about 15% faster than the
selected incumbent.

This validates Low as an exact schedule representation. Spike 004 then tests
the required maintained form: High ownership of semantics and memory with a
small Low microkernel, compiled by the default pipeline and executed only by
sanctioned Loom runners:

| Family witness | Fresh incumbent | Default-pipeline result | Conclusion |
| --- | --- | --- | --- |
| gfx12 / gfx1201, `1024x1024x1024` | 28.78 us | 25.04 us device p50, correct, access-sanitized, no spill/private traffic | Accepted. It is faster, so the non-congruent normalized K32 schedule (239 versus 147 instructions) is diagnostic rather than blocking. |
| gfx11 / gfx1100, `1024x960x1024` | 45.46 us | 61.0 us device p50, correct, 16 bytes private traffic | Performance and schedule fail; normalized K32 is 219.5 instructions versus 134. |

In both default artifacts the normalized matrix/global/LDS/barrier/branch
counts match the incumbent. The deltas are waits and miscellaneous address,
move, and control packets, plus four spill/reload pairs on gfx11. A
single-block `low.invoke` helper now works on HRX main for both exact and
family-generic targets. #513 also preserves `schedule(locked)` source order
across required inlining. The full gfx1201 candidate emits the same target
artifact with a family-generic helper and no explicit workaround fences.
Multi-block outer CFG and nested Low calls remain outside this first contract;
that redirects loop recovery toward structured High/source control flow plus
straight-line locked fragments.

The original compiler controls remain as archaeology on HRX branch
`loom-blas/spike4-low-invoke-experiment` at `db115431e`, based on `b422b5056`.
The validated upstream implementation is main commit `f17f69e82` (#513) and
requires no experiment environment gate.

The shareable Spike 4 packet includes the default-pipeline sources, source
lineage, reproduction commands, compact results, and minimized compiler cases
under
[`research/spike-004-default-pipeline-microkernels/`](../../research/spike-004-default-pipeline-microkernels/README.md).

Native access verification has a split status. The maintained Spike 004 gfx12
candidate keeps memory in High and passes Loom's access sanitizer. A preserved
LDS-read-plus-pack helper proves that authored Low memory packets remain
outside sanitizer instrumentation even when the surrounding High accesses are
covered. Therefore register-only Low microkernels are acceptable now;
memory-bearing Low helpers need Low instrumentation or a machine-checkable
static access contract.

The accelerated-datatype basket remains open. A gfx11 BF16 retarget of the
terminal schedule compiles with native BF16 WMMA and an explicit VALU
round-to-nearest-even output conversion. BF16 runtime parity, I8, and gfx12
FP8/BF8 are the next schedule conclusions; gfx906 enablement follows those.

## Scope

### In scope

- Source-led reconstruction of GEMM selections and kernels for the three
  installed Radeon GPUs.
- Primitive unfused coverage of every provider-visible dense GEMM arithmetic
  signature backed by native WMMA on `gfx1100` or `gfx1201`.
- rocBLAS/Tensile for `gfx906` and any other incumbent path actually selected
  by the probed installation.
- hipBLASLt/TensileLite for `gfx1100` and `gfx1201`, including Equality and
  GridBased logic, epilogues, workspace, and auxiliary kernels when selected.
- A custom API-level correctness, enumeration, selection, and measurement
  harness. Vendor benchmark and tuning programs may be consulted as source,
  but are not the experimental interface.
- A normalized corpus and selection explainer derived from YAML, runtime
  libraries, generator source, and native artifacts.
- The agent research and implementation loop used to translate and improve
  kernels in Loom.
- JIT artifact/cache requirements that the provider implementation will need.

### Deferred

- Instinct targets and broad datacenter tuning.
- Non-GEMM BLAS levels and operations.
- Replacing every legacy solution before the Loom provider can ship.
- Reproducing Tensile's tuning infrastructure as such.
- Designing the public BLAS provider interface.

Instinct is deferred, but it constrains the evidence format. The Radeon
kernels sampled here are relatively basic: their YAML recipes and generator
paths expose much of the schedule. Important Instinct kernels encode exact
native schedules whose issue order, wait placement, clauses, MFMA dependency
spacing, LDS phases, register lifetime, and publication protocol must be
researched from generated assembly and disassembly rather than invented. The
normalized schedule packet below must be able to carry those facts even though
the first implementation will not need all of them.

## Source and machine baseline

This design was checked against the following local revisions and installation:

| Item | Observed identity |
| --- | --- |
| TheRock source | `9a50c0e9e` |
| rocm-libraries source | `bb5babaf8` |
| hrx-system / Loom source | `bc59ef458` |
| llama.cpp reference branch | `33a1f2b23` |
| selected ROCm | `<selected-rocm>` |
| rocBLAS runtime | 5.7.0 |
| hipBLASLt runtime | 1.4.1 |
| Device 0 | `gfx906`, Radeon VII |
| Device 1 | `gfx1100`, Radeon Pro W7900 |
| Device 2 | `gfx1201`, Radeon RX 9070 XT |

The revisions above are observations, not project dependencies. Every durable
experiment must record full revisions, paths, hashes, and target properties
rather than relying on this table.

### Incumbent corpus census

The current installed artifact and source census is:

| Target | Installed rocBLAS target files | Installed hipBLASLt target files | Source logic files mentioning target |
| --- | ---: | ---: | ---: |
| `gfx906` | 210 | 0 | 130 rocBLAS, 0 hipBLASLt |
| `gfx1100` | 0 | 98 | 40 rocBLAS, 46 hipBLASLt |
| `gfx1201` | 0 | 292 | 0 rocBLAS, 150 hipBLASLt |

These counts are useful for scoping only. A source logic file does not prove
that its solution was built or selected, and a target directory does not prove
that every public operation works. The laboratory must establish support by
calling the public API and joining the call to an observed selection and code
object.

The primary local corpora are:

- rocBLAS logic:
  `<workspace>/sources/TheRock/rocm-libraries/projects/rocblas/library/src/blas3/Tensile/Logic`
- classic Tensile generator and host library:
  `<workspace>/sources/TheRock/rocm-libraries/shared/tensile/Tensile`
- hipBLASLt logic:
  `<workspace>/sources/TheRock/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic`
- TensileLite generator and host library:
  `<workspace>/sources/TheRock/rocm-libraries/projects/hipblaslt/tensilelite`
- Loom workflow:
  `<workspace>/sources/hrx-system/loom/docs/src/workflows/agent-driven-kernel-development.md`

### What is present in the logic

Three sampled files establish the important forms:

- The rocBLAS `gfx906` HHS logic
  `vega20_Cijk_Alik_Bljk_HHS_BH.yaml` contains 152 solution recipes and 3,309
  exact-logic rows. Of those solutions, 120 are assembly kernels and 32 are
  source kernels. The sampled recipes use ordinary VALU schedules, explicit
  macro tiles, unroll depths, workgroups, global vector widths, LDS layout,
  and prefetch controls; they do not use matrix instructions.
- A hipBLASLt `gfx1201` GridBased HHS/bias/scales/UserArgs file contains 778
  assembly solution recipes and 9,699 routing rows. All sampled solutions use
  wave32 `[16, 16, 16, 1]` matrix instructions. The recipes vary macro tile,
  depth, direct-to-VGPR, global reads, LDS, prefetch, and occasionally dynamic
  or fixed global split-U.
- A hipBLASLt `gfx1201` Equality HHS file contains four recipes and four exact
  shape rows. Equality and GridBased libraries are distinct ordered routing
  strata, not interchangeable tuning annotations.

This is enough to reject two tempting shortcuts: porting only a nominal tile
and trusting the default API heuristic.

The sampled 1024-cubed winners are also joined to their shipped code objects
and exact external symbols. Because the ELF function sizes are zero, the
research tool bounds each symbol by the next external text symbol and retains
the complete mnemonic histogram. The gfx1100 and gfx1201 spans contain 54 and
144 static `v_wmma_f32_16x16x16_f16` instructions respectively; the gfx906
span contains 1,056 static `v_dot2_f32_f16` instructions. These are complete
multi-path symbol counts, not dynamic execution counts. The hipBLASLt symbols
include runtime argument, GSU, activation, and epilogue paths, which makes
control-flow extraction part of schedule reconstruction and identifies work
that exact Loom specialization should erase.

### Accelerated arithmetic inventory

The matrix-instruction inventory, not SGEMM, organizes the first kernel set.
The current Loom target contracts expose the following dense signatures. The
laboratory must still confirm which combinations the public provider and the
installed rocBLAS/hipBLASLt build expose; a hardware instruction alone does not
create an API contract.

| Target | Native matrix input signatures | Accumulator/result domain relevant to the primitive |
| --- | --- | --- |
| `gfx1100` | FP16×FP16, BF16×BF16, signed/unsigned I8 combinations, signed/unsigned I4 combinations | FP32 or FP16 for FP16; FP32 or BF16 for BF16; I32 for integer forms |
| `gfx1201` | All `gfx1100` forms, plus FP8×FP8, FP8×BF8, BF8×FP8, and BF8×BF8 | Same FP16/BF16/integer domains; FP8/BF8 forms accumulate to FP32 in the current gfx12 contracts |
| `gfx906` | No native matrix-instruction profile in the current Loom target model | Separate VALU/dot-product primitive; it does not define the WMMA design |

For integer WMMA, signedness selection and packed I4 representation are part
of the arithmetic signature. For FP8/BF8, encoding, conversion, NaN/Inf, and
flush behavior are likewise semantic facts, not storage aliases. If the public
provider does not expose a listed hardware form, retain it in the target
capability inventory but do not invent a public operation for this project.

`gfx1201` also exposes sparse matrix instructions for several of the same
types. Sparsity changes metadata movement and the core dataflow, so it is a
separate algorithm variant rather than another data type. Dense primitives are
completed first; sparse coverage is added only if it belongs to the provider's
GEMM scope or the selected demand corpus.

Maintain this as a generated capability matrix rather than a hand-maintained
list. Each arithmetic row records: hardware/LLVM instruction, Loom matrix
contract and fragment layout, Tensile/TensileLite recipe presence, installed
runtime/API exposure, numerical semantics, primitive implementation, emitted
instruction proof, correctness status, and measured coverage. A missing row is
then visibly a compiler, library, API, or implementation gap. Because operand
packing and fragment ownership drive the core loop, completing this matrix
precedes broad layout or fusion work.

## How incumbent selection actually works

The extractor must model the runtime library, not infer it from filenames.
The TensileLite path currently composes approximately as follows:

```text
public matmul request
  -> normalized ContractionProblemGemm
  -> architecture / CU / problem-type / performance-metric predicates
  -> ordered ExactLogicLibrary rows
       -> Equality matching (exact normalized key), when present
       -> Range / Prediction / GridBased / FreeSize fallbacks as configured
  -> matching-table key properties and index order
  -> table search and candidate nested library
  -> solution problem predicate
  -> task predicate, including workspace/launch facts
  -> hardware predicate
  -> lazy solution shard and assigned code object
  -> one or more kernel launches
```

The exact order is serialized into the built library and is part of the
selection contract. `ExactLogicLibrary` returns the first successful
non-fallback match and only retains the first successful fallback. A selected
matching-table row can still fail when its nested solution library applies the
problem, task, or hardware predicate.

For Equality, the normalized key must match exactly. For the current
TensileLite GridBased specialization, the optimized lookup is more involved
than a scalar distance:

- it searches an M/N grid (optionally with a KD tree);
- within selected M/N points it brackets batch and chooses nearby K;
- the distance used within that slice is absolute K distance;
- it can retry a batched request after folding batch into the larger of M or N;
- it de-duplicates solutions for top-N enumeration; and
- the chosen nested library may reject the solution, causing further search.

Lazy loading adds another identity boundary. The architecture master library
loads `TensileLiteLibrary_lazy_<arch>_Mapping.dat`, maps solution-index ranges
to serialized shards, loads the shard on demand, and assigns its `.co` file to
the loaded solutions. An index is only stable relative to this complete library
build. Code-object and mapping hashes are therefore required provenance.

Classic Tensile differs in details and needs its own adapter, but the same
rule applies: preserve normalization, hierarchy, predicates, row metric,
solution identity, and native artifact as one trace.

### Required selection trace

For one API request, the explainer must emit an ordered trace with at least:

1. original API descriptor and normalized GEMM problem;
2. every derived matching property and its index order;
3. architecture and problem-library path;
4. each visited ordered predicate row and pass/fail reason;
5. matching mode and considered table rows;
6. table row key, recorded metric, and referenced solution index;
7. solution problem/task/hardware predicate results;
8. required device and host workspace;
9. library-logic index, API algorithm index, solution/kernel names;
10. serialized shard, code object, and kernel symbol;
11. complete launch sequence, including conversion, reduction, or epilogue
    kernels; and
12. agreement or disagreement among source prediction, runtime selection, and
    observed dispatch.

Any disagreement is a finding to explain. It must not be silently repaired by
changing the expected result to whatever the runtime returned.

## What JIT should remove from the router

Reproducing the incumbent router is an analysis milestone, not the intended
Loom architecture. Tensile routes among a finite set of precompiled kernels.
Its library hierarchy must encode distinctions needed to find a compatible
artifact, control installed binary size, survive missing shards, and fall back
across solutions whose code cannot change. A JIT can instead turn many request
properties into compiler facts and emit one compatible artifact on demand.

The extraction must classify every predicate and routing edge by reason:

| Class | Examples | Loom disposition |
| --- | --- | --- |
| Semantic legality | types, transpose/layout, beta behavior, epilogue contract, aliasing | Part of the authored contract or specialization signature; never a performance router decision. |
| Hardware legality | ISA feature, wave size, instruction and LDS limits | Target profile plus compiler verification; retain as hard constraints. |
| Request facts | alignment, leading-dimension multiples, exact/tail dimensions, alpha/beta class | Bind as JIT facts when stable; otherwise retain a guarded version or generic path. |
| Workspace policy | split-K buffers, host/device workspace cap, auxiliary launches | Retain as a runtime/provider constraint and compilation choice. |
| Static packaging | architecture library, lazy shard, code object, symbol, fallback file | Eliminate from the logical router; replaced by the JIT cache and loaded-artifact identity. |
| Generator specialization | separate kernels for edge/no-edge, beta zero, fixed strides, epilogue variants | Prefer specialization or guarded multi-versioning from shared Loom source. |
| Empirical performance | Equality/GridBased row, tile family, unroll, prefetch, mapping, split strategy | Preserve initially as a prior, then reduce to the schedule policy demonstrated by measurement. |

This suggests a smaller two-stage Loom decision:

```text
canonical request + exact target
  -> semantic/specialization signature
  -> legal schedule-family candidates
  -> schedule policy or bounded autotune
  -> specialize and canonicalize a kernel request
  -> derived compiled-program key
  -> evaluated launch
```

The first stage is declarative and should mostly disappear into Loom
specialization, verification, and the cache key. M, N, K and semantic choices
are exact compile-time facts for the normal provider path. A Loom
`config.get`-controlled `scf.if` may express a generic source choice such as an
optional epilogue; after the config is bound, canonicalization must remove the
unselected branch and its unused work. Only physical schedule choice remains a
performance-routing concern.

JIT does **not** remove the need to choose a physical schedule. Macro tile,
wave/workgroup topology, VALU versus matrix path, prefetch pipeline,
workgroup mapping, and split-K/StreamK strategy can still vary materially with
M, N, K, batch, workspace, and target. Exact request specialization remains
practical, but latency must be measured through the production path. Spike
004 indexes Loom bytecode once, links a selected root, applies an exact
bytecode config and target profile, runs the default source-to-prepared-Low
pipeline, and emits HSACO in memory through the Loom C API. With the host
compiler built in Bazel `opt` mode, median link+compile+emit latency is 22.09 ms
for the 58.1 KiB gfx12 motif and 11.77 ms for the 16.1 KiB gfx11 motif; p95 is
23.26 and 12.91 ms respectively. A non-optimized control was 62.71 and 45.81
ms, so host build mode is part of the measurement contract. The
earlier 4.67--8.47 ms prepared-Low measurements bypass the default pipeline
and are only HSACO-emission floors. Detailed current measurements are in
[`results/loombc-to-hsaco-latency.json`](../../research/spike-004-default-pipeline-microkernels/results/loombc-to-hsaco-latency.json).

Artifact cardinality, rather than single-object emission latency, is the
integration gate. The synchronous path must derive the post-specialization
program key before native emission, coalesce equal in-flight misses, and consult
a bounded persistent artifact cache. For popular targets, initialize that
cache by compiling the expected artifact set in parallel. If the working set is
on the order of a dozen artifacts per deployed target, tens of milliseconds
per true miss remain a bounded startup transient and may be
cheaper than navigating and loading a large precompiled Tensile library.
Explicit preparation or an asynchronous incumbent fallback is required only
for deployments whose first-call latency budget cannot absorb that transient;
code-object load time must be added before setting that policy.

Do not preserve Tensile's adjacency buckets merely to avoid compilation:
generality belongs in shared source and exactness in the specialization input.
Exactness need not imply one artifact per request because specialization can
erase distinctions that do not change the compiled program. If that collapse
does not reduce the real demand corpus to a small, bounded number of HSACOs per
family/type/fusion class, the design must revisit schedule/configuration
factoring or its miss policy before provider integration.

In this document, an **authored kernel** is a Loom algorithm/source body. A
**configuration** selects schedule parameters for that body. A **specialized
executable** is one JIT result for a target and bounded request facts. Tensile's
hundreds of solution records may collapse to a few authored kernels, dozens of
useful configurations, many specialization requests, and only a handful of
distinct program-derived executable entries per family/type/fusion class.
Those counts must not be conflated.

### Target breadth policy

Source target scope and emitted artifact scope are intentionally different:

1. Author a mechanism against the broadest legal generic target, such as
   `gfx11-generic` or `gfx12-generic`.
2. Bind the exact device profile at JIT specialization and include it in
   evidence and cache identity.
3. Allow ordinary configuration and provider selection to choose schedules
   that happen to perform best on one ASIC without renaming them as that ASIC.
4. Introduce an exact-ISA source definition only with a recorded legality or
   structural witness: for example an instruction absent from siblings, an
   incompatible fragment carrier, or a target-only memory/synchronization
   contract.

The first mechanism probe demonstrates this split. Its Loom target declarations
are `gfx11-generic` and `gfx12-generic`; they compile and run as exact
`gfx1100` and `gfx1201` HSACOs. The family split is justified by an actual WMMA
operand-fragment ABI difference, while the physical device names remain
specialization parameters. A macro-tile or prefetch preference would not by
itself justify another source target.

### Epilogue specialization experiment

TensileLite's separation is evidence to extract, not an architecture to copy.
In the installed `gfx1201` library, the FP16-input/FP16-output/HPA NN form is
split at the semantic router into distinct lazy shards:

- no bias, no alpha-vector scale: 30 solutions behind an Equality table with
  25 exact rows;
- bias-capable plus alpha-vector scale: 517 solutions behind a GridBased table
  with 6,696 rows; and
- bias plus auxiliary output: one solution behind one Equality row.

The primitive public request can nevertheless force supported algorithms from
both the bias-capable and no-bias shards. At the first sampled shapes, the best
forced result changes shard: `gfx1201` solution `133764` (no bias) wins at
256 cubed, while solution `133222` (bias-capable) wins at 1024 cubed. Thus shard
identity is neither core-algorithm identity nor a safe performance partition.
It is partly a consequence of static artifact generation and corpus coverage.

The first Loom family will therefore make the epilogue an explicit closed
specialization choice, initially:

```text
has_bias = false | true
bias_axis = row | column       # meaningful only for bias
bias_type = <provider type>    # meaningful only for bias
```

The Loom file declares the choice as configuration and reads it in an ordinary
`scf.if`; the provider binds it before compilation. The function signature may
carry a stable superset argument bundle, but the selected source region must
erase unused operands and control flow from the emitted executable.
The experiment will compile matched core schedules with `none` and `bias`, then
compare native matrix instructions, load/store paths, VGPR/LDS allocation, and
latency. The intended invariant is that core load/MMA scheduling remains shared
unless the epilogue's live-range or publication cost measurably requires a
different configuration. This same method decides whether activation, scales,
or auxiliary output join the family; it does not assume that every nominally
fusable operation is free.

### Router-reduction experiment

For each incumbent problem family, produce a **router reduction report**:

1. Flatten the runtime tree into leaf paths, retaining the reason for every
   predicate and fallback edge.
2. Remove packaging-only nodes and show that this does not change the set of
   semantically eligible recipe families.
3. Convert semantic, target, alignment, stride, beta, edge, and epilogue nodes
   into an explicit proposed Loom specialization signature.
4. Group remaining leaves by normalized physical schedule family, ignoring
   names and parameters that specialization derives mechanically.
5. Replay the router corpus and count how many distinct choices remain at each
   reduction step.
6. Compile representative family members with exact facts and verify that the
   predicted dead paths, edge handling, address arithmetic, and epilogue
   differences actually disappear from compiler/native evidence.
7. Measure whether one specialized family covers neighboring incumbent cells
   without regression. Split the family again when physical evidence requires
   it.
8. Compare three policies on held-out requests: incumbent routing, a reduced
   static policy, and reduced-policy-plus-bounded-autotune.
9. Attempt to express every remaining leaf as a configuration or specialization
   of the primitive gfx-family kernel. Require a new authored kernel only after
   identifying an incompatible dataflow, synchronization/publication protocol,
   or instruction family.

The report should include a funnel such as:

```text
serialized leaf paths
  -> semantically distinct request forms
  -> legal recipe families
  -> distinct physical schedule families
  -> measured schedule-policy regions
```

No target counts are assumed in advance. The important result is which
dimensions disappear and why, plus held-out evidence that the reduced policy
retains performance.

The expected result for the initial Radeon scope is a small family count. A
large count is not accepted merely because Tensile names many solutions: the
report must identify the irreducible algorithmic differences. Likely reasons
for an additional authored body include VALU versus matrix-instruction
accumulation, a genuinely different split-K/StreamK publication protocol, or a
fusion whose dataflow cannot be expressed as an epilogue composition.

### Proposed miss and tuning behavior

The synchronous provider path should never search the full extracted corpus.
On an artifact-cache miss it may either compile the policy's first candidate
within a declared latency budget or use the incumbent immediately while a
bounded worker compiles and evaluates candidates. The exact product behavior
depends on the provider's eventual asynchronous/fallback contract.

The incumbent tables are valuable priors for candidate generation:

- start with the routed winning recipe;
- add recipes that win adjacent router cells or share the same schedule family;
- specialize those recipes to eliminate irrelevant generic paths; and
- permit a small set of Loom-native mutations justified by compiler evidence.

Persist a tuning result by target, specialization class, workload policy, and
compiler/source identity. Do not generalize an online winner beyond the
measured region without boundary tests. Background tuning must use bounded
memory, compilation, and device time and must not perturb scored foreground
measurements.

## Deliverables

### 1. `blas-lab`: an explicit API comparison tool

Build a small standalone executable and supporting Python/report tools against
the public rocBLAS, hipBLASLt, HIP, and Loom embedding APIs. It is not a wrapper
around `rocblas-bench`, `hipblaslt-bench`, or tuning tools. Those programs hide
selection, allocation, data, timing, and workspace choices that must be part of
the experiment.

The executable should have backend adapters for:

- rocBLAS default selection;
- rocBLAS forced solution index;
- hipBLASLt default heuristic selection;
- hipBLASLt forced algorithm;
- a direct loaded Loom artifact; and
- the eventual Loom BLAS provider path.

The harness owns device selection, allocation, initialization, descriptors,
enumeration, forcing, workspace, launches, timing, readback, and result
serialization. It must be possible to run the same materialized request and
input bytes through every applicable backend in one process.

#### Canonical request

The request schema must retain all values that can change semantics,
applicability, routing, or timing:

- M, N, K, batch count, grouped/strided-batched form;
- A/B transposes, logical layouts, leading dimensions, strides, byte offsets,
  and alignment;
- A, B, C, D, compute, scale, bias, and auxiliary data types;
- alpha and beta values, types, pointer modes, and zero/one classification;
- C/D aliasing and overwrite behavior;
- epilogue, activation, bias source/orientation, scales, auxiliary output, and
  amax behavior;
- math mode, atomics/determinism policy, stochastic behavior, and API flags;
- maximum device and host workspace;
- requested algorithm/index or default-selection mode; and
- stream and graph/capture mode when those become supported.

The canonical serialization is versioned and hashes into the experiment ID.
Unknown descriptor attributes are errors, not dropped fields.

#### Provenance

Every result records:

- PCI identity, device ordinal, exact ISA/target ID, CU count, wave mode, and
  relevant clock/power state;
- HIP runtime/driver identity;
- absolute loaded library paths, semantic versions, and content hashes;
- logic, mapping, shard, and code-object paths and hashes;
- source revisions and dirty state;
- Loom compiler, target profile, pass program, source, and configuration
  hashes;
- process environment variables that affect selection or code generation; and
- host, clock, thermal, exclusivity, warmup, and timing policy.

This is required because algorithm indices and library-logic indices are not
stable across library builds.

#### Incumbent algorithm enumeration

For rocBLAS, use `rocblas_gemm_ex_get_solutions` (or the applicable typed
variant) for the exact request, then invoke each returned solution through
`rocblas_gemm_ex` with its `solution_index`. Measure the default path
separately. Record unsupported or failed forced solutions rather than
discarding them.

For hipBLASLt, use the extension enumeration path (`getAllAlgos`), validate
each candidate with `matmulIsAlgoSupported`, retain workspace and support
status, and explicitly invoke every eligible algorithm. Join
`getIndexFromAlgo`, solution name, and kernel name to the source/runtime trace.
Also run the ordinary public heuristic path and record its ordered results.

The best measured eligible forced solution is the performance oracle. The
default-selected solution is a separate product behavior measurement. This
distinction detects stale or weak routing and prevents the Loom target from
being set by an arbitrary heuristic limit.

#### Timing

- Time GPU completion with events on the actual submission stream.
- Measure first-call selection, lazy loading, JIT, and steady-state kernel
  execution as separate regions.
- Warm caches and cold/streaming operands are distinct workload policies.
- Use rotating allocations whose aggregate working set exceeds the intended
  cache when measuring streaming behavior.
- Compare baseline and candidate in an interleaved schedule such as ABABA,
  with randomized outer ordering across repetitions.
- Preserve raw samples; report distributions and paired ratios, not only a
  minimum or median.
- Keep profiling/counter runs out of the score. Join them by artifact and
  request identity.
- Time an entire solution launch sequence when split-K or auxiliary kernels
  are used, not just its main contraction kernel.

### 2. Corpus extractor and normalized IR

Create versioned parsers for the classic list-format and current dict-format
logic. Reuse the projects' own readers where practical, but serialize a stable,
tool-independent result so later analysis is not coupled to mutable Python
objects or msgpack implementation details.

The normalized corpus has four linked record types.

#### Problem-family record

- problem type and index assignments;
- data/compute types and transpose/layout contract;
- batched/grouped and strided forms;
- beta, high-precision accumulation, stochastic, sparse, and math flags;
- epilogue, bias, scale, activation, gradient, auxiliary, and UserArgs
  contracts; and
- all normalization rules needed to construct a matching key.

#### Router record

- ordered enclosing predicates and fallback classification;
- matching mode, key properties, and index order;
- every table row, recorded performance metric, source location, and solution
  reference;
- GridBased batch-folding and search mode;
- solution/task/hardware predicates and workspace requirements; and
- lazy shard and code-object resolution.

Each path and predicate also carries the classification from “What JIT should
remove from the router,” so the corpus can produce router-reduction reports
rather than merely deserialize the incumbent tree.

#### Kernel recipe record

- macro and thread tiles, workgroup and wave topology;
- depth-U, local split-U, global split-U, StreamK, persistent, and workgroup
  mapping choices;
- matrix instruction or VALU operation and fragment geometry;
- global load vector widths, coalescing, address behavior, and edge policy;
- LDS size, shapes, padding/swizzling, write/read ownership, and barriers;
- direct-to-LDS/direct-to-VGPR paths;
- global/local prefetch distance, buffering, and loop scheduling;
- accumulator and output vectorization, store/remap, atomic/reduction, and
  epilogue policy; and
- generator defaults and derived values, with their source revision.

Omitted recipe fields are not assumed to have a universal default. Defaults
are resolved using the generator revision and stored explicitly.

#### Exact schedule record

The schedule record is mandatory evidence even when a Radeon translation can
begin from the recipe. It stores:

- expanded instruction classes and loop phases;
- memory address forms and per-lane/coalesced request shape;
- issue order, dependency chains, waits, barriers, and clauses;
- matrix/VALU instruction spacing and independent accumulator chains;
- LDS producer/consumer stages and bank mapping;
- SGPR/VGPR/AGPR/LDS allocation, spills, scratch, and occupancy;
- edge, tail, split/reduction, and output-publication paths;
- kernel descriptor and launch metadata; and
- source recipe, generated assembly, code object, and disassembly hashes.

Initially this may combine structured extraction with annotated native regions.
It must not claim semantic equivalence merely because instruction mnemonics or
counts look similar.

### 3. Router fidelity test suite

Before kernel work, test the explainer against the actual runtime:

- every sampled Equality row selects its referenced eligible solution;
- sampled GridBased points, between-points, outside-grid points, batch folds,
  tails, and workspace boundaries reproduce runtime selection;
- default API routing, direct library routing, and observed dispatch agree;
- forced indices resolve to the expected shard, symbol, and launch sequence;
- nonstandard leading dimensions, alignments, beta modes, and epilogues hit the
  expected predicates; and
- selection is repeated with cold and warm lazy-library state.

The initial sampling strategy should cover every distinct selected solution,
every predicate boundary, every fallback edge, and a representative point per
router cell. Exhaustively replaying thousands of duplicate grid rows is useful
as an offline test but is not the only coverage metric.

Router fidelity is followed by router reduction. For every initially targeted
problem family, the reduced policy must be evaluated on held-out routing rows
and synthesized boundary points. Report both performance regret versus the
best enumerated incumbent and the number of schedule families/artifacts needed
to achieve it. This tradeoff, not fidelity to the incumbent tree shape, decides
the Loom router.

### 4. Loom recipe and schedule vocabulary

Start with one primitive dataflow per required gfx family. Instantiate it for
each accelerated arithmetic signature before adding fusions. Every
instantiation begins from the same deliberately narrow semantic seed:

```text
batch = 1, NN, aligned dense storage
alpha = 1, beta = 0
no bias, activation, scales, auxiliary output, or split workspace
D = A * B
```

This is a bring-up boundary, not the final GEMM contract. It isolates operand
movement, accumulation, and publication before descriptor variants obscure
the schedule. Add transpose/layout views, dynamic sizes and edges, general
alpha/beta, batching, and fusions in that order unless incumbent evidence
shows that a later feature requires a different core dataflow.

Once the primitive mechanism passes, add `none` and `bias` as two
specializations of the same Loom file before inventing a second FP16 GEMM body.
Treat this as a generalization gate: the compile evidence must show removal of
the unselected epilogue, and the physical comparison must show whether the same
schedule configuration remains competitive for both choices.

The initial arithmetic ladder is:

1. FP16 inputs with FP32 accumulation on `gfx1100` and `gfx1201`.
2. BF16 inputs with FP32 accumulation on both targets.
3. Native low-precision accumulation variants (FP16 and BF16 accumulation)
   where exposed by the provider.
4. I8 signedness combinations with I32 accumulation on both targets.
5. Packed I4 signedness combinations with I32 accumulation where exposed.
6. FP8/BF8 homogeneous and mixed input pairs with FP32 accumulation on
   `gfx1201`, preserving each exposed format's exact semantics.
7. The separate `gfx906` VALU/dot-product primitive, beginning with the
   highest-value source-backed arithmetic signature.

The order makes the common floating-point pipeline work first, then varies
fragment and accumulation ownership, then packed integer and 8-bit floating
formats. It is not a license to stop after FP16: completion of the primitive
milestone requires every provider-visible native WMMA signature in the target
capability inventory.

Implement reusable Loom motifs/components for the mechanisms shared by those
primitive kernels and their variants:

- target-specific VALU or matrix fragment operation;
- tiled global reads with explicit lane ownership and alignment facts;
- LDS staging, padding, and ping-pong/multi-buffered pipelines;
- direct-to-VGPR paths where profitable;
- unrolled K loop with explicit prefetch and wait placement;
- fragment accumulation and vectorized/edge-safe stores;
- beta and initial epilogue variants;
- workgroup mapping and batched addressing; and
- split-K plus reduction only when an incumbent winner requires it.

The vocabulary should express a schedule, not recreate the complete Tensile
parameter surface. Add a mechanism when an extracted winning family requires
it or a controlled experiment demonstrates it. Preserve enough explicit
schedule control for later Instinct work; do not hide issue order and memory
phase boundaries behind a high-level GEMM operation whose lowering cannot be
examined or constrained.

The default extension rule is:

- a new type changes element conversion, accumulator/fragments, and instruction
  selection but reuses the enclosing load/pipeline/store structure where
  physically valid;
- a fusion composes an epilogue with the primitive and becomes a separate
  authored body only when it changes ownership, liveness, synchronization, or
  launch structure materially;
- a tile, depth, vector width, prefetch distance, or workgroup mapping is a
  configuration, not a new kernel; and
- an exact M/N/K, alignment, transpose, beta class, or tail is preferably a
  specialization of a kernel/configuration pair.

### 5. Evidence store and agent interface

Each research task produces a self-contained evidence bundle:

```text
<experiment-id>/
  witness.json
  request.json
  incumbent-selection.json
  router-trace.json
  recipe.json
  schedule.json
  candidate-record.md
  source.loom
  correctness.json
  compile-report.json
  native-metadata.json
  candidate.hsaco
  candidate.disasm
  benchmark.json
  decision.md
```

Large input buffers may be content-addressed outside the directory. Every
record refers to hashes, never an unversioned `latest` path. The schema is
intended to become a goal/skill interface later: an agent receives a witness
and a bounded question, and returns an evidence bundle and decision rather
than a conversational transcript.

### 6. JIT provider integration

Use Loom's embedded compiler API. Immutable source, context, target profile,
compiler, and pass programs may be shared. Each compilation worker owns its
workspace and mutable module. Compilation returns two coupled products:

- a prepared target module emitted as an in-memory HSACO; and
- a launch-config artifact evaluated by the provider for the concrete
  workload.

The provider/runtime remains responsible for allocation, executable loading,
argument serialization, submission, and synchronization.

Do not key the executable cache directly from the raw request tuple. Use a
pre-emission specialization stage analogous to a C preprocessor feeding
`ccache`:

1. Load the immutable linked `.loombc` and bind exact target, M/N/K-derived
   facts, arithmetic, epilogue, and schedule configuration.
2. Run the same specialization/canonicalization boundary that precedes target
   emission, requesting transformed Loom bytecode but no HSACO.
3. Hash that canonical compiled-program representation together with the
   non-serialized compiler inputs below.
4. Look up or coalesce compilation by that derived key. Emit and load HSACO
   only on a miss.

The checked-out public API exposes the ingredients, but not a single polished
"derive executable key" call. The closest concrete path is
`loomc_compile_module` with a typed config module, target-specialization
options, and `LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE`, followed by hashing
the returned bytecode artifact; `loomc_emit_module` is a separate later step.
The exact current API recipe is an implementation assumption until the spike
verifies its pass boundary, cost, determinism, and key stability. The
pre-emission identity itself is a final design invariant: raw request
cardinality must not determine HSACO cardinality. If the current APIs cannot
expose that identity cheaply and deterministically, the project changes Loom
rather than adopting a tuple-keyed executable cache. Compiler facts that
bytecode serialization does not retain must remain explicit key components.
Product-frontier kernel requests may provide a cheaper canonical input when the
BLAS integration adopts that API.

A first canonical-text proxy demonstrates the required behavior and why the
boundary must be explicit. Config values M=32 and M=48 both specialize an
intrinsic `M % 16` partition to zero. After `canonicalize,dce` their serialized
modules still differ because resolved `config.def` symbols remain. After
`symbol-dce`, the dead bindings disappear and the two canonical program hashes
are identical; M=33 produces partition one and a different hash. The production
experiment must reproduce this through `loomc`, bytecode, exact target
specialization, and the actual pre-emission pipeline.

This optimization is not on the critical path for the first kernel spike.
Spike 001 fully specializes each candidate request and configuration through
HSACO emission and records the resulting artifact hash. Benchmarking candidate
program-identity forms, moving the lookup ahead of emission, and measuring
collapse across a shape corpus belong to the later JIT/provider-hardening
experiment. Until that experiment, phase-one tooling intentionally treats the
HSACO as the specialization result and does not claim cache reuse across raw
requests.

The derived key must include or be namespaced by:

- exact target profile and relevant device properties;
- specialized canonical kernel-request/bytecode hash;
- compiler and pass-program identity;
- device ABI and public export; and
- every emission option or retained compiler fact not represented in the
  serialized specialized program.

Raw M/N/K and configuration values are inputs to specialization, not necessarily
fields in the final executable key. Two requests alias only when Loom produces
the same canonical specialized program and all external key components match.
This permits exact request reasoning while naturally collapsing intrinsic
M/N/K partitions, layout classes, or disabled epilogues into the small number
of artifacts that actually differ. Record both the raw specialization input
and the derived key so a collision or missed reuse is explainable. The design
expects a small number of distinct HSACOs--provisionally on the order of a
dozen--per deployed target across the arithmetic/fusion classes in active use.
The derived-key experiment quantifies that collapse and identifies the
intrinsic partitions; it does not relax the program-derived-key invariant.

Use bounded memory and disk caches, atomic publication, failure/negative cache
entries, and per-key compilation coalescing. Separate compile latency from
first-load and steady-state launch latency. A provider request for an
unvalidated cell, a failed compilation, or an exceeded latency/workspace bound
falls back to the incumbent provider.

## BLAS adaptation of the Loom agent workflow

The base workflow is:

```text
production witness
  -> semantic cut
  -> checked source and workloads
  -> compiler hypothesis
  -> compiler and native evidence
  -> controlled physical experiment
  -> integrated result
```

For BLAS reconstruction, use the following concrete gates.

### Gate 0: freeze an API witness

Materialize one canonical request, deterministic nonzero inputs, workspace
policy, target, library identity, and timing policy. Include neighboring
requests across the suspected router or tail boundary. The production witness
may come from a client trace, a logic row, or a deliberately selected coverage
cell, but its origin and weighting must be explicit.

### Gate 1: establish numerical truth

For small and moderate cases use an independent high-precision CPU reference.
For large cases use sampled/blocked CPU checks plus cross-backend comparison;
agreement between rocBLAS and hipBLASLt is supporting evidence, not an
independent oracle. Test nonzero alpha/beta, distinct A/B/C values, overwrite
and accumulation, tails, odd leading dimensions, NaN/Inf policy, and every
included epilogue.

The tolerance model is type- and accumulation-aware and recorded with the
case. A kernel cannot earn performance measurement after an unexplained
numerical difference.

### Gate 2: reproduce incumbent selection

Enumerate eligible incumbent solutions, capture the default selection, force
the alleged winner, and observe its dispatch. The source explainer must join
the request through the router to the same recipe, shard, code object, symbol,
and launch sequence. Freeze the best eligible measured incumbent as the
performance oracle for this witness.

### Gate 3: extract the mechanism

Produce the normalized recipe and exact schedule record. State which features
are required for the candidate and which are incidental. For Radeon, begin
with the YAML plus generator source and confirm against disassembly. For a
future Instinct task, assembly/disassembly research is the primary schedule
source, with the YAML acting as an index and parameter summary.

### Gate 4: write the candidate record before editing

Every candidate starts with:

```text
Production boundary:  exact witness and neighboring coverage cells
Incumbent identity:    router trace, recipe, schedule, and native hashes
Independent variable: one causal schedule/compiler/provider change
Physical hypothesis:  mechanism expected to improve
Compiler consequence: report or native delta expected before execution
Correctness gates:    local, sanitized, boundary, and API-integrated checks
Discriminator:        cheapest request that executes the changed path
Success and stop:     thresholds fixed before results are observed
```

“One variable” means one causal schedule change. A wave topology change may
require coordinated fragment ownership, LDS exchange, and loop changes; it is
not usefully tested as a one-line subgroup edit.

### Gate 5: ask the compiler first

Format and plan the source, run cheap correctness cases, and capture a compile
report before running broad GPU benchmarks. Confirm:

- intended provider/instruction selection;
- target and wave size;
- launch geometry and dynamic launch facts;
- memory/vector shape and alignment consequences;
- register, LDS, spill, scratch, and occupancy state;
- barriers, waits, and loop structure in native evidence; and
- the predicted consequence in the candidate record.

If Loom cannot express or lower a required mechanism, reduce it to a standalone
compiler packet containing the smallest source, target/profile, command,
expected result, actual report/IR/native result, and a correctness case only
when execution is necessary. Do not bury a compiler feature request inside a
large GEMM tuning branch.

### Gate 6: earn physical measurement

After numerical and compiler gates pass, run a short paired pilot on the
cheapest discriminating witness. Stop early when it misses the predeclared
threshold. Promote promising candidates to interleaved full sampling across
the target cell, its router boundaries, cache policies, and tails.

Use load-only, cache-resident, dispatch/publication, bandwidth, and relevant
compute probes to classify the performance regime before hill climbing.
Resource counts are constraints and explanatory evidence, not objectives.

### Gate 7: generalize before publication

A winner must pass:

- all correctness cases in its declared domain;
- every router boundary and alignment/workspace predicate it claims;
- cold JIT/load and steady-state execution limits;
- repeat measurements on the target device; and
- the provider's fallback and cache behavior.

Publish a narrow provider cell with its exact predicate. Broaden it only after
new evidence. Do not infer that the full region represented by one incumbent
solution index is safe if the Loom kernel has not been tested over that region.

### Compact loop

```text
freeze request and incumbent identity
  -> explain route and extract recipe/schedule
  -> write candidate hypothesis and stop rule
  -> plan/check
  -> inspect compile/native evidence
  -> short paired pilot
  -> interleaved boundary and corpus measurement
  -> publish narrow cell or retain incumbent fallback
```

## Workload and coverage construction

There are three distinct corpora; none should silently substitute for another:

1. **Router corpus:** Equality rows, GridBased points and boundaries, fallback
   edges, and every distinct selected solution in the incumbent logic.
2. **API capability corpus:** layouts, types, beta modes, batching, alignments,
   tails, epilogues, workspace limits, and behavior flags promised by the
   provider.
3. **Demand corpus:** weighted real client requests. This determines which
   valid cells matter enough to optimize first.

The initial router corpus can be generated without guessing popular matrix
sizes. Cluster rows by normalized problem family and selected recipe, select
representative interior points, and synthesize points immediately across each
decision boundary. Add small, skinny, square, large, K-small, K-large,
non-multiple, and batch-fold cases even when they are sparse in the tables.

The demand corpus remains an explicit input to the project. Until one is
chosen, report unweighted per-cell results and distributions; do not collapse
them into a single marketing-style geomean.

The first vertical slice is the primitive boundary above with FP16 inputs and
FP32 accumulation on RDNA3/RDNA4. Use it to prove the load/compute/store
schedule and evidence loop, then walk the accelerated arithmetic ladder before
the first fusion. Each arithmetic variant acquires general alpha/beta and edge
forms, but those forms remain specializations of the primitive. A scalar FP32
smoke test may precede this work but does not satisfy the primitive milestone
because it does not exercise the target's dominant matrix path.

## Performance and acceptance policy

“Meets or exceeds ROCm BLAS” applies to the best eligible incumbent solution
for the identical request, not only to its default-selected heuristic. It also
requires correct behavior and bounded integration costs.

For each promoted provider cell:

- numerical acceptance passes for every request in the declared domain;
- router/source/runtime/dispatch identity is explained with no unresolved
  mismatch;
- no request exceeds the agreed hard slowdown threshold versus the paired best
  incumbent;
- the cell meets the agreed aggregate target on the chosen demand corpus;
- workspace and allocation behavior do not exceed the declared provider bound;
- steady-state timing includes the complete launch sequence;
- JIT, code-object load, and first-call latency are reported separately and
  satisfy their own bounds; and
- artifacts and measurements reproduce from the evidence bundle.

Schedule congruence is not a separate promotion requirement. It is a powerful
reconstruction method when a candidate is slower or its mechanism is not
understood. A candidate that passes correctness, legality, resource, and
integration gates and statistically beats the best eligible incumbent is
accepted even if it uses a different schedule. The schedule delta should still
be retained because it may expose reusable compiler improvements or explain
why the result does not generalize to adjacent cells.

Numeric thresholds should be fixed with the initial demand corpus rather than
invented during tuning. A reasonable staging policy is: parity permits an
experimental cell, a statistically supported win permits default-on for that
cell, and any unexplained correctness or hard-tail regression disables it.

## Representability assessment by target

### `gfx1201`

**Assessment:** best first end-to-end target.

It has a large TensileLite corpus with explicit Equality and GridBased routing,
wave32 matrix-instruction recipes, fused descriptors, and installed runtime
artifacts. The sampled HHS recipes predominantly use one-level split and
ordinary staged or direct-to-VGPR pipelines; a smaller set uses dynamic or
fixed global split-U. These are credible Loom mechanisms.

The default-pipeline FP16 anchor now beats the selected incumbent, establishing
an end-to-end performance witness. Family-generic Low carrier specialization
and locked-order preservation are validated on main. Remaining risks are
structured composition for exact loops, matching wait density across a broader
shape basket, fused epilogue ABI, and multi-kernel split/reduction paths. Start
with unsplit winners before adding the less common mechanisms.

### `gfx1100`

**Assessment:** feasible second target, and useful evidence of cross-generation
reuse.

Both rocBLAS and hipBLASLt source logic exist, while this installation packages
hipBLASLt target artifacts. The laboratory must determine the actual public API
coverage rather than encode the original rocBLAS-only assumption. Compare
recipe families with `gfx1201` to decide which Loom motifs are genuinely shared
and which require RDNA3 specialization. The all-High FP16 anchor is correct but
1.34x slower and spills four values; the prepared-Low oracle proves the target
schedule. The next experiment reconstructs it with structured source control
flow and straight-line locked Low fragments, since arbitrary multi-block
locked helpers are outside #513's deliberate contract.

### `gfx906`

**Assessment:** kernel structure appears straightforward; compiler target
enablement is the blocking dependency.

The sampled rocBLAS HHS family uses assembly/source VALU kernels without matrix
instructions. Loom's vector, view, unrolling, LDS, and explicit target concepts
are sufficient in principle. However, the current target catalog has an exact
processor/code-object mapping without a complete exposed physical target row,
and compilation fails. Add target descriptors, emission/metadata behavior,
occupancy/resource modeling, instruction legality, and tests before promising
performance. A scalar/vector GEMM smoke test must compile, load, execute, and
report correct descriptor/resource data before starting tuning.

## Project sequence and evidence gates

Elapsed human estimates are deliberately omitted. This workflow is intended
for an agent research loop, and progress is measured by reproducible evidence
rather than guessed calendar duration.

| Work package | Exit condition | Current state |
| --- | --- | --- |
| Laboratory and schemas | One request enumerates, forces, checks, and measures each installed incumbent; a loaded Loom kernel runs on each supported target | Public API comparison harness forces hipBLASLt/rocBLAS solutions, launches Loom HSACOs, performs sampled or full CPU differentials, and records paired timings; BF16 support is in progress |
| Source/runtime extractor and router explainer | Sampled Equality/GridBased/classic Tensile routes agree with runtime/dispatch and emit normalized packets | Runtime shards, source recipes, code objects, and bounded native symbol summaries joined for sampled winners; broader route replay and dynamic schedule extraction remain |
| Derived-key experiment | Config and exact shape facts specialize a `.loombc`; equal resulting programs alias before HSACO emission and unequal programs do not | Optimized-host bytecode link/default-compile/emit latency is 11.77--22.09 ms median; pre-emission key stability, demand-corpus cardinality, parallel startup behavior, and code-object load remain open |
| `gfx906` Loom enablement | Compile/load/correctness/resource-report smoke tests pass | Blocked on missing physical target support, isolated from GEMM work |
| First `gfx1201` vertical slice | Primitive FP16-to-FP32 WMMA GEMM reaches parity across a bounded cell through the default pipeline | The 1024-cubed point is accepted: 25.04 us versus 28.78 us, correct and spill-free. #513 validates its family-generic helper and locked ordering with a byte-identical artifact; broader-cell validation remains open. |
| gfx11 family specialization | The shared family passes on the RDNA3 witness with structural variants only where justified | Not complete: default-pipeline High is 61.0 us versus 45.46 us and spills; exact prepared-Low schedule remains an oracle for reconstruction as structured source control flow plus straight-line Low fragments. |
| `gfx906` VALU family | One useful family reaches parity across a bounded cell | Incumbent schedule recovered; Loom target enablement precedes implementation |
| Accelerated arithmetic expansion | Every provider-visible native matrix signature has a correct primitive and measured representative cells | Instruction inventory and compile witnesses exist; gfx11 BF16 schedule port compiles, while BF16/I8/FP8 timing conclusions remain open |
| Layout/epilogue expansion | Agreed key demand corpus is covered with provider fallback elsewhere | Standalone bias/no-bias specialization witness passes and removes the bias load; composed GEMM row/column differential remains open. |
| JIT/provider hardening | Bounded cache, concurrency, failure, first-call, and packaging behavior pass | Design only |

The dominant early uncertainty is whether incumbent leaves collapse to one
primitive per broad gfx family plus type/fusion configurations. The
router-reduction and derived-key experiments measure that directly before the
project multiplies authored algorithms.

### Milestones

1. **Selection truth:** custom laboratory and router explainer reproduce one
   representative route on each usable incumbent target.
2. **Compiler truth:** required Loom mechanisms and `gfx906` target gaps have
   small passing or explicitly blocked compiler packets.
3. **First primitive:** unfused FP16→FP32 WMMA on RDNA3/RDNA4 and the chosen
   `gfx906` VALU/dot path pass correctness and expose the intended native
   mechanism.
4. **First parity cell:** the first primitive plus its required edge and
   alpha/beta specializations meets paired performance gates.
5. **Accelerated type coverage:** every provider-visible native WMMA arithmetic
   signature has a correct primitive instantiation and measured representative
   cells before fusion work begins.
6. **Small-family proof:** held-out incumbent leaves map to configurations or
   type/fusion variants without inventing new core algorithms.
7. **Incremental provider:** the Loom provider owns a narrow, explicit set of
   target/problem cells with reliable fallback.
8. **Coverage expansion:** prioritize uncovered cells by demand weight and
   incumbent advantage, not by parameter-grid completeness.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| YAML is mistaken for the runtime route | Join source prediction to API selection and observed dispatch; hash runtime shards and code objects. |
| Recipe fields hide generator defaults or derived behavior | Resolve defaults against the exact generator revision and confirm the expanded schedule in assembly/disassembly. |
| A translation matches tiles but not schedule | Require expected compiler/native consequences before benchmarking. |
| The best forced solution differs from the default | Track both; optimize against best eligible and file router improvements separately. |
| Algorithm indices drift | Treat index as build-relative and retain names, paths, versions, and hashes. |
| Benchmark tools hide favorable state | Own allocations, data, workspace, selection, launch sequence, and timing in `blas-lab`. |
| Cache-hot inputs or zero data inflate results | Deterministic nonzero inputs and explicit rotating/cold versus resident policies. |
| Tuning overfits YAML points | Test cell interiors, boundaries, tails, alignments, and a separately weighted demand corpus. |
| Loom lacks a required target or schedule control | Reduce to a standalone compiler packet; land the capability independently. |
| JIT latency erases kernel gains | Separate compile/load/steady-state budgets, cache by exact identity, and retain incumbent fallback. |
| Broad abstraction weakens future Instinct schedules | Keep exact schedule records and explicit issue/memory phase controls even when Radeon recipes need less detail. |
| Scope expands into all of Tensile | Add only mechanisms required by winning demand-weighted families; do not clone the full parameter language. |

## Alternatives Considered

### Treat rocBLAS and hipBLASLt as black-box performance oracles

Rejected. It can find a number to beat but does not expose the selection
boundary or the mechanism that achieved it. It also confuses default heuristic
quality with kernel quality and cannot produce an auditable port.

### Drive the project through vendor benchmark/tuning tools

Rejected as the primary workflow. They are useful source references and can
cross-check isolated findings, but hide API descriptors, solution enumeration,
workspace, data, and timing choices that this project must control explicitly.

### Port generated assembly directly

Rejected for Radeon as the default. It would preserve one artifact while
discarding Loom's structured specialization and JIT advantages, and would be
fragile across targets and compiler changes. Generated assembly/disassembly
remains mandatory schedule evidence. For future Instinct kernels it may also
motivate new exact Loom schedule controls or narrowly encapsulated low-level
operations, but copying text is not the research method.

### Translate every Tensile parameter into Loom

Rejected. The result would recreate a large historical generator before
proving which mechanisms matter. Normalize the full incumbent corpus for
analysis, but implement Loom motifs in demand order.

### Tune a generic GEMM from scratch

Rejected as the initial strategy. It ignores years of architecture-specific
recipe and routing information. Novel schedules are welcome only as bounded
candidates after the incumbent mechanism and performance envelope are known.

### Generate all kernels ahead of time

Rejected as the target architecture. AOT artifacts can bootstrap comparison
and reduce deployment risk, but specialization and incremental JIT coverage
are central benefits. The same Loom source and emission boundary should support
both cached JIT and optional prewarming/precompilation.

### Wait for complete BLAS coverage before integration

Rejected. Per-target/provider routing makes narrow validated cells useful and
keeps an incumbent fallback. Early integration is also necessary to measure
the real JIT, cache, ABI, and launch costs.

## Open decisions

1. **Demand corpus:** Which clients or captured traces define “key” GEMMs and
   their weights? Logic-table coverage can start the project, but cannot define
   product importance.
2. **Provider-visible MMA surface:** Does the new provider expose packed I4 and
   every FP8/BF8 mixed pair, or should some hardware-capability rows remain
   internal until the public type surface grows?
3. **Performance gate:** What paired slowdown ceiling, aggregate target, and
   confidence rule should govern experimental and default-on cells?
4. **JIT budget:** What first-use latency and persistent cache/storage limits
   are acceptable to the provider?
5. **Numerical contract:** Should initial cells require bitwise agreement for
   deterministic modes where achievable, or a type-aware error contract plus
   behavioral equivalence?
6. **Repository placement:** Should `blas-lab`, corpus schemas, and evidence
   tooling begin as workspace tools, in hrx-system beside Loom, or in the new
   BLAS API repository once available?

The first three decisions materially affect the size and acceptance criteria
of the effort. The laboratory and source-extraction work can begin before they
are final, provided results remain per-request and are not summarized into a
premature overall score.
