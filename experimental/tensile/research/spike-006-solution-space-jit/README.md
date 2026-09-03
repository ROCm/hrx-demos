# Spike 006: solution space and JIT cardinality

## Outcome

The Radeon approach still looks small at the authored-kernel level, but the
single FP16 motif from each prior spike is not a complete schedule policy. A
bounded, unweighted router corpus exposes one normalized mechanism family for
the gfx11 classic HHS source and five mechanism families covering 93.3% of the
aligned gfx1201 GridBased rows. Most of the remaining variation is ordinary
tile, depth, vector-width, mapping, and publication configuration rather than
a reason for another authored GEMM algorithm.

The first runtime basket is deliberately narrower than the static census:
FP16 NN, FP32 accumulation, FP16 output, alpha one, beta zero, no bias, aligned
interiors. On gfx1201 the existing `128x128x32` motif is already competitive
over a useful region, but skinny and deep-K cells expose the expected need for
smaller tile and direct-to-VGPR configurations. On gfx1100 the existing
`64x96x32` motif is a useful anchor but does not match the ecosystem-best
rocBLAS result across the initial basket. Its failures correlate with the
incumbent's `32x32`/`64x64` tiles, different mapping choices, and two
shape-sensitive range-proof failures. This is evidence for a small schedule
policy, not evidence that every Tensile recipe should be ported.

Concretely, the gfx1201 motif is within 1.05x of the best eligible hipBLASLt
kernel on 10 of 12 cells, is 1.189x on one deep-K cell, and 1.667x on the
skinny `640x128x2048` cell. The gfx1100 motif compiles on six of eight cells:
one is within 1.05x, three are between 1.10x and 1.25x, and two exceed 1.25x
against the better of rocBLAS and hipBLASLt. The other two fail range proof.
These unweighted counts bound mechanism work; they are not an aggregate
performance score.

The production-shaped bytecode experiment establishes a real pre-emission
program boundary and a remaining LoomC gap. Ten repeated compilations are
byte-deterministic on both targets. However, 48 gfx1201 requests produce 48
distinct transformed-program hashes but only 26 distinct HSACOs. The current
serialized module retains resolved configuration symbols that do not affect
the executable, even after a `canonicalize,dce,symbol-dce` cleanup. Hashing the
bytecode is therefore safe but materially over-specific. Hashing the HSACO
would discover the collapse too late and is not an accepted substitute.

The separately requested launch-config artifacts have 22 identities, and the
pair `(launch config, HSACO)` is unique for all 48 requests. That diversity is
legitimate at dispatch time, but it must not prevent the executable cache from
reusing the 26 HSACOs independently.

The epilogue fixture does achieve the intended identity: row and column bias
metadata produce the same transformed program and HSACO when bias is disabled;
enabled row and column bias are distinct. The prior native report also shows
that disabling bias removes its load. This validates the configuration style
while isolating the GEMM shape-key issue from epilogue specialization.

## Evidence summary

- Static census: all source logic under classic navi31 and TensileLite gfx1201,
  restricted only to solutions with a nonempty `MatrixInstruction`.
  It covers 174 logic files, 149 arithmetic/semantic signatures, 4,840
  accelerated gfx1100 solutions and 54,584 accelerated gfx1201 solutions.
- Dynamic corpora: at most 48 deterministic cells per target, with anchors,
  dominant mechanism rows, geometry strata, and router transitions. Router-row
  frequency is explicitly topology evidence and is not a demand distribution.
- Vendor comparison: every eligible public-API solution is timed in a short
  selection sweep; the winner is then correctness-checked and remeasured.
- Loom comparison: generated exact-shape cases execute only through
  `iree-benchmark-loom`, with the default compiler pipeline and an embedded
  numerical check.
- Keying: exact `.loombc` source and config inputs pass through bytecode link,
  exact target/config compilation, key cleanup, transformed-bytecode
  serialization, and separate HSACO emission.
- Startup: independent compiler processes with one worker per context scale
  from 2.035 s to 0.514 s for all 48 gfx1201 requests at one versus four
  workers. The six successful gfx1100 requests scale from 0.459 s to 0.135 s.
  These are intentionally pessimistic request counts because bytecode
  over-keying has not yet been fixed.
- Loop form: at the gfx1100 anchor, the structured source emits a 13,312-byte
  object in 8.68 ms median total, while the accepted fully unrolled form emits
  46,080 bytes in 59.39 ms. The structured form is not the performance winner;
  this quantifies the value of fixing its loop-wait schedule.

## Small-family decision

The authored-family hypothesis is supported; the executable-cardinality
hypothesis is not yet supported by the current key representation.

The initial policy should carry no more than four FP16 configurations per
target family before adding a structural motif:

1. the accepted large-tile staged motif;
2. a smaller-tile staged configuration for skinny/small geometry;
3. a direct-to-VGPR configuration for the gfx1201 deep-K/skinny region; and
4. one mapping/depth variant chosen from held-out measurements.

This is a research bound, not a claim that these four are sufficient. Add a
second authored body only if a split/reduction protocol or incompatible
dataflow survives configuration and specialization. Tails, split-K, StreamK,
and nontrivial epilogues remain explicit incumbent fallbacks in this spike.

## Blocking findings

1. **Program identity is over-specific.** LoomC needs a canonical executable
   identity that excludes dead resolved config definitions and includes all
   target facts required for later emission. See
   [`compiler-usability-program-key.md`](compiler-usability-program-key.md).
2. **gfx11 legality is not shape-uniform.** Some aligned configurations fail
   `SUBRANGE/010` on flattened vector accesses. The source needs a reusable
   proof or address form before those cells become candidates.
3. **One tile is not a router.** The performance misses align with incumbent
   tile/DTV classes. The next kernel work is configuration recovery, not more
   key plumbing.
4. **Load latency is not separately observable through the current sanctioned
   runner/API evidence.** End-to-end runner startup includes it, but a provider
   load-only phase and warm persistent-cache lookup remain integration work.
5. **In-flight miss coalescing is not yet measurable.** The current public
   pre-emission identity is over-specific, so a coalescing benchmark would
   validate the wrong boundary. The compiler handoff makes pre-emission
   coalescing an explicit acceptance condition.
6. **No demand trace exists.** Nothing in this spike may be read as weighted
   application coverage.

## Directory map

- [`corpus/`](corpus/) — deterministic gfx1100 and gfx1201 FP16 router corpora.
- [`config/`](config/) — bias/no-bias exact configuration fixtures.
- [`results/`](results/) — static census, performance baskets, program-key,
  startup, epilogue, and loop-form records.
- [`reproduce.md`](reproduce.md) — commands and validation contract.
- [`compiler-usability-program-key.md`](compiler-usability-program-key.md) —
  should-work LoomC boundary and acceptance fixture.
- Shared [`../tools/`](../tools/) — corpus, shape-fixture, performance, and JIT
  probes retained for later spikes.

## Alternatives Considered

- A cache keyed by raw M/N/K/config tuples was rejected because it cannot
  discover erased distinctions and violates the design invariant.
- Hashing emitted HSACO was rejected as the primary key because it requires the
  expensive work the cache is intended to avoid.
- Treating every exact normalized recipe as an authored Loom kernel was
  rejected; the census intentionally separates mechanisms from configuration.
- Timing only the vendor default was rejected; the comparison uses the best
  eligible measured solution, even where that makes a prior anchor look worse.
- Direct HIP execution of Loom artifacts was rejected for acceptance evidence;
  generated cells use the sanctioned Loom benchmark runner.
- Calling router-row frequency a demand weight was rejected. It measures logic
  topology only until an actual client corpus is supplied.
