# Spike 004: default-pipeline Low microkernels

## Question

Can a small, manually scheduled Low microkernel carry the irreducible Radeon
inner-loop schedule while the public GEMM, configuration, launch geometry,
buffer handling, and epilogue remain ordinary High Loom—and can that composed
program survive the default pipeline?

Prepared Low from Spike 003 is an acceptance oracle and a compiler reproducer.
It is not the maintained solution. `--pipeline=none`, direct HIP launch of a
Loom artifact, and uninstrumented prepared Low do not establish Loom
correctness, sanitizer, performance, or product evidence.

## Outcome

The smallest High+Low boundary now works through the default pipeline on an
experimental HRX branch. gfx1201 accepts a register-only Low fragment-pack
helper inside the full High GEMM and reaches 25.04 us device p50 versus
28.78 us for hipBLASLt 133309. It is correct, has no spill/private traffic,
and passes the performance gate. It is not schedule-congruent: after
normalizing the candidate's K64 loop to K32, all matrix and memory operation
counts match the incumbent, but Loom emits 239 instructions versus 147,
including 24 waits versus 11.

gfx1100 remains conclusive negative evidence for the all-High form: 61.0 us
device p50 versus 45.46 us for solution 1675, four 4-byte spill/reload pairs,
and a normalized 219.5 instructions per K32 versus 134. The exact prepared-Low
oracle from Spike 003 cannot yet be raised as a maintained microkernel because
required-inline Low calls reject multi-block CFG. That is now isolated by a
two-block reproducer rather than attributed generically to “the compiler.”

The completion condition is met by one accepted performance witness plus exact
compiler-contract packets. Schedule congruence has not been achieved through
the default pipeline on either target.

## Anchors

| Target | Request | Incumbent oracle | Spike 004 default-pipeline result |
| --- | --- | --- | --- |
| gfx12-generic / gfx1201 | FP16, `1024x1024x1024` | hipBLASLt 133309: 28.78 us | Default High+Low: 25.04 us; performance pass, schedule fail |
| gfx11-generic / gfx1100 | FP16, `1024x960x1024` | hipBLASLt 1675: 45.46 us | Default High: 61.0 us; performance and schedule fail |

The performance gate is an upper 95% bootstrap confidence bound of 1.05 for
the candidate/incumbent median ratio. Vendor measurements bracket Loom
measurements so drift remains visible; they are not described as same-process
paired samples.

The exact values and bootstrap summaries are in
[`results/gfx1201-anchor.json`](results/gfx1201-anchor.json) and
[`results/gfx1100-anchor.json`](results/gfx1100-anchor.json).
Post-format correctness, sanitizer, parser, and expected-failure checks are
indexed in [`results/validation.json`](results/validation.json).

The source-to-motif derivation is summarized in
[`source-lineage.md`](source-lineage.md). It separates facts copied from the
routed recipe/native object, facts retained only because an experiment showed
they were load-bearing, and choices that remain ordinary Loom policy.

## Boundary-ladder result

The search proceeds from the smallest schedule-sensitive boundary. A larger
boundary is justified only by a named compiler contract that prevents the
smaller one.

- gfx12: fragment pack; LDS read plus pack; one K16 step; one K32 step; full
  main loop only if required.
- gfx11: one scheduled K32 body with carried state; whole peeled main loop only
  if required.

gfx12's fragment-pack rung works only with an exact target and explicit
schedule fences. Moving LDS reads into Low also executes, but the access
sanitizer cannot cover those Low memory packets. The full-loop rung is required
for schedule control and is blocked by multi-block Low inlining.

gfx11 begins at a loop-shaped boundary because its load/fragment placement is
the schedule. The prepared oracle is multi-block and reaches the same CFG
inlining blocker immediately.

The High wrapper owns exact-shape specialization and a stable argument
superset. It selects `has_bias` and `bias_axis = row | column` as compile-time
configuration. Bias is FP16, is accumulated before the final FP16 conversion,
and must compile away in the no-bias specialization.

## Evidence results

An anchor is accepted only when all of these hold:

| Gate | gfx1201 High+Low | gfx1100 High |
| --- | --- | --- |
| Default pipeline | Pass on exact target; generic helper blocked | Pass on exact target |
| No-bias correctness | Pass | Pass |
| Access sanitizer | Pass; all maintained memory is High | Pass |
| No spills/private memory | Pass | Fail: 16 bytes, 4 stores, 4 reloads |
| Performance ratio upper CI <= 1.05 | Pass; captured-sample upper CI 0.870 | Fail; lower CI 1.329 |
| Schedule congruence | Fail | Fail |

Bias/no-bias configuration is proven separately by
[`epilogue-f16-bias-config.loom`](loom/epilogue-f16-bias-config.loom): all
three cases pass correctness and access sanitization, and the no-bias compile
has one global load versus two with bias. It is not yet fused into the GEMM,
and its row/column cases use uniform bias; this is representation evidence, not
a completed GEMM epilogue differential.

With the host compiler built in Bazel `opt` mode, default-pipeline Loom C API
bytecode-to-HSACO medians are 22.09 ms for gfx1201 and 11.77 ms for gfx1100,
including root linking, specialization, compilation, and emission. See
[`results/loombc-to-hsaco-latency.json`](results/loombc-to-hsaco-latency.json).
The commands and required revisions are in [`reproduce.md`](reproduce.md).

## Compiler experiment discipline

Each experiment records the should-work source, the first failing compiler
contract, and any narrowly gated experimental fix on a dedicated HRX branch.
Once a missing contract is isolated, work stops stacking unrelated rewrites.
Compiler changes in this spike are disposable evidence for Loom maintainers;
they are not proposed for direct landing.

## Compiler packets

- [`experiments/000-low-invoke-contract.md`](experiments/000-low-invoke-contract.md):
  the working exact-target, single-block, register-only path and its
  experimental lowering.
- [`experiments/001-family-carrier-rebinding.md`](experiments/001-family-carrier-rebinding.md):
  generic Low carrier identity is not rebound for an exact target.
- [`experiments/002-schedule-lock-across-inline.md`](experiments/002-schedule-lock-across-inline.md):
  a helper's `schedule(locked)` does not survive inlining.
- [`experiments/003-cfg-low-invoke.md`](experiments/003-cfg-low-invoke.md):
  required-inline calls reject multi-block Low CFG.
- [`compiler-usability-access-sanitizer.md`](compiler-usability-access-sanitizer.md):
  authored Low memory operations are not access-sanitizer covered.
- [`experiments/004-loombc-link-compile.md`](experiments/004-loombc-link-compile.md):
  production-shaped bytecode link/compile/emission timing and the necessary
  link-before-compile API path.

## Completion disposition

gfx1201 has an accepted default-pipeline performance candidate and precise
schedule delta. gfx1100 has a measured default-pipeline baseline and an exact
minimal CFG-inlining packet that blocks composition of the retained congruent
oracle. gfx906 is intentionally deferred: work there would require the same
microkernel composition contracts plus gfx9 target enablement, and neither
gfx11 nor gfx12 benefits from starting that compiler surgery now.

## Alternatives considered

- Maintaining the prepared-Low whole kernel was rejected as the default path:
  it freezes policy and composition at the point where Loom should specialize.
- Treating the native disassembly as a verbatim port was rejected: it is an
  oracle for load-bearing facts, not a limit on compiler optimization.
- Continuing with `--pipeline=none` was rejected for acceptance evidence; it
  bypasses the compiler behavior this spike is intended to validate.
