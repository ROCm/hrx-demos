# Suggested `loom-compile-report` knowledge

The current suggestion engine is useful but deliberately small. Running it on
the two best schedule witnesses produces:

| Report | Default and experimental output |
| --- | --- |
| gfx11 cross-iteration PLR | One `amdgpu.spill_traffic` finding: shorten live ranges or reduce fragment state. |
| gfx12 parity Low | No findings. |
| gfx11 terminal exact-schedule Low | No findings, despite the preceding clean-report form spending roughly 11.7 us in a poorly coalesced cross-lane epilogue. |

Those answers are internally consistent, but they miss the mechanisms that
matter in these cases. The following rules are proposed in priority order.

## 1. Contiguous register-run fragmentation

Trigger when a materialized spill exists, aggregate scheduled pressure is
below target capacity, and an allocation high-water row reports enough free
units in total but a largest free run smaller than `required_unit_count`.

For the gfx11 witness:

```text
value: even_second_rhs2
required run: 8 VGPR
peak scheduled pressure: 151
final allocation: 160 VGPR
lower free units: 14 in 3 runs
largest lower run: 6
active storage leases: 48 / 96 units
branch-edge moves: 71 units
```

Suggested action: identify this as register-run fragmentation, list the value,
required width, free-run histogram, assignment/lease blockers, and recommend
placement/lease or representation changes. Do not make “shorten live ranges”
the sole high-confidence recommendation when aggregate pressure is not the
limiter.

The report should also expose the architectural allocatable capacity used by
the comparison. Today `target_resources.vector.final.register_count` describes
the result, not the maximum available range, so a consumer cannot formulate
the rule robustly without target knowledge.

## 2. Loop-carried wide-state / branch-edge materialization

Trigger when `move_causes.branch_edge.unit_count` is large relative to peak
pressure or when wide x4/x8 values cross a backedge and coincide with spills.
Report the widest crossing values and whether they are accumulator, matrix
operand, global payload, or address state.

For gfx11, 71 units of branch-edge moves are exactly the rotating PLR fragment
state the source is trying to preserve. A recommendation to collapse or split
that state needs to acknowledge that doing so may destroy the desired native
schedule. Useful actions include loop-edge bank rotation, coalescing compatible
successor/predecessor leases, or a microkernel boundary that makes the ring
explicit.

The native-payload-order ablation strengthens this rule: moving both five-value
publication groups to the exact incumbent phase leaves four spill objects but
raises branch-edge moves from 71 to 95 and shifts the x8 high-water value. A
useful suggestion should identify that trade rather than declaring the local
source order independently better.

## 3. Storage-lease fragmentation versus live pressure

At an allocation failure/high-water point, present active assignment blockers
and active storage-lease blockers separately. Flag cases where leases occupy a
large portion of the constrained prefix despite low semantic live pressure.
The gfx11 row has 96 lease-blocked units. That is more actionable than the
post-repair fact that four scalar addresses were spilled.

### Preserve the pre-repair allocation snapshot

The final gfx11 report says `spill_count: 0` and `spill_plan_count: 0` while
also reporting four materialized spill objects. The failed allocation frame
and its exact blocking interval have already been discarded by spill repair.
`details` should retain a compact pre-repair snapshot for every repair round:
failed value, width/alignment, candidate prefix, free-run histogram, active
assignments, active leases, chosen spill/rematerialization victim, and the
change in the next round. This would have made the x8 contiguous-run failure
visible without a custom allocator trace.

## 3a. Structural move round trip

Trigger when a placement change reduces one move cause but increases another,
or when an aggregate `low.concat` and its branch successor together materialize
the same unit mapping. The first successful-looking gfx11 recoloring moved the
five aggregate results into loop-header banks but left every `ds_read` packet
in its old bank. All 40 recurrent copies remained, merely moving from
`branch_edge` to `low_concat`.

The report should name the non-edge placement component and give both logical
and materialized move counts before and after late recoloring. For this
fixture, the actionable component is 43 assignments: five x8 aggregates,
twelve RHS B128 packet pieces, sixteen scalar/D16 LHS pieces, and their in-place
placement relations. A suggestion should say “recolor the placement component
atomically,” not report the aggregate assignment as coalesced.

## 3b. Projected lease conflict in late recoloring

Any report for a late allocation proposal should evaluate leases at proposed
locations, not only current locations. The first atomic gfx11 experiment moved
`next_bv1` from VGPRs 36:39 to 112:115 and independently moved `%1170` to VGPR
113. Their SSA intervals were disjoint, but the asynchronous global-load lease
for `next_bv1` remained active through `%1170` and the kernel faulted.

This deserves both a verifier invariant and a diagnostic row containing the
two values, proposed ranges, SSA intervals, lease interval, and owning
operation. The corrected experiment places `next_bv1` in 160:163 and `%1170`
in 113 after checking the whole proposal. `loom-compile-report suggest` should
never endorse a move reduction when the associated allocation table contains
a proposed lease/assignment overlap.

## 3c. Unroll amplification

When otherwise identical reports differ only by `scf.for` unroll mode, flag
superlinear spill amplification. On this fixture, factor two retains four
spill objects, factors four/eight produce 49/73, and full unrolling produces
154. This is strong evidence that unroll is exposing a placement/allocation
problem rather than resolving the intended schedule. A useful suggestion
would point back to the smallest factor and structural backedge state instead
of recommending more unrolling.

## 4. Wait serialization hotspot

Trigger when a memory counter has many full drains, a maximum full drain much
larger than operand consumption width, or the majority of waits are due to
`read_result_reuse`/`memory_source_reuse` instead of explicit barriers.

The gfx11 count-matched witness has 28 full drains and drains as many as 38 LDS
operations at once. Its instruction count is native-like, yet its 1.35x timing
is explained by banded issue and drain-before-use. The suggestion should name
the counter and top reason, then recommend distinct result/address banks or a
more local producer-consumer schedule.

For gfx12, the report should similarly call out full `kmcnt`/loadcnt/dscnt
drains that dominate an otherwise spill-free matrix loop. ATT corroboration is
optional; the static wait plan is enough to make the suggestion.

## 5. Matrix-fragment packetization opportunity

Trigger when repeated High fragment selections lower to many D16 LDS packets
and the target advertises a packed matrix ABI. Report selected plan key,
fragment role, emitted LDS packet count/bytes, and alternate legal packet
widths when known.

The gfx12 source selected `strided_d16_packed_b16_fragment_load`: four LHS
fragments per half became 64 D16 packets, while the native representation is
eight B64 reads plus sixteen permutes. This should become a knowledge rule once
the alternate plan exists, and a “missing grouped fragment plan” finding until
then.

## 6. Invariant address arithmetic in a dynamic tile loop

Trigger when equivalent LDS address arithmetic is recomputed in each unrolled
fragment group or dynamic iteration. The gfx11 address-hoist experiment drops
the dynamic descriptor count from roughly 188 to 133 per K=32 with no spills.
The rule must be lease-aware: blindly hoisting the five store addresses in the
native-order loop increased spills from four to ten. Recommend hoisting only
when the projected lifetime/contiguous-run budget remains feasible.

## 7. Fixed-register diagnostic quality

When fixed placement fails, report the exact requested base/range/alignment,
the already assigned overlapping ranges, and whether the rejection is a
verifier alignment policy or an allocator conflict. The current generic
failure required multiple verifier-bypass builds to learn that unaligned x8
WMMA operand bases are ISA-legal but still do not solve the underlying
fragmentation.

## 8. Sanitizer pipeline coverage

If `--sanitizer=access` is requested for a pipeline which no longer contains
instrumentable memory operations, report that instrumentation was not applied
and fail verification rather than silently emitting an uninstrumented object.
For prepared Low, the suggestion should identify the latest supported input
stage or state that Low sanitization is unsupported. This is essential because
`--pipeline=none` currently emits a runnable but unsanitized kernel.

When instrumentation causes `BACKEND/021`, report uninstrumented versus
instrumented packet/pressure/spill growth and recommend a reduced complete
address-map witness. Do not suggest merely increasing the spill iteration
limit.

## 9. Performance-clean report with an external schedule delta

The gfx12 Low report has no suggestion because it is spill-free and legal, yet
ATT shows matrix, LDS-read, and permute stalls above the incumbent. A future
optional `suggest --compare-native <witness>` mode could ingest a normalized
schedule or a second compile report and flag service/order deltas. This should
remain opt-in: a single compile report cannot infer which external schedule is
the oracle.

## 10. `low.invoke` microkernel boundary

When a target-bound kernel contains `low.invoke`, report the selected callee
representation, whether the body was inlined, which input facts crossed the
boundary, and whether the call remains a scheduling/allocation barrier. If the
target has no rule, `suggest` should still be able to consume a failure report
and identify a missing generic Low-invocation bridge rather than presenting it
as an unsupported arithmetic operation.

The current AMDGPU probe fails with `TARGET/001` before a compile report is
written. Syntax and ABI verification exist, but there is no selected AMDGPU
contract/emitter path. See
[`compiler-usability-low-invoke-amdgpu.md`](compiler-usability-low-invoke-amdgpu.md).
For the GEMM use case, a successful result must inline before scheduling and
must expose Low helper memory packets to native access instrumentation.

## 11. Address-mode substitution cliffs

When a compile-report diff replaces generic/global loads with fewer-address-op
buffer loads but increases materialized spills or waits, `suggest` should not
report the packet selection as an unconditional win. Attribute the new cliff
to the live SRD, scalar-offset, and payload lease intervals that cross the
pressure peak, and recommend joint placement or a carried address-state form.

The gfx11 isolation changes exactly 15 `global_load_b128_saddr` packets to
`buffer_load_b128`. Without coordinated SRD placement, materialized spill
objects rise from 4 to 20, private storage from 16 to 80 bytes, and wait
actions from 55 to 124. The correct kernel regresses from 72 to about 131 us.
The relevant compact result is
[`results/gfx1100-buffer-address-mode-summary.json`](results/gfx1100-buffer-address-mode-summary.json).

## 12. Wave-coalescing quality of result publication

Trigger when a matrix epilogue uses cross-lane exchange followed by stores
from a strict subset of lanes, or when adjacent active lanes have a large
output-address stride despite a contiguous destination layout. Report active
lane count, per-lane store width/count, lane-to-address stride, cross-lane
packet count, and a candidate direct fragment-to-address mapping when the
matrix contract provides one.

The pre-terminal gfx11 report was spill-free and otherwise looked healthy, but
its epilogue used 48 `ds_bpermute` operations and 48 32-bit stores from only 16
lanes, with a 2,048-byte stride between active lanes. The terminal mapping uses
48 16-bit stores from all 32 lanes and makes adjacent lanes write adjacent
rows. That paired change moves the candidate/incumbent ratio from roughly
1.08 to 0.851. A clean allocation report must not suppress a publication-
coalescing finding this large.

## 13. Sanitized artifact runtime requirements

When sanitization introduces runtime globals, emit a machine-readable manifest
field naming the initialization/reporting ABI and supported loaders. A direct
HIP-module launch should be rejected or diagnosed before execution if those
globals have not been initialized. The earlier gfx11 page fault came from
launching a sanitized HSACO through the raw comparison harness; the same High
case passes under AMDGPU-enabled `iree-test-loom`.

This rule belongs in both the artifact manifest and `suggest`: the compiler
report alone said `OK`, so an agent had no evidence that the object was not a
standalone loadable kernel. Also report the Bazel/runtime capability mismatch
when `iree-test-loom` was built without `--//runtime/config/hal:drivers=amdgpu`.

## 14. High fragment-role escape for physical microkernels

When High fragment-role validation rejects an operand order that is legal for
a physically symmetric target instruction, name both semantic roles, the
selected instruction, and the supported escape hatch (`low.invoke`, explicit
physical-fragment cast, or another deliberate operation). Do not reduce this
to a generic `TARGET/039` failure.

The preserved gfx11 paired source swaps the two x8 WMMA source vectors and
changes the output map to undo the resulting 16x16 microtile transpose. Low
emits and validates the intended native instruction, while High rejects every
`vector.mma` before lowering. The address-only High witness can be sanitized,
but the semantically complete paired witness cannot currently cross this
contract. This is precisely the microkernel-boundary case an agent needs the
report to explain.

## 15. Zero-copy compare/branch forwarding

Flag a target-inserted delay between a scalar compare and its sole conditional
branch consumer when the target model permits direct SCC forwarding and no
intervening SCC writer exists. The gfx11 terminal loop needed a gated compiler
experiment to remove one such insertion; the incumbent demonstrates the legal
adjacency. A comparison mode should identify a lone target insertion even when
all source-scheduled categories otherwise match.

## 16. Specialization-aware terminal prefetch

When exact K specialization proves that a final prefetch iteration is not
consumed, report global-load/LDS-publication packets whose values die on the
exit edge. Recommend peeling or predicating the terminal stage and include the
exact specialization facts that justify removal. This is distinct from a
generic dead-code rule because asynchronous memory leases and waits may keep
the packets superficially live.

## 17. Workgroup-count and flattening consistency

Cross-check artifact workgroup-count metadata against the kernel's decoded
workgroup mapping and the launch configuration supplied by the harness. The
gfx11 path required an explicit flattened grid plus WGM8 mapping. A report
should make a 2-D/flattened mismatch visible before a numerically plausible
partial launch is timed.

## Candidate fixtures

- Contiguous-run/loop-edge rule:
  `artifacts/gfx11-cross-iteration-plr/report.json`.
- Structural-move and projected-lease rules:
  `artifacts/gfx11-cross-iteration-plr-wide-relocation-trace/report.json`, its
  `trace.log`, and
  `results/gfx1100-loop-structural-recolor-summary.json`.
- Unroll amplification:
  reports under `artifacts/gfx11-cross-iteration-plr-unroll-*` and
  `artifacts/gfx11-cross-iteration-plr-full-unroll`.
- Wait-serialization and safe address-hoist rule:
  `artifacts/gfx11-native-lds-address-hoisted/report.json` plus its unhoisted
  report.
- Clean-report comparison case:
  `artifacts/gfx12-uniform-bpad-scalar-saddr-native-rhs-lhs-first/report.json`.
- Fragment plan case:
  `artifacts/gfx12-scalar-acc-k32-trace/report.json`.
- Instrumentation explosion:
  `artifacts/gfx12-scalar-acc-k-unroll/access-full-report.json`.
- Sanitizer runtime-initialization case:
  compare the obsolete raw-HIP artifact under
  `artifacts/gfx11-native-lds-high-access-one-tile/` with the passing supported
  runs in `artifacts/gfx11-high-access-testbench/`.
- Wave-coalescing and exact external schedule comparison:
  `artifacts/gfx11-native-k32-wgm8-swapped-native-epilogue-exact-branch/report.json`,
  `results/gfx1100-att-terminal-comparison.json`, and
  `results/gfx1100-terminal-exact-branch-schedule-comparison.json`.
- High fragment-role escape:
  `loom/gemm-f16-f32-mt64x96x32-gfx11-native-epilogue.loom` and
  `artifacts/gfx11-high-native-epilogue/compile.log`.
- Should-work `low.invoke` form:
  `low/gfx12-pack-invoke-probe.loom-test` (currently fails before report
  creation with `TARGET/001`).
- Address-mode substitution cliff:
  compare `artifacts/gfx11-cross-iteration-plr-wide-relocation-trace/report.json`
  with `artifacts/gfx11-cross-iteration-plr-buffer-loads-dce/report.json`.

These are small enough to make regression fixtures after paths and source
identities are normalized.
