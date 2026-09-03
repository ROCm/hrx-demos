# Compiler usability: preserving legal loop-carried LDS work

## User intent

The should-work source is
[`loom/gemm-f16-f32-mt64x96x32-gfx11-high-k32-ring.loom`](loom/gemm-f16-f32-mt64x96x32-gfx11-high-k32-ring.loom).
It expresses a two-stage K32 ring in structured High Loom. Global loads for
the next tile, current-tile fragment reads, WMMAs, LDS publication, and a
workgroup barrier are explicit. The loop carries accumulator and next-fragment
SSA values. No authored wait or illegal memory packet is involved.

Desired behavior is not verbatim reproduction of Tensile assembly. The
acceptance requirement is that the default pipeline preserve enough legal LDS
work across the loop backedge to match the fully unrolled counterfactual
without forcing the whole exact-K body to expand.

## Current behavior and smoking gun

The structured source is correct but measures approximately 61.588 us. Its
steady report batches 38 LDS-read packets and emits two full drains. In current
HRX `loom/src/loom/target/arch/amdgpu/planning/wait_plan.c`:

1. `loom_amdgpu_wait_plan_relocate_loop_entry_dependencies` identifies
   single-counter SSA-use links crossing a natural-loop entry and relocates
   them to a preheader-indexed table.
2. `loom_amdgpu_wait_plan_handle_loop_entry_dependencies` emits the relocated
   wait with `target_count=0`.

For this source, target zero means every outstanding `lgkmcnt` operation is
drained at the backedge. The policy is conservative and explicit; this is the
root cause, not an unexplained compiler-quality label.

Adding `unroll(%thirty_two)` leaves the High dataflow intact but removes the
backedge. The resulting report has 95 partial waits and improves to 53.8 us
before the direct epilogue. That A/B experiment is the acceptance fixture.

## Requested compiler contract

For a loop-carried SSA dependency, derive the minimum safe remaining counter
value from the scheduled producer position and other outstanding operations,
or prove a narrower version for this common single-natural-loop pipeline.
Reject or fully drain only when the required ordering cannot be represented.

A passing implementation should:

- compile the retained structured ring through the default pipeline;
- keep correctness and native access-sanitizer checks passing;
- introduce no private memory or spill traffic;
- avoid the two 38-operation LDS drains at the backedge;
- approach the fully unrolled High-only schedule and runtime without requiring
  exact native instruction identity; and
- preserve family-generic `gfx11-generic` source specialization to gfx1100.

Once this contract is implemented, compare the structured and unrolled
artifacts on the same correctness/performance fixture. Do not stack allocation
or hazard workarounds before that comparison.

## Compile-report request

The report currently counts full drains and assigns an SSA-use reason, but it
does not make the loop-entry policy obvious. Add a wait reason/provenance row
such as `loop_entry_relocated_ssa_use`, with the source producer/consumer,
natural-loop header, chosen target count, and whether the count is conservative
or derived. A summary warning should fire when a loop-entry action drains the
maximum outstanding count and exact unrolling would remove the reason.

## Related allocator reproducer

[`loom/gemm-f16-f32-mt64x96x32-gfx11-peeled-structured-allocator-reproducer.loom`](loom/gemm-f16-f32-mt64x96x32-gfx11-peeled-structured-allocator-reproducer.loom)
is the same structured loop with the final K32 tile peeled, but without full
unrolling. It currently fails in `codegen/low/allocation/coalescing.c` with:

```text
FAILED_PRECONDITION; low tied result cannot share the operand location
without overlapping another live interval
```

The unrolled form compiles and spills nothing. This looks like a cyclic
tied-result coalescing issue exposed by peeling, not evidence that peeling is
illegal. The compiler should either allocate it, repair it, or diagnose the
specific source value and live interval in the compile report rather than
terminating with an unnamed internal value.
