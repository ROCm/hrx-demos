# Experiment 000: establish the `low.invoke` contract

## Intended contract

A High kernel invokes a target-specific, schedule-locked Low object function.
The default pipeline must preserve or legally inline that boundary and produce
an executable AMDGPU kernel. This is a compiler-contract experiment only; it
does not yet claim GPU correctness or performance.

## Observations

1. The original Spike 003 probe failed `TARGET/001` before lowering because
   `low.invoke` was not classified as a structural operation by source-to-Low
   lowering.
2. Translating it to `low.func.call` exposed a second issue: source selection
   treated the already-Low helper as a new High source root and attempted to
   lower its `low.slice` operations.
3. Excluding Low definitions from High source selection reached ABI checking.
   An implicit helper ABI was interpreted as a HAL kernel ABI, producing
   `AMDGPU/025` for its VGPR vector arguments.

4. Making the helper `abi(object_function)` and teaching source-to-Low to
   lower `low.invoke` produces a `low.func.call` as intended. AMDGPU has no
   emitted object-call ABI, so the call must be consumed before packetization.
5. A narrowly gated post-source-to-Low `inline-callables` pass successfully
   composes a single-block exact-target helper. Pure divergent operands must
   be marked storage-required during structural lowering; otherwise their
   values are elided before the call.
6. Fragment role facts do not cross the boundary. The caller must currently
   reattach `vector.fragment<lhs>` before `vector.mma`.

This established a usable register-only exact-target microkernel path. The
family-target, schedule, CFG, and sanitizer contracts were separated into
experiments 001 through 003 and the access-sanitizer usability report.

## Upstream resolution

HRX main `f17f69e82` (PR #513) implements required-inline Low schedule
fragments in the default pipeline without the experimental environment gate.
Both the exact fixture and the full gfx1201 GEMM execute through sanctioned
Loom runners. The disposable branch below remains useful archaeology for the
sequence of compiler contracts it exposed, but it is no longer required.

## Historical experimental compiler branch

`loom-blas/spike4-low-invoke-experiment` at `db115431e`, based on `b422b5056`.
Changes are disposable compiler evidence and are not intended to land
verbatim. Thirty focused lower/Low-op/target/symbol test targets pass.
