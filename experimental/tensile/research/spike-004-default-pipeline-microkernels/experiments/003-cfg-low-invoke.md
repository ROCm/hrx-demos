# Experiment 003: multi-block Low microkernel invocation

## Should-work source

[`low-invoke-cfg-helper-gfx1201.loom`](../reproducers/low-invoke-cfg-helper-gfx1201.loom)
is a two-block, schedule-locked Low identity function called by a High kernel.
It is intentionally smaller than a GEMM loop: the only property under test is
that Low CFG and block arguments survive composition.

## Original observation

The experimental `low.invoke` lowering reaches the default-pipeline inliner,
which rejects every multi-block callee:

```text
LOWERING/044: inline-callables cannot inline low.func.call to @identity_cfg:
callee_body_not_single_block
```

The prepared-Low gfx12 and gfx11 schedule oracles both contain multi-block CFG
for loop setup, steady state, peel, and exit. Consequently moving the boundary
from a register-only fragment pack to a complete K loop is blocked by this
same structural restriction. Stacking more packet rewrites below this failure
would not answer the composition question.

## Upstream disposition

HRX main `f17f69e82` (PR #513) deliberately defines a narrower contract:
required-inline helpers have exactly one outer body block. Schedule-free
helpers may contain supported structured nested regions, while locked helpers
are straight-line schedule fragments. The retained two-block reproducer now
fails with the direct `TARGET/072` diagnostic describing that contract.

This changes the next experiment. Multi-block CFG inlining is no longer the
presumed fix. First express loop and carried-state semantics in High/source
structured control flow and invoke the smallest straight-line locked K-step or
publication fragment that must retain exact packet order. A schedule-free,
single-outer-block helper with structured nested regions is a second option.
Only if those forms cannot recover the gfx1100 schedule should this fixture be
promoted from an unsupported-shape probe to a request for general CFG inlining.

Nested Low-to-Low calls are likewise not projected by this first contract;
`gfx12-fragment-pack-exact.loom` preserves that separate rejection. Flattening
such a small fragment is mechanically possible, but composition should prefer
one High-to-Low boundary per irreducible schedule unit.
