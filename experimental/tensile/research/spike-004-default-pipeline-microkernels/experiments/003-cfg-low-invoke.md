# Experiment 003: multi-block Low microkernel invocation

## Should-work source

[`low-invoke-cfg-helper-gfx1201.loom`](../reproducers/low-invoke-cfg-helper-gfx1201.loom)
is a two-block, schedule-locked Low identity function called by a High kernel.
It is intentionally smaller than a GEMM loop: the only property under test is
that Low CFG and block arguments survive composition.

## Observation

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

## Required compiler contract

Required-inline Low calls need CFG inlining: split the caller block, clone the
callee region, map entry operands and block arguments, redirect every
`low.return` to a continuation block, and preserve schedule/allocation
contracts on the cloned region. The acceptance fixture must then compile and
run through `iree-test-loom` with the default pipeline and Loom access
sanitizer.

This is the smallest missing mechanism that prevents a maintained High wrapper
from invoking the exact gfx11 peeled schedule or the gfx12 scheduled main
loop.
