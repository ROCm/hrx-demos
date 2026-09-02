# `low.invoke` AMDGPU microkernel usability report

## Intended use

Use a small `low.func.def schedule(locked)` as the register-level escape hatch
for gfx12 fragment packing, invoked from an otherwise High Loom GEMM. This is
the natural raising path for the eight B64 LDS reads plus sixteen
`v_perm_b32` operations that the current High fragment selection cannot emit.
It should also allow the outer GEMM to remain config-specialized and keep
sanitizer-visible High memory operations.

The preserved should-work probe is
[`low/gfx12-pack-invoke-probe.loom-test`](low/gfx12-pack-invoke-probe.loom-test).
It defines a target-bound Low register helper and invokes it with two High
vectors from a target-bound kernel.

## Observed behavior

Text verification succeeds, but normal AMDGPU compilation fails during target
contract selection before source-to-Low lowering:

```text
error [TARGET/001]: target 'target' export 'caller' config 'target' has no
target-low contract for 'low.invoke' in '@caller'
```

Repository search finds `low.invoke` syntax, ABI verification, call-graph, and
symbol-boundary handling, but no AMDGPU target-contract rule or source-to-Low
emitter for the operation. Consequently the documented/author-suggested
microkernel path is not currently usable by an AMDGPU kernel in this checkout.

This is not an omitted inline keyword in the reproducer. Neither
`low.func.def` nor `low.invoke` currently has an inline-policy attribute, and
the generic `inline-callables` pass only accepts semantic and template call
kinds. Therefore there is no authored policy that can make this Low invocation
disappear before target-contract selection.

No conclusion can yet be drawn about specialization-fact propagation or
inlining because compilation does not reach either stage.

## Requested behavior

1. A target-bound `low.invoke` whose callee has a compatible selected
   target-Low representation should be accepted without requiring an ordinary
   source-op descriptor rule.
2. Before final allocation, the call should either inline the Low body or
   expose an explicit call ABI. The GEMM use case requires inlining so packet
   scheduling and placement can cross the boundary. This could be an explicit
   `inline` policy on `low.invoke`, a required-inline rule for a
   `low.func.def`, or a dedicated Low-invoke expansion pass; the authored
   behavior must be visible rather than inferred from callee size.
3. High operand facts and config-derived constants should be mapped onto Low
   argument facts. If they cannot propagate before inlining, the report should
   state that and show whether post-inline refinement recovered them.
4. Compile reports should name the callee, chosen representation contract,
   inline outcome, surviving unknown facts, and any scheduling boundary.
5. Native access instrumentation should account for memory packets inside an
   invoked/inlined Low helper; otherwise this route does not solve the prepared
   Low sanitizer gap.

## Reproduction

```sh
../hrx-system/bazel-bin/loom/src/loom/tools/loom-compile/loom-compile \
  research/spike-003-native-upward/low/gfx12-pack-invoke-probe.loom-test \
  --backend=amdgpu-hal \
  --compile-report=details \
  --compile-report-output=research/spike-003-native-upward/artifacts/gfx12-pack-invoke-probe/report.json \
  --output=research/spike-003-native-upward/artifacts/gfx12-pack-invoke-probe/kernel.hsaco
```

This fails before producing the report or HSACO. Once the basic probe compiles,
the next fixture should move the complete gfx12 B64-read/permute fragment
loader behind `low.invoke` and compare its inlined prepared Low against the
retained all-Low parity witness.
