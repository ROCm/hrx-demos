# Experiment 001: family-generic Low carrier rebinding

## Should-work source

[`gfx12-pack-invoke-family-generic.loom-test`](../loom/gfx12-pack-invoke-family-generic.loom-test)
defines both the High caller and Low helper against `gfx12-generic`, then asks
the default pipeline to specialize `@caller` for `gfx1201`.

## Observation

The High caller specializes to `amdgpu.rdna4.core`. The Low helper retains
`amdgpu.gfx12.generic.core` carrier identities. Inlining rejects operands whose
printed types are both `reg<amdgpu.vgpr x2>` because their descriptor stable
IDs differ:

```text
LOWERING/044: inline-callables cannot inline low.func.call to @pack_pair:
operand_type_mismatch
```

Changing only the declaration and helper target to exact `gfx1201`/
`amdgpu.rdna4.core` makes the same boundary compile and execute. That exact
witness is
[`gfx12-pack-invoke-object-abi.loom-test`](../loom/gfx12-pack-invoke-object-abi.loom-test).

## Required compiler contract

Exact target specialization must clone/rebind a reachable family-generic Low
object function and its physical carrier types to the selected exact target,
or define a target-family carrier identity that remains valid for exact child
targets. The source must not duplicate the helper for every exact ISA unless a
packet is actually illegal on sibling targets.

An acceptance fixture compiles the family-generic source with target
`gfx1201`, executes `@caller`, and compares the eight FP16 outputs.
