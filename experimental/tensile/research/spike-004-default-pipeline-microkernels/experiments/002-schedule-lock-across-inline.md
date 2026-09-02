# Experiment 002: schedule locking across `low.invoke`

## Should-work source

[`gfx12-pack-invoke-object-abi.loom-test`](../loom/gfx12-pack-invoke-object-abi.loom-test)
authors four independent `v_perm_b32` packets in the order `packed0`,
`packed1`, `packed2`, `packed3` inside a `schedule(locked)` Low object
function. The helper is necessarily inlined because AMDGPU has no object-call
ABI at this boundary.

## Original observation

After default-pipeline inlining the emitted packet order is `packed0`,
`packed2`, `packed1`, `packed3`:

```text
v_perm_b32 v8,  v4, v2, 0x05040100  // packed0
v_perm_b32 v10, v4, v2, 0x07060302  // packed2 moved early
v_perm_b32 v9,  v5, v3, 0x05040100  // packed1
v_perm_b32 v11, v5, v3, 0x07060302  // packed3
```

The inliner moves operations into a caller that is not schedule-locked, so the
callee's lock is not represented after the boundary disappears. Explicit
`low.schedule.fence` operations between packets preserve the authored order;
the fragment-pack and GEMM witnesses currently use that workaround.

## Required compiler contract

Inlining a schedule-locked Low helper must preserve the callee's ordered
region in the caller, either by propagating a scoped schedule constraint or by
materializing equivalent boundaries. `schedule(locked)` should be sufficient;
an author should not need a fence after every independent packet.

The acceptance fixture compares the four packet identities in emitted target
order, not just numerical output, because all four permutations can be
numerically correct after reordering.

## Resolution

HRX main `f17f69e82` (PR #513) materializes conservative source-order
boundaries when a `schedule(locked)` helper is required-inline. With all
explicit `low.schedule.fence` operations removed, the acceptance fixture emits
`packed0`, `packed1`, `packed2`, `packed3` in authored order. Removing the same
fences from the full family-generic gfx1201 fragment-pack helper produces a
byte-identical target artifact, so the maintained motif now relies on the
declared lock rather than per-packet workaround fences.
