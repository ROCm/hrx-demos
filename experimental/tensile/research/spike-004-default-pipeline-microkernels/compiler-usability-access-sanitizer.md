# Compiler usability report: access sanitizer coverage at High/Low boundaries

## Summary

Loom's access sanitizer successfully checks and executes the High memory
portion of a composed High+Low kernel, but authored target-Low memory packets
inside a `low.invoke` helper are outside its instrumentation domain. A passing
sanitizer run therefore does not presently certify Low-authored LDS or global
accesses. Register-only Low microkernels avoid this gap; memory-bearing Low
microkernels need a separate static proof until coverage is implemented.

## Preserved should-work case

[`gfx12-fragment-pack-exact.loom`](loom/gfx12-fragment-pack-exact.loom)
contains two boundaries:

- `@fragment_pack` keeps all memory in High and invokes a register-only Low
  pack helper.
- `@lds_fragment_load_pack` performs High global loads and High LDS stores,
  then invokes `@load_and_pack_lhs_fragments`, which contains eight authored
  `amdgpu.ds_read_b64` packets.

Both cases compile through the default pipeline and pass
`iree-test-loom --sanitizer=asan` on gfx1201. The latter is the source we think
should eventually be fully covered.

## Evidence of the gap

The sanitizer assertion-selection pass runs before source-to-Low. Its site
table for `@lds_fragment_load_pack` contains the twelve High global accesses:
eight reads and four writes. The eight Low `ds_read_b64` packets remain
verbatim in prepared Low and have no corresponding access sites. The run
passes because those particular addresses are legal; it does not test whether
the sanitizer could diagnose an illegal authored-Low address.

No deliberately out-of-bounds Low packet was executed. Doing so would use an
uninstrumented device operation as the detector and would not be sanitizer
evidence.

## Expected contract

One of these mechanisms is needed before Low memory helpers can carry the same
acceptance weight as High memory:

1. instrument target-Low memory descriptors after `low.invoke` is inlined,
   using resource/LDS bounds projected from the caller; or
2. require a machine-checkable memory-effect/access contract on the Low object
   function and statically verify every packet against it.

The sanitizer report should state coverage explicitly: eligible High access
count, instrumented High count, authored-Low memory count, verified Low count,
and unverified Low count. A requested sanitizer run with nonzero unverified
Low memory should warn or fail in strict mode instead of looking identical to
full coverage.

`loom-compile-report suggest` should also emit one focused finding when a
sanitizer is requested and authored-Low memory remains uncovered. The finding
should name the helper, packet count, memory spaces, and the exact verification
mechanism required; it should not recommend adding more High assertions.

## Current Spike 4 policy

The maintained gfx12 GEMM candidate uses Low only for register permutation;
global and LDS memory remain High and are sanitizer-visible. The
memory-bearing `@load_and_pack_lhs_fragments` case is retained only as a
compiler acceptance fixture, not promoted as a validated production motif.
