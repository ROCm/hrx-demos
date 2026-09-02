# Compiler usability report: access sanitizer coverage at High/Low boundaries

## Summary

Loom's access sanitizer successfully checks and executes the High memory
portion of a composed High+Low kernel, but authored target-Low memory packets
inside a `low.invoke` helper are outside its instrumentation domain. A passing
sanitizer run therefore does not presently certify Low-authored LDS or global
accesses. Register-only Low microkernels avoid this gap; memory-bearing Low
microkernels need a separate static proof until coverage is implemented.

## Preserved should-work cases

[`low-invoke-lds-access-gfx1201.loom`](reproducers/low-invoke-lds-access-gfx1201.loom)
is the minimal current acceptance probe. A High kernel writes one legal LDS
value and invokes a single-block Low helper containing one `ds_read_b64`. It
passes both ordinary and access-sanitized execution through the default
pipeline on HRX main `f17f69e82`.

[`gfx12-fragment-pack-exact.loom`](loom/gfx12-fragment-pack-exact.loom)
preserves the larger original ladder:

- `@fragment_pack` keeps all memory in High and invokes a register-only Low
  pack helper.
- `@lds_fragment_load_pack` performs High global loads and High LDS stores,
  then invokes `@load_and_pack_lhs_fragments`, which contains eight authored
  `amdgpu.ds_read_b64` packets and a nested Low-to-Low pack call.

The register-only case compiles. The larger memory case is now independently
rejected because #513 does not yet project nested Low calls. It remains useful
source for the desired composed form, while the minimal probe isolates
sanitizer coverage without that unrelated failure.

## Evidence of the gap

The sanitizer assertion-selection pass runs before source-to-Low. In the
minimal probe its site table contains two High global accesses: the input read
and output write. The authored Low `ds_read_b64` remains in the module with no
corresponding access site. The detailed compile report sees two local-memory
packets in the final instruction mix, but its source-Low memory economics
contain only the three memory operations lowered from High; the authored Low
read is not one of them. `loom-compile-report suggest` emits only the unrelated
single-subgroup workgroup-communication finding and does not identify this
coverage gap.

The sanitized run passes because the Low address is legal; it does not prove
that the sanitizer could diagnose an illegal authored-Low address. The larger
original fixture showed the same phase ordering with eight omitted Low reads
before the upstream required-inline implementation landed.

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
