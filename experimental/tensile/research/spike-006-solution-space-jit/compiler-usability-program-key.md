# Compiler usability: canonical pre-emission program identity

## Should-work source path

The retained probe uses only public LoomC operations:

1. index immutable kernel `.loombc`;
2. link one exported root with an exact target profile;
3. apply an exact bytecode config through the default target pipeline;
4. run `canonicalize,dce,symbol-dce` on the transformed in-memory module;
5. request `LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE`;
6. hash that bytecode plus target/compiler/pass/ABI/export/emission namespace;
7. emit the still-live module only after the key decision.

The implementation is
[`../tools/loomc_jit_benchmark.c`](../tools/loomc_jit_benchmark.c). It preserves
the form that should become the provider path rather than replacing it with a
raw tuple or post-emission key.

## Observed disconnect

The boundary is deterministic but not canonical enough for executable reuse.
In the gfx1201 corpus, 48 transformed program hashes map to 26 distinct HSACO
hashes. Different bytecode programs can therefore emit byte-identical objects.
Inspection from the earlier canonical-text experiment and this bytecode result
indicates that resolved configuration definitions remain serialized after
their uses are folded. The cleanup pass handles dead epilogue metadata, but
does not collapse all exact GEMM shape inputs that cease to affect emitted
code.

The largest observed alias groups make the missing boundary concrete. For
fixed gfx1201 M128/K256, six N values from 256 through 896 emit one HSACO; a
second six-way group does the same for K64. Launch-grid differences are real
per-request products, but they belong in the launch-config result rather than
forcing six copies of the target executable. A useful API should expose those
two identities independently.

This is not a hash collision: equal derived keys always emit byte-identical
HSACOs, and ten repeated anchor compilations produce one program hash and one
HSACO hash on each target. It is missed reuse caused by excess serialized
identity.

There is a second lifecycle wrinkle. Target compiler facts retained on the
live module are required by `loomc_emit_module`; serializing and deserializing
the transformed bytecode does not retain them. A polished API must either
return a stable executable key directly, preserve an opaque prepared product,
or define which serialized program plus explicit facts is sufficient to
resume emission.

## Requested acceptance fixture

An upstream test should compile a parameterized kernel through the default
pipeline and establish all of the following:

- exact inputs that affect emitted instructions produce different keys;
- exact inputs folded from the executable produce equal keys;
- disabled epilogue metadata aliases, as the retained no-bias row/column
  fixture already does;
- ten repeats produce byte-identical key material;
- equal keys imply byte-identical emitted artifacts for each exact target; and
- the key can be obtained without native emission and used to coalesce misses
  before emission.

For BLAS specifically, use at least two M/N/K requests that currently emit the
same HSACO and two that emit different HSACOs. Keep the target profile,
compiler/pass identity, ABI, export, and emission options explicit even if a
future helper packages them.

The checked acceptance pair is
[`config/gemm-gfx1201-128x256x256.loom`](config/gemm-gfx1201-128x256x256.loom)
and
[`config/gemm-gfx1201-128x384x256.loom`](config/gemm-gfx1201-128x384x256.loom)
against the retained Spike 4 gfx1201 source: their transformed program hashes
differ and their HSACO hashes are equal.

## Suggested compile-report knowledge

Report a compact program-identity section containing:

- canonical program byte count and digest;
- retained config symbols and the live operations they influence;
- target/compiler facts excluded from serialization;
- identity namespace/version; and
- an explicit warning when dead resolved configuration symbols remain in the
  key artifact.

This would make key inflation visible without requiring a corpus-wide HSACO
differential.

## Alternatives Considered

- Re-keying on HSACO bytes is a useful diagnostic oracle but too late for the
  production lookup.
- Stripping arbitrary bytecode sections in the provider is rejected because
  the provider cannot safely infer compiler semantics.
- Omitting target facts from the key is rejected even when two devices happen
  to emit identical bytes.
