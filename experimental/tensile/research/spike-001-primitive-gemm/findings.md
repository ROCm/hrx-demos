# Spike 001 findings

## State of the question

The complete kernel-reconstruction loop is not yet closed, but the first spike
has crossed four important boundaries: public-library selection truth, source
and serialized-recipe recovery, physical execution of the native Loom WMMA
mechanism on both Radeon matrix-instruction families, and a correct full-K
gfx12 GEMM. A cooperative parity schedule and paired boundary corpus remain to
be built.

The environment captured here uses ROCm
`<selected-rocm>`, Loom revision
`bc59ef458288987a0ee03456e6337ba2b338fd1a`, and rocm-libraries revision
`bb5babaf8af2dd24d07617191e1fbd731c978cb3`.

## Public API selection truth

The custom probe fixes one column-major NN request: FP16 A/B/C/D, FP32
accumulation and scalars, alpha one, beta zero, batch one. It enumerates and
forces public solutions, compares each output with an independent CPU
reference, and times with HIP events. These first two square sizes validate the
laboratory; they are not a coverage corpus.

| Device | Size | Backend | Tested / enumerated | Incorrect | Default (us) | Best (us) | Best ID | Default slowdown |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gfx1100 | 256 cubed | hipBLASLt | 70 / 140 | 0 | 14.960 | 14.401 | 1692 | 3.88% |
| gfx1100 | 1024 cubed | hipBLASLt | 70 / 140 | 0 | 47.641 | 42.280 | 1675 | 12.68% |
| gfx1201 | 256 cubed | hipBLASLt | 536 / 537 | 0 | 12.480 | 12.120 | 133764 | 2.97% |
| gfx1201 | 1024 cubed | hipBLASLt | 536 / 537 | 0 | 35.000 | 30.320 | 133222 | 15.44% |
| gfx906 | 256 cubed | rocBLAS | 204 / 204 | 0 | 20.800 | 20.160 | -605555914 | 3.17% |
| gfx906 | 1024 cubed | rocBLAS | 204 / 204 | 0 | 241.600 | 171.200 | -605555978 | 41.12% |

The default is therefore not a sufficient oracle. The best eligible solution
must be forced and paired with every candidate measurement.

## Runtime route and source recipe joins

The installed gfx1201 lazy master contains a nested semantic `Problem` router
for `Contraction_l_Ailk_Bljk_Cijk_Dijk`. Its HHS rows separate no-bias,
bias/alpha-vector, and bias/auxiliary placeholders. The corresponding installed
shards contain 30 Equality-routed no-bias solutions, 517 GridBased bias-capable
solutions, and one Equality-routed auxiliary solution. The unfused public
request can force solutions from both major shards: the 256-cubed winner is
no-bias solution 133764, while the 1024-cubed winner is bias-capable solution
133222. Static shard identity is not a sound authored-kernel boundary.

Runtime gfx1201 solution 133222 is a 128x128x32, four-wave, wave32 WMMA
schedule. Normalized fields match source YAML solutions 4 and 92 with zero
differences across 20 compared schedule fields. This establishes a structural
join despite global solution-index drift.

The installed gfx1100 winner 1675 is a 64x96x32, four-wave, wave32 WMMA
schedule. The current classic navi31 source corpus is materially older or
different: its nearest source entries differ in macro tile, prefetch depth, and
wave-separated global reads. Runtime artifacts, current source, and filenames
must remain separately hashed; raw solution indices are not durable identity.

rocBLAS exposes its gfx906 solution IDs as negative public integers. Their
absolute values join directly to positive serialized-library indices in the
sampled HHS shard. The 1024-cubed winner `-605555978` is a VALU assembly
schedule with macro tile 128x64x16, workgroup 16x16, thread tile 8x4, and no
workspace. Its normalized core fields have zero-difference matches in the
current vega20 YAML. This resolves the previously opaque gfx906 public ID.

The shipped code objects have also been unbundled and the winning symbols
bounded by the next external text symbol, because their ELF function sizes are
zero. The static native spans provide three useful mechanism witnesses:

| Target / winner | Static span | Instructions | Accelerated multiply | LDS reads / writes | Barriers |
| --- | ---: | ---: | ---: | ---: | ---: |
| gfx1100 / 1675 | 44,032 bytes | 8,130 | 54 `v_wmma_f32_16x16x16_f16` | 332 / 28 | 14 |
| gfx1201 / 133222 | 115,968 bytes | 20,201 | 144 `v_wmma_f32_16x16x16_f16` | 268 / 40 | 34 |
| gfx906 / -605555978 | 18,432 bytes | 2,761 | 1,056 `v_dot2_f32_f16` | 192 / 48 | 5 |

These are static counts across each complete symbol region, not dynamic counts
for the sampled request. In particular, the hipBLASLt winners contain runtime
GSU, activation, argument, and epilogue paths plus internal calls: each has
five static `s_endpgm` instructions and hundreds of branch/call sites. The
large spans are direct evidence of opportunity for exact Loom specialization,
but a control-flow/dynamic trace is still needed before copying an issue
schedule. The tracked native summaries retain the code-object hash, bounds,
complete mnemonic histogram, and coarse memory/synchronization counts.

## Loom mechanism result and target scope

The authored sources declare `gfx11-generic` and `gfx12-generic`, then compile
to exact `gfx1100` and `gfx1201` profiles. Both HSACOs load through the HIP
module API and produce an exact 16 for every element of an all-ones 16x16x16
multiply, with zero mismatches. Disassembly proves one native
`v_wmma_f32_16x16x16_f16` in each kernel.

The family split has a structural cause: the gfx11 FP16 operand carrier is
`vector<16xf16>` (eight VGPRs per lane), while gfx12 uses `vector<8xf16>` (four
VGPRs per lane). Both return `vector<8xf32>`. This is a legitimate family
variant. Exact physical target names remain JIT specialization and evidence
facts; tuning preferences such as macro tile or prefetch depth are not grounds
for an exact-ISA source.

Compilation for gfx906 fails before kernel lowering with `AMDGPU target
'gfx906' is not supported`. The incumbent recipe is representable as ordinary
VALU/LDS machinery, but Loom target enablement is a real prerequisite.

## First full GEMM vertical slice

One gfx12-family Loom source now expresses an exactly specialized
column-major NN GEMM through `gemm.m`, `gemm.n`, and `gemm.k` config values.
The first candidate deliberately assigns one wave to one 16x16 output tile,
loops over K in 16-element WMMA steps, reads fragments directly from global
memory, and publishes FP32 accumulators as FP16. The 256-cubed and 1024-cubed
specializations compile to separate HSACOs and run through the same explicit
probe as the incumbent.

| Shape | Loom median | Best forced hipBLASLt median | Loom / incumbent | Loom correctness |
| --- | ---: | ---: | ---: | --- |
| 256 cubed | 16.0605 us (2.089 TF/s) | 14.7600 us (2.273 TF/s), solution 133764 | 1.088x | zero mismatches |
| 1024 cubed | 112.2400 us (19.133 TF/s) | 29.5405 us (72.696 TF/s), solution 133222 | 3.800x | zero mismatches |

These adjacent runs are a discriminator, not the final interleaved benchmark.
The large-shape gap is nevertheless structurally explained by compiler
evidence: the 1024 specialization issues an estimated 270,532,608 bytes across
the dispatch because every 16x16 output wave reloads its own A and B fragments.
It has no LDS, uses 28 VGPRs without spills, and executes 64 WMMA instructions
per workgroup. The incumbent winner instead assigns a 128x128 macro tile to a
four-wave workgroup and cooperatively stages a depth-32 tile. The next candidate
should reconstruct that output ownership and LDS reuse, not tune the direct
kernel blindly. The small 256 result is launch/overhead dominated and is not
evidence that the direct schedule generalizes.

## Program-derived cache identity

The provider design requires exact request facts to enter Loom specialization
without making every raw M/N/K tuple a distinct HSACO. Cache identity must be
derived from the specialized canonical program, analogous to hashing
preprocessed C for `ccache`.

A minimal tool-level proxy confirms both the intended collapse and a pipeline
pitfall. Exact config values M=32 and M=48 both specialize `M % 16` to zero,
while M=33 specializes it to one. `canonicalize,dce` alone leaves the resolved
`config.def` in serialized output, so all three raw hashes differ. Adding
`symbol-dce` removes that dead binding: M=32 and M=48 then have identical
canonical hashes, while M=33 remains distinct. The pass boundary is therefore
part of the key contract, not incidental cleanup.

This is a canonical-text proxy, not yet the final `loomc` bytecode and target
specialization path. The checked-out public API exposes compilation to a
transformed bytecode artifact without HSACO emission. Benchmarking that and
other identity forms is explicitly deferred: this spike fully specializes each
candidate through HSACO and records the artifact hash. This is the deliberate
phase-one implementation rule, not merely a temporary tool limitation. The
eventual provider must repeat the equality/inequality test through its
production pipeline. If it cannot provide a stable and cheap identity, Loom
must be extended; tuple-keying is not an acceptable final design.

## What this proves and does not prove

It proves that the primary FP16 matrix primitive is expressible and physically
executable through Loom on gfx11 and gfx12, that the public incumbent selection
can be exhaustively checked, and that sampled serialized schedules can be
joined to source recipes. It also makes the gfx906 dependency concrete.

It does not prove a parity tiled GEMM, edge handling, alpha/beta, bias erasure,
the final bytecode/native cache-key path, broader shapes, or any accelerated
type beyond FP16. Those are the next executable gates.
