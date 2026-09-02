# Spike 001: primitive GEMM evidence loop

## Question

Can one source-led, auditable loop identify the best incumbent primitive FP16
GEMM, recover its routing and schedule evidence, express the mechanism in Loom,
and compare it on `gfx906`, `gfx1100`, and `gfx1201`?

The initial semantic boundary is column-major NN GEMM with FP16 A/B/C/D, FP32
alpha/beta and accumulation, batch one, alpha one, beta zero, and no epilogue.
This is the HHS-style path present in the sampled logic; output conversion to
FP16 is part of the boundary.

## Gates

1. Enumerate and force every eligible public-library algorithm for an exact
   request. Preserve names, indices, workspace, raw timings, and correctness.
2. Join the winning algorithm to source logic, serialized runtime library, code
   object, symbol, recipe, and generated/native schedule.
3. Classify router nodes and demonstrate what exact-shape JIT specialization
   removes.
4. Compile and execute the primitive Loom mechanism with compiler and native
   evidence.
5. Compare with paired physical measurements over router-derived shapes.

## Initial request

The first cheap discriminator is `M=N=K=256`. It is large enough to execute a
tiled matrix path on RDNA while keeping full algorithm enumeration cheap. It is
not a performance conclusion. Router interior, boundary, tail, skinny, and
large cases follow after the tool is validated.

## Status

The first evidence pass is complete. See [`findings.md`](findings.md) and the
tracked summaries under [`results/`](results/). The laboratory has enumerated
and checked the primitive request on all three GPUs; runtime solutions have
been joined to sampled source recipes; and family-generic Loom WMMA mechanism
probes compile and execute on the exact gfx1100 and gfx1201 devices. A first
full-K, column-major gfx12 GEMM is correct at 256 and 1024 cubed; its direct
global-memory schedule exposes the expected large-shape gap and makes the
incumbent macro-tile/LDS pipeline the next implementation step. A parity Loom
GEMM and router-boundary corpus remain. This spike may fully specialize through
HSACO; optimized pre-emission program hashing is deferred.

## Reproduce the gfx1201 direct vertical slice

Consult the compiler's agent help first:

```bash
<workspace>/sources/hrx-system/bazel-bin/loom/src/loom/tools/loom-compile/loom-compile \
  --agents_md
```

Then compile one exact request (change all three config values and the artifact
directory together for another square witness):

```bash
mkdir -p experimental/tensile/research/spike-001-primitive-gemm/artifacts/loom-gfx1201-gemm256
<workspace>/sources/hrx-system/bazel-bin/loom/src/loom/tools/loom-compile/loom-compile \
  experimental/tensile/research/spike-001-primitive-gemm/loom/gemm-f16-f32-direct-gfx12.loom \
  --backend=amdgpu-hal --target=gfx1201 \
  --config=gemm.m=256 --config=gemm.n=256 --config=gemm.k=256 \
  --output=experimental/tensile/research/spike-001-primitive-gemm/artifacts/loom-gfx1201-gemm256/gemm.hsaco \
  --compile-report=details \
  --compile-report-output=experimental/tensile/research/spike-001-primitive-gemm/artifacts/loom-gfx1201-gemm256/compile-report.json \
  --artifact-manifest=details \
  --emit-artifact-manifest=experimental/tensile/research/spike-001-primitive-gemm/artifacts/loom-gfx1201-gemm256/manifest.json
```

Run it through the same input, CPU-reference, and HIP-event machinery as the
incumbents:

```bash
experimental/tensile/build/blas-probe \
  --backend loom --device 2 --m 256 --n 256 --k 256 \
  --warmup 10 --iterations 20 \
  --hsaco experimental/tensile/research/spike-001-primitive-gemm/artifacts/loom-gfx1201-gemm256/gemm.hsaco \
  --kernel gemm_f16_f32_direct_gfx12 \
  --tile-m 16 --tile-n 16 --workgroup-size 32

experimental/tensile/build/blas-probe \
  --backend hipblaslt --device 2 --m 256 --n 256 --k 256 \
  --warmup 10 --iterations 20 --solution-index 133764
```
