# Reproducing Spike 006

Commands assume the hrx-demos repository root, the selected ROCm installation
at `<rocm>`, and HRX main at `f17f69e82` (#513).

## Static census and corpora

```shell
python experimental/tensile/research/tools/analyze_tensile_solution_space.py \
  <rocblas-navi31-logic-root> <hipblaslt-gfx1201-logic-root> \
  --output experimental/tensile/research/spike-006-solution-space-jit/results/accelerated-solution-space.json

python experimental/tensile/research/tools/build_router_corpus.py \
  <f16-nn-logic.yaml> --target gfx1201 --backend hipblaslt \
  --tile-m 128 --tile-n 128 --tile-k 32 --anchor 1024x1024x1024 \
  --limit 48 --output <output>/gfx1201-corpus.json
```

The checked corpus commands additionally use `--align-router-cells` for the
gfx1100 `64x96` motif because classic router points are not generally N96
multiples. The generated cell retains its original source row and shape.

## LoomC key and startup study

Build `//experimental/tensile:loomc-jit-benchmark` against the same HRX
revision in Bazel `opt` mode. The checked module pin uses the older module name,
so a local module override may be required; do not leave that override in the
repository.

```shell
python experimental/tensile/research/tools/run_program_key_study.py \
  experimental/tensile/research/spike-006-solution-space-jit/corpus/gfx1201-f16-nn.json \
  --source experimental/tensile/research/spike-004-default-pipeline-microkernels/loom/gemm-f16-f32-mt128x128x32-gfx12-pack-microkernel-exact.loom \
  --target gfx1201 \
  --symbol gemm_f16_f32_mt128x128x32_gfx12_scalar_acc \
  --compiler-id hrx-f17f69e82-default-pipeline \
  --formatter <hrx>/bazel-bin/loom/src/loom/tools/loom-format/loom-format \
  --benchmark bazel-bin/experimental/tensile/loomc-jit-benchmark \
  --rocm-lib <rocm>/lib --workers 4 --output <output>/keys.json
```

Run workers 1, 2, and 4 in fresh processes. Each compiler subprocess owns its
context/workspace; source parsing is excluded because the input is `.loombc`.

## Performance basket

```shell
python experimental/tensile/research/tools/run_performance_basket.py \
  experimental/tensile/research/spike-006-solution-space-jit/corpus/gfx1201-f16-nn.json \
  --source <gfx12-motif.loom> \
  --symbol gemm_f16_f32_mt128x128x32_gfx12_scalar_acc \
  --device 2 --visible-device 2 --vendor-backend hipblaslt \
  --probe experimental/tensile/build/blas-probe \
  --runner <hrx>/bazel-bin/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom \
  --fixture-tool experimental/tensile/research/tools/make_shape_benchmark_fixture.py \
  --rocm-lib <rocm>/lib --limit 12 --output <output>/performance.json
```

The Loom runner uses `--pipeline=default`, 20 warmup batches, and a 500 ms
minimum measurement window. Shorter windows did not reliably raise gfx1201
clocks and produced a false anchor regression. Vendor sweeps skip per-candidate
readback only while selecting; the winning forced solution is re-run with 1024
reference samples.

## Validation boundary

The prior motifs retain full numerical and native access-sanitizer evidence.
Every generated dynamic benchmark adds an exact checked output before timing.
No Loom artifact is launched through HIP in this spike. Vendor artifacts use
the public HIP/rocBLAS/hipBLASLt APIs.

## Alternatives Considered

- `--pipeline=none` is intentionally absent.
- Text compilation is not used for the JIT timing/key result; text is used only
  to generate temporary correctness-gated runner fixtures.
- Generated HSACOs and transformed bytecode remain temporary artifacts; compact
  hashes, sizes, timings, and failures are retained in JSON.
