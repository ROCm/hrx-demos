# Reproduction

Commands assume the hrx-demos repository root, a selected ROCm installation,
and HRX tools built from commit `f17f69e82` (#513).

```shell
HRX=../hrx-system
ROCM=<selected-rocm>
LOOM_DIR=experimental/tensile/research/spike-005-gfx11-structured-recovery
FINAL=$LOOM_DIR/loom/gemm-f16-f32-mt64x96x32-gfx11-k32-unrolled-locked-native-pub.loom
VALIDATION=$LOOM_DIR/loom/gemm-f16-f32-mt64x96x32-gfx11-k32-unrolled-locked-native-pub-validation.loom
```

## Rebuild the retained source chain

```shell
python "$LOOM_DIR/tools/make_gfx11_high_k32_ring.py" \
  experimental/tensile/research/spike-004-default-pipeline-microkernels/loom/gemm-f16-f32-mt64x96x32-gfx11-high-exact.loom /tmp/ring.loom
python "$LOOM_DIR/tools/add_k32_unroll_policy.py" --schedule linear \
  /tmp/ring.loom /tmp/unrolled.loom
python "$LOOM_DIR/tools/reorder_gfx11_mma_wavefront.py" \
  /tmp/unrolled.loom /tmp/reordered.loom
python "$LOOM_DIR/tools/peel_gfx11_high_k32_tail.py" \
  /tmp/reordered.loom /tmp/high-only.loom
python experimental/tensile/research/spike-003-native-upward/tools/use_gfx11_native_high_epilogue.py \
  /tmp/high-only.loom /tmp/native-pub.loom
python "$LOOM_DIR/tools/use_gfx11_wmma_helper.py" --schedule locked \
  --expected-count 24 /tmp/native-pub.loom /tmp/helper.loom
python "$LOOM_DIR/tools/fix_gfx11_high_publication_rows.py" \
  /tmp/helper.loom /tmp/final.loom
"$HRX/bazel-bin/loom/src/loom/tools/loom-format/loom-format" \
  --in-place /tmp/final.loom
cmp /tmp/final.loom "$FINAL"
```

## Correctness and native access sanitizer

```shell
LD_LIBRARY_PATH="$ROCM/lib" ROCR_VISIBLE_DEVICES=1 \
  "$HRX/bazel-bin/loom/src/loom/tools/iree-test-loom/iree-test-loom" \
  "$VALIDATION" --device=amdgpu --pipeline=default \
  --case=@gemm_f16_f32_mt64x96x32_gfx11_min_nonuniform_case \
  --config=gemm.m=64 --config=gemm.n=96 --config=gemm.k=64

LD_LIBRARY_PATH="$ROCM/lib" ROCR_VISIBLE_DEVICES=1 \
  "$HRX/bazel-bin/loom/src/loom/tools/iree-test-loom/iree-test-loom" \
  "$VALIDATION" --device=amdgpu --pipeline=default --sanitizer=asan \
  --case=@gemm_f16_f32_mt64x96x32_gfx11_min_nonuniform_case \
  --config=gemm.m=64 --config=gemm.n=96 --config=gemm.k=64

LD_LIBRARY_PATH="$ROCM/lib" ROCR_VISIBLE_DEVICES=1 \
  "$HRX/bazel-bin/loom/src/loom/tools/iree-test-loom/iree-test-loom" \
  "$VALIDATION" --device=amdgpu --pipeline=default \
  --case=@gemm_f16_f32_mt64x96x32_gfx11_nonuniform_case \
  --config=gemm.m=1024 --config=gemm.n=960 --config=gemm.k=1024
```

The full CPU oracle intentionally takes tens of seconds. It checks every
output and is not part of the timing loop.

## Loom and vendor timing

Use a fresh artifact directory for each Loom run:

```shell
LD_LIBRARY_PATH="$ROCM/lib" ROCR_VISIBLE_DEVICES=1 \
  "$HRX/bazel-bin/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom" \
  "$FINAL" --device=amdgpu --pipeline=default \
  --benchmark=@gemm_f16_f32_mt64x96x32_gfx11_pgr2_1024x960x1024 \
  --measure=dispatch_complete --batch-size=64 --iterations=20 \
  --warmup-iterations=5 --min-time-ms=200 --input-ring-count=1 \
  --config=gemm.m=1024 --config=gemm.n=960 --config=gemm.k=1024 \
  --compile-report=details --artifact-manifest=analysis \
  --artifact-bundle-policy=debug --artifact-bundle-dir=/tmp/gfx11-final

LD_LIBRARY_PATH="$ROCM/lib" \
  experimental/tensile/build/blas-probe --backend hipblaslt --device 1 \
  --m 1024 --n 960 --k 1024 --warmup 20 --iterations 40 \
  --reference-samples 1024 --solution-index 1675 --max-algorithms 1
```

Run five independent blocks of each. The checked result bootstraps the median
of the five per-run medians independently with 100,000 resamples and seed
23063. A separate full-reference vendor run also passed; sampled references
are used for timing so CPU reference work does not idle the GPU between the
default and forced algorithm measurements.

```shell
python "$LOOM_DIR/tools/analyze_block_medians.py" \
  "$LOOM_DIR/results/gfx1100-performance.json"
```

## Production-shaped bytecode-to-HSACO timing

```shell
"$HRX/bazel-bin/loom/src/loom/tools/loom-format/loom-format" "$FINAL" \
  --from=text --to=bc --output=/tmp/kernel.loombc
"$HRX/bazel-bin/loom/src/loom/tools/loom-format/loom-format" \
  experimental/tensile/research/spike-004-default-pipeline-microkernels/config/gfx1100-f16-1024x960x1024.loom \
  --from=text --to=bc --output=/tmp/config.loombc
bazel-bin/experimental/tensile/loomc-jit-benchmark \
  /tmp/kernel.loombc /tmp/config.loombc gfx1100 \
  gemm_f16_f32_mt64x96x32_gfx11_pgr2 30 /tmp/kernel.hsaco
```

Build the benchmark against the same HRX revision with Bazel optimization.
The hrx-demos module pin still names the pre-rename module and may require a
temporary local override; do not compare a stale benchmark/bytecode format or
a non-optimized host build.

## Alternatives Considered

- Direct HIP launch is used only by `blas-probe` for vendor artifacts. It is
  not a fallback runner for Loom correctness or performance.
- `--pipeline=none` is intentionally absent: it would not validate the
  maintained source or compiler contracts.
- Raw bundles and HSACOs are regenerated rather than committed.
