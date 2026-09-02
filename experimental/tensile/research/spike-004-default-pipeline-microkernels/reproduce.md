# Reproducing Spike 4

## Revisions and build

Use this hrx-demos revision together with HRX main at
`f17f69e82df39f31b9b3ee404c5a33ab7c2c47b6` (#513). The historical
`loom-blas/spike4-low-invoke-experiment` branch at `db115431e` remains an
archaeological mechanism witness; it is not required for these commands.

From the HRX checkout, build the Loom runners with the AMDGPU provider:

```shell
bazel build --//runtime/config/hal:drivers=amdgpu,task \
  //loom/src/loom/tools/iree-test-loom \
  //loom/src/loom/tools/iree-benchmark-loom
```

If the Bazel sandbox cannot write the default ccache temporary directory,
provide writable, task-specific `CCACHE_DIR` and `CCACHE_TEMPDIR` values
through `--action_env`. The historical experimental branch was also checked
with these focused suites:

```shell
bazel test \
  //loom/src/loom/codegen/low/lower:all \
  //loom/src/loom/transforms/symbol:all \
  //loom/src/loom/ops/low/test:all \
  //loom/src/loom/target:all
```

Those suites contained 30 tests and all passed at the recorded experimental
revision.

The examples below assume `HRX` names that checkout, `TENSILE` names this
Spike 4 directory, and `LD_LIBRARY_PATH` explicitly includes the selected ROCm
installation's `lib` directory. No ROCm installation is selected implicitly.
No experiment environment variable is needed on the upstream revision.

## Correctness and access sanitizer

Use GPU 2 for gfx1201 and GPU 1 for gfx1100 on the recorded three-GPU host:

```shell
ROCR_VISIBLE_DEVICES=2 \
  "$HRX/bazel-bin/loom/src/loom/tools/iree-test-loom/iree-test-loom" \
  "$TENSILE/loom/gfx12-pack-invoke-object-abi.loom-test" \
  --device=amdgpu --pipeline=default

ROCR_VISIBLE_DEVICES=2 \
  "$HRX/bazel-bin/loom/src/loom/tools/iree-test-loom/iree-test-loom" \
  "$TENSILE/loom/gemm-f16-f32-mt128x128x32-gfx12-pack-microkernel-exact.loom" \
  --device=amdgpu --pipeline=default --sanitizer=asan \
  --case=@gemm_f16_f32_mt128x128x32_gfx12_scalar_acc_access_tile_case \
  --config=gemm.m=128 --config=gemm.n=128 --config=gemm.k=64

ROCR_VISIBLE_DEVICES=1 \
  "$HRX/bazel-bin/loom/src/loom/tools/iree-test-loom/iree-test-loom" \
  "$TENSILE/loom/gemm-f16-f32-mt64x96x32-gfx11-high-exact.loom" \
  --device=amdgpu --pipeline=default \
  --config=gemm.m=1024 --config=gemm.n=960 --config=gemm.k=1024
```

The sanitizer command intentionally exercises one complete macro-tile. Low
memory coverage is not implied; see `compiler-usability-access-sanitizer.md`.

## Default-pipeline benchmark evidence

Run the embedded correctness-gated benchmarks through the sanctioned runner:

```shell
ROCR_VISIBLE_DEVICES=2 \
  "$HRX/bazel-bin/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom" \
  "$TENSILE/loom/gemm-f16-f32-mt128x128x32-gfx12-pack-microkernel-exact.loom" \
  --device=amdgpu \
  --benchmark=@gemm_f16_f32_mt128x128x32_gfx12_scalar_acc_1024 \
  --pipeline=default --measure=dispatch_complete --batch-size=64 \
  --iterations=20 --warmup-iterations=5 --min-time-ms=200 \
  --input-ring-count=1 --profile-final-batch=true \
  --config=gemm.m=1024 --config=gemm.n=1024 --config=gemm.k=1024 \
  --artifact-bundle-policy=debug --artifact-bundle-dir=/tmp/loom-gfx12

ROCR_VISIBLE_DEVICES=1 \
  "$HRX/bazel-bin/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom" \
  "$TENSILE/loom/gemm-f16-f32-mt64x96x32-gfx11-high-exact.loom" \
  --device=amdgpu \
  --benchmark=@gemm_f16_f32_mt64x96x32_gfx11_pgr2_1024x960x1024 \
  --pipeline=default --measure=dispatch_complete --batch-size=64 \
  --iterations=20 --warmup-iterations=5 --min-time-ms=200 \
  --input-ring-count=1 --profile-final-batch=true \
  --config=gemm.m=1024 --config=gemm.n=960 --config=gemm.k=1024 \
  --artifact-bundle-policy=debug --artifact-bundle-dir=/tmp/loom-gfx11
```

Vendor comparison uses the explicit public-API harness retained by the earlier
spikes; do not substitute rocBLAS/hipBLASLt tune or bench utilities because
they hide selection and launch details.

## Expected compiler-contract results

Both exact-target and family-generic object fixtures now compile and execute.
The latter directly checks target/carrier projection:

```shell
ROCR_VISIBLE_DEVICES=2 \
  "$HRX/bazel-bin/loom/src/loom/tools/iree-test-loom/iree-test-loom" \
  "$TENSILE/loom/gfx12-pack-invoke-family-generic.loom-test" \
  --device=amdgpu --pipeline=default
```

The unfenced `schedule(locked)` helper emits its four permutations in authored
order. Inspect the emitted target artifact when validating that property; a
numeric check alone cannot distinguish legal reorderings.

The CFG reproducer remains an expected rejection because #513 supports one
outer helper block:

```shell
ROCR_VISIBLE_DEVICES=2 \
  "$HRX/bazel-bin/loom/src/loom/tools/iree-test-loom/iree-test-loom" \
  "$TENSILE/reproducers/low-invoke-cfg-helper-gfx1201.loom" \
  --device=amdgpu --pipeline=default
```

is expected to fail with `TARGET/072` and “required inlining supports exactly
one outer body block.” This is an unsupported-shape probe, not by itself a
request to add general CFG inlining. `--pipeline=none` is not a substitute.

The minimal authored-Low LDS access probe executes under the sanitizer while
retaining the known coverage limitation:

```shell
ROCR_VISIBLE_DEVICES=2 \
  "$HRX/bazel-bin/loom/src/loom/tools/iree-test-loom/iree-test-loom" \
  "$TENSILE/reproducers/low-invoke-lds-access-gfx1201.loom" \
  --device=amdgpu --pipeline=default --sanitizer=asan
```

Its sanitizer site table covers the High global input and output operations,
not the authored Low `ds_read_b64`; see the compiler usability report.

## Bytecode-to-HSACO latency

Compile the Loom and config sources to bytecode, build the retained
`research/tools/loomc_jit_benchmark.c` against the same HRX revision, then pass
the two bytecode files to the benchmark. The hrx-demos Bazel label is
`//experimental/tensile:loomc-jit-benchmark`, but its checked-in submodule pin
still uses HRX's older `iree` module identity; update that pin/override to a
compatible landed HRX revision before expecting the label to build directly.
Build the host binary with `--compilation_mode=opt`; non-optimized host builds
materially inflate the result. The checkpoint binary was built from this exact
source in the experimental HRX tree.

The tool prepares the provider index, target profile, compiler, linker, and
source-to-prepared-Low pass program once. Each measured sample links one root,
applies exact target/config specialization, compiles, and emits an in-memory
HSACO:

```shell
loomc-jit-benchmark kernel.loombc config.loombc gfx1201 \
  gemm_f16_f32_mt128x128x32_gfx12_scalar_acc 30 output.hsaco
```

The compact phase distributions and byte sizes are retained in
`results/loombc-to-hsaco-latency.json`.

## Alternatives considered

- Direct HIP launch is reserved for vendor artifacts and historical oracle
  classification, not Loom acceptance evidence.
- `--pipeline=none` is useful for compiler debugging but cannot establish
  correctness, sanitizer coverage, or performance.
- Generated HSACOs and raw bundles are not committed; regenerate them from the
  retained source and revisions, then compare their reports rather than
  relying on stale binary identity.
