# Compiler usability: `access` sanitizer on High and prepared Low

## Gfx12 High summary

Loom's native `access` sanitizer correctly compiles and executes the complete
gfx12 GEMM address map for one `128x128x32` macro-tile. The same sanitizer is
not currently usable on the production `1024x1024x1024` specialization after
the K loop has been fully unrolled: compiler-inserted instrumentation expands
the kernel from 1,024 static WMMAs to 75,189 scheduled packets, triggers 1,662
spill warnings, and terminates at the spill-materialization iteration limit.

This is a compiler usability failure, not a reported illegal device access.
It also does not invalidate the uninstrumented production result: that kernel
has zero spill storage and passes the independent CPU differential. It means
that full unrolling and full-shape access instrumentation cannot currently be
composed into one verification cut.

## Reproducer

Source:
[`loom/gemm-f16-f32-mt128x128x32-gfx12-scalar-acc.loom`](loom/gemm-f16-f32-mt128x128x32-gfx12-scalar-acc.loom)

Target and device:

- source target: `gfx12-generic`
- selected processor: `gfx1201`
- target configuration: `amdgpu.rdna4.core`
- device: Radeon RX 9070 XT

The production case is invoked with:

```shell
ROCR_VISIBLE_DEVICES=2 iree-test-loom \
  loom/gemm-f16-f32-mt128x128x32-gfx12-scalar-acc.loom \
  --device=amdgpu \
  --case=@gemm_f16_f32_mt128x128x32_gfx12_scalar_acc_access_case \
  --config=gemm.m=1024 --config=gemm.n=1024 --config=gemm.k=1024 \
  --sanitizer=access
```

Observed structured result:

```json
{
  "failed_sample_count": 1,
  "issues": [{
    "category": "compile_rejected",
    "provider": "actual",
    "stage": "emit",
    "kind": "emit_diagnostics"
  }]
}
```

The terminal diagnostic is:

```text
BACKEND/021: spill-materialization-iteration-limit; iteration 8 of 8,
2 pending spill plans, 2 spill-slot assignments, and 75189 scheduled packets
```

Raw evidence:

- [`access-full-report.json`](artifacts/gfx12-scalar-acc-k-unroll/access-full-report.json)
- [`access-full-stderr.txt`](artifacts/gfx12-scalar-acc-k-unroll/access-full-stderr.txt)

## Reduced sanitizer witness

The reduced case changes only specialization facts and tensor extents:
`M=N=128`, `K=32`. It retains all eight 128-bit global-load address maps, all
eight LDS stores, both barriers, both K=16 fragment-load/MMA phases, and all
sixteen result-fragment stores. In other words, it exercises every address
formula and memory space in the production kernel without cloning the body 32
times.

```shell
ROCR_VISIBLE_DEVICES=2 iree-test-loom \
  loom/gemm-f16-f32-mt128x128x32-gfx12-scalar-acc.loom \
  --device=amdgpu \
  --case=@gemm_f16_f32_mt128x128x32_gfx12_scalar_acc_access_tile_case \
  --config=gemm.m=128 --config=gemm.n=128 --config=gemm.k=32 \
  --sanitizer=access
```

Observed result: one case planned, one sample passed, one expectation passed,
and no failures, skips, planning issues, or sanitizer events. See
[`access-tile-report.json`](artifacts/gfx12-scalar-acc-k-unroll/access-tile-report.json).

## Usability assessment

The one-tile witness is sufficient to validate the current address map while
schedule work remains in High Loom. It is not a satisfactory permanent escape
hatch: a production verification workflow must permit instrumentation at the
same specialization used for timing.

The preferred compiler remedy is to preserve sanitizer checks through the
structured K loop or otherwise avoid cloning invariant check machinery during
`scf.for` full unrolling. A second acceptable route is to repair allocation of
the loop-carried accumulator bank, eliminating the need to fully unroll K in
the first place. Raising the spill iteration limit is not a remedy; the
instrumented kernel already has pathological code and allocation growth.

Until one of those routes lands, report the two gates separately:

1. full-shape uninstrumented independent differential for numerical evidence;
2. complete one-tile native `access` run for address-space legality.

Do not describe the full production specialization as access-sanitized.

## Gfx11 runtime initialization pitfall and passing witness

The earlier gfx11 page fault was a harness error. A sanitized HSACO is not a
standalone HIP module: Loom's test runtime initializes shadow/report globals
required by the instrumented accesses. Launching that HSACO directly through
`hipModuleLaunchKernel` bypassed this initialization and faulted in sanitizer
metadata, not in a demonstrated source access.

The supported testbench must be built with the AMDGPU driver enabled:

```shell
bazel build --//runtime/config/hal:drivers=amdgpu \
  //loom/src/loom/tools/iree-test-loom:iree-test-loom
```

Without that flag, the binary lacks the AMDGPU dialect/driver and rejects the
case before execution. With it, the reduced `64x96x64` case embedded in
[`loom/gemm-f16-f32-mt64x96x32-gfx11-double-buffer-native-lds.loom`](loom/gemm-f16-f32-mt64x96x32-gfx11-double-buffer-native-lds.loom)
passes both unsanitized and `--sanitizer=access
--sanitizer-reporting=report-only` runs. The numerical expectation passes and
there are no sanitizer events. Evidence is under
[`artifacts/gfx11-high-access-testbench/`](artifacts/gfx11-high-access-testbench/).

The terminal Low epilogue's exact scalar address formulas were also raised to
a High address-only witness:
[`loom/gemm-f16-f32-mt64x96x32-gfx11-native-epilogue-address-witness.loom`](loom/gemm-f16-f32-mt64x96x32-gfx11-native-epilogue-address-witness.loom).
Its unsanitized and access-sanitized testbench runs both pass; evidence is in
[`artifacts/gfx11-high-native-epilogue-address-witness/`](artifacts/gfx11-high-native-epilogue-address-witness/).

This address-only source deliberately retains the original High WMMA semantic
roles. The exact paired terminal transform must also swap the two physically
symmetric gfx11 WMMA operands, but High target validation rejects that source
with `TARGET/039 fragment_roles`. The should-work paired source and compile log
are preserved as
[`loom/gemm-f16-f32-mt64x96x32-gfx11-native-epilogue.loom`](loom/gemm-f16-f32-mt64x96x32-gfx11-native-epilogue.loom)
and `artifacts/gfx11-high-native-epilogue/compile.log`. This is a separate High
representation usability gap, not a sanitizer failure.

## Prepared-Low coverage gap

The schedule-congruent gfx12 witness and the native-order gfx11 witness are
prepared Low kernels. Compiling them with `--pipeline=none` is required to
preserve the authored schedule, but that pipeline does not apply access
instrumentation. Passing an access-sanitizer option alongside `pipeline=none`
therefore proves only that the existing Low kernel compiled; it does not prove
that it was instrumented.

Feeding the same prepared Low through the default access-sanitizer pipeline
does not provide an alternate path: lowering reaches an internal assertion in
the Low lowering path. The retained probe is
[`artifacts/access-low-default-probe/`](artifacts/access-low-default-probe/).

This gap matters more than a convenience feature. The exact schedules we need
to validate may require Low, and accidental success on one GPU is not a
legality proof. The required contract is one of:

1. an access-instrumentation pass that accepts prepared Low memory operations
   and preserves locked ordering; or
2. a documented split pipeline that instruments the last address-bearing High
   representation, then permits the inner loop to be replaced through
   `low.invoke` while retaining compatible checks.

The compiler must also make instrumentation coverage observable. If no
eligible operations were instrumented, a requested sanitizer verification
should fail or emit a structured `not_applied` result. Silent success is too
easy to misinterpret.

## Current verification statement

- Gfx12 High: the complete one-tile address map passed native access; the
  fully unrolled production specialization did not compile under access.
- Gfx11 High: the reduced exact-LDS and scalar-address witnesses pass through
  the supported AMDGPU testbench; the earlier raw-HIP fault is root-caused as
  missing sanitizer runtime initialization.
- Gfx11/gfx12 prepared Low: not access-sanitized because the preserving
  pipeline does not instrument Low.

Consequently, numerical correctness and schedule timing are established for
the best Low kernels, but their required native-access legality gate remains
open. No document should call those Low kernels sanitizer-validated.
