# Experiment 004: production-shaped Loom C API compilation

Production does not compile Loom text. The retained
[`loomc_jit_benchmark.c`](../../tools/loomc_jit_benchmark.c) therefore loads
the motif and config as Loom bytecode, prepares a provider index, linker,
compiler, target profile, and default pass program once, then measures each
root through link, compile, and in-memory HSACO emission.

Linking the selected root before compilation is material. Calling
`loomc_compile_module` directly on the unlinked four-buffer research module
left the unused C input absent from Low resources while D retained binding
index 3, producing `AMDGPU/010` and `AMDGPU/012`. `loom-compile` and the
production-shaped C API link path both compact the reachable root consistently
and succeed. This was an API-usage trap, not evidence against stable provider
arguments.

The 30-sample results are in
[`loombc-to-hsaco-latency.json`](../results/loombc-to-hsaco-latency.json).
With the benchmark and compiler built in Bazel `opt` mode, median
link+compile+emit time is 22.09 ms for the 58.1 KiB gfx12 motif and 11.77 ms
for the 16.1 KiB gfx11 motif. Those are ordinary default-pipeline latencies,
not the Spike 003 no-pipeline HSACO emission floor. A non-optimized host build
measured 62.71 ms and 45.81 ms respectively; host build mode must therefore be
part of any JIT latency record.

The retained hrx-demos Bazel target still resolves its checked-in HRX
submodule through the older module name `iree`; the experimental compiler
checkout has since renamed that module to `hrx`. For this checkpoint the exact
retained C source was built as a temporary `cc_binary` in the experimental HRX
checkout and then the temporary target was removed. The program compiled and
produced both recorded distributions. Aligning the hrx-demos dependency pin
with a landed HRX revision is packaging work, not part of the latency result.
