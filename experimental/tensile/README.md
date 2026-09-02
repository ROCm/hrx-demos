# Tensile-to-Loom BLAS research

Research and implementation of BLAS kernels compiled with Loom. This directory
is self-contained within `hrx-demos`; use it as the root for the CMake probes
and all paths in the research notes.

The current architecture and execution plan are in
[`docs/design/loom-blas.md`](docs/design/loom-blas.md). The shortest evidence
handoff is
[`research/spike-003-native-upward/README.md`](research/spike-003-native-upward/README.md).

The repository begins as an experimental notebook with executable probes. See
[`research/`](research/) for captured questions, methods, evidence, and
findings. Provider implementation will grow from evidence retained here.

Nothing in `research/` should depend on vendor benchmark defaults. Experiments
record the public API request, enumerated/forced algorithms, inputs, timing
policy, runtime identity, and source identity explicitly.

Research records use portable path placeholders:

- `<tensile-root>` is this directory.
- `<workspace>` is a development workspace containing the referenced source
  checkouts.
- `<selected-rocm>` is the ROCm installation selected explicitly for a run.
- `<scratch>` is an ephemeral experiment directory outside the checkout.
