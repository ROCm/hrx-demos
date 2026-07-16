# HRX Demos

Standalone, production-shaped applications built on
[HRX](https://github.com/ROCm/hrx-system). The first application is an
Ideogram 4 text-to-image implementation with direct AMDGPU execution, Loom
kernels, dynamic structured prompts, compact FP8 checkpoints, and LoRA support.

HRX Demos keeps model scheduling, parameter residency, kernel selection, and
diagnostics explicit in a small C runtime. Ideogram 4 accepts a prompt or a
structured request and emits an image through the `hrx-id4` command-line tool.

## Performance

Reproducible latency, throughput, and memory results will be published here.
The current implementation has completed 1024x1024 FP8 generation on a Radeon
Pro W7900 with an observed physical VRAM peak of approximately 25.8 GiB.

## Supported GPUs

| GPU | Architecture | Status |
| --- | --- | --- |
| Radeon Pro W7900 | gfx1100 | Verified |

Other AMD GPUs are not yet part of the supported and measured configuration.

## Installation

Manylinux wheels will be the primary distribution under the `hrx-demos`
package name. PyPI publication is not part of the current release.

```bash
pip install hrx-demos
```

The current source tree can produce a Python-minor-independent, Linux x86-64
wheel. From a clone with initialized submodules, configure the local ROCm and
Bazel paths once and then build:

```bash
python build_tools/setup_python.py --rocm=/path/to/rocm
python -m pip install build
python -m build --wheel
python -m pip install dist/hrx_demos-0.1.0-py3-none-linux_x86_64.whl
```

This development wheel bundles the HSA runtime and its ROCm support libraries.
It deliberately uses a `linux_x86_64` platform tag; a later release automation
step will build and validate the publishable manylinux wheel in the pinned
manylinux container.

## Build From Source

Clone with submodules, select a ROCm installation, and build with Bazel 9.1:

```bash
git clone --recurse-submodules https://github.com/ROCm/hrx-demos.git
cd hrx-demos
export IREE_ROCM_PATH=/path/to/rocm
export CC="$IREE_ROCM_PATH/lib/llvm/bin/clang"
export CXX="$IREE_ROCM_PATH/lib/llvm/bin/clang++"
bazel build -c opt //binding/cli:id4
```

To develop against a local HRX checkout instead of the pinned submodule:

```bash
bazel build --override_module=iree=/path/to/hrx-system -c opt //binding/cli:id4
```

The setup command uses Bazel from `BAZEL`, `PATH`, or a containing workspace's
`programs/bazel/bin/bazel`. It validates Bazel 9.1 and writes the ignored
`.bazelrc.local` with the selected ROCm root and compilers. See
[DEVELOPMENT.md](DEVELOPMENT.md) for the pinned Bazelisk installation command.

## Quick Start

The model weights are gated. Accept the
[Ideogram 4 FP8 license](https://huggingface.co/ideogram-ai/ideogram-4-fp8),
download the required checkpoints as described in the
[quickstart](docs/quickstart.md), then generate an image:

```bash
hrx-id4 \
  --flagfile=docs/ideogram4-fp8.flags \
  --device=amdgpu:// \
  --prompt_json_file=docs/requests/long_1024.json \
  --output=ideogram4.ppm
```

## Documentation

- [Build, model download, and generation quickstart](docs/quickstart.md)
- [Ideogram 4 FP8 performance case study](docs/case_study.md)
- [Production-shaped model porting guide](docs/porting_guide.md)
- [Target and scheduling generalization roadmap](docs/generalization_roadmap.md)
- [Architecture and contributor guide](DEVELOPMENT.md)

## License

Apache 2.0 with LLVM Exceptions. Model weights are external and retain their
respective licenses; Ideogram 4 weights are licensed for non-commercial use.
