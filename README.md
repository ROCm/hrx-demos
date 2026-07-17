# HRX Demos

Standalone, demo applications built on
[HRX](https://github.com/ROCm/hrx-system). The first application is an
Ideogram 4 text-to-image implementation with direct AMDGPU execution, Loom
kernels, dynamic structured prompts, compact FP8 checkpoints, and LoRA support.

HRX Demos keeps model scheduling, parameter residency, kernel selection, and
diagnostics explicit in a small C runtime. Ideogram 4 accepts a prompt or a
structured request and emits an image through the `hrx-id4` command-line tool.

## Performance

Performance data coming soon.

## Installation

`hrx-demos` is available on PyPI. Install it with using your favorite python
package manager:

```bash
pip install hrx-demos
```

## Available Demos

### Ideogram-4 FP8

#### Download models (see docs/quickstart.md for full details):

```
# ideogram-4 must have its license accepted:
#   https://huggingface.co/ideogram-ai/ideogram-4-fp8
hf download ideogram-ai/ideogram-4-fp8 \
  transformer/diffusion_pytorch_model.safetensors \
  unconditional_transformer/diffusion_pytorch_model.safetensors \
  --revision ee79a7237b519f1402ceacf952f30c8a31ec5073 \
  --local-dir models/ideogram-4-fp8

# Qwen with stock, block-scaled FP8 checkpoints
hf download Qwen/Qwen3-VL-8B-Instruct-FP8 \
  tokenizer.json \
  model.safetensors.index.json \
  model-00001-of-00002.safetensors \
  model-00002-of-00002.safetensors \
  --revision 9cdc6310a8cb770ce18efaf4e9935334512aee45 \
  --local-dir models/qwen3-vl-8b-instruct-fp8

# Accept the FLUX.2 non-commercial license and download its
# single-file autoencoder: 
#   https://huggingface.co/black-forest-labs/FLUX.2-dev
hf download black-forest-labs/FLUX.2-dev \
  ae.safetensors \
  --revision 26afe3a78bb242c0a8bb181dcc8937bb16e5c66c \
  --local-dir models/flux2-dev
```

#### Generate an Image from Sample Flags

You may copy the flag and prompt files and customize as desired. The defaults
assume you are in a PWD with a models/ sub-directory as downloaded above.

```bash
hrx-id4 \
  --flagfile=docs/ideogram4-fp8.flags \
  --device=amdgpu:// \
  --prompt_json_file=docs/requests/long_1024.json \
  --output=ideogram4.ppm
```

## Building From Source

### Python Wheels

Python wheels are typically built in a manylinux container but can be built for
your machine:

```bash
python build_tools/setup_python.py --rocm=/path/to/rocm
python -m pip install build
python -m build --wheel
python -m pip install dist/hrx_demos-0.1.0-py3-none-linux_x86_64.whl
```

This development wheel bundles the HSA runtime and its required ROCm support
library closure.

### Building for Development

Clone with submodules, select a ROCm installation, and build with Bazel 9.1:

```bash
git clone --recurse-submodules https://github.com/ROCm/hrx-demos.git
cd hrx-demos
export IREE_ROCM_PATH=/path/to/rocm
export CC="$IREE_ROCM_PATH/lib/llvm/bin/clang"
export CXX="$IREE_ROCM_PATH/lib/llvm/bin/clang++"
bazel run -c opt //binding/cli:id4
```

To develop against a local HRX checkout instead of the pinned submodule:

```bash
bazel build --override_module=iree=/path/to/hrx-system -c opt //binding/cli:id4
```

The setup command uses Bazel from `BAZEL`, `PATH`. It validates Bazel 9.1 and 
writes the ignored `.bazelrc.local` with the selected ROCm root and compilers. See
[DEVELOPMENT.md](DEVELOPMENT.md) for the pinned Bazelisk installation command.

## Documentation

- [Build, model download, and generation quickstart](docs/quickstart.md)
- [Ideogram 4 FP8 performance case study](docs/case_study.md)
- [Production-shaped model porting guide](docs/porting_guide.md)
- [Target and scheduling generalization roadmap](docs/generalization_roadmap.md)
- [Architecture and contributor guide](DEVELOPMENT.md)

## License

Apache 2.0 with LLVM Exceptions. Model weights are external and retain their
respective licenses; Ideogram 4 weights are licensed for non-commercial use.
