# Apple Silicon with Metal (MLX)

This document describes how run SGLang on Apple Silicon using [Metal (MLX)](https://opensource.apple.com/projects/mlx/). If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## Install SGLang

You can install SGLang using one of the methods below.

### Install from Source

```bash
# Use the default branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install sglang python package
pip install --upgrade pip
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"
```

## Launch of the Serving Engine

MLX inference is gated by the `SGLANG_USE_MLX=1` environment variable. Launch the server with:

```bash
SGLANG_USE_MLX=1 python -m sglang.launch_server \
  --model <MODEL_ID_OR_PATH> \
  --host 0.0.0.0 \
```

## Features

Overlap scheduling using `mlx.async_eval()` is enabled by default. To disable this, use the `--disable-overlap-schedule` flag.

## Benchmarking

For benchmarking performance without overlap scheduling (synchronous MLX evaluation), use `sglang.benchmark_one_batch` as it calls the synchronous prefill/decode methods directly without going through the scheduler and the overlap code path.

For benchmarking performance with overlap scheduling, use `sglang.benchmark_offline_throughput` as it uses the scheduler and the overlap code path.
