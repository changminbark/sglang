# Apple Silicon with Metal

This document describes how run SGLang on Apple Silicon using [Metal](https://developer.apple.com/metal/). If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

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

## Features

Overlap scheduling using `mlx.async_eval()` is enabled by default. To disable this, use the `--disable-overlap` flag.

## Benchmarking

For benchmarking performance without overlap scheduling (synchronous MLX evaluation), use `sglang.benchmark_one_batch` as it calls the synchronous prefill/decode methods directly without going through the scheduler and the overlap code path.

For benchmarking performance with overlap scheduling, use `sglang.benchmark_offline_throughput` as it uses the scheduler and the overlap code path.
