# DGX Spark ML Training Setup

A comprehensive guide for setting up ML training on NVIDIA DGX Spark with GB10 (Blackwell, sm_121, CUDA 13.0, aarch64).

This guide was created while getting [open-instruct](https://github.com/allenai/open-instruct) running on DGX Spark for SFT and GRPO/RL training.

## Hardware Overview

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GB10 (Blackwell architecture) |
| Compute Capability | sm_121 (12.1) |
| GPU Memory | 119.68 GB (unified with CPU) |
| CPU | 20 ARM cores (10x Cortex-X925 + 10x Cortex-A725) |
| Architecture | aarch64 / ARM64 / SBSA |
| CUDA Version | 13.0 |
| Driver | 580.95.05 |
| OS | Ubuntu 24.04 LTS |

## The Core Problem

**CUDA ABI mismatch**: Packages built against CUDA 12.x are linked to `libcudart.so.12`, but DGX Spark only has `libcudart.so.13`. Result: imports fail before you even get to "does this kernel support sm_121?".

```bash
# This is the error you'll see constantly:
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory
```

## Important: Fragile Ecosystem

**The DGX Spark ML ecosystem is fragile.** There are no stable cu130 aarch64 vLLM releases yet. We pin to a specific nightly wheel that may stop being hosted at any time.

**Fallback**: If the pinned wheel stops working, see [Building vLLM from Source](#building-vllm-from-source-fallback) (~25-35 min build time).

## Quick Start

```bash
# 1. Create virtualenv with uv
uv venv .venv --python 3.12
source .venv/bin/activate

# 2. Install PyTorch cu130 (aarch64 wheels exist!)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 3. Install vLLM cu130 (pinned to v0.13.0 stable)
uv pip install "vllm @ https://wheels.vllm.ai/72506c98349d6bcd32b4e33eec7b5513453c1502/vllm-0.13.0%2Bcu130-cp38-abi3-manylinux_2_35_aarch64.whl"

# 4. DON'T install flash-attn (vLLM bundles FlashInfer, SDPA is faster on Blackwell anyway)
```

## Package Compatibility Matrix

| Package | Status | Notes |
|---------|--------|-------|
| **PyTorch 2.9.0+cu130** | Working | Warns about sm_121 but safe to ignore |
| **DeepSpeed** | Working | Builds successfully via uv sync |
| **Accelerate** | Working | Use `uv run python -m accelerate.commands.launch` |
| **Transformers** | Working | Standard HF transformers work fine |
| **vLLM 0.13.0+cu130** | Working | Use nightly cu130 wheels from wheels.vllm.ai |
| **flash-attn** | Skip It | SDPA is faster on Blackwell, flash-attn causes libcudart.so.12 errors |
| **Triton** | Working | Needs TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas |

## PyTorch sm_121 Warning

You'll see this warning every time:
```
Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

**This is safe to ignore.** sm_120 and sm_121 are binary compatible. A build that contains sm_120 kernels runs fine on sm_121. This is confirmed by PyTorch maintainers.

## vLLM Setup

### Why pip install vllm Fails

The default PyPI wheel is compiled against CUDA 12.x:
```bash
# DON'T do this - will fail with libcudart.so.12 error
pip install vllm==0.12.0
```

### The Fix: Use cu130 Wheels

vLLM provides CUDA 13.0 wheels at their nightly index:

```bash
# Use uv (pip's --extra-index-url behavior is unreliable)
uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly/cu130
```

Or install a specific wheel directly:
```bash
# Check https://wheels.vllm.ai/nightly/cu130/ for available versions
pip install https://wheels.vllm.ai/nightly/cu130/vllm-0.14.0rc1.dev530%2Bcu130-cp312-cp312-manylinux_2_35_aarch64.whl
```

### Alternative: NVIDIA Container

For inference-only workloads, NVIDIA's container is known-good:

```bash
docker pull nvcr.io/nvidia/vllm:25.11-py3
docker run -it --gpus all -p 8000:8000 \
    nvcr.io/nvidia/vllm:25.11-py3 \
    vllm serve "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --gpu-memory-utilization 0.7
```

Note: Use `--gpu-memory-utilization 0.7` or lower due to unified memory.

## Flash Attention: Don't Install It

**Key insight**: vLLM does NOT require the upstream `flash-attn` pip package. It bundles its own FlashInfer kernels.

If you install `flash-attn`, it will:
1. Try to load CUDA 12 libraries (libcudart.so.12 error)
2. Even if you build from source, it's **slower than SDPA on Blackwell**

### Use SDPA Instead

PyTorch's native Scaled Dot Product Attention with cuDNN 9.13 is currently faster on GB10:

```python
# In your model loading code
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="sdpa",  # NOT "flash_attention_2"
    ...
)
```

For training scripts that default to flash_attention_2, use:
```bash
--attn_implementation sdpa
```

## Environment Variables

```bash
# For building CUDA code targeting Blackwell
export TORCH_CUDA_ARCH_LIST="12.1a"

# CUDA paths
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Critical for Triton kernels
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# vLLM specific (if building from source)
export VLLM_USE_FLASHINFER_MXFP4_MOE=1
```

## Example: open-instruct on DGX Spark

We successfully ran SFT and GRPO training on DGX Spark with these modifications:

### pyproject.toml Changes

```toml
[project]
dependencies = [
    # vLLM: different versions for x86_64 vs aarch64
    "vllm>=0.12.0; platform_system != 'Darwin' and platform_machine != 'aarch64'",
    "vllm>=0.13.0; platform_system != 'Darwin' and platform_machine == 'aarch64'",

    # flash-attn: exclude aarch64 (vLLM uses FlashInfer, SDPA is faster)
    "flash-attn>=2.8.3; platform_system != 'Darwin' and platform_machine != 'aarch64'",
]

[tool.uv.sources]
vllm = [
  { index = "vllm-cu130", marker = "platform_system == 'Linux' and platform_machine == 'aarch64'"},
]

[[tool.uv.index]]
name = "vllm-cu130"
url = "https://wheels.vllm.ai/nightly/cu130"
explicit = true
```

### Code Changes

1. **Add GB10 to GPU specs** (for memory estimation):
```python
GPU_SPECS = {
    # ... existing entries ...
    "gb10": {"flops": 104e12, "memory_size": 128e9, "memory_bandwidth": 273e9},
}
```

2. **Make attention implementation configurable** (default to sdpa):
```python
attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] | None = None
```

3. **Change default from flash_attention_2 to sdpa**

### Running Training

```bash
# SFT
uv run python -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/finetune.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --use_flash_attn false \
    --with_tracking \
    ...

# GRPO (single GPU)
uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --attn_implementation sdpa \
    --vllm_enforce_eager \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --single_gpu_mode \
    --with_tracking \
    ...
```

## Troubleshooting

### "libcudart.so.12 not found"
You installed a CUDA 12 wheel. Find and uninstall it:
```bash
pip uninstall flash-attn vllm-flash-attn vllm
# Then reinstall vLLM from cu130 index
```

### "sm_121a is not defined for option 'gpu-name'"
Triton is using the wrong ptxas. Set:
```bash
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

### "FlashAttention2 has been toggled on, but flash_attn is not installed"
Your code is trying to use flash_attention_2. Change to sdpa:
```python
attn_implementation="sdpa"
```

### vLLM model loading hangs
Try `--vllm_enforce_eager` to disable CUDA graphs.

## Timeline Expectations

| Component | Expected Support | Source |
|-----------|------------------|--------|
| PyTorch official sm_121 | Q1 2026 | PyTorch forums |
| Flash-Attention 4 | Unknown | Maintainers transitioning to CuTe DSL |
| vLLM native pip | Available now | cu130 nightly wheels |
| NVIDIA containers | Available now | NGC catalog |

## Building vLLM from Source (Fallback)

If the pinned wheel stops working, you can build vLLM from source. **Estimated time: 25-35 minutes.**

### Automated Script

```bash
cd ~/dev/dgx-spark-setup
./scripts/build_vllm_from_source.sh
```

This script:
1. Creates a fresh virtualenv
2. Installs PyTorch cu130
3. Clones vLLM v0.13.0
4. Builds with CUDA 13 / sm_121a support
5. Reports timing for each step

### Manual Build

```bash
# 1. Create environment
uv venv vllm-build --python 3.12
source vllm-build/bin/activate

# 2. Install PyTorch cu130
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 3. Clone vLLM
git clone --depth 1 --branch v0.13.0 https://github.com/vllm-project/vllm.git
cd vllm

# 4. Set CUDA 13 environment
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="12.1a"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export MAX_JOBS=4  # Prevent OOM during compilation

# 5. Build
python use_existing_torch.py
uv pip install -r requirements/build.txt
uv pip install --no-build-isolation -e .
```

## References

- [PyTorch Forums: DGX Spark GB10 CUDA 13.0](https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744)
- [vLLM Issue #31128: Blackwell SM121 Support](https://github.com/vllm-project/vllm/issues/31128)
- [Flash-Attention Issue #1969: SM_121 Support](https://github.com/Dao-AILab/flash-attention/issues/1969)
- [NVIDIA DGX Spark vLLM Guide](https://build.nvidia.com/spark/vllm)
- [vLLM Installation Docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
- [vLLM Wheel Index](https://wheels.vllm.ai/nightly/cu130/)

## Contributing

Found something that works (or doesn't)? Open an issue or PR!
