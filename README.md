# DGX Spark ML Training Setup

## Why This Repo Exists

*As of 15 Jan. 2026*

The NVIDIA DGX Spark is a unique machine: it pairs a Blackwell GPU (GB10, sm_121) with an ARM CPU (aarch64), unified 128GB memory, and **requires CUDA 13.0**. This combination creates significant challenges for ML practitioners because the broader ecosystem hasn't caught up yet.

**The core problem is a CUDA version mismatch.** Blackwell (sm_121) is only supported in CUDA 13.0+, but nearly all pip-installable ML packages—vLLM, flash-attn, and many others—ship wheels compiled against CUDA 12.x. When you try to import them, you get:

```
ImportError: libcudart.so.12: cannot open shared object file
```

This happens because CUDA 12 libraries simply don't exist on DGX Spark.

**Additional challenges:**

1. **Unified memory behaves differently.** The 128GB is shared between CPU and GPU. Standard GPU OOM errors become system-wide memory exhaustion, which can freeze the entire machine rather than just killing your job.

2. **Flash Attention doesn't work** (and isn't needed). The upstream `flash-attn` package fails to load, but PyTorch's native SDPA with cuDNN 9.13 is actually faster on Blackwell anyway.

3. **The ecosystem is fragile.** There are no stable cu130 aarch64 vLLM releases on PyPI. We rely on nightly wheels hosted at `wheels.vllm.ai` that could change at any time.

4. **ARM + CUDA 13 is a rare combination.** Most CI/CD systems don't test this configuration, so you're often the first to encounter issues.

This guide documents working configurations, safe batch sizes, and fallback procedures developed while getting [open-instruct](https://github.com/allenai/open-instruct) running on DGX Spark. The goal is to save you the days of debugging we went through.

---

## Contents

| Section | Description |
|---------|-------------|
| [Hardware Overview](#hardware-overview) | GB10 specs and system info |
| [Quick Start](#quick-start) | Get running in 4 commands |
| [Package Compatibility](#package-compatibility-matrix) | What works, what doesn't |
| [vLLM Setup](#vllm-setup) | cu130 wheels and containers |
| [Flash Attention](#flash-attention-dont-install-it) | Why to skip it, use SDPA |
| [Environment Variables](#environment-variables) | Required exports |
| [Memory Management](#memory-management--oom-prevention) | OOM prevention, profiling results |
| [open-instruct Example](#example-open-instruct-on-dgx-spark) | Real-world training setup |
| [Troubleshooting](#troubleshooting) | Common errors and fixes |
| [Building from Source](#building-vllm-from-source-fallback) | Fallback if wheels break |

**Scripts:**
| File | Description |
|------|-------------|
| [`scripts/build_vllm_from_source.sh`](scripts/build_vllm_from_source.sh) | Build vLLM with CUDA 13 support |
| [`scripts/oom_protection.sh`](scripts/oom_protection.sh) | Memory checks and cleanup utilities |
| [`scripts/memory_profile_sft.sh`](scripts/memory_profile_sft.sh) | Profile memory for SFT training |
| [`scripts/memory_profile_dpo.sh`](scripts/memory_profile_dpo.sh) | Profile memory for DPO training |
| [`scripts/memory_profile_lora.sh`](scripts/memory_profile_lora.sh) | Profile memory for LoRA training |

**Docs:**
| File | Description |
|------|-------------|
| [`docs/memory_profiling_methodology.md`](docs/memory_profiling_methodology.md) | How to safely profile batch sizes |

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

## Memory Management & OOM Prevention

### Why DGX Spark OOMs Are Different

DGX Spark is **not a normal GPU workstation**. It's a Grace Blackwell system with **128GB of unified memory**—the GPU and CPU share the same DRAM pool (no dedicated VRAM).

**Two critical consequences:**

1. **"GPU OOM" = "System OOM"**: When training allocates "GPU memory," it's consuming system RAM that the OS, SSH, and your agents also need.

2. **Memory telemetry is weird**: `nvidia-smi` shows "Memory-Usage: N/A" on unified systems, even though memory is being used.

**The failure mode**: Instead of a clean `RuntimeError: CUDA out of memory`, the machine goes "zombie" (SSH hangs, job looks alive externally, hard reboot needed).

### Why It Bricks Instead of Failing Cleanly

From NVIDIA forum reports, the root cause is the **swap death spiral**:

1. Training exceeds available memory
2. System starts swapping aggressively
3. Swap thrashing makes the system unresponsive
4. SSH dies before any cleanup can happen
5. Hard reboot required

**The fix**: Disable swap. This converts "brick the box" into "job dies, OS lives."

### Defense-in-Depth: 5 Layers of Protection

#### Layer 1: Disable Swap (Most Important)

```bash
# Disable immediately (temporary)
sudo swapoff -a
swapon --show  # should print nothing

# Make permanent: edit /etc/fstab and comment out swap entries
sudoedit /etc/fstab
```

#### Layer 2: Run Experiments in Memory Jails (cgroups)

Put every training run in a memory-capped scope:

```bash
sudo systemd-run --scope \
  -p MemoryMax=100G \
  -p MemorySwapMax=0 \
  -p OOMScoreAdjust=500 \
  bash -lc 'cd ~/open-instruct && uv run python ...'
```

- **MemoryMax=100G**: Hard cap (leaves ~19GB for OS)
- **MemorySwapMax=0**: No swap for this job
- **OOMScoreAdjust=500**: Kernel prefers killing this over other services

#### Layer 3: Protect SSH and Agents

Create SSH override to make it unkillable:

```bash
sudo mkdir -p /etc/systemd/system/ssh.service.d
sudo tee /etc/systemd/system/ssh.service.d/oom.conf >/dev/null <<'EOF'
[Service]
OOMScoreAdjust=-1000
MemoryMin=512M
EOF

sudo systemctl daemon-reload
sudo systemctl restart ssh.service
```

#### Layer 4: Memory Watchdog

Kill jobs before they hit the cliff:

```bash
# watchdog.sh - run in separate terminal
WATCH_PID=$1
THRESHOLD_KB=$((16 * 1024 * 1024))  # 16GB

while true; do
  avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if [ "$avail_kb" -lt "$THRESHOLD_KB" ]; then
    echo "MemAvailable low (${avail_kb}kB). Killing $WATCH_PID"
    kill -TERM "$WATCH_PID" 2>/dev/null || true
    sleep 5
    kill -KILL "$WATCH_PID" 2>/dev/null || true
    break
  fi
  sleep 1
done
```

#### Layer 5: Drop Caches Between Runs

```bash
# Reset memory state between experiments
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

### Recommended Default Posture

For sweep-style work on DGX Spark:

1. **Swap off** (until NVIDIA says otherwise)
2. **Memory-capped scope** for every experiment (~100G max)
3. **Protect SSH** with OOMScoreAdjust=-1000
4. **Watchdog** to kill early
5. **Hard sequence length cap** in data preprocessing
6. **drop_caches** between runs

### Memory Profiling Results (2026-01-14)

#### Qwen3-0.6B SFT (seq_len=1024, gradient_checkpointing=true)

| batch | grad_accum | total_batch | peak_mem | headroom | status |
|-------|------------|-------------|----------|----------|--------|
| 2 | 1 | 2 | 21GB | 98GB | ✅ safe |
| 4 | 1 | 4 | 29GB | 90GB | ✅ safe |
| 8 | 1 | 8 | 47GB | 72GB | ✅ recommended |
| 16 | 1 | 16 | 81GB | 38GB | ⚠️ limit |

**Key insight**: Memory scales **super-linearly** with batch size. Doubling batch from 8→16 adds 34GB, not 24GB.

#### Qwen3-0.6B DPO (seq_len=1024, gradient_checkpointing=true)

| batch | grad_accum | total_batch | peak_mem | headroom | status |
|-------|------------|-------------|----------|----------|--------|
| 2 | 1 | 2 | 24GB | 95GB | ✅ safe |
| 4 | 1 | 4 | 27GB | 92GB | ✅ safe |
| 8 | 1 | 8 | 62GB | 57GB | ✅ recommended |

**DPO vs SFT**: DPO uses ~1.3x more memory at batch=8 (62GB vs 47GB).

**Safe defaults for Qwen3-0.6B SFT:**
```bash
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 4 \  # total_batch=32
--max_seq_length 1024 \
--gradient_checkpointing
```

**Safe defaults for Qwen3-0.6B DPO:**
```bash
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 8 \  # total_batch=32
--max_seq_length 1024 \
--gradient_checkpointing
```

### Memory Estimation

```
SFT:  ~6 * model_params_B + activation_memory
DPO:  ~1.3 * SFT (policy + frozen reference, less than expected!)
GRPO: SFT + vllm_gpu_memory_utilization * 119GB
```

### Pre-flight Checks

```bash
# Check memory
free -h

# Kill leftover processes
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "VLLM" 2>/dev/null || true
sleep 10

# Clear caches
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Verify >80GB free before starting
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

## Real-World Implementations

### open-instruct (AI2)

The [open-instruct](https://github.com/allenai/open-instruct) repo has DGX Spark support on the `dgx-spark-support` branch:

- SFT, DPO, and GRPO training scripts for single-GPU Blackwell
- Memory profiling scripts and documentation
- pyproject.toml modifications for aarch64 + CUDA 13

Key changes:
```toml
# vLLM: use cu130 nightly for aarch64
"vllm>=0.13.0; platform_machine == 'aarch64'"

# Exclude flash-attn on aarch64 (use SDPA instead)
"flash-attn>=2.8.3; platform_machine != 'aarch64'"
```

### Trained Models

Models trained on DGX Spark using this setup:

| Model | Base | Dataset | Training |
|-------|------|---------|----------|
| [natolambert/qwen3-dgx-spark-sft](https://huggingface.co/natolambert/qwen3-dgx-spark-sft) | Qwen3-0.6B | no_robots | SFT |

## Contributing

Got a framework working on DGX Spark? Open a PR to add it here!
