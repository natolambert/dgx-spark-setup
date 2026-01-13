#!/bin/bash
# Build vLLM from source for DGX Spark (CUDA 13, aarch64)
# This script times the full build process as a fallback when wheels aren't available

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${REPO_DIR}/vllm-build"
VENV_DIR="${BUILD_DIR}/venv"
VLLM_VERSION="${VLLM_VERSION:-v0.13.0}"

echo "============================================"
echo "vLLM Build from Source for DGX Spark"
echo "============================================"
echo "Build directory: ${BUILD_DIR}"
echo "vLLM version: ${VLLM_VERSION}"
echo "Started at: $(date)"
echo ""

# Track total time
TOTAL_START=$(date +%s)

# Clean up previous builds
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning previous build directory..."
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Step 1: Create virtual environment
echo ""
echo "[1/5] Creating virtual environment..."
STEP_START=$(date +%s)
~/.local/bin/uv venv "$VENV_DIR" --python 3.12
source "$VENV_DIR/bin/activate"
STEP_END=$(date +%s)
echo "  -> Done in $((STEP_END - STEP_START)) seconds"

# Step 2: Install PyTorch cu130
echo ""
echo "[2/5] Installing PyTorch cu130..."
STEP_START=$(date +%s)
~/.local/bin/uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
STEP_END=$(date +%s)
echo "  -> Done in $((STEP_END - STEP_START)) seconds"

# Step 3: Clone vLLM
echo ""
echo "[3/5] Cloning vLLM (${VLLM_VERSION})..."
STEP_START=$(date +%s)
git clone --depth 1 --branch "$VLLM_VERSION" https://github.com/vllm-project/vllm.git
cd vllm
STEP_END=$(date +%s)
echo "  -> Done in $((STEP_END - STEP_START)) seconds"

# Step 4: Set up CUDA 13 environment
echo ""
echo "[4/5] Setting up CUDA 13 build environment..."
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="12.1a"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export MAX_JOBS=4  # Limit parallel jobs to avoid OOM

echo "  CUDA_HOME=$CUDA_HOME"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  TRITON_PTXAS_PATH=$TRITON_PTXAS_PATH"
echo "  MAX_JOBS=$MAX_JOBS"

# Install build dependencies
~/.local/bin/uv pip install -r requirements/build.txt
python use_existing_torch.py

# Step 5: Build vLLM
echo ""
echo "[5/5] Building vLLM (this takes 20-30 minutes)..."
STEP_START=$(date +%s)
~/.local/bin/uv pip install --no-build-isolation -e .
STEP_END=$(date +%s)
BUILD_TIME=$((STEP_END - STEP_START))
echo "  -> Build completed in ${BUILD_TIME} seconds ($((BUILD_TIME / 60)) minutes)"

# Verify installation
echo ""
echo "============================================"
echo "Verifying installation..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo ""
echo "============================================"
echo "BUILD COMPLETE"
echo "============================================"
echo "Total time: ${TOTAL_TIME} seconds ($((TOTAL_TIME / 60)) minutes)"
echo "Build step: ${BUILD_TIME} seconds ($((BUILD_TIME / 60)) minutes)"
echo "Finished at: $(date)"
echo ""
echo "To use this environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To test vLLM:"
echo "  python -c \"from vllm import LLM; print('vLLM loaded successfully')\""
