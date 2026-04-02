#!/bin/bash
# ============================================================
# BioReview SFT: One-Time Environment Setup for Cayuga HPC
# ============================================================
#
# Run this ONCE on the login node (NOT as a SLURM job):
#   bash slurm/setup_cayuga.sh
#
# Prerequisites:
#   - Miniconda installed and activatable via `conda`
#   - Cayuga VPN connected
#
# After setup, submit training:
#   /opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch slurm/train_sft.sh
# ============================================================

set -euo pipefail

CONDA_ENV="bioreview-sft"
PYTHON_VERSION="3.11"
USER_NAME="${USER:-$(id -un)}"
DEFAULT_SCRATCH_DIR="/athena/masonlab/scratch/users/${USER_NAME}"
if [ ! -d "${DEFAULT_SCRATCH_DIR}" ] && [ -d "/athena/cayuga_0003/scratch/users/${USER_NAME}" ]; then
    DEFAULT_SCRATCH_DIR="/athena/cayuga_0003/scratch/users/${USER_NAME}"
fi
SCRATCH_DIR="${SCRATCH_DIR:-${DEFAULT_SCRATCH_DIR}}"
PROJECT_DIR="${SCRATCH_DIR}/BioReview_Training"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"

echo "============================================================"
echo "BioReview SFT Environment Setup — Cayuga"
echo "============================================================"
echo "conda_base=${CONDA_BASE:-'(from PATH)'}"

# ── Step 1: Create conda environment ─────────────────────────
echo ""
echo "[Step 1/10] Creating conda env: ${CONDA_ENV} (Python ${PYTHON_VERSION})..."
if conda env list | grep -q "${CONDA_ENV}"; then
    echo "  Environment already exists. Skipping creation."
    echo "  To recreate: conda env remove -n ${CONDA_ENV} && bash $0"
else
    conda create -n "${CONDA_ENV}" python="${PYTHON_VERSION}" -y
fi

# ── Step 2: Activate ─────────────────────────────────────────
echo ""
echo "[Step 2/10] Activating ${CONDA_ENV}..."
source ~/.bashrc
conda activate "${CONDA_ENV}"

# ── Step 3: Set cache paths (must match train_sft.sh / run_inference.sh) ──
echo ""
echo "[Step 3/10] Configuring cache directories..."
export HF_HOME="${SCRATCH_DIR}/huggingface"
export TORCH_HOME="${SCRATCH_DIR}/cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"
echo "  HF_HOME:    ${HF_HOME}"
echo "  TORCH_HOME: ${TORCH_HOME}"

# ── Step 4: Install PyTorch with CUDA 12.1 ───────────────────
echo ""
echo "[Step 4/10] Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── Step 5: Install training dependencies ─────────────────────
echo ""
echo "[Step 5/10] Installing training dependencies..."
pip install \
    "transformers>=4.51.0" \
    "datasets>=2.20.0" \
    "trl>=0.12.0,<0.16.0" \
    "peft>=0.13.0" \
    "accelerate>=0.34.0" \
    "bitsandbytes>=0.43.0" \
    "pyyaml>=6.0" \
    "sentencepiece>=0.2.0" \
    "protobuf>=4.25.0" \
    "tiktoken>=0.7.0"

# ── Step 6: Install evaluation dependencies ───────────────────
echo ""
echo "[Step 6/10] Installing evaluation dependencies..."
pip install \
    "sentence-transformers>=2.2.0" \
    "scipy>=1.10.0" \
    "pydantic>=2.0.0" \
    "openai>=1.0.0" \
    "anthropic>=0.34.0" \
    "google-generativeai>=0.8.0"

# ── Step 7: Install Unsloth (recommended for 2-5x faster training) ──
echo ""
echo "[Step 7/10] Installing Unsloth (recommended for 2-5x faster training)..."
if pip install unsloth 2>&1; then
    echo "  pip install completed. Verifying import..."
    if python -c "from unsloth import FastLanguageModel; print('  Unsloth import OK')" 2>&1; then
        echo "  Training will use optimized backend (~12-16GB VRAM)."
    else
        echo "  WARNING: pip install succeeded but import failed."
        echo "  Training will fall back to PEFT+bitsandbytes until this is fixed."
    fi
else
    echo "  Unsloth install failed. Training will use PEFT+bitsandbytes fallback."
    echo "  Note: Fallback uses ~20-24GB VRAM and is 2-5x slower."
    echo "  This is fine for A40 (48GB) but be aware of performance difference."
fi

# ── Step 8: Pre-cache all comparison models ───────────────────
echo ""
echo "[Step 8/10] Pre-caching models for multi-model comparison..."
echo "  This may take 30-60 minutes on first download."
echo "  Files are cached to: ${HF_HOME}"
python -c "
from huggingface_hub import snapshot_download
models = [
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen3-8B',
    'Qwen/Qwen3.5-9B',
    'Qwen/Qwen2.5-14B-Instruct',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
]
for model_id in models:
    print(f'  Downloading {model_id}...')
    try:
        path = snapshot_download(model_id)
        print(f'    Cached at: {path}')
    except Exception as e:
        print(f'    WARNING: Failed ({e}). Will download during training.')
" || echo "  WARNING: Model pre-download had errors."

# ── Step 9: Create project directories ────────────────────────
echo ""
echo "[Step 9/10] Setting up project directories..."
mkdir -p "${PROJECT_DIR}"/{logs,models,data,results/sft_eval}

# ── Step 10: Verify installation ──────────────────────────────
echo ""
echo "[Step 10/10] Verifying installation..."
python -c "import torch; print(f'  PyTorch:        {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  CUDA version:   {torch.version.cuda}')" 2>/dev/null || true
python -c "import transformers; print(f'  Transformers:   {transformers.__version__}')"
python -c "import peft; print(f'  PEFT:           {peft.__version__}')"
python -c "import trl; print(f'  TRL:            {trl.__version__}')"
python -c "import bitsandbytes; print(f'  bitsandbytes:   {bitsandbytes.__version__}')"
python -c "
try:
    from unsloth import FastLanguageModel
    print('  Unsloth:        OK')
except ImportError:
    print('  Unsloth:        NOT INSTALLED (will use fallback)')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "Project dir: ${PROJECT_DIR}"
echo "Conda env:   ${CONDA_ENV}"
echo ""
echo "Next steps:"
echo "  1. Sync data:  bash slurm/sync_to_hpc.sh  (from local machine)"
echo "  2. Train:       /opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch slurm/train_sft.sh"
echo "  3. Evaluate:    /opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch slurm/run_inference.sh"
echo "============================================================"
