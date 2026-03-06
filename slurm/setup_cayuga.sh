#!/bin/bash
# ============================================================
# BioReview SFT: One-Time Environment Setup for Cayuga HPC
# ============================================================
#
# Run this ONCE on the login node (NOT as a SLURM job):
#   bash slurm/setup_cayuga.sh
#
# Prerequisites:
#   - Miniconda installed at /athena/masonlab/scratch/users/jak4013/miniconda3
#   - Cayuga VPN connected
#
# After setup, submit training:
#   sbatch slurm/train_sft.sh
# ============================================================

set -euo pipefail

CONDA_ENV="bioreview-sft"
PYTHON_VERSION="3.11"
SCRATCH_DIR="/athena/masonlab/scratch/users/jak4013"
PROJECT_DIR="${SCRATCH_DIR}/BioReview_Training"

echo "============================================================"
echo "BioReview SFT Environment Setup — Cayuga"
echo "============================================================"

# ── Step 1: Create conda environment ─────────────────────────
echo ""
echo "[Step 1] Creating conda env: ${CONDA_ENV} (Python ${PYTHON_VERSION})..."
if conda env list | grep -q "${CONDA_ENV}"; then
    echo "  Environment already exists. Skipping creation."
    echo "  To recreate: conda env remove -n ${CONDA_ENV} && bash $0"
else
    conda create -n "${CONDA_ENV}" python="${PYTHON_VERSION}" -y
fi

# ── Step 2: Activate ─────────────────────────────────────────
echo ""
echo "[Step 2] Activating ${CONDA_ENV}..."
source ~/.bashrc
conda activate "${CONDA_ENV}"

# ── Step 3: Install PyTorch with CUDA 12.1 ───────────────────
echo ""
echo "[Step 3] Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── Step 4: Install training dependencies ─────────────────────
echo ""
echo "[Step 4] Installing training dependencies..."
pip install \
    "transformers>=4.45.0" \
    "datasets>=2.20.0" \
    "trl>=0.12.0,<0.16.0" \
    "peft>=0.13.0" \
    "accelerate>=0.34.0" \
    "bitsandbytes>=0.43.0" \
    "pyyaml>=6.0" \
    "sentencepiece>=0.2.0" \
    "protobuf>=4.25.0" \
    "tiktoken>=0.7.0"

# ── Step 5: Install evaluation dependencies ───────────────────
echo ""
echo "[Step 5] Installing evaluation dependencies..."
pip install \
    "sentence-transformers>=2.2.0" \
    "scipy>=1.10.0"

# ── Step 6: Install Unsloth (recommended) ─────────────────────
echo ""
echo "[Step 6] Installing Unsloth (optional, recommended)..."
pip install unsloth \
    || echo "  WARNING: Unsloth install failed. Will use PEFT+bitsandbytes fallback."

# ── Step 7: Create project directories ────────────────────────
echo ""
echo "[Step 7] Setting up project directories..."
mkdir -p "${PROJECT_DIR}"/{logs,models,data,results/sft_eval,cache}

# ── Step 8: Verify installation ───────────────────────────────
echo ""
echo "[Step 8] Verifying installation..."
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
echo "  1. Sync data:  bash slurm/sync_to_hpc.sh"
echo "  2. Train:       sbatch slurm/train_sft.sh"
echo "  3. Evaluate:    sbatch slurm/run_inference.sh"
echo "============================================================"
