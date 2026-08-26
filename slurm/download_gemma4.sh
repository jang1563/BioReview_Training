#!/bin/bash
# ── Download Gemma 4 31B for BioReview training ──────────────────────────
# Run on Cayuga LOGIN NODE (not in sbatch — needs internet):
#   bash slurm/download_gemma4.sh
# ────────────────────────────────────────────────────────────────────────
set -euo pipefail

USER_NAME="${USER:-$(id -un)}"
SCRATCH_DIR="/athena/masonlab/scratch/users/${USER_NAME}"
CONDA_ENV_PATH="${SCRATCH_DIR}/conda_envs/bioreview-sft"

export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
export HF_HOME="${SCRATCH_DIR}/huggingface"

# MUST disable offline mode for download
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

MODEL_ID="unsloth/gemma-4-31B-it-unsloth-bnb-4bit"

echo "============================================================"
echo "Gemma 4 31B Download — Cayuga"
echo "============================================================"
echo "Model:     ${MODEL_ID}"
echo "HF cache:  ${HF_HOME}"
echo ""

# Step 1: Upgrade Unsloth + transformers for Gemma 4 support
echo "=== Step 1: Upgrading Unsloth + transformers ==="
pip install --upgrade unsloth transformers 2>&1 | tail -5
echo ""

# Step 2: Download model
echo "=== Step 2: Downloading ${MODEL_ID} ==="
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('${MODEL_ID}')
print()
print('=' * 60)
print('Download complete!')
print(f'Snapshot path: {path}')
print()
print('Update configs/gemma4_31b_all_nonfig.yaml model.name to:')
print(f'  {path}')
print('=' * 60)
"
