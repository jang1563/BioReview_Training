#!/bin/bash
# ── Shared environment setup for all BioReview SLURM scripts ─────────────
#
# Sources the conda environment from scratch storage to avoid NFS /home
# metadata slowness. Also sets CUDA_MODULE_LOADING=LAZY to speed up
# torch import (~73s → faster).
#
# Usage (from other SLURM scripts):
#   source slurm/env_setup.sh
#
# Override conda env location:
#   CONDA_ENV_PATH=/path/to/env source slurm/env_setup.sh
# ────────────────────────────────────────────────────────────────────────────

SCRATCH_DIR="${SCRATCH_DIR:-/path/to/scratch}"

# Conda env: prefer scratch copy (avoids NFS /home metadata latency)
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${SCRATCH_DIR}/conda_envs/bioreview-sft}"

if [ ! -d "${CONDA_ENV_PATH}/bin" ]; then
    echo "ERROR: Conda env not found at ${CONDA_ENV_PATH}/bin"
    echo "  Expected scratch copy at: ${SCRATCH_DIR}/conda_envs/bioreview-sft"
    echo "  Fallback: set CONDA_ENV_PATH to a valid conda env directory"
    exit 1
fi

# Activate by prepending to PATH (no need to source conda.sh)
export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV_PATH}"
export CONDA_DEFAULT_ENV="bioreview-sft"

# HuggingFace / torch caches → scratch
export HF_HOME="${SCRATCH_DIR}/huggingface"
export TORCH_HOME="${SCRATCH_DIR}/cache/torch"

# Speed up torch import: lazy-load CUDA modules (~73s → faster)
export CUDA_MODULE_LOADING=LAZY

# Offline mode: don't try to download models during jobs
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Environment: ${CONDA_ENV_PATH}"
echo "Python:      $(which python) — $(python --version 2>&1)"
