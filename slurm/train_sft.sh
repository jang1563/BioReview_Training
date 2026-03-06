#!/bin/bash
# ── BioReview SFT Training Job — Cornell Cayuga HPC ────────────────────────
#
# Cluster: Cayuga (Cornell CAC) — Slurm v25.05.0
# GPU nodes (scu-gpu partition):
#   g0001:       4× A100 80GB PCIe
#   g0002-g0003: 4× A40  48GB PCIe each
#
# VRAM estimate: Qwen2.5-7B + QLoRA 4-bit + seq=16384 + batch=1
#   Unsloth:         ~12-16GB → A40 (48GB) sufficient
#   PEFT+bitsandbytes: ~20-24GB → A40 (48GB) sufficient
#
# Usage:
#   sbatch slurm/train_sft.sh                     # full training (3 epochs)
#   sbatch slurm/train_sft.sh --export=MAX_STEPS=50  # quick test
#   sbatch slurm/train_sft.sh --export=GPU_TYPE=a100  # use A100
#
# Interactive debug (1h):
#   srun -p scu-gpu --gres=gpu:a40:1 --mem=48G --time=01:00:00 --pty bash
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=bioreview_sft
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/BioReview_Training/logs/train_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/BioReview_Training/logs/train_%j.err

# ── Configuration (overridable via sbatch --export) ────────────────────────
CONDA_ENV="${CONDA_ENV:-bioreview-sft}"
SCRATCH_DIR="/athena/masonlab/scratch/users/jak4013"
PROJECT_DIR="${SCRATCH_DIR}/BioReview_Training"
CONFIG="${CONFIG:-configs/qwen7b_qlora.yaml}"
MAX_STEPS="${MAX_STEPS:--1}"
DRY_RUN="${DRY_RUN:-false}"
GPU_TYPE="${GPU_TYPE:-a40}"

# HuggingFace cache → scratch (avoid filling home dir)
export HF_HOME="${SCRATCH_DIR}/cache/huggingface"
export TRANSFORMERS_CACHE="${SCRATCH_DIR}/cache/transformers"
export TORCH_HOME="${SCRATCH_DIR}/cache/torch"

echo "============================================================"
echo "BioReview SFT Training — Cayuga"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURMD_NODENAME}"
echo "GPU type:     ${GPU_TYPE}"
echo "Config:       ${CONFIG}"
echo "Max steps:    ${MAX_STEPS}"
echo "Dry run:      ${DRY_RUN}"
echo "Start time:   $(date)"
echo "============================================================"

# ── Environment setup ──────────────────────────────────────────────────────
source ~/.bashrc
conda activate "${CONDA_ENV}" \
    || { echo "ERROR: conda env '${CONDA_ENV}' not found. Run setup_cayuga.sh first."; exit 1; }

cd "${PROJECT_DIR}" \
    || { echo "ERROR: project dir not found: ${PROJECT_DIR}. Run sync_to_hpc.sh first."; exit 1; }

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" logs

# ── GPU verification ───────────────────────────────────────────────────────
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
echo ""
echo "Python: $(which python) — $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"

# Check Unsloth availability
python -c "
try:
    from unsloth import FastLanguageModel
    print('Backend: Unsloth (optimized)')
except ImportError:
    print('Backend: PEFT + bitsandbytes (fallback)')
"

# ── Verify data files ─────────────────────────────────────────────────────
echo ""
for f in data/sft_train.jsonl data/sft_val.jsonl "${CONFIG}"; do
    if [ -f "$f" ]; then
        echo "  OK: $f ($(wc -l < "$f") lines)"
    else
        echo "  MISSING: $f"
        echo "ERROR: Required file not found. Run sync_to_hpc.sh first."
        exit 1
    fi
done

# ── Build training command ─────────────────────────────────────────────────
CMD="python scripts/train_sft.py --config ${CONFIG}"

if [ "${MAX_STEPS}" != "-1" ]; then
    CMD="${CMD} --max-steps ${MAX_STEPS}"
fi

if [ "${DRY_RUN}" = "true" ]; then
    CMD="${CMD} --dry-run"
fi

echo ""
echo "Command: ${CMD}"
echo ""

# ── Run training ───────────────────────────────────────────────────────────
eval "${CMD}"
EXIT_CODE=$?

echo ""
echo "============================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Model saved to: ${PROJECT_DIR}/models/qwen7b_bioreview_v1/"
    # Show model size
    if [ -d "models/qwen7b_bioreview_v1" ]; then
        echo "Model size: $(du -sh models/qwen7b_bioreview_v1 | cut -f1)"
    fi
else
    echo "Training FAILED with exit code ${EXIT_CODE}"
    echo "Check: ${PROJECT_DIR}/logs/train_${SLURM_JOB_ID}.err"
fi
echo "End time: $(date)"
echo "============================================================"
exit ${EXIT_CODE}
