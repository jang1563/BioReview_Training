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
#   sbatch slurm/train_sft.sh                                     # full training (3 epochs)
#   sbatch slurm/train_sft.sh --export=MAX_STEPS=50               # quick test
#   sbatch slurm/train_sft.sh --export=RESUME=1                   # resume from latest checkpoint
#   sbatch slurm/train_sft.sh --export=RESUME=models/qwen7b_bioreview_v1/checkpoint-300  # specific checkpoint
#   sbatch --gres=gpu:a100:1 --mem=80G slurm/train_sft.sh         # use A100 instead
#
# Note: Config uses save_strategy="epoch" (3 epochs, ~2h each on A40).
# If training is slower than expected, consider:
#   - Increase wall time: sbatch --time=12:00:00 slurm/train_sft.sh
#   - Or change config save_strategy to "steps" with save_steps=100
#   - Resume interrupted training: sbatch --export=RESUME=1 slurm/train_sft.sh
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
#SBATCH --time=40:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

# ── Configuration (overridable via sbatch --export) ────────────────────────
USER_NAME="${USER:-$(id -un)}"
DEFAULT_SCRATCH_DIR="/athena/masonlab/scratch/users/${USER_NAME}"
if [ ! -d "${DEFAULT_SCRATCH_DIR}" ] && [ -d "/athena/cayuga_0003/scratch/users/${USER_NAME}" ]; then
    DEFAULT_SCRATCH_DIR="/athena/cayuga_0003/scratch/users/${USER_NAME}"
fi
SCRATCH_DIR="${SCRATCH_DIR:-${DEFAULT_SCRATCH_DIR}}"
PROJECT_DIR="${SCRATCH_DIR}/BioReview_Training"
CONFIG="${CONFIG:-configs/qwen3_8b_all_nonfig.yaml}"
MAX_STEPS="${MAX_STEPS:--1}"
DRY_RUN="${DRY_RUN:-false}"
NO_EVAL="${NO_EVAL:-true}"
RESUME="${RESUME:-}"

echo "============================================================"
echo "BioReview SFT Training — Cayuga"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURMD_NODENAME}"
echo "Config:       ${CONFIG}"
echo "Max steps:    ${MAX_STEPS}"
echo "Dry run:      ${DRY_RUN}"
echo "No eval:      ${NO_EVAL}"
echo "Resume:       ${RESUME:-no}"
echo "Start time:   $(date)"
echo "============================================================"

# ── Environment setup ──────────────────────────────────────────────────────
cd "${PROJECT_DIR}" \
    || { echo "ERROR: project dir not found: ${PROJECT_DIR}. Run sync_to_hpc.sh first."; exit 1; }
source slurm/env_setup.sh

mkdir -p "${HF_HOME}" "${TORCH_HOME}" logs

# ── GPU verification ───────────────────────────────────────────────────────
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
echo ""
echo "Python: $(which python) — $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"

# Check Unsloth availability without importing it in the shell preflight.
# A full import can trigger heavyweight initialization before training starts.
python -c "
import importlib.util
if importlib.util.find_spec('unsloth') is not None:
    print('Backend: Unsloth (optimized)')
else:
    print('Backend: PEFT + bitsandbytes (fallback)')
"

# ── Verify config + data files ────────────────────────────────────────────
echo ""
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG}"
    echo "  Run sync_to_hpc.sh first."
    exit 1
fi

TRAIN_PATH=$(python -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg['data']['train_path'])")
VAL_PATH=$(python -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg['data']['val_path'])")

for f in "${TRAIN_PATH}" "${VAL_PATH}" "${CONFIG}"; do
    if [ -f "$f" ]; then
        echo "  OK: $f ($(wc -l < "$f") lines)"
    else
        echo "  MISSING: $f"
        echo "ERROR: Required file not found. Run sync_to_hpc.sh first."
        exit 1
    fi
done

# Quick JSONL integrity check (parse first + last line)
for f in "${TRAIN_PATH}" "${VAL_PATH}"; do
    python -c "
import json, sys
with open('$f') as fh:
    first = fh.readline()
    json.loads(first)
    last = first
    for last in fh: pass
    json.loads(last)
print(f'  JSONL OK: $f')
" || { echo "ERROR: Malformed JSONL: $f"; exit 1; }
done

# ── Disk space check ────────────────────────────────────────────────────
avail_gb=$(df -BG "${PROJECT_DIR}" 2>/dev/null | tail -1 | awk '{gsub(/G/,""); print $4}')
if [ -n "${avail_gb}" ] && [ "${avail_gb}" -lt 30 ]; then
    echo ""
    echo "WARNING: Only ${avail_gb}GB available. Training may need ~20GB for model + checkpoints."
fi

# ── Build training command ─────────────────────────────────────────────────
CMD="python scripts/train_sft.py --config ${CONFIG}"

if [ "${MAX_STEPS}" != "-1" ]; then
    CMD="${CMD} --max-steps ${MAX_STEPS}"
fi

if [ "${DRY_RUN}" = "true" ]; then
    CMD="${CMD} --dry-run"
fi

if [ "${NO_EVAL}" = "true" ]; then
    CMD="${CMD} --no-eval"
fi

if [ -n "${RESUME}" ]; then
    if [ -d "${RESUME}" ]; then
        # User provided explicit checkpoint path
        echo "Resuming from: ${RESUME}"
        CMD="${CMD} --resume ${RESUME}"
    else
        # Auto-detect latest checkpoint from config output_dir
        OUTPUT_DIR=$(python -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('output',{}).get('dir','models/output'))")
        LATEST_CKPT=$(ls -td "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | head -1)
        if [ -n "${LATEST_CKPT}" ]; then
            echo "Resuming from latest checkpoint: ${LATEST_CKPT}"
            CMD="${CMD} --resume ${LATEST_CKPT}"
        else
            echo "WARNING: No checkpoints found in ${OUTPUT_DIR}/, starting fresh."
        fi
    fi
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
    # Extract output dir from config
    MODEL_OUTPUT_DIR=$(python -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('output',{}).get('dir','models/output'))")
    echo "Model saved to: ${PROJECT_DIR}/${MODEL_OUTPUT_DIR}/"
    if [ -d "${MODEL_OUTPUT_DIR}" ]; then
        echo "Model size: $(du -sh ${MODEL_OUTPUT_DIR} | cut -f1)"
    fi
else
    echo "Training FAILED with exit code ${EXIT_CODE}"
    echo "Check: ${PROJECT_DIR}/logs/train_${SLURM_JOB_ID}.err"
fi
echo "End time: $(date)"
echo "============================================================"
exit ${EXIT_CODE}
