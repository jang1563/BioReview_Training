#!/bin/bash
# ── BioReview SFT Sweep Array Job — Cornell Cayuga HPC ─────────────────────
#
# Runs multiple training + inference experiments from a sweep manifest.
# Each SLURM array task: (1) trains model, (2) runs inference, (3) logs result.
#
# Prerequisites:
#   python scripts/sweep_manager.py generate-configs --sweep configs/sweep/stage1_14b.yaml
#   → generates configs/sweep/stage1_14b/sweep_manifest.csv
#
# Usage:
#   IMPORTANT: Always specify --array matching the number of experiments in the manifest.
#   Run `python scripts/sweep_manager.py generate-configs --sweep ...` first — it prints
#   the recommended --array spec.
#
#   # stage1_14b (4 experiments, A100):
#   sbatch --array=1-4%2 --gres=gpu:a100:1 --mem=80G \
#       --export=ALL,MANIFEST=configs/sweep/stage1_14b/sweep_manifest.csv \
#       slurm/sweep_array.sh
#
#   # stage1_9b (2 experiments, A40):
#   sbatch --array=1-2%2 --gres=gpu:a40:1 --mem=48G \
#       --export=ALL,MANIFEST=configs/sweep/stage1_9b/sweep_manifest.csv \
#       slurm/sweep_array.sh
#
#   # Single experiment (no concurrency):
#   sbatch --array=1-1%1 --export=ALL,MANIFEST=...,TAG=baseline slurm/sweep_array.sh
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=bioreview_sweep
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/sweep_%A_%a.log
#SBATCH --error=logs/sweep_%A_%a.err

# IMPORTANT: Override --array via sbatch CLI to match manifest size.
# This default (1-4%2) is for stage1_14b only. Use --array=1-N%2 for N experiments.
#SBATCH --array=1-4%2

# ── Configuration ────────────────────────────────────────────────────────────
USER_NAME="${USER:-$(id -un)}"
DEFAULT_SCRATCH_DIR="/athena/masonlab/scratch/users/${USER_NAME}"
if [ ! -d "${DEFAULT_SCRATCH_DIR}" ] && [ -d "/athena/cayuga_0003/scratch/users/${USER_NAME}" ]; then
    DEFAULT_SCRATCH_DIR="/athena/cayuga_0003/scratch/users/${USER_NAME}"
fi
SCRATCH_DIR="${SCRATCH_DIR:-${DEFAULT_SCRATCH_DIR}}"
PROJECT_DIR="${SCRATCH_DIR}/BioReview_Training"
MANIFEST="${MANIFEST:-configs/sweep/stage1_14b/sweep_manifest.csv}"
SPLIT="${SPLIT:-val}"
EVALUATE="${EVALUATE:-true}"
SWEEP_CSV="${SWEEP_CSV:-results/sweep_results.csv}"

echo "============================================================"
echo "BioReview SFT Sweep — Task ${SLURM_ARRAY_TASK_ID}"
echo "============================================================"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID:      ${SLURM_ARRAY_TASK_ID}"
echo "Node:         ${SLURMD_NODENAME}"
echo "Manifest:     ${MANIFEST}"
echo "Start time:   $(date)"
echo "============================================================"

# ── Environment setup ────────────────────────────────────────────────────────
cd "${PROJECT_DIR}" \
    || { echo "ERROR: project dir not found"; exit 1; }
source slurm/env_setup.sh

mkdir -p logs results/sft_eval

# ── GPU info ──────────────────────────────────────────────────────────────────
echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ── Read manifest ─────────────────────────────────────────────────────────────
if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Manifest not found: ${MANIFEST}"
    exit 1
fi

# Warn if task ID exceeds manifest size (common mistake: --array not overridden)
MANIFEST_SIZE=$(awk -F',' 'NR>1 && $1~/^[0-9]+$/ {count++} END {print count+0}' "${MANIFEST}")
if [ "${SLURM_ARRAY_TASK_ID}" -gt "${MANIFEST_SIZE}" ]; then
    echo "ERROR: Task ID ${SLURM_ARRAY_TASK_ID} > manifest size ${MANIFEST_SIZE}."
    echo "  Did you forget --array=1-${MANIFEST_SIZE}%2 when submitting?"
    echo "  Re-submit with: sbatch --array=1-${MANIFEST_SIZE}%2 --export=ALL,MANIFEST=${MANIFEST} slurm/sweep_array.sh"
    exit 1
fi

# Read the row matching SLURM_ARRAY_TASK_ID (index column)
TASK_ID="${SLURM_ARRAY_TASK_ID}"
ROW=$(awk -F',' -v id="${TASK_ID}" 'NR>1 && $1==id {print; exit}' "${MANIFEST}")

if [ -z "${ROW}" ]; then
    echo "ERROR: No manifest row for task index ${TASK_ID}"
    echo "Manifest contents:"
    cat "${MANIFEST}"
    exit 1
fi

TAG=$(echo "${ROW}" | cut -d',' -f2)
CONFIG_PATH=$(echo "${ROW}" | cut -d',' -f3)
OUTPUT_DIR=$(echo "${ROW}" | cut -d',' -f5)

echo "Tag:          ${TAG}"
echo "Config:       ${CONFIG_PATH}"
echo "Output dir:   ${OUTPUT_DIR}"

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config not found: ${CONFIG_PATH}"
    exit 1
fi

# ── Phase 1: Training ────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Phase 1: Training (tag=${TAG})"
echo "============================================================"

python scripts/train_sft.py \
    --config "${CONFIG_PATH}" \
    --no-eval

TRAIN_EXIT=$?
if [ ${TRAIN_EXIT} -ne 0 ]; then
    echo "TRAINING FAILED (exit=${TRAIN_EXIT})"
    exit ${TRAIN_EXIT}
fi

echo "Training complete for ${TAG}"

# ── Phase 2: Inference ───────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Phase 2: Inference (tag=${TAG})"
echo "============================================================"

BENCH_DIR="${SCRATCH_DIR}/peer-review-benchmark"
SPLITS_DIR="${BENCH_DIR}/data/splits/v3"

INF_CMD="python scripts/run_sft_inference.py"
INF_CMD="${INF_CMD} --model-dir ${OUTPUT_DIR}"
INF_CMD="${INF_CMD} --split ${SPLIT}"
INF_CMD="${INF_CMD} --splits-dir ${SPLITS_DIR}"
INF_CMD="${INF_CMD} --tag ${TAG}"

if [ "${EVALUATE}" = "true" ]; then
    INF_CMD="${INF_CMD} --evaluate"
fi

echo "Command: ${INF_CMD}"
eval "${INF_CMD}"

INF_EXIT=$?
if [ ${INF_EXIT} -ne 0 ]; then
    echo "INFERENCE FAILED (exit=${INF_EXIT})"
    exit ${INF_EXIT}
fi

# ── Phase 3: Log result ──────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Phase 3: Logging result (tag=${TAG})"
echo "============================================================"

# Find the summary JSON (output by run_sft_inference.py)
MODEL_NAME=$(basename "${OUTPUT_DIR}")
SUMMARY_FILE="results/sft_eval/${MODEL_NAME}_${SPLIT}_${TAG}.summary.json"

if [ -f "${SUMMARY_FILE}" ]; then
    python scripts/sweep_manager.py log-result \
        --summary "${SUMMARY_FILE}" \
        --tag "${TAG}" \
        --csv "${SWEEP_CSV}"
else
    echo "WARNING: Summary not found at ${SUMMARY_FILE}"
    echo "Available summary files:"
    ls -la results/sft_eval/*.summary.json 2>/dev/null | tail -5
fi

echo ""
echo "============================================================"
echo "Sweep task ${TAG} complete!"
echo "End time: $(date)"
echo "============================================================"
