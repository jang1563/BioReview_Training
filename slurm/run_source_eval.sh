#!/bin/bash
# ── BioReview Source-Sliced Evaluation — Cornell Cayuga HPC ────────────────
#
# Runs by-source evaluation for an existing inference JSONL using the current
# benchmark split and the local SPECTER2 model on Cayuga.

#SBATCH --job-name=bioreview_src
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=/athena/masonlab/scratch/users/jak4013/BioReview_Training/logs/source_eval_%j.log
#SBATCH --error=/athena/masonlab/scratch/users/jak4013/BioReview_Training/logs/source_eval_%j.err

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-bioreview-sft}"
SCRATCH_DIR="/athena/masonlab/scratch/users/jak4013"
PROJECT_DIR="${SCRATCH_DIR}/BioReview_Training"
BENCH_DIR="${SCRATCH_DIR}/peer-review-benchmark"
MODEL_DIR="${MODEL_DIR:-models/qwen3_8b_all_nonfig_v1}"
SPLIT="${SPLIT:-val}"
TAG="${TAG:-}"
TOOL_OUTPUT="${TOOL_OUTPUT:-}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-}"

CONDA_BASE="${CONDA_BASE:-/home/fs01/jak4013/miniconda3/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh" \
    || { echo "ERROR: conda not found at ${CONDA_BASE}"; exit 1; }
conda activate "${CONDA_ENV}" \
    || { echo "ERROR: conda env '${CONDA_ENV}' not found."; exit 1; }

cd "${PROJECT_DIR}" \
    || { echo "ERROR: project dir not found: ${PROJECT_DIR}"; exit 1; }

mkdir -p logs results/source_eval

MODEL_BASENAME="$(basename "${MODEL_DIR}")"
if [ -z "${TOOL_OUTPUT}" ]; then
    TAG_SUFFIX=""
    if [ -n "${TAG}" ]; then
        TAG_SUFFIX="_${TAG}"
    fi
    TOOL_OUTPUT="results/sft_eval/${MODEL_BASENAME}_${SPLIT}${TAG_SUFFIX}.jsonl"
fi

if [ -z "${OUTPUT_PREFIX}" ]; then
    TOOL_STEM="$(basename "${TOOL_OUTPUT}" .jsonl)"
    OUTPUT_PREFIX="results/source_eval/${TOOL_STEM}_by_source"
fi

SPLITS_DIR="${BENCH_DIR}/data/splits/v3"
EMBED_MODEL="${PROJECT_DIR}/models/specter2_base"

echo "============================================================"
echo "BioReview Source-Sliced Evaluation — Cayuga"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Tool output:   ${TOOL_OUTPUT}"
echo "Split:         ${SPLIT}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Embed model:   ${EMBED_MODEL}"
echo "Start time:    $(date)"
echo "============================================================"

if [ ! -f "${TOOL_OUTPUT}" ]; then
    echo "ERROR: tool output not found: ${TOOL_OUTPUT}"
    exit 1
fi

if [ ! -f "${SPLITS_DIR}/${SPLIT}.jsonl" ]; then
    echo "ERROR: split file not found: ${SPLITS_DIR}/${SPLIT}.jsonl"
    exit 1
fi

if [ ! -d "${EMBED_MODEL}" ]; then
    echo "ERROR: local SPECTER2 model not found: ${EMBED_MODEL}"
    exit 1
fi

python scripts/evaluate_by_source.py \
    "${TOOL_OUTPUT}" \
    --splits-dir "${SPLITS_DIR}" \
    --split "${SPLIT}" \
    --embed-model "${EMBED_MODEL}" \
    --output-prefix "${OUTPUT_PREFIX}"

echo ""
echo "Output files:"
ls -lh "${OUTPUT_PREFIX}".json "${OUTPUT_PREFIX}".md
echo "End time: $(date)"
