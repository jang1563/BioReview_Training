#!/bin/bash
# ── BioReview SFT Inference + Evaluation — Cornell Cayuga HPC ──────────────
#
# Runs inference on the fine-tuned model, then evaluates against ground truth.
# Requires: trained model in the configured output directory
#
# Usage:
#   sbatch slurm/run_inference.sh                                  # full val set
#   sbatch --export=MODEL_DIR=models/qwen3.5_9b_all_nonfig_v1 slurm/run_inference.sh
#   sbatch slurm/run_inference.sh --export=MAX_ARTICLES=5          # quick test
#   sbatch slurm/run_inference.sh --export=SPLIT=test              # test set
#   sbatch --gres=gpu:a100:1 --mem=80G slurm/run_inference.sh     # use A100 instead
#
# Interactive debug:
#   srun -p scu-gpu --gres=gpu:a40:1 --mem=48G --time=01:00:00 --pty bash
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=bioreview_inf
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/inference_%j.log
#SBATCH --error=logs/inference_%j.err

# ── Configuration (overridable via sbatch --export) ────────────────────────
USER_NAME="${USER:-$(id -un)}"
DEFAULT_SCRATCH_DIR="/athena/masonlab/scratch/users/${USER_NAME}"
if [ ! -d "${DEFAULT_SCRATCH_DIR}" ] && [ -d "/athena/cayuga_0003/scratch/users/${USER_NAME}" ]; then
    DEFAULT_SCRATCH_DIR="/athena/cayuga_0003/scratch/users/${USER_NAME}"
fi
SCRATCH_DIR="${SCRATCH_DIR:-${DEFAULT_SCRATCH_DIR}}"
PROJECT_DIR="${SCRATCH_DIR}/BioReview_Training"
MODEL_DIR="${MODEL_DIR:-models/qwen3_8b_all_nonfig_v1}"
SPLIT="${SPLIT:-val}"
MAX_ARTICLES="${MAX_ARTICLES:-0}"
EVALUATE="${EVALUATE:-true}"
TAG="${TAG:-}"
RESUME="${RESUME:-false}"
ENGINE="${ENGINE:-}"   # "" = auto (tries unsloth→hf), "hf" = force HuggingFace, "unsloth", "vllm" (merged full model only)
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-}"  # optional override for scripts/run_sft_inference.py
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"  # optional override for scripts/run_sft_inference.py

echo "============================================================"
echo "BioReview SFT Inference — Cayuga"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURMD_NODENAME}"
echo "Model dir:    ${MODEL_DIR}"
echo "Split:        ${SPLIT}"
echo "Max articles: ${MAX_ARTICLES} (0=all)"
echo "Tag:          ${TAG:-'(none)'}"
echo "Resume:       ${RESUME}"
echo "Evaluate:     ${EVALUATE}"
echo "Engine:       ${ENGINE:-auto}"
echo "Max seq len:  ${MAX_SEQ_LENGTH:-16384(default)}"
echo "Max new toks: ${MAX_NEW_TOKENS:-4096(default)}"
echo "Start time:   $(date)"
echo "============================================================"

# ── Environment setup ──────────────────────────────────────────────────────
cd "${PROJECT_DIR}" \
    || { echo "ERROR: project dir not found: ${PROJECT_DIR}"; exit 1; }
source slurm/env_setup.sh

mkdir -p logs results/sft_eval

# ── GPU verification ───────────────────────────────────────────────────────
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ── Verify model exists ───────────────────────────────────────────────────
if [ ! -d "${MODEL_DIR}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_DIR}"
    echo "Train first: sbatch slurm/train_sft.sh"
    exit 1
fi

echo "Model files:"
ls -lh "${MODEL_DIR}/" | head -10
echo ""

# ── Verify peer-review-benchmark for evaluation ──────────────────────────
BENCH_DIR="${SCRATCH_DIR}/peer-review-benchmark"
SPLITS_DIR="${SPLITS_DIR:-${BENCH_DIR}/data/splits/v3}"

# Splits are required for both inference (article loading) and evaluation
if [ ! -d "${SPLITS_DIR}" ]; then
    echo "ERROR: peer-review-benchmark splits not found at ${SPLITS_DIR}"
    echo "  Inference requires split files to load articles."
    echo "  To fix: bash slurm/sync_to_hpc.sh (from local machine)"
    exit 1
fi

if [ ! -f "${SPLITS_DIR}/${SPLIT}.jsonl" ]; then
    echo "ERROR: Split file not found: ${SPLITS_DIR}/${SPLIT}.jsonl"
    echo "  Available splits:"
    ls "${SPLITS_DIR}"/*.jsonl 2>/dev/null | while read f; do
        echo "    $(basename "$f" .jsonl) ($(wc -l < "$f") lines)"
    done
    exit 1
fi

echo "Splits dir: ${SPLITS_DIR}"
echo "Split file: ${SPLITS_DIR}/${SPLIT}.jsonl ($(wc -l < "${SPLITS_DIR}/${SPLIT}.jsonl") lines)"

if [ "${EVALUATE}" = "true" ]; then
    echo "Evaluation: enabled"
fi

# ── Build inference command ────────────────────────────────────────────────
CMD="python -u scripts/run_sft_inference.py"
CMD="${CMD} --model-dir ${MODEL_DIR}"
CMD="${CMD} --split ${SPLIT}"
CMD="${CMD} --splits-dir ${SPLITS_DIR}"

if [ "${MAX_ARTICLES}" != "0" ]; then
    CMD="${CMD} --max-articles ${MAX_ARTICLES}"
fi

if [ -n "${TAG}" ]; then
    CMD="${CMD} --tag ${TAG}"
fi

if [ "${RESUME}" = "true" ]; then
    CMD="${CMD} --resume"
fi

if [ "${EVALUATE}" = "true" ]; then
    CMD="${CMD} --evaluate"
fi

if [ -n "${ENGINE}" ]; then
    CMD="${CMD} --engine ${ENGINE}"
fi

if [ -n "${MAX_SEQ_LENGTH}" ]; then
    CMD="${CMD} --max-seq-length ${MAX_SEQ_LENGTH}"
fi

if [ -n "${MAX_NEW_TOKENS}" ]; then
    CMD="${CMD} --max-new-tokens ${MAX_NEW_TOKENS}"
fi

echo ""
echo "Command: ${CMD}"
echo ""

# ── Run inference ──────────────────────────────────────────────────────────
eval "${CMD}"
EXIT_CODE=$?

echo ""
echo "============================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Inference completed successfully!"
    # Show output files
    echo "Output files:"
    ls -lh results/sft_eval/ 2>/dev/null | tail -5
else
    echo "Inference FAILED with exit code ${EXIT_CODE}"
    echo "Check: ${PROJECT_DIR}/logs/inference_${SLURM_JOB_ID}.err"
fi
echo "End time: $(date)"
echo "============================================================"
exit ${EXIT_CODE}
