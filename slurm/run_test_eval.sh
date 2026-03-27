#!/bin/bash
# ── BioReview Test Set Evaluation Pipeline ───────────────────────────────
#
# Full pipeline: inference → postprocess (dedup+cap) → by-source eval
# Only run after a model passes the gate on val set (F1 ≥ 0.58 or Recall ≥ 0.45).
#
# Usage:
#   sbatch --export=MODEL_DIR=models/qwen3_8b_all_nonfig_v1 slurm/run_test_eval.sh
#   sbatch --export=MODEL_DIR=models/qwen3.5_9b_all_nonfig_v1,CAP=20 slurm/run_test_eval.sh
#   sbatch --gres=gpu:a100:1 --mem=80G slurm/run_test_eval.sh
# ─────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=bioreview_test
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --output=${SCRATCH_DIR:-/path/to/scratch}/BioReview_Training/logs/test_eval_%j.log
#SBATCH --error=${SCRATCH_DIR:-/path/to/scratch}/BioReview_Training/logs/test_eval_%j.err

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────
SCRATCH_DIR="${SCRATCH_DIR:-/path/to/scratch}"
PROJECT_DIR="${SCRATCH_DIR}/BioReview_Training"
BENCH_DIR="${SCRATCH_DIR}/peer-review-benchmark"
MODEL_DIR="${MODEL_DIR:-models/qwen3_8b_all_nonfig_v1}"
SPLIT="test"
CAP="${CAP:-20}"
DEDUP="${DEDUP:-true}"
RESUME="${RESUME:-false}"

# ── Setup ────────────────────────────────────────────────────────────────
cd "${PROJECT_DIR}" \
    || { echo "ERROR: project dir not found"; exit 1; }
source slurm/env_setup.sh

mkdir -p logs results/sft_eval results/source_eval results/test_results

MODEL_BASENAME="$(basename "${MODEL_DIR}")"
RAW_OUTPUT="results/sft_eval/${MODEL_BASENAME}_test.jsonl"
PP_OUTPUT="results/sft_eval/${MODEL_BASENAME}_test.dedup_cap${CAP}.jsonl"
SPLITS_DIR="${BENCH_DIR}/data/splits/v3"
EMBED_MODEL="${PROJECT_DIR}/models/specter2_base"

echo "============================================================"
echo "BioReview Test Set Evaluation Pipeline"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Model:         ${MODEL_DIR}"
echo "Split:         ${SPLIT}"
echo "Dedup:         ${DEDUP}"
echo "Resume:        ${RESUME}"
echo "Cap:           ${CAP}"
echo "Raw output:    ${RAW_OUTPUT}"
echo "PP output:     ${PP_OUTPUT}"
echo "Start time:    $(date)"
echo "============================================================"

# ── GPU info ──────────────────────────────────────────────────────────────
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ── Verify prerequisites ─────────────────────────────────────────────────
if [ ! -d "${MODEL_DIR}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_DIR}"
    exit 1
fi

if [ ! -f "${SPLITS_DIR}/${SPLIT}.jsonl" ]; then
    echo "ERROR: Split file not found: ${SPLITS_DIR}/${SPLIT}.jsonl"
    exit 1
fi

if [ ! -d "${EMBED_MODEL}" ]; then
    echo "WARNING: local SPECTER2 not found, will use HF cache"
fi

N_TEST=$(wc -l < "${SPLITS_DIR}/${SPLIT}.jsonl")
echo "Test articles: ${N_TEST}"
echo ""

# ── Step 1: Inference ─────────────────────────────────────────────────────
echo "============================================================"
echo "Step 1/3: Running inference on test set (${N_TEST} articles)"
echo "============================================================"

INFER_ARGS="--model-dir ${MODEL_DIR} --split ${SPLIT} --splits-dir ${SPLITS_DIR} --evaluate"
if [ "${RESUME}" = "true" ] || [ "${RESUME}" = "1" ]; then
    INFER_ARGS="${INFER_ARGS} --resume"
fi
python scripts/run_sft_inference.py ${INFER_ARGS}

echo ""
echo "Raw inference complete: ${RAW_OUTPUT}"
echo "Raw concerns: $(python3 -c "
import json
total=0
with open('${RAW_OUTPUT}') as f:
    for line in f:
        row=json.loads(line)
        total+=len(row.get('concerns',[]))
print(total)
")"

# ── Step 2: Postprocess (dedup + cap + fix categories) ────────────────────
echo ""
echo "============================================================"
echo "Step 2/3: Postprocessing (dedup=${DEDUP}, cap=${CAP})"
echo "============================================================"

PP_ARGS="--input ${RAW_OUTPUT} --output ${PP_OUTPUT} --cap ${CAP} --evaluate"
PP_ARGS="${PP_ARGS} --splits-dir ${SPLITS_DIR} --split ${SPLIT}"
if [ "${DEDUP}" != "true" ]; then
    PP_ARGS="${PP_ARGS} --no-dedup-exact"
fi

python scripts/postprocess_inference_output.py ${PP_ARGS}

echo ""
echo "Postprocessed output: ${PP_OUTPUT}"

# ── Step 3: By-source evaluation ──────────────────────────────────────────
echo ""
echo "============================================================"
echo "Step 3/3: By-source evaluation"
echo "============================================================"

# Evaluate both raw and postprocessed
for VARIANT in "${RAW_OUTPUT}" "${PP_OUTPUT}"; do
    VARIANT_STEM="$(basename "${VARIANT}" .jsonl)"
    OUT_PREFIX="results/source_eval/${VARIANT_STEM}_by_source"

    echo ""
    echo "--- Evaluating: ${VARIANT_STEM} ---"

    EVAL_CMD="python scripts/evaluate_by_source.py ${VARIANT}"
    EVAL_CMD="${EVAL_CMD} --splits-dir ${SPLITS_DIR}"
    EVAL_CMD="${EVAL_CMD} --split ${SPLIT}"
    if [ -d "${EMBED_MODEL}" ]; then
        EVAL_CMD="${EVAL_CMD} --embed-model ${EMBED_MODEL}"
    fi
    EVAL_CMD="${EVAL_CMD} --output-prefix ${OUT_PREFIX}"

    eval "${EVAL_CMD}"
done

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Test Evaluation Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Raw:  results/source_eval/${MODEL_BASENAME}_test_by_source.json"
echo "  PP:   results/source_eval/${MODEL_BASENAME}_test.dedup_cap${CAP}_by_source.json"
echo ""

# Copy key results to test_results for easy access
cp -v results/source_eval/${MODEL_BASENAME}_test*_by_source.json results/test_results/ 2>/dev/null || true
cp -v results/source_eval/${MODEL_BASENAME}_test*_by_source.md results/test_results/ 2>/dev/null || true

echo ""
echo "End time: $(date)"
echo "============================================================"
