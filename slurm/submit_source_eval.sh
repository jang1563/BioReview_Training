#!/bin/bash
# Submit a by-source evaluation job against an inference JSONL on Cayuga.
#
# Run from LOCAL machine:
#   bash slurm/submit_source_eval.sh --afterok 2702592
#   bash slurm/submit_source_eval.sh --tool-output results/sft_eval/foo_val.jsonl

set -euo pipefail

CAYUGA_USER="${USER}"
CAYUGA_HOST="${HPC_LOGIN:-cayuga-login1}"
REMOTE_BASE="/athena/masonlab/scratch/users/${CAYUGA_USER}/BioReview_Training"
SLURM_BIN="/opt/ohpc/pub/software/slurm/24.05.2/bin"

CONFIG="configs/qwen3_8b_all_nonfig.yaml"
SPLIT="val"
TAG=""
MODEL_DIR=""
TOOL_OUTPUT=""
OUTPUT_PREFIX=""
AFTEROK=""

while [ $# -gt 0 ]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --tool-output)
            TOOL_OUTPUT="$2"
            shift 2
            ;;
        --output-prefix)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        --afterok)
            AFTEROK="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

REMOTE="${CAYUGA_USER}@${CAYUGA_HOST}"
SSH_CMD=(ssh -o ControlMaster=no -o ControlPath=none)

if [ -z "${MODEL_DIR}" ]; then
    MODEL_DIR="$(
    "${SSH_CMD[@]}" "${REMOTE}" "cd '${REMOTE_BASE}' && source slurm/env_setup.sh && python - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('${CONFIG}').read_text())
print(cfg['output']['dir'])
PY"
    )"
fi

if [ -z "${MODEL_DIR}" ]; then
    echo "ERROR: could not resolve model dir from ${CONFIG}" >&2
    exit 1
fi

MODEL_BASENAME="$(basename "${MODEL_DIR}")"
if [ -z "${TAG}" ]; then
    TAG="final_${MODEL_BASENAME}_${SPLIT}"
fi

if [ -z "${TOOL_OUTPUT}" ]; then
    TOOL_OUTPUT="results/sft_eval/${MODEL_BASENAME}_${SPLIT}_${TAG}.jsonl"
fi

if [ -z "${OUTPUT_PREFIX}" ]; then
    TOOL_STEM="$(basename "${TOOL_OUTPUT}" .jsonl)"
    OUTPUT_PREFIX="results/source_eval/${TOOL_STEM}_by_source"
fi

DEPENDENCY_ARG=""
if [ -n "${AFTEROK}" ]; then
    DEPENDENCY_ARG="--dependency=afterok:${AFTEROK}"
fi

echo "============================================================"
echo "Submitting source-sliced evaluation"
echo "============================================================"
echo "Config:        ${CONFIG}"
echo "Model dir:     ${MODEL_DIR}"
echo "Split:         ${SPLIT}"
echo "Tag:           ${TAG}"
echo "Tool output:   ${TOOL_OUTPUT}"
echo "Output prefix: ${OUTPUT_PREFIX}"
if [ -n "${AFTEROK}" ]; then
    echo "Dependency:    afterok:${AFTEROK}"
else
    echo "Dependency:    (none)"
fi
echo "============================================================"

"${SSH_CMD[@]}" "${REMOTE}" \
    "cd '${REMOTE_BASE}' && \
     '${SLURM_BIN}/sbatch' \
       ${DEPENDENCY_ARG} \
       --export=ALL,MODEL_DIR='${MODEL_DIR}',SPLIT='${SPLIT}',TAG='${TAG}',TOOL_OUTPUT='${TOOL_OUTPUT}',OUTPUT_PREFIX='${OUTPUT_PREFIX}' \
       slurm/run_source_eval.sh"
