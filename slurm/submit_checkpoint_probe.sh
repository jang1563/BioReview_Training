#!/bin/bash
# Submit a quick inference/eval probe against the latest training checkpoint.
#
# Run from LOCAL machine:
#   bash slurm/submit_checkpoint_probe.sh
#   bash slurm/submit_checkpoint_probe.sh --max-articles 100
#   bash slurm/submit_checkpoint_probe.sh --config configs/qwen3_8b_hi_conf.yaml

set -euo pipefail

CAYUGA_USER="${USER}"
CAYUGA_HOST="${HPC_LOGIN:-cayuga-login1}"
REMOTE_BASE="/athena/masonlab/scratch/users/${CAYUGA_USER}/BioReview_Training"
SLURM_BIN="/opt/ohpc/pub/software/slurm/24.05.2/bin"

CONFIG="configs/qwen3_8b_all_nonfig.yaml"
SPLIT="val"
MAX_ARTICLES="50"
TAG=""
EVALUATE="true"
SPLITS_DIR=""
CHECKPOINT_STEP=""
CHECKPOINT_DIR=""

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
        --max-articles)
            MAX_ARTICLES="$2"
            shift 2
            ;;
        --splits-dir)
            SPLITS_DIR="$2"
            shift 2
            ;;
        --checkpoint-step)
            CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --no-eval)
            EVALUATE="false"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

REMOTE="${CAYUGA_USER}@${CAYUGA_HOST}"
SSH_CMD=(ssh -o ControlMaster=no -o ControlPath=none)

OUTPUT_DIR="$(
"${SSH_CMD[@]}" "${REMOTE}" "cd '${REMOTE_BASE}' && source slurm/env_setup.sh && python - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('${CONFIG}').read_text())
print(cfg['output']['dir'])
PY"
)"

if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: could not resolve output dir from ${CONFIG}" >&2
    exit 1
fi

if [ -n "${CHECKPOINT_DIR}" ] && [ -n "${CHECKPOINT_STEP}" ]; then
    echo "ERROR: use only one of --checkpoint-dir or --checkpoint-step" >&2
    exit 1
fi

if [ -n "${CHECKPOINT_DIR}" ]; then
    LATEST_CKPT="${CHECKPOINT_DIR}"
elif [ -n "${CHECKPOINT_STEP}" ]; then
    LATEST_CKPT="${OUTPUT_DIR}/checkpoint-${CHECKPOINT_STEP}"
else
    LATEST_CKPT="$(
    "${SSH_CMD[@]}" "${REMOTE}" "cd '${REMOTE_BASE}' && ls -td '${OUTPUT_DIR}'/checkpoint-* 2>/dev/null | head -1"
    )"
fi

if [ -z "${LATEST_CKPT}" ]; then
    echo "No checkpoints found yet under ${OUTPUT_DIR}/"
    exit 2
fi

READY="$(
"${SSH_CMD[@]}" "${REMOTE}" "cd '${REMOTE_BASE}' && if [ -d '${LATEST_CKPT}' ]; then echo ready; fi"
)"
if [ "${READY}" != "ready" ]; then
    echo "ERROR: checkpoint not found: ${LATEST_CKPT}" >&2
    exit 2
fi

CKPT_NAME="$(basename "${LATEST_CKPT}")"
if [ -z "${TAG}" ]; then
    TAG="probe_${CKPT_NAME}_n${MAX_ARTICLES}"
fi

echo "============================================================"
echo "Submitting checkpoint probe"
echo "============================================================"
echo "Config:        ${CONFIG}"
echo "Checkpoint:    ${LATEST_CKPT}"
if [ -n "${CHECKPOINT_STEP}" ]; then
    echo "Step pin:      ${CHECKPOINT_STEP}"
fi
echo "Split:         ${SPLIT}"
if [ -n "${SPLITS_DIR}" ]; then
    echo "Splits dir:    ${SPLITS_DIR}"
fi
echo "Max articles:  ${MAX_ARTICLES}"
echo "Evaluate:      ${EVALUATE}"
echo "Tag:           ${TAG}"
echo "============================================================"

EXPORTS="ALL,MODEL_DIR='${LATEST_CKPT}',SPLIT='${SPLIT}',MAX_ARTICLES='${MAX_ARTICLES}',EVALUATE='${EVALUATE}',TAG='${TAG}'"
if [ -n "${SPLITS_DIR}" ]; then
    EXPORTS="${EXPORTS},SPLITS_DIR='${SPLITS_DIR}'"
fi

"${SSH_CMD[@]}" "${REMOTE}" \
    "cd '${REMOTE_BASE}' && \
     '${SLURM_BIN}/sbatch' \
       --export=${EXPORTS} \
       slurm/run_inference.sh"
