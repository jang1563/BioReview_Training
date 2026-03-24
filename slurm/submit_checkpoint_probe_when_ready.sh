#!/bin/bash
# Wait until a specific checkpoint exists on Cayuga, then submit a probe job.
#
# Run from LOCAL machine:
#   bash slurm/submit_checkpoint_probe_when_ready.sh \
#     --checkpoint-step 300 \
#     --splits-dir ${SCRATCH_DIR:-/path/to/scratch}/peer-review-benchmark/data/splits/probe50 \
#     --max-articles 50 \
#     --tag probe50_current_v3_checkpoint300

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CAYUGA_USER="${USER}"
CAYUGA_HOST="${HPC_LOGIN:-cayuga-login1}"
REMOTE_BASE="/athena/masonlab/scratch/users/${CAYUGA_USER}/BioReview_Training"

CONFIG="configs/qwen3_8b_all_nonfig.yaml"
CHECKPOINT_STEP=""
POLL_SECONDS="120"
SPLIT="val"
SPLITS_DIR=""
MAX_ARTICLES="50"
TAG=""
NO_EVAL=""

while [ $# -gt 0 ]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint-step)
            CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --poll-seconds)
            POLL_SECONDS="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --splits-dir)
            SPLITS_DIR="$2"
            shift 2
            ;;
        --max-articles)
            MAX_ARTICLES="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --no-eval)
            NO_EVAL="--no-eval"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "${CHECKPOINT_STEP}" ]; then
    echo "ERROR: --checkpoint-step is required" >&2
    exit 1
fi

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

TARGET_CKPT="${OUTPUT_DIR}/checkpoint-${CHECKPOINT_STEP}"

echo "============================================================"
echo "Waiting for checkpoint"
echo "============================================================"
echo "Config:          ${CONFIG}"
echo "Checkpoint step: ${CHECKPOINT_STEP}"
echo "Checkpoint dir:  ${TARGET_CKPT}"
echo "Poll seconds:    ${POLL_SECONDS}"
echo "============================================================"

while true; do
    READY="$(
    "${SSH_CMD[@]}" "${REMOTE}" "cd '${REMOTE_BASE}' && if [ -d '${TARGET_CKPT}' ]; then echo ready; fi"
    )"
    if [ "${READY}" = "ready" ]; then
        break
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] checkpoint-${CHECKPOINT_STEP} not ready yet"
    sleep "${POLL_SECONDS}"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] checkpoint-${CHECKPOINT_STEP} detected, submitting probe"

CMD=(bash "${ROOT_DIR}/slurm/submit_checkpoint_probe.sh"
    --config "${CONFIG}"
    --checkpoint-step "${CHECKPOINT_STEP}"
    --split "${SPLIT}"
    --max-articles "${MAX_ARTICLES}"
)

if [ -n "${SPLITS_DIR}" ]; then
    CMD+=(--splits-dir "${SPLITS_DIR}")
fi
if [ -n "${TAG}" ]; then
    CMD+=(--tag "${TAG}")
fi
if [ -n "${NO_EVAL}" ]; then
    CMD+=("${NO_EVAL}")
fi

"${CMD[@]}"
