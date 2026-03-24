#!/bin/bash
# Submit a full validation inference/eval job against the final training output.
#
# Run from LOCAL machine:
#   bash slurm/submit_final_eval.sh
#   bash slurm/submit_final_eval.sh --afterok 2702484
#   bash slurm/submit_final_eval.sh --config configs/qwen3_8b_hi_conf.yaml

set -euo pipefail

CAYUGA_USER="${USER}"
CAYUGA_HOST="${HPC_LOGIN:-cayuga-login1}"
REMOTE_BASE="/athena/masonlab/scratch/users/${CAYUGA_USER}/BioReview_Training"
SLURM_BIN="/opt/ohpc/pub/software/slurm/24.05.2/bin"
CONDA_BASE="${CONDA_PREFIX:-/path/to/conda}"
CONDA_ENV="bioreview-sft"

CONFIG="configs/qwen3_8b_all_nonfig.yaml"
SPLIT="val"
MAX_ARTICLES="0"
TAG=""
EVALUATE="true"
RESUME="false"
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
        --max-articles)
            MAX_ARTICLES="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --afterok)
            AFTEROK="$2"
            shift 2
            ;;
        --resume)
            RESUME="true"
            shift
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
"${SSH_CMD[@]}" "${REMOTE}" "source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${CONDA_ENV}' && cd '${REMOTE_BASE}' && python - <<'PY'
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

MODEL_BASENAME="$(basename "${OUTPUT_DIR}")"
if [ -z "${TAG}" ]; then
    if [ "${MAX_ARTICLES}" = "0" ]; then
        TAG="final_${MODEL_BASENAME}_${SPLIT}"
    else
        TAG="final_${MODEL_BASENAME}_${SPLIT}_n${MAX_ARTICLES}"
    fi
fi

if [ -z "${AFTEROK}" ]; then
    READY="$(
    "${SSH_CMD[@]}" "${REMOTE}" "cd '${REMOTE_BASE}' && if [ -f '${OUTPUT_DIR}/adapter_config.json' ] || [ -f '${OUTPUT_DIR}/config.json' ]; then echo ready; fi"
    )"
    if [ "${READY}" != "ready" ]; then
        echo "ERROR: final model files not found under ${OUTPUT_DIR}/" >&2
        echo "Use --afterok <train_job_id> to queue this eval behind a running training job." >&2
        exit 2
    fi
fi

DEPENDENCY_ARGS=()
if [ -n "${AFTEROK}" ]; then
    DEPENDENCY_ARGS+=(--dependency "afterok:${AFTEROK}")
fi

echo "============================================================"
echo "Submitting final inference/eval"
echo "============================================================"
echo "Config:        ${CONFIG}"
echo "Model dir:     ${OUTPUT_DIR}"
echo "Split:         ${SPLIT}"
echo "Max articles:  ${MAX_ARTICLES} (0=all)"
echo "Evaluate:      ${EVALUATE}"
echo "Resume:        ${RESUME}"
echo "Tag:           ${TAG}"
if [ -n "${AFTEROK}" ]; then
    echo "Dependency:    afterok:${AFTEROK}"
else
    echo "Dependency:    (none)"
fi
echo "============================================================"

"${SSH_CMD[@]}" "${REMOTE}" \
    "cd '${REMOTE_BASE}' && \
     '${SLURM_BIN}/sbatch' \
       ${DEPENDENCY_ARGS[*]:-} \
       --export=ALL,MODEL_DIR='${OUTPUT_DIR}',SPLIT='${SPLIT}',MAX_ARTICLES='${MAX_ARTICLES}',EVALUATE='${EVALUATE}',TAG='${TAG}',RESUME='${RESUME}' \
       slurm/run_inference.sh"
