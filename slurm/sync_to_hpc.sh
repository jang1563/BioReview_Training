#!/bin/bash
# ── Sync BioReview_Training to Cayuga HPC ──────────────────────────────────
#
# Run from LOCAL machine (not on HPC):
#   bash slurm/sync_to_hpc.sh
#   bash slurm/sync_to_hpc.sh --dry-run     # preview without copying
#   bash slurm/sync_to_hpc.sh --download     # download results from HPC
#
# Prerequisites:
#   - Cayuga VPN connected
#   - SSH key configured for Cayuga
#
# What gets synced:
#   UP:   scripts/, configs/, slurm/, requirements-train.txt, phase-specific SFT corpora
#   DOWN: models/, results/sft_eval/, results/source_eval/, logs/
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
CAYUGA_USER="${USER}"
CAYUGA_HOST="${HPC_LOGIN:-cayuga-login1}"
REMOTE_BASE="/athena/masonlab/scratch/users/${CAYUGA_USER}/BioReview_Training"
LOCAL_BASE="$(cd "$(dirname "$0")/.." && pwd)"

# peer-review-benchmark (needed for evaluation splits + SPECTER2)
LOCAL_BENCH="$(cd "${LOCAL_BASE}/.." && pwd)/peer-review-benchmark"
REMOTE_BENCH="/athena/masonlab/scratch/users/${CAYUGA_USER}/peer-review-benchmark"

RSYNC_OPTS="-avz --progress"
DRY_RUN=""
DOWNLOAD=false

# ── Parse args ─────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run"; echo "[DRY RUN MODE]" ;;
        --download) DOWNLOAD=true ;;
    esac
done

REMOTE="${CAYUGA_USER}@${CAYUGA_HOST}"
SSH_CMD=(ssh -o ControlMaster=no -o ControlPath=none)
export RSYNC_RSH="ssh -o ControlMaster=no -o ControlPath=none"

if [ "${DOWNLOAD}" = true ]; then
    # ── Download results from HPC ──────────────────────────────────────────
    echo "============================================================"
    echo "Downloading results from Cayuga"
    echo "============================================================"

    # Create local directories (may be gitignored and absent)
    mkdir -p \
        "${LOCAL_BASE}/models" \
        "${LOCAL_BASE}/results/sft_eval" \
        "${LOCAL_BASE}/results/source_eval" \
        "${LOCAL_BASE}/logs"

    echo ""
    echo "[1/3] Downloading trained model..."
    rsync ${RSYNC_OPTS} ${DRY_RUN} \
        "${REMOTE}:${REMOTE_BASE}/models/" \
        "${LOCAL_BASE}/models/"

    echo ""
    echo "[2/3] Downloading evaluation results..."
    rsync ${RSYNC_OPTS} ${DRY_RUN} \
        "${REMOTE}:${REMOTE_BASE}/results/sft_eval/" \
        "${LOCAL_BASE}/results/sft_eval/"

    echo ""
    echo ""
    echo "[3/4] Downloading source-sliced evaluation results..."
    rsync ${RSYNC_OPTS} ${DRY_RUN} \
        "${REMOTE}:${REMOTE_BASE}/results/source_eval/" \
        "${LOCAL_BASE}/results/source_eval/"

    echo ""
    echo "[4/4] Downloading logs..."
    rsync ${RSYNC_OPTS} ${DRY_RUN} \
        "${REMOTE}:${REMOTE_BASE}/logs/" \
        "${LOCAL_BASE}/logs/"

    echo ""
    echo "Download complete!"
else
    # ── Upload project to HPC ──────────────────────────────────────────────
    echo "============================================================"
    echo "Uploading BioReview_Training to Cayuga"
    echo "============================================================"
    echo "Local:  ${LOCAL_BASE}"
    echo "Remote: ${REMOTE}:${REMOTE_BASE}"
    echo ""

    # Create remote directories
    echo "[0/4] Creating remote directories..."
    "${SSH_CMD[@]}" "${REMOTE}" "mkdir -p ${REMOTE_BASE}/{scripts,configs,slurm,data,data/corpus_all_nonfig,data/corpus_hi_conf,models,logs,results/sft_eval,results/source_eval}"

    echo ""
    echo "[1/4] Uploading scripts, configs, slurm..."
    rsync ${RSYNC_OPTS} ${DRY_RUN} \
        "${LOCAL_BASE}/scripts/" "${REMOTE}:${REMOTE_BASE}/scripts/"
    rsync ${RSYNC_OPTS} ${DRY_RUN} \
        "${LOCAL_BASE}/configs/" "${REMOTE}:${REMOTE_BASE}/configs/"
    rsync ${RSYNC_OPTS} ${DRY_RUN} \
        "${LOCAL_BASE}/slurm/" "${REMOTE}:${REMOTE_BASE}/slurm/"
    rsync ${RSYNC_OPTS} ${DRY_RUN} \
        "${LOCAL_BASE}/requirements-train.txt" "${REMOTE}:${REMOTE_BASE}/"

    echo ""
    echo "[2/4] Uploading SFT training data..."
    if [ -d "${LOCAL_BASE}/data/corpus_all_nonfig" ]; then
        rsync ${RSYNC_OPTS} ${DRY_RUN} \
            "${LOCAL_BASE}/data/corpus_all_nonfig/" \
            "${REMOTE}:${REMOTE_BASE}/data/corpus_all_nonfig/"
    else
        echo "  WARNING: missing local corpus_all_nonfig directory"
    fi

    if [ -d "${LOCAL_BASE}/data/corpus_hi_conf" ]; then
        rsync ${RSYNC_OPTS} ${DRY_RUN} \
            "${LOCAL_BASE}/data/corpus_hi_conf/" \
            "${REMOTE}:${REMOTE_BASE}/data/corpus_hi_conf/"
    else
        echo "  WARNING: missing local corpus_hi_conf directory"
    fi

    # Legacy flat data files are still synced if present for backward compatibility.
    rsync ${RSYNC_OPTS} ${DRY_RUN} --ignore-missing-args \
        "${LOCAL_BASE}/data/sft_train.jsonl" \
        "${LOCAL_BASE}/data/sft_val.jsonl" \
        "${LOCAL_BASE}/data/sft_train_stats.json" \
        "${LOCAL_BASE}/data/sft_val_stats.json" \
        "${REMOTE}:${REMOTE_BASE}/data/" 2>/dev/null || true

    echo ""
    echo "[3/4] Uploading SPECTER2 model (for evaluation)..."
    if [ -d "${LOCAL_BASE}/models/specter2_base" ]; then
        rsync ${RSYNC_OPTS} ${DRY_RUN} \
            "${LOCAL_BASE}/models/specter2_base/" \
            "${REMOTE}:${REMOTE_BASE}/models/specter2_base/"
    else
        echo "  SPECTER2 not found locally, skipping."
    fi

    echo ""
    echo "[4/4] Uploading peer-review-benchmark (for evaluation)..."
    if [ -d "${LOCAL_BENCH}" ]; then
        "${SSH_CMD[@]}" "${REMOTE}" "mkdir -p ${REMOTE_BENCH}/{bioreview_bench,data/splits/v3}"

        # bioreview_bench package (imported via sys.path in inference script)
        echo "  Syncing bioreview_bench package..."
        rsync ${RSYNC_OPTS} ${DRY_RUN} \
            --exclude="__pycache__" --exclude="*.pyc" \
            "${LOCAL_BENCH}/bioreview_bench/" "${REMOTE}:${REMOTE_BENCH}/bioreview_bench/"

        # pyproject.toml (metadata; pip install not used — requires Python >=3.12)
        rsync ${RSYNC_OPTS} ${DRY_RUN} \
            "${LOCAL_BENCH}/pyproject.toml" "${REMOTE}:${REMOTE_BENCH}/"

        # v3 splits: val + test only (skip 512MB train.jsonl — not needed for eval)
        echo "  Syncing v3 evaluation splits (val + test)..."
        rsync ${RSYNC_OPTS} ${DRY_RUN} \
            --exclude="train.jsonl" \
            "${LOCAL_BENCH}/data/splits/v3/" "${REMOTE}:${REMOTE_BENCH}/data/splits/v3/"
    else
        echo "  peer-review-benchmark not found at ${LOCAL_BENCH}, skipping."
        echo "  Evaluation will not be available on HPC."
    fi

    echo ""
    echo "============================================================"
    echo "Upload complete!"
    echo ""
    echo "Next steps on Cayuga:"
    echo "  ssh ${REMOTE}"
    echo "  cd ${REMOTE_BASE}"
    echo "  bash slurm/setup_cayuga.sh      # one-time env setup"
    echo "  /opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch slurm/train_sft.sh"
    echo "  /opt/ohpc/pub/software/slurm/24.05.2/bin/squeue -u \$USER"
    echo "============================================================"
fi
