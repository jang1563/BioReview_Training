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
#   UP:   scripts/, configs/, slurm/, requirements-train.txt, data/*.jsonl
#   DOWN: models/, results/sft_eval/, logs/
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
CAYUGA_USER="jak4013"
CAYUGA_HOST="cayuga.cac.cornell.edu"
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

if [ "${DOWNLOAD}" = true ]; then
    # ── Download results from HPC ──────────────────────────────────────────
    echo "============================================================"
    echo "Downloading results from Cayuga"
    echo "============================================================"

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
    echo "[3/3] Downloading logs..."
    mkdir -p "${LOCAL_BASE}/logs"
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
    ssh ${REMOTE} "mkdir -p ${REMOTE_BASE}/{scripts,configs,slurm,data,models,logs,results/sft_eval}"

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
    rsync ${RSYNC_OPTS} ${DRY_RUN} \
        "${LOCAL_BASE}/data/sft_train.jsonl" \
        "${LOCAL_BASE}/data/sft_val.jsonl" \
        "${REMOTE}:${REMOTE_BASE}/data/"
    # Also upload stats files if present
    rsync ${RSYNC_OPTS} ${DRY_RUN} --ignore-missing-args \
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
        ssh ${REMOTE} "mkdir -p ${REMOTE_BENCH}"
        # Only sync evaluation essentials (not the full repo)
        rsync ${RSYNC_OPTS} ${DRY_RUN} \
            --include="bioreview_bench/" \
            --include="bioreview_bench/***" \
            --include="data/" \
            --include="data/splits/" \
            --include="data/splits/***" \
            --include="setup.py" \
            --include="setup.cfg" \
            --include="pyproject.toml" \
            --exclude="*" \
            "${LOCAL_BENCH}/" "${REMOTE}:${REMOTE_BENCH}/"
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
    echo "  sbatch slurm/train_sft.sh       # submit training"
    echo "  squeue -u \$USER                 # check job status"
    echo "============================================================"
fi
