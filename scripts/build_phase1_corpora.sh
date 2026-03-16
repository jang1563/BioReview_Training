#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SPLITS_DIR="${SPLITS_DIR:-../peer-review-benchmark/data/splits/v3}"
LOG_DIR="${LOG_DIR:-results/phase1_logs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%dT%H%M%S)"

echo "Phase 1 corpus build"
echo "root: ${ROOT_DIR}"
echo "splits: ${SPLITS_DIR}"
echo "python: ${PYTHON_BIN}"
echo "logs: ${LOG_DIR}"

run_build() {
    local name="$1"
    shift
    local log_path="${LOG_DIR}/${name}_${TS}.log"
    echo ""
    echo "[build] ${name}"
    echo "[log]   ${log_path}"
    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" scripts/prepare_sft_data.py "$@" | tee "${log_path}"
}

run_build \
    corpus_all_nonfig \
    --splits train val \
    --splits-dir "${SPLITS_DIR}" \
    --output-dir data/corpus_all_nonfig \
    --min-resolution-confidence 0.0 \
    --min-concerns 1 \
    --drop-title-only

run_build \
    corpus_hi_conf \
    --splits train val \
    --splits-dir "${SPLITS_DIR}" \
    --output-dir data/corpus_hi_conf \
    --min-resolution-confidence 0.8 \
    --min-concerns 3 \
    --drop-title-only

echo ""
echo "Done."
