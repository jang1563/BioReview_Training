#!/bin/bash
# ── vLLM Speed Test — merge LoRA to a full model then test with vLLM ─────
#
# Usage:
#   sbatch --export=ALL slurm/test_vllm_speed.sh
#
# Requires: trained LoRA adapter, vLLM installed in conda env
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=vllm_test
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00

# ── Configuration ────────────────────────────────────────────────────────
USER_NAME="${USER:-$(id -un)}"
DEFAULT_SCRATCH_DIR="/athena/masonlab/scratch/users/${USER_NAME}"
if [ ! -d "${DEFAULT_SCRATCH_DIR}" ] && [ -d "/athena/cayuga_0003/scratch/users/${USER_NAME}" ]; then
    DEFAULT_SCRATCH_DIR="/athena/cayuga_0003/scratch/users/${USER_NAME}"
fi
SCRATCH_DIR="${SCRATCH_DIR:-${DEFAULT_SCRATCH_DIR}}"
PROJECT_DIR="${SCRATCH_DIR}/BioReview_Training"

ADAPTER_DIR="${ADAPTER_DIR:-models/qwen3.5_9b_all_nonfig_v1}"
MERGED_DIR="${MERGED_DIR:-models/qwen3.5_9b_merged_bf16}"
MAX_ARTICLES="${MAX_ARTICLES:-5}"
MERGE_SAVE_METHOD="${MERGE_SAVE_METHOD:-merged_16bit}"

echo "============================================================"
echo "vLLM Speed Test — Cayuga"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURMD_NODENAME}"
echo "Adapter:      ${ADAPTER_DIR}"
echo "Merged:       ${MERGED_DIR}"
echo "Save method:  ${MERGE_SAVE_METHOD}"
echo "Max articles: ${MAX_ARTICLES}"
echo "Start time:   $(date)"
echo "============================================================"

cd "${PROJECT_DIR}" \
    || { echo "ERROR: project dir not found: ${PROJECT_DIR}"; exit 1; }
source slurm/env_setup.sh

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

merged_model_ready() {
    local model_dir="$1"
    [ -f "${model_dir}/config.json" ] || return 1
    compgen -G "${model_dir}/*.safetensors" > /dev/null && return 0
    compgen -G "${model_dir}/pytorch_model*.bin" > /dev/null && return 0
    [ -f "${model_dir}/model.safetensors.index.json" ] && return 0
    [ -f "${model_dir}/pytorch_model.bin.index.json" ] && return 0
    return 1
}

# ── Step 1: Merge LoRA to bf16 (if not already done) ────────────────────
if ! merged_model_ready "${MERGED_DIR}"; then
    echo ""
    echo "=== Step 1: Merge LoRA into a full model ==="
    if [ -f "${MERGED_DIR}/config.json" ]; then
        echo "Existing merged directory is incomplete; rebuilding ${MERGED_DIR}"
    fi
    python scripts/merge_lora_adapter.py \
        --adapter-dir "${ADAPTER_DIR}" \
        --output-dir "${MERGED_DIR}" \
        --max-seq-length 16384 \
        --save-method "${MERGE_SAVE_METHOD}"
    if [ $? -ne 0 ]; then
        echo "ERROR: Merge failed"
        exit 1
    fi
    echo "Merged model size:"
    du -sh "${MERGED_DIR}"
else
    echo ""
    echo "=== Step 1: Merged model already exists, skipping ==="
    du -sh "${MERGED_DIR}"
fi

# ── Step 2: Quick vLLM import check ────────────────────────────────────
echo ""
echo "=== Step 2: vLLM availability check ==="
python -c "import vllm; print(f'vLLM {vllm.__version__} OK')" || {
    echo "ERROR: vLLM not installed"
    exit 1
}

# ── Step 3: Inference speed test with vLLM ──────────────────────────────
echo ""
echo "=== Step 3: vLLM inference speed test (${MAX_ARTICLES} articles) ==="

SPLITS_DIR="${SCRATCH_DIR}/peer-review-benchmark/data/splits/v3"

python scripts/run_sft_inference.py \
    --model-dir "${MERGED_DIR}" \
    --engine vllm \
    --no-4bit \
    --split val \
    --splits-dir "${SPLITS_DIR}" \
    --max-articles "${MAX_ARTICLES}" \
    --tag vllm_speed_test \
    --log-every 1

echo ""
echo "=== Step 4: Baseline comparison (Unsloth, same articles) ==="
python scripts/run_sft_inference.py \
    --model-dir "${ADAPTER_DIR}" \
    --split val \
    --splits-dir "${SPLITS_DIR}" \
    --max-articles "${MAX_ARTICLES}" \
    --tag unsloth_speed_test \
    --log-every 1

echo ""
echo "============================================================"
echo "Speed test complete!"
echo "End time: $(date)"
echo "============================================================"
