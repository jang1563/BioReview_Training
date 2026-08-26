#!/bin/bash
# ── Merge LoRA adapters into base models for inference ────────────────────
# Produces full-model artifacts that can be used with vLLM or plain HF.
#
# Usage:
#   sbatch slurm/merge_adapters.sh
#   sbatch --export=ALL,MODEL=gemma2 slurm/merge_adapters.sh
#   sbatch --export=ALL,MODEL=gemma4 slurm/merge_adapters.sh
#   sbatch --export=ALL,MODEL=gemma2,MERGE_SAVE_METHOD=merged_4bit_forced slurm/merge_adapters.sh
# ────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=merge_adapters
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00

USER_NAME="${USER:-$(id -un)}"
DEFAULT_SCRATCH_DIR="/athena/masonlab/scratch/users/${USER_NAME}"
if [ ! -d "${DEFAULT_SCRATCH_DIR}" ] && [ -d "/athena/cayuga_0003/scratch/users/${USER_NAME}" ]; then
    DEFAULT_SCRATCH_DIR="/athena/cayuga_0003/scratch/users/${USER_NAME}"
fi
SCRATCH_DIR="${SCRATCH_DIR:-${DEFAULT_SCRATCH_DIR}}"
PROJECT_DIR="${SCRATCH_DIR}/BioReview_Training"
MODEL="${MODEL:-both}"
MERGE_SAVE_METHOD="${MERGE_SAVE_METHOD:-merged_16bit}"

cd "${PROJECT_DIR}" || { echo "ERROR: project dir not found: ${PROJECT_DIR}"; exit 1; }
source slurm/env_setup.sh

echo "============================================================"
echo "Merge LoRA Adapters — Cayuga"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "Model:  ${MODEL}"
echo "Save:   ${MERGE_SAVE_METHOD}"
echo "============================================================"
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

merge_gemma2() {
    local ADAPTER_DIR="models/gemma2_9b_all_nonfig_v1_8k_left"
    local MERGED_DIR="models/gemma2_9b_merged_bf16"

    if merged_model_ready "${MERGED_DIR}"; then
        echo "Gemma-2 already merged: ${MERGED_DIR}"
        return 0
    elif [ -f "${MERGED_DIR}/config.json" ]; then
        echo "Gemma-2 merge output is incomplete; rebuilding: ${MERGED_DIR}"
    fi

    echo ""
    echo "=== Merging Gemma-2-9B ==="
    python scripts/merge_lora_adapter.py \
        --adapter-dir "${ADAPTER_DIR}" \
        --output-dir "${MERGED_DIR}" \
        --max-seq-length 8192 \
        --save-method "${MERGE_SAVE_METHOD}"
    return $?
}

merge_gemma4() {
    local ADAPTER_DIR="models/gemma4_31b_all_nonfig_v1_4k"
    local MERGED_DIR="models/gemma4_31b_merged_bf16"

    if merged_model_ready "${MERGED_DIR}"; then
        echo "Gemma 4 already merged: ${MERGED_DIR}"
        return 0
    elif [ -f "${MERGED_DIR}/config.json" ]; then
        echo "Gemma 4 merge output is incomplete; rebuilding: ${MERGED_DIR}"
    fi

    echo ""
    echo "=== Merging Gemma 4 31B ==="
    echo "NOTE: Gemma 4 merge typically needs an A100 80GB; override sbatch resources if needed."
    python scripts/merge_lora_adapter.py \
        --adapter-dir "${ADAPTER_DIR}" \
        --output-dir "${MERGED_DIR}" \
        --max-seq-length 4096 \
        --save-method "${MERGE_SAVE_METHOD}"
    return $?
}

EXIT=0
if [ "${MODEL}" = "gemma2" ] || [ "${MODEL}" = "both" ]; then
    merge_gemma2 || EXIT=1
fi
if [ "${MODEL}" = "gemma4" ] || [ "${MODEL}" = "both" ]; then
    merge_gemma4 || EXIT=1
fi

echo ""
echo "============================================================"
if [ ${EXIT} -eq 0 ]; then
    echo "All merges completed successfully!"
    echo "Merged models:"
    ls -lhd models/*merged* 2>/dev/null
else
    echo "Some merges FAILED (exit code ${EXIT})"
fi
echo "End time: $(date)"
echo "============================================================"
exit ${EXIT}
