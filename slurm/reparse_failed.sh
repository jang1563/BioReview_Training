#!/bin/bash
# ── Reparse 10 failed articles from Qwen3-8B all_nonfig inference ──────────
# Uses doubled max_new_tokens (8192) to handle longer outputs
# Uses a separate splits dir with only the 10 failed articles as val.jsonl
#SBATCH --job-name=reparse_8b
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=03:00:00
#SBATCH --output=/athena/cayuga_0003/scratch/users/jak4013/BioReview_Training/logs/reparse_%j.log
#SBATCH --error=/athena/cayuga_0003/scratch/users/jak4013/BioReview_Training/logs/reparse_%j.err

SCRATCH_DIR="/athena/masonlab/scratch/users/jak4013"
PROJECT_DIR="/athena/cayuga_0003/scratch/users/jak4013/BioReview_Training"
MODEL_DIR="models/qwen3_8b_all_nonfig_v1"
# Use a dedicated splits dir containing only the 10 failed articles
SPLITS_DIR="${PROJECT_DIR}/data/splits_reparse"

export HF_HOME="${SCRATCH_DIR}/huggingface"
export TORCH_HOME="${SCRATCH_DIR}/cache/torch"
# Force Python to flush stdout/stderr immediately for SLURM log visibility
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "BioReview Reparse — 10 failed articles (Qwen3-8B all_nonfig)"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURMD_NODENAME}"
echo "max_new_tokens: 8192 (doubled from 4096)"
echo "Wall time:    3 hours"
echo "Start time:   $(date)"
echo "============================================================"

# ── Conda ──────────────────────────────────────────────────────────────────
source /home/fs01/jak4013/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate bioreview-sft

cd "${PROJECT_DIR}" || { echo "ERROR: project dir not found"; exit 1; }
mkdir -p logs results/sft_eval

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

echo "Split file: ${SPLITS_DIR}/val.jsonl"
echo "Articles: $(wc -l < ${SPLITS_DIR}/val.jsonl)"
echo ""

python scripts/run_sft_inference.py \
    --model-dir ${MODEL_DIR} \
    --split val \
    --splits-dir ${SPLITS_DIR} \
    --output results/sft_eval/qwen3_8b_all_nonfig_v1_val_reparse.jsonl \
    --max-new-tokens 8192 \
    --resume

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Reparse completed successfully!"
    echo "Output:"
    ls -lh results/sft_eval/qwen3_8b_all_nonfig_v1_val_reparse.jsonl
    echo ""
    echo "Parse results:"
    python3 -c "
import json
ok = fail = 0
with open(results/sft_eval/qwen3_8b_all_nonfig_v1_val_reparse.jsonl) as f:
    for line in f:
        row = json.loads(line.strip())
        if row.get(parse_ok):
            ok += 1
        else:
            fail += 1
        print(%s: parse_ok=%s, concerns=%d % (row.get(article_id), row.get(parse_ok), len(row.get(concerns,[]))))
print(nTotal: %d ok, %d failed % (ok, fail))
"
else
    echo "Reparse FAILED with exit code ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "============================================================"
exit ${EXIT_CODE}
