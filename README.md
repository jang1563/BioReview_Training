# BioReview Training

QLoRA SFT (Supervised Fine-Tuning) pipeline for training biomedical peer-review LLMs on the [peer-review-benchmark](../peer-review-benchmark/) dataset.

## Overview

Fine-tunes open-source LLMs to identify specific scientific concerns in biomedical papers, evaluated against human reviewer annotations using SPECTER2 semantic matching.

**Baselines (peer-review-benchmark val split, 982 articles):**

| Model | F1 | Recall | Precision |
|-------|-----|--------|-----------|
| GPT-4o-mini | 0.6962 | — | — |
| Gemini-2.5-Flash | 0.4489 | — | — |
| **Ensemble Union v1** (9B+14B) | **0.5403** | **0.385** | **0.903** |
| Qwen3.5-9B v1 (SFT) | 0.4248 | 0.274 | 0.947 |
| Qwen2.5-14B v1 (SFT) | 0.3809 | 0.238 | 0.962 |
| Ensemble Vote2 v1 | 0.0904 | 0.047 | 0.999 |

---

## Project Structure

```
BioReview_Training/
├── configs/                   # Training configurations
│   ├── qwen3.5_9b_qlora.yaml       # Qwen3.5-9B QLoRA (A40 48G)
│   ├── qwen2.5_14b_qlora.yaml      # Qwen2.5-14B QLoRA (A100 80G)
│   ├── deepseek_r1_14b_qlora.yaml  # DeepSeek-R1-14B QLoRA (A100 80G)
│   ├── qwen3_8b_qlora.yaml         # Qwen3-8B QLoRA (A40)
│   ├── qwen7b_qlora.yaml           # Qwen2.5-7B QLoRA (A40)
│   └── sweep/                      # Hyperparameter sweep configs
│       ├── stage1_9b.yaml / stage1_14b.yaml   # Sweep specs
│       └── stage1_9b/ stage1_14b/              # Generated variant YAMLs
│
├── scripts/
│   ├── prepare_sft_data.py    # Convert benchmark splits → ShareGPT JSONL
│   ├── train_sft.py           # QLoRA SFT training (Unsloth or standard PEFT)
│   ├── run_sft_inference.py   # Inference + evaluation on val/test splits
│   ├── compare_models.py      # Side-by-side F1/Recall/Precision table
│   ├── ensemble_concerns.py   # Multi-model union/vote ensemble
│   ├── error_analysis.py      # Per-category P/R, failure modes (HPC + local)
│   ├── sweep_manager.py       # Generate sweep configs, log results
│   ├── run_baselines.py       # Evaluate GPT/Gemini baselines
│   └── download_specter2.py   # Cache SPECTER2 model locally
│
├── slurm/
│   ├── train_sft.sh           # SLURM training job (A40 default, A100 option)
│   ├── run_inference.sh       # SLURM inference job (supports RESUME=true)
│   ├── sweep_array.sh         # SLURM array job for hyperparameter sweeps
│   ├── sync_to_hpc.sh         # rsync: local → HPC (or --download)
│   └── setup_cayuga.sh        # One-time HPC environment setup
│
├── data/                      # SFT training data (gitignored, see below)
│   ├── sft_train.jsonl        # 701 articles, ShareGPT format
│   ├── sft_val.jsonl          # 121 articles
│   ├── sft_train_stats.json   # Category distribution, token stats
│   └── sft_val_stats.json
│
├── models/                    # Model weights (gitignored)
│   └── specter2_base/         # Local SPECTER2 cache (required for eval)
│
├── results/
│   ├── sft_eval/              # Inference outputs (gitignored)
│   │   └── *.jsonl + *.summary.json
│   ├── baseline_eval/         # Baseline evaluation outputs (gitignored)
│   └── error_analysis/        # Error analysis JSON outputs
│
└── requirements-train.txt     # pip dependencies
```

---

## Quick Start

### 1. Setup (local)

```bash
# Clone alongside peer-review-benchmark
git clone https://github.com/jang1563/BioReview_Training
cd BioReview_Training

# Download SPECTER2 model (required for evaluation)
python scripts/download_specter2.py
```

### 2. Prepare training data

Requires `peer-review-benchmark` in the sibling directory (`../peer-review-benchmark/`).

```bash
# Generate SFT JSONL from benchmark splits
python scripts/prepare_sft_data.py --splits train val

# Options:
#   --min-concerns 3          minimum concerns per article (default: 3)
#   --min-resolution-confidence 0.8  confidence filter (default: 0.8)
#   --token-budget 15000      max input tokens per article (default: 15000)
#   --preview 2               show 2 example articles
```

**Training data stats (v2, 2026-03-09):**

| Split | Articles | Avg concerns | Top categories |
|-------|----------|--------------|----------------|
| train | 701 | 6.88 | missing_exp 31.2%, writing_clarity 18.9%, interpretation 18.9% |
| val | 121 | 6.82 | missing_exp 28.3%, writing_clarity 16.4%, interpretation 18.3% |

### 3. Train locally (for testing)

```bash
# Quick 50-step test
python scripts/train_sft.py --config configs/qwen3.5_9b_qlora.yaml --max-steps 50 --no-eval

# Full training (requires GPU)
python scripts/train_sft.py --config configs/qwen3.5_9b_qlora.yaml
```

### 4. Sync and train on HPC (Cornell Cayuga)

```bash
# Upload to HPC
bash slurm/sync_to_hpc.sh

# On HPC — submit training
sbatch --gres=gpu:a40:1 --mem=48G --export=ALL,CONFIG=configs/qwen3.5_9b_qlora.yaml slurm/train_sft.sh
sbatch --gres=gpu:a100:1 --mem=80G --export=ALL,CONFIG=configs/qwen2.5_14b_qlora.yaml slurm/train_sft.sh

# Download results
bash slurm/sync_to_hpc.sh --download
```

**HPC SLURM note:** Use full path `/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch` when submitting via SSH (local `sbatch` version mismatch causes protocol error).

### 5. Run inference and evaluate

```bash
# Run on HPC (requires trained model in models/)
sbatch --export=ALL,MODEL_DIR=models/qwen3.5_9b_bioreview_v1 slurm/run_inference.sh

# Resume interrupted job
sbatch --time=06:00:00 --gres=gpu:a40:1 --mem=48G \
    --export=ALL,MODEL_DIR=models/qwen3.5_9b_bioreview_v1,RESUME=true \
    slurm/run_inference.sh

# Compare models (local, after downloading results)
python scripts/compare_models.py \
    results/sft_eval/qwen3.5_9b_bioreview_v1_val.jsonl \
    results/sft_eval/qwen2.5_14b_bioreview_v1_val.jsonl
```

---

## Models Trained

### Version 1 (trained on 839-article split, 2026-03-08/09)

| Model | Config | Training time | F1 | Recall | Precision |
|-------|--------|---------------|-----|--------|-----------|
| Qwen3.5-9B-v1 | A40, 3 epochs | 883 min | 0.4248 | 0.2738 | 0.9467 |
| Qwen2.5-14B-v1 | A100, 3 epochs | 408 min | 0.3809 | 0.2375 | 0.9617 |
| Qwen3-8B-v1 | A40, 3 epochs | — | 50-step only | — | — |
| Qwen2.5-7B-v1 | A40, 3 epochs | — | 50-step only | — | — |
| DeepSeek-R1-14B-v1 | A100, 3 epochs | 409 min | inference in progress | — | — |

### Version 2 (submitted 2026-03-09, training in progress)

Training data: 701 articles, avg 6.88 concerns, updated system prompt (10-15 concerns, anti-repetition).

| Model | SLURM job | Status |
|-------|-----------|--------|
| Qwen2.5-14B-v2 | 2701664 | PENDING |
| Qwen3.5-9B-v2 | 2701665 | PENDING |

---

## Training Data Pipeline

```
peer-review-benchmark/data/splits/v3/train.jsonl
         │
         ▼ scripts/prepare_sft_data.py
         │   - Filter: resolution_confidence ≥ 0.8, no figure concerns
         │   - Filter: ≥ 3 concerns per article
         │   - Truncate to 15,000 token budget (priority: methods > results > intro > ...)
         │   - Format: ShareGPT (system / human / gpt turns)
         ▼
data/sft_train.jsonl  ──►  scripts/train_sft.py  ──►  models/<name>/
```

### System prompt (v2)

Located in `../peer-review-benchmark/bioreview_bench/baseline/reviewer.py` (`REVIEWER_SYSTEM`).

Key rules:
1. Generate **10–15** specific, actionable scientific concerns
2. Cover diverse types: design, methods, statistics, interpretation, **writing clarity**, reagent specificity
3. Do NOT generate concerns about figures
4. Do NOT repeat the same concern for multiple figures/sections/experiments

### Output format (SFT target)

```json
[
  {"text": "The statistical analysis uses t-tests...", "category": "statistical_methodology", "severity": "major"},
  {"text": "Missing negative controls for...", "category": "missing_experiment", "severity": "major"}
]
```

**Categories:** `design_flaw`, `statistical_methodology`, `missing_experiment`, `prior_art_novelty`, `writing_clarity`, `reagent_method_specificity`, `interpretation`, `other`

**Severity:** `major`, `minor`, `optional`

---

## Evaluation

Evaluation uses **SPECTER2** semantic embeddings + Hungarian algorithm matching (threshold 0.65) from `bioreview_bench.evaluate.runner`.

```bash
# After inference, summary.json is auto-generated:
cat results/sft_eval/qwen3.5_9b_bioreview_v1_val.summary.json

# Error analysis (run on HPC with SPECTER2):
python scripts/error_analysis.py \
    --models results/sft_eval/qwen3.5_9b_bioreview_v1_val.jsonl \
             results/sft_eval/qwen2.5_14b_bioreview_v1_val.jsonl \
    --model-labels "Qwen3.5-9B" "Qwen2.5-14B" \
    --splits-dir /path/to/peer-review-benchmark/data/splits/v3 \
    --output-json results/error_analysis/analysis.json

# Ensemble (union of multiple models):
python scripts/ensemble_concerns.py \
    --models results/sft_eval/qwen3.5_9b_bioreview_v1_val.jsonl \
             results/sft_eval/qwen2.5_14b_bioreview_v1_val.jsonl \
    --labels "9B" "14B" --strategy union \
    --output results/sft_eval/ensemble_union_val.jsonl \
    --evaluate --splits-dir /path/to/splits/v3
```

---

## Hyperparameter Sweeps

```bash
# Generate sweep variant configs
python scripts/sweep_manager.py generate-configs \
    --sweep configs/sweep/stage1_9b.yaml \
    --output-dir configs/sweep/stage1_9b

# Submit SLURM array job (on HPC)
sbatch --array=1-2%2 --gres=gpu:a40:1 --mem=48G \
    --export=ALL,MANIFEST=configs/sweep/stage1_9b/sweep_manifest.csv \
    slurm/sweep_array.sh

# View results
python scripts/sweep_manager.py show-results
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `scripts/prepare_sft_data.py` | Data preprocessing; `--min-resolution-confidence` controls quality filter |
| `scripts/train_sft.py` | Training; auto-detects Unsloth vs standard PEFT |
| `scripts/run_sft_inference.py` | Inference with `--tag` suffix, `--resume`, per-article logging |
| `scripts/compare_models.py` | Prints F1/R/P table from multiple JSONL files |
| `scripts/ensemble_concerns.py` | union / intersection / vote-k ensemble strategies |
| `scripts/error_analysis.py` | HPC mode (SPECTER2) + local mode (pre-computed JSON) |
| `scripts/sweep_manager.py` | `generate-configs` / `log-result` / `show-results` subcommands |
| `slurm/sync_to_hpc.sh` | rsync wrapper; `--download` flag for results |
| `../peer-review-benchmark/bioreview_bench/baseline/reviewer.py` | `REVIEWER_SYSTEM` prompt used for SFT |

---

## Dependencies

```bash
# Create conda environment
conda create -n bioreview-sft python=3.11
conda activate bioreview-sft
pip install -r requirements-train.txt

# Optional: Unsloth for faster training
pip install unsloth
```

Requires **sibling directory** `../peer-review-benchmark/` for:
- Training/val splits: `data/splits/v3/{train,val,test}.jsonl`
- Evaluation runner: `bioreview_bench.evaluate.runner`
- System prompt: `bioreview_bench/baseline/reviewer.py`
