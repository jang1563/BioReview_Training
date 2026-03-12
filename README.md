# BioReview Training

QLoRA SFT (Supervised Fine-Tuning) pipeline for training biomedical peer-review LLMs on the [peer-review-benchmark](../peer-review-benchmark/) dataset.

## Overview

Fine-tunes open-source LLMs to identify specific scientific concerns in biomedical papers, evaluated against human reviewer annotations using SPECTER2 semantic matching.

**Results (peer-review-benchmark val split, 982 articles):**

| Model | F1 | Recall | Precision | concerns/art |
|-------|-----|--------|-----------|--------------|
| GPT-4o-mini (baseline) | 0.6962 | — | — | — |
| Gemini-2.5-Flash (baseline) | 0.4489 | — | — | — |
| **Ensemble Union v2** (9B-v2+14B-v2) | **0.5831** | **0.433** | **0.891** | 6.9 |
| **Ensemble Union v1** (9B+14B) | **0.5403** | 0.385 | 0.903 | 6.1 |
| DeepSeek-R1-14B v1 (SFT) | 0.4316 | 0.280 | 0.936 | 4.3 |
| Qwen3.5-9B v1 (SFT) | 0.4248 | 0.274 | 0.947 | 4.1 |
| Qwen3.5-9B v2 (SFT) | 0.4019 | 0.255 | 0.946 | 3.8 |
| Qwen2.5-14B v1 (SFT) | 0.3809 | 0.238 | 0.962 | 3.5 |
| Qwen2.5-14B v2 (SFT) | 0.3636 | 0.225 | 0.942 | 3.4 |
| Ensemble Vote2 v1 | 0.0904 | 0.047 | 0.999 | 0.7 |

> Ensemble uses `--cluster-threshold 0.98` (near-exact dedup). Lower thresholds cause over-clustering via connected components transitivity in SPECTER2 space.

---

## Project Structure

```
BioReview_Training/
├── configs/                   # Training configurations
│   ├── qwen3.5_9b_qlora.yaml       # Qwen3.5-9B QLoRA (A100 80G — A40 OOM at seq=16384)
│   ├── qwen2.5_14b_qlora.yaml      # Qwen2.5-14B QLoRA (A100 80G)
│   ├── deepseek_r1_14b_qlora.yaml  # DeepSeek-R1-14B QLoRA (A100 80G)
│   └── sweep/                      # Hyperparameter sweep configs
│       ├── stage1_9b.yaml / stage1_14b.yaml
│       └── stage1_9b/ stage1_14b/  # Generated variant YAMLs
│
├── scripts/
│   ├── prepare_sft_data.py    # Convert benchmark splits → ShareGPT JSONL
│   ├── train_sft.py           # QLoRA SFT training (Unsloth or standard PEFT)
│   ├── run_sft_inference.py   # Inference + evaluation on val/test splits
│   ├── compare_models.py      # Side-by-side F1/Recall/Precision table
│   ├── ensemble_concerns.py   # Multi-model union/vote ensemble
│   ├── reevaluate_ensemble.py # Re-run ensemble eval with SPECTER2 on HPC
│   ├── error_analysis.py      # Per-category P/R, failure modes (HPC + local)
│   ├── sweep_manager.py       # Generate sweep configs, log results
│   ├── run_baselines.py       # Evaluate GPT/Gemini baselines
│   └── download_specter2.py   # Cache SPECTER2 model locally
│
├── slurm/
│   ├── train_sft.sh           # SLURM training job (A100 recommended)
│   ├── run_inference.sh       # SLURM inference job (supports RESUME=true)
│   ├── sweep_array.sh         # SLURM array job for hyperparameter sweeps
│   ├── sync_to_hpc.sh         # rsync: local → HPC (or --download)
│   └── setup_cayuga.sh        # One-time HPC environment setup
│
├── data/                      # SFT training data (gitignored)
│   ├── sft_train.jsonl        # 701 articles, ShareGPT format (v2)
│   ├── sft_val.jsonl          # 121 articles
│   └── sft_train_stats.json   # Category distribution stats
│
├── models/                    # Model weights (gitignored)
│   └── specter2_base/         # Local SPECTER2 cache (required for evaluation)
│
├── results/
│   ├── sft_eval/              # Inference outputs + summary.json (gitignored)
│   ├── baseline_eval/         # Baseline evaluation outputs (gitignored)
│   └── lessons_learned_*.md   # Per-iteration lessons
│
└── requirements-train.txt
```

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/jang1563/BioReview_Training
cd BioReview_Training

# Cache SPECTER2 locally (required for evaluation — Jaccard fallback gives wrong metrics)
python scripts/download_specter2.py
```

### 2. Prepare training data

Requires `peer-review-benchmark/` in the sibling directory.

```bash
python scripts/prepare_sft_data.py --splits train val

# Key options:
#   --min-concerns 3                    minimum concerns per article (default: 3)
#   --min-resolution-confidence 0.8     confidence filter (default: 0.8)
#   --token-budget 15000                max input tokens (default: 15000)
```

**Training data stats (v2, 2026-03-09):**

| Split | Articles | Avg concerns | writing_clarity | missing_experiment |
|-------|----------|--------------|-----------------|-------------------|
| train | 701 | 6.88 | 18.9% | 31.2% |
| val | 121 | 6.82 | 16.4% | 28.3% |

### 3. Train on HPC (Cornell Cayuga)

```bash
# Sync to HPC
bash slurm/sync_to_hpc.sh

# Submit training (use A100 for all models — A40 OOM at max_seq_length=16384)
/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch \
    --gres=gpu:a100:1 --mem=80G \
    --export=ALL,CONFIG=configs/qwen2.5_14b_qlora.yaml \
    slurm/train_sft.sh

/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch \
    --gres=gpu:a100:1 --mem=80G \
    --export=ALL,CONFIG=configs/qwen3.5_9b_qlora.yaml,PYTORCH_ALLOC_CONF=expandable_segments:True \
    slurm/train_sft.sh
```

> **Note:** Use the full sbatch path when submitting via SSH. `--export` must include `ALL` or PATH/conda will be lost.

### 4. Run inference and evaluate

```bash
# Submit inference (on HPC)
/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch \
    --export=ALL,MODEL_DIR=models/qwen2.5_14b_bioreview_v1,SPLIT=val \
    slurm/run_inference.sh

# Resume interrupted inference
/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch \
    --time=06:00:00 --gres=gpu:a40:1 --mem=48G \
    --export=ALL,MODEL_DIR=models/qwen3.5_9b_bioreview_v1,RESUME=true \
    slurm/run_inference.sh

# Download results and compare locally
bash slurm/sync_to_hpc.sh --download

python scripts/compare_models.py \
    results/sft_eval/qwen3.5_9b_bioreview_v1_val.summary.json \
    results/sft_eval/qwen2.5_14b_bioreview_v1_val.summary.json
```

### 5. Ensemble

```bash
python scripts/ensemble_concerns.py \
    --models results/sft_eval/qwen3.5_9b_bioreview_v1_val.jsonl \
             results/sft_eval/qwen2.5_14b_bioreview_v1_val.jsonl \
    --labels "9B" "14B" --strategy union \
    --output results/sft_eval/ensemble_union_val.jsonl \
    --evaluate --splits-dir /path/to/peer-review-benchmark/data/splits/v3

# If --evaluate was run without SPECTER2 (gave wrong metrics), re-run:
python scripts/reevaluate_ensemble.py  # on HPC where SPECTER2 is accessible
```

### 6. Baselines

`scripts/run_baselines.py` needs provider SDKs in addition to the training stack:

```bash
pip install openai anthropic google-generativeai
```

---

## Models

### v1 — trained on 839-article split (2026-03-08/09)

| Model | GPU | Train time | F1 | Recall | Precision |
|-------|-----|------------|----|--------|-----------|
| Qwen3.5-9B-v1 | A100 | 883 min | 0.4248 | 0.2738 | 0.9467 |
| Qwen2.5-14B-v1 | A100 | 408 min | 0.3809 | 0.2375 | 0.9617 |
| DeepSeek-R1-14B-v1 | A100 | 409 min | 0.4316 | 0.2804 | 0.9364 |
| **Ensemble Union v1** | — | — | **0.5403** | **0.385** | **0.903** |

### v2 — 701-article split, updated system prompt (2026-03-12)

Improvements: 10–15 concerns (was 5–15), anti-repetition rule, full v3 split for more writing_clarity examples.

| Model | GPU | Train time | F1 | Recall | Precision |
|-------|-----|------------|----|--------|-----------|
| Qwen3.5-9B-v2 | A100 | 731 min | 0.4019 | 0.2551 | 0.9458 |
| Qwen2.5-14B-v2 | A100 | 342 min | 0.3636 | 0.2253 | 0.9422 |
| **Ensemble Union v2** | — | — | **0.5831** | **0.433** | **0.891** |

> These v2 job statuses refer to external HPC runs and the corresponding model directories/configs are not checked into this repository snapshot.

---

## Training Data Pipeline

```
peer-review-benchmark/data/splits/v3/train.jsonl  (4740 articles)
         │
         ▼ scripts/prepare_sft_data.py
         │   - Filter: resolution_confidence ≥ 0.8
         │   - Filter: ≥ 3 concerns per article, no figure concerns
         │   - Truncate: 15,000 token budget (methods > results > intro > ...)
         │   - Format: ShareGPT (system / human / gpt turns)
         ▼
data/sft_train.jsonl  (701 articles, avg 6.88 concerns)
         │
         ▼ scripts/train_sft.py
         ▼
models/<name>/  (LoRA adapter)
```

### System prompt (v2)

Located in `../peer-review-benchmark/bioreview_bench/baseline/reviewer.py` (`REVIEWER_SYSTEM`).

Key rules:
1. Generate **10–15** specific, actionable concerns (raised from 5–15 to improve recall)
2. Cover diverse types: design, methods, statistics, interpretation, writing clarity, reagent specificity
3. Do NOT generate concerns about figures
4. Do NOT repeat the same concern across figures/sections/experiments

### Output format

```json
[
  {"text": "The statistical analysis uses t-tests without verifying normality...",
   "category": "statistical_methodology", "severity": "major"},
  {"text": "Missing negative controls for the knockdown experiment...",
   "category": "missing_experiment", "severity": "major"}
]
```

**Categories:** `design_flaw`, `statistical_methodology`, `missing_experiment`, `prior_art_novelty`, `writing_clarity`, `reagent_method_specificity`, `interpretation`, `other`

**Severity:** `major`, `minor`, `optional`

---

## Evaluation

Uses **SPECTER2** semantic embeddings + Hungarian algorithm (threshold 0.65).

> **Critical:** SPECTER2 must be available. Without it, evaluation silently falls back to Jaccard similarity (word overlap), giving misleadingly low scores (~F1=0.03 instead of ~0.54). Always run `scripts/download_specter2.py` first, or ensure `allenai/specter2_base` is accessible from HuggingFace.

```bash
# View inference summary
cat results/sft_eval/qwen3.5_9b_bioreview_v1_val.summary.json

# Per-category error analysis (run on HPC with SPECTER2)
python scripts/error_analysis.py \
    --models results/sft_eval/qwen3.5_9b_bioreview_v1_val.jsonl \
             results/sft_eval/qwen2.5_14b_bioreview_v1_val.jsonl \
    --model-labels "Qwen3.5-9B" "Qwen2.5-14B" \
    --splits-dir /path/to/peer-review-benchmark/data/splits/v3 \
    --output-json results/error_analysis/analysis.json
```

---

## Hyperparameter Sweeps

```bash
# Generate sweep configs
python scripts/sweep_manager.py generate-configs \
    --sweep configs/sweep/stage1_9b.yaml \
    --output-dir configs/sweep/stage1_9b

# Submit SLURM array (on HPC)
sbatch --array=1-2%2 --gres=gpu:a100:1 --mem=80G \
    --export=ALL,MANIFEST=configs/sweep/stage1_9b/sweep_manifest.csv \
    slurm/sweep_array.sh

python scripts/sweep_manager.py show-results
```

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/prepare_sft_data.py` | Data preprocessing; `--min-resolution-confidence` / `--min-concerns` |
| `scripts/train_sft.py` | Training; auto-detects Unsloth vs standard PEFT |
| `scripts/run_sft_inference.py` | Inference with `--tag` suffix, `--resume`, per-article logging |
| `scripts/compare_models.py` | F1/R/P table from a summary directory or explicit `.summary.json` / `.jsonl` paths |
| `scripts/ensemble_concerns.py` | union / intersection / vote-k ensemble; requires SPECTER2 for `--evaluate` |
| `scripts/reevaluate_ensemble.py` | Re-run ensemble evaluation with SPECTER2 (use when original eval used Jaccard) |
| `scripts/error_analysis.py` | HPC mode (SPECTER2) + local mode (pre-computed JSON) |
| `scripts/sweep_manager.py` | `generate-configs` / `log-result` / `show-results` |
| `slurm/sync_to_hpc.sh` | rsync wrapper; `--download` to pull results |
| `../peer-review-benchmark/bioreview_bench/baseline/reviewer.py` | `REVIEWER_SYSTEM` prompt used for SFT |

---

## Known Issues / Lessons Learned

- **Qwen3.5-9B on A40**: OOM at step 0 with `max_seq_length=16384` (training data p99 ≈ 15K tokens). Use A100 (80GB).
- **SPECTER2 fallback**: If `sentence-transformers` unavailable, evaluation silently uses Jaccard → ~5% precision. Always verify SPECTER2 before trusting eval metrics.
- **Ensemble cluster threshold**: SPECTER2 cosine similarity for short biomedical concerns is uniformly high (0.85–0.95+). Connected-components clustering with threshold ≤ 0.90 causes transitivity chaining → all concerns merge into 1 per article. Use `--cluster-threshold 0.98` for near-exact dedup only.
- **DeepSeek-R1 output format**: Model outputs bare JSON objects `{…}, {…}]` without a leading `[`. Fixed in `parse_model_output()` step 5; use `scripts/reparse_inference.py` to retroactively fix existing JSONL files.
- **writing_clarity imbalance**: 96% of writing_clarity concerns have `resolution_confidence=0.10` (no annotator response). Lowering the confidence threshold to include them causes 80%+ writing_clarity domination. Fix: use a larger split (full v3 → 18.9% naturally).
- **14B figure repetition**: v1 model repeated the same concern template per figure ("Figure X shows dramatic differences…"). Fixed in v2 via Rule 8 in system prompt.
- **v2 individual models slightly worse than v1**: System prompt change (5–15 → 10–15 concerns) did not improve individual F1; v2 ensemble gains are higher (0.54 → 0.58) because models generate more concerns to combine.
- See `results/lessons_learned_2026-03-09.md` for full analysis.

---

## Dependencies

```bash
conda create -n bioreview-sft python=3.11
conda activate bioreview-sft
pip install -r requirements-train.txt

# Optional: Unsloth for faster training (requires trl≥0.18.2)
pip install unsloth
```

Requires **sibling directory** `../peer-review-benchmark/` for:
- Splits: `data/splits/v3/{train,val,test}.jsonl`
- Evaluation: `bioreview_bench.evaluate.runner`
- System prompt: `bioreview_bench/baseline/reviewer.py`
