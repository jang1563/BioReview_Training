# BioReview Training

QLoRA SFT (Supervised Fine-Tuning) pipeline for training biomedical peer-review LLMs on the [peer-review-benchmark](https://github.com/jang1563/peer-review-benchmark) dataset.

## Overview

Fine-tunes open-source LLMs to identify specific scientific concerns in biomedical papers, evaluated against human reviewer annotations using SPECTER2 semantic matching + Hungarian algorithm.

**Leaderboard (peer-review-benchmark v3 test split, 981 articles):**

| Rank | Model | F1 | Recall | Precision | Recall (major) | Gate |
|---:|-------|---:|-------:|----------:|---------------:|------|
| -- | GPT-4o-mini (baseline) | 0.696* | 0.647* | 0.753* | — | PASS |
| 1 | **8B+9B Ensemble** (union, dedup+cap20) | **0.704** | **0.695** | 0.713 | **0.814** | **PASS** |
| 2 | **Qwen3.5-9B** (SFT, dedup+cap20) | **0.621** | 0.498 | 0.827 | 0.638 | **PASS** |
| 3 | Qwen3-8B (SFT, dedup+cap20) | 0.557 | 0.409 | 0.871 | 0.548 | FAIL |

> *\*GPT-4o-mini baseline evaluated on val split only. Test results pending.*

**Val/Test consistency (no overfitting):**

| Model | Val F1 | Test F1 | Delta |
|-------|-------:|--------:|------:|
| 8B+9B Ensemble | 0.694 | 0.704 | +0.010 |
| Qwen3.5-9B | 0.625 | 0.621 | -0.004 |
| Qwen3-8B | 0.556 | 0.557 | +0.001 |

**Key findings:**
- **8B+9B ensemble exceeds GPT-4o-mini** (F1=0.704 vs 0.696) with **81.4% recall on major concerns**
- **Data–task alignment is critical** — Corpus A (all non-figure) improved single-model F1 from 0.43 → 0.63
- **Models are complementary** — only 8.4% of combined concerns overlap; union ensemble captures both
- **Dedup+cap20 is essential** — removes ~50% of raw output, improving F1 by +0.11 (9B: 0.514 → 0.625)
- **9B precision is exceptional** — 0.827 precision after postprocessing vs GPT-4o-mini's 0.753

**Success gates:** F1 >= 0.58 or Recall >= 0.45

---

## Current Status (2026-03-29)

### Phase 2: Task-aligned corpus training — COMPLETE ✓

**Test set evaluation complete.** No overfitting — val and test metrics within 0.01.

| Model | Corpus | Training | Val F1 | Test F1 | Gate |
|-------|--------|----------|-------:|--------:|------|
| **8B+9B Ensemble** | — | — | 0.694 | **0.704** | **PASS** |
| Qwen3.5-9B all_nonfig | A (4,734 articles) | 1,773 steps, 3 epochs, 35h | 0.625 | **0.621** | **PASS** |
| Qwen3-8B all_nonfig | A (4,734 articles) | 1,773 steps, 3 epochs, ~18h | 0.556 | 0.557 | FAIL |

**8B+9B Ensemble test set (union, cluster-threshold=0.98):**

| Metric | Val | Test |
|--------|----:|-----:|
| F1 | 0.694 | **0.704** |
| Recall | 0.695 | 0.695 |
| Precision | 0.692 | 0.713 |
| Recall (major concerns) | 0.811 | **0.814** |

**Ensemble test set per-category breakdown:**

| Category | GT count | Recall | Precision | F1 |
|----------|---------|-------:|----------:|---:|
| interpretation | 2,240 | 0.842 | 0.814 | **0.828** |
| missing_experiment | 2,146 | 0.845 | 0.791 | **0.817** |
| design_flaw | 1,431 | 0.805 | 0.793 | **0.799** |
| prior_art_novelty | 1,113 | 0.829 | 0.771 | **0.799** |
| writing_clarity | 5,172 | 0.747 | 0.801 | 0.774 |
| reagent_method_specificity | 1,042 | 0.719 | 0.727 | 0.723 |
| statistical_methodology | 764 | 0.709 | 0.705 | 0.707 |
| other | 39 | 0.625 | 0.588 | 0.606 |

**Test set by-source breakdown (dedup+cap20):**

| Source | N | 8B F1 | 9B F1 | Ensemble F1 | Ensemble Recall |
|--------|--:|------:|------:|------------:|----------------:|
| eLife | 273 | 0.606 | **0.680** | **0.737** | 0.813 |
| PLOS | 260 | 0.524 | **0.600** | **0.721** | 0.654 |
| F1000 | 403 | 0.586 | **0.636** | **0.699** | 0.725 |
| PeerJ | 37 | 0.464 | **0.551** | **0.643** | 0.573 |
| Nature | 8 | 0.224 | **0.390** | **0.455** | 0.303 |

> Nature remains weakest (8 test articles, 66/4,734 train = 1.4%). Ensemble precision on Nature is 0.913.

### Phases 0–1 (complete)

- **Phase 0**: Experimental contract frozen — v3 split (4,740/838/981), SPECTER2 matching
- **Phase 1**: Task-aligned corpora rebuilt — Corpus A (all non-figure, 4,734 train) and Corpus B (high-confidence, 700 train)

### Phase 3 (planned)

- **Ensemble improvement**: Add 3rd model (different architecture) for diversity; explore weighted merging
- **Inference speed**: Investigate vLLM/SGLang or fix flash-linear-attention for 9B (~281s → target <60s/article)
- **Scale to 14B+**: Larger single model to close gap with ensemble

---

## Project Structure

```
BioReview_Training/
├── configs/                   # Training configurations
│   ├── qwen3_8b_all_nonfig.yaml    # 8B on Corpus A (task-aligned)
│   ├── qwen3.5_9b_all_nonfig.yaml  # 9B on Corpus A
│   ├── qwen3.5_9b_all_nonfig_fast.yaml  # 9B fast variant (batch=2, grad_accum=4)
│   ├── qwen3.5_9b_hi_conf.yaml     # 9B on Corpus B (curriculum)
│   ├── qwen2.5_14b_qlora.yaml      # 14B QLoRA
│   ├── deepseek_r1_14b_qlora.yaml  # DeepSeek-R1-14B QLoRA
│   └── sweep/                      # Hyperparameter sweep configs
│
├── scripts/
│   ├── prepare_sft_data.py    # Convert benchmark splits → ShareGPT JSONL
│   ├── train_sft.py           # QLoRA SFT training (Unsloth or standard PEFT)
│   ├── run_sft_inference.py   # Inference + evaluation on val/test splits
│   ├── postprocess_inference_output.py  # Dedup + cap postprocessing
│   ├── ensemble_concerns.py   # Multi-model union/vote ensemble
│   ├── reevaluate_ensemble.py # Re-run ensemble eval with SPECTER2
│   ├── error_analysis.py      # Per-category P/R, failure modes
│   ├── evaluate_by_source.py  # Per-source evaluation breakdown
│   ├── compare_models.py      # Side-by-side F1/Recall/Precision table
│   ├── generate_comparison_report.py  # Automated leaderboard report
│   ├── compare_step_probes.py # Checkpoint probe comparison
│   ├── reparse_inference.py   # Re-parse JSONL with updated parser
│   ├── sweep_manager.py       # Generate sweep configs, log results
│   ├── run_baselines.py       # Evaluate GPT/Gemini baselines
│   ├── download_specter2.py   # Cache SPECTER2 model locally
│   └── build_phase1_corpora.sh  # Build Corpus A & B from v3 splits
│
├── slurm/
│   ├── train_sft.sh           # SLURM training job
│   ├── run_inference.sh       # SLURM inference job (supports RESUME=true)
│   ├── sweep_array.sh         # SLURM array job for hyperparameter sweeps
│   ├── submit_checkpoint_probe.sh       # Probe specific checkpoints
│   ├── submit_final_eval.sh   # Final evaluation job
│   ├── submit_source_eval.sh  # Submit per-source evaluation
│   ├── run_source_eval.sh     # Per-source evaluation script
│   ├── run_test_eval.sh       # Full test-set evaluation pipeline
│   ├── sync_to_hpc.sh         # rsync: local → HPC (or --download)
│   └── setup_cayuga.sh        # One-time HPC environment setup
│
├── data/                      # SFT training data (gitignored)
│   ├── corpus_all_nonfig/     # Corpus A: all non-figure concerns
│   │   ├── sft_train.jsonl    # 4,734 articles, ShareGPT format
│   │   └── sft_val.jsonl      # 835 articles
│   └── corpus_hi_conf/        # Corpus B: high-confidence subset
│       ├── sft_train.jsonl    # 700 articles
│       └── sft_val.jsonl      # 118 articles
│
├── models/                    # Model weights (gitignored)
│   └── specter2_base/         # Local SPECTER2 cache (required for evaluation)
│
├── results/
│   ├── sft_eval/              # Inference outputs + summaries (gitignored)
│   ├── baseline_eval/         # Baseline outputs (gitignored)
│   ├── progress_report_*.md   # Progress reports
│   ├── model_comparison_*.md  # Leaderboard snapshots
│   ├── error_analysis_*.md    # Error analysis reports
│   ├── lessons_learned_*.md   # Per-iteration lessons
│   └── next_steps_plan_*.md   # Phase plans
│
└── requirements-train.txt
```

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/jang1563/BioReview_Training
cd BioReview_Training
pip install -r requirements-train.txt

# Cache SPECTER2 locally (REQUIRED for evaluation)
# Without it, evaluation silently falls back to Jaccard → garbage scores (~F1=0.03)
python scripts/download_specter2.py
```

### 2. Prepare training data

Requires [`peer-review-benchmark/`](https://github.com/jang1563/peer-review-benchmark) in the sibling directory.

```bash
# Corpus A: task-aligned (all non-figure concerns, all 5 sources)
python scripts/prepare_sft_data.py \
  --splits train val \
  --splits-dir ../peer-review-benchmark/data/splits/v3 \
  --output-dir data/corpus_all_nonfig \
  --min-resolution-confidence 0.0 \
  --min-concerns 1 \
  --drop-title-only

# Corpus B: high-confidence subset (curriculum warm-start)
python scripts/prepare_sft_data.py \
  --splits train val \
  --splits-dir ../peer-review-benchmark/data/splits/v3 \
  --output-dir data/corpus_hi_conf \
  --min-resolution-confidence 0.8 \
  --min-concerns 3 \
  --drop-title-only
```

**Corpus statistics (v3 split):**

| Corpus | Split | Articles | Avg concerns | Source coverage |
|--------|-------|----------|--------------|----------------|
| A: All non-figure | train | 4,734 | 14.10 | eLife 1304, F1000 1933, PLOS 1255, PeerJ 176, Nature 66 |
| A: All non-figure | val | 835 | 14.27 | eLife 232, F1000 341, PLOS 221, PeerJ 31, Nature 10 |
| B: High-confidence | train | 700 | 6.87 | eLife 653, Nature 46, PLOS 1 |
| B: High-confidence | val | 118 | 6.74 | eLife 110, Nature 8 |

### 3. Train on HPC

```bash
# Sync to HPC
bash slurm/sync_to_hpc.sh

# Submit training (A100 recommended — A40 OOM at max_seq_length=16384)
/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch \
    --gres=gpu:a100:1 --mem=80G \
    --export=ALL,CONFIG=configs/qwen3_8b_all_nonfig.yaml \
    slurm/train_sft.sh
```

> **Note:** Use the full sbatch path. `--export` must include `ALL` or PATH/conda will be lost.

### 4. Run inference and evaluate

```bash
# Submit inference
/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch \
    --export=ALL,MODEL_DIR=models/qwen3_8b_all_nonfig_v1,SPLIT=val \
    slurm/run_inference.sh

# Resume interrupted inference
/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch \
    --time=48:00:00 --gres=gpu:a100:1 --mem=80G \
    --export=ALL,MODEL_DIR=models/qwen3.5_9b_all_nonfig_v1,RESUME=true \
    slurm/run_inference.sh

# Download results locally
bash slurm/sync_to_hpc.sh --download
```

### 5. Postprocess and evaluate

```bash
# Postprocess: dedup + cap20 (optimal pipeline)
python scripts/postprocess_inference_output.py \
    --input results/sft_eval/qwen3_8b_all_nonfig_v1_val.jsonl \
    --dedup --cap 20

# Compare models
python scripts/compare_models.py \
    results/sft_eval/model_a_val.summary.json \
    results/sft_eval/model_b_val.summary.json

# Generate full leaderboard
python scripts/generate_comparison_report.py
```

### 6. Ensemble

```bash
python scripts/ensemble_concerns.py \
    --model-a results/sft_eval/qwen3.5_9b_all_nonfig_v1_val_dedup_cap20.jsonl \
    --model-b results/sft_eval/qwen3_8b_all_nonfig_v1_val.dedup_cap20.jsonl \
    --strategy union \
    --cluster-threshold 0.98 \
    --output results/sft_eval/ensemble_8b_9b_union_val.jsonl \
    --evaluate --split val
```

> **Critical:** Use `--cluster-threshold 0.98`. Lower thresholds cause transitivity chaining in connected components, merging all concerns into one cluster.

---

## Models

### Phase 2 — Corpus A (all non-figure), v3 split

| Model | Base | GPU | Train time | Val F1 | Test F1 | Test Recall | Test Precision |
|-------|------|-----|------------|-------:|--------:|------------:|---------------:|
| **8B+9B Ensemble** (union) | — | — | — | 0.694 | **0.704** | **0.695** | 0.713 |
| **Qwen3.5-9B all_nonfig** | Qwen/Qwen3.5-9B | A100 | 35h | 0.625 | **0.621** | 0.498 | 0.827 |
| Qwen3-8B all_nonfig | Qwen/Qwen3-8B | A100 | ~18h | 0.556 | 0.557 | 0.409 | 0.871 |

> Inference speed: 8B ~48s/article, 9B ~281s/article (no flash-linear-attention on HPC).

### Phase 1 — Corpus B (high-confidence), legacy split

> *Legacy results on 982-article val split. Not directly comparable to Phase 2.*

| Model | F1 | Recall | Precision |
|-------|----|--------|-----------|
| Ensemble Union v2 (9B+14B) | **0.583** | 0.433 | 0.891 |
| Ensemble Union v1 (9B+14B) | 0.540 | 0.385 | 0.903 |
| DeepSeek-R1-14B v1 | 0.432 | 0.280 | 0.936 |
| Qwen3.5-9B v1 | 0.425 | 0.274 | 0.947 |
| Qwen3.5-9B v2 | 0.402 | 0.255 | 0.946 |
| Qwen2.5-14B v1 | 0.381 | 0.238 | 0.962 |

---

## Training Data Pipeline

```
peer-review-benchmark/data/splits/v3/train.jsonl  (4,740 articles)
         │
         ▼ scripts/prepare_sft_data.py
         │
    ┌────┴────────────────────────────────────┐
    │ Corpus A (all non-figure)               │ Corpus B (high-confidence)
    │   - conf ≥ 0.0, ≥ 1 concern            │   - conf ≥ 0.8, ≥ 3 concerns
    │   - All 5 sources                       │   - 93% eLife/Nature
    │   - Avg 14.1 concerns/art               │   - Avg 6.9 concerns/art
    │   - 4,734 train articles                │   - 700 train articles
    └────┬────────────────────────────────────┘
         │   - Truncate: 15,000 token budget (methods > results > intro > ...)
         │   - Format: ShareGPT (system / human / gpt turns)
         ▼
data/corpus_{all_nonfig,hi_conf}/sft_train.jsonl
         │
         ▼ scripts/train_sft.py  (QLoRA, Unsloth)
         ▼
models/<name>/  (LoRA adapter)
         │
         ▼ scripts/run_sft_inference.py
         ▼
results/sft_eval/<name>_val.jsonl  (raw output)
         │
         ▼ scripts/postprocess_inference_output.py  (dedup + cap)
         ▼
results/sft_eval/<name>_val_dedup_cap20.jsonl  (postprocessed)
```

### System prompt

Located in `../peer-review-benchmark/bioreview_bench/baseline/reviewer.py` (`REVIEWER_SYSTEM`).

Key rules:
1. Generate **10–15** specific, actionable concerns
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

> **Critical:** SPECTER2 must be available. Without it, evaluation silently falls back to Jaccard similarity (word overlap), giving misleadingly low scores (~F1=0.03 instead of ~0.55). Always run `scripts/download_specter2.py` first.

### Postprocessing pipeline

| Step | Effect |
|------|--------|
| Dedup (exact text, case-insensitive) | Removes 35% of concerns, +0.062 F1 |
| Cap20 (per-article limit) | +0.035 F1 on top of dedup |
| Cap15 | Marginal vs cap20 |
| Source-adaptive cap | Worse than uniform cap20 |

**Optimal pipeline:** dedup + cap20

### Error analysis

Three failure classes identified:

1. **Parse failure** (10/838 articles, 1.2%): Model cannot produce valid JSON. Recall ceiling: ~0.98
2. **Under-generation** (eLife/Nature): Conservative 3–4 concerns/article vs GT ~9–14
3. **Over-generation** (F1000/PLOS/PeerJ): 50–140 concerns/article, many duplicates. Fixed by dedup+cap20

**Weakest categories:** `reagent_method_specificity` (R=0.33) and `statistical_methodology` (R=0.39) — both sparse in training data (absent in 60%+ of articles).

**Severity prioritization** (positive): Model correctly finds major concerns (R=0.45) better than minor (R=0.35) and optional (R=0.29).

---

## Known Issues / Lessons Learned

### Data–Task Alignment (primary issue)
- Phase 1 training used `resolution_confidence ≥ 0.8` → 93% eLife, avg 6.9 concerns/article
- Benchmark evaluates all non-figure concerns → avg 14.2 concerns across all 5 sources
- **Fix (Phase 2):** Corpus A (conf ≥ 0.0) restores source balance and concern density → F1 improved from 0.43 to 0.56

### SPECTER2 Evaluation
- **Silent fallback**: Without weight files, evaluation uses Jaccard → ~F1=0.03 (garbage)
- **Matching threshold insensitive**: 0.50–0.75 all give identical F1 (bimodal distribution)
- **Ensemble cluster threshold**: Must use 0.98. Lower thresholds cause transitivity chaining

### Model-Specific Notes
- **Qwen3.5-9B**: Vision-language model (`Qwen3_5ForConditionalGeneration`). Inference requires `_unwrap_processor()` to extract text tokenizer from multimodal processor
- **Qwen3.5-9B on A40**: Mamba-hybrid architecture needs `flash-linear-attention` for fast inference. Without it, falls back to slow torch implementation (~281s/article)
- **All models on A40**: OOM at `max_seq_length=16384`. Use A100 (80GB)
- **DeepSeek-R1**: Outputs bare JSON `{…}, {…}]` without leading `[`. Fixed in parser

### Output Format
- 35% of raw model output are duplicate concerns → dedup is essential
- Over-generation concentrated in F1000/PLOS/PeerJ (multi-reviewer, long papers)

---

## Dependencies

```bash
conda create -n bioreview-sft python=3.11
conda activate bioreview-sft
pip install -r requirements-train.txt

# Optional: Unsloth for faster training (recommended)
pip install unsloth
```

Requires **sibling directory** `../peer-review-benchmark/` for:
- Splits: `data/splits/v3/{train,val,test}.jsonl`
- Evaluation: `bioreview_bench.evaluate.runner`
- System prompt: `bioreview_bench/baseline/reviewer.py`

---

## Citation

If you use this work, please cite:

```bibtex
@software{bioreview_training_2026,
  title = {BioReview Training: QLoRA SFT Pipeline for Biomedical Peer-Review LLMs},
  author = {Jang, Andrew},
  year = {2026},
  url = {https://github.com/jang1563/BioReview_Training}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
