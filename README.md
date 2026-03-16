# BioReview Training

QLoRA SFT (Supervised Fine-Tuning) pipeline for training biomedical peer-review LLMs on the [peer-review-benchmark](../peer-review-benchmark/) dataset.

## Overview

Fine-tunes open-source LLMs to identify specific scientific concerns in biomedical papers, evaluated against human reviewer annotations using SPECTER2 semantic matching.

**Leaderboard (peer-review-benchmark val split, 982 articles, legacy v3):**

| Rank | Model | F1 | Recall | Precision | Recall (major) |
|---:|-------|---:|--------|-----------|----------------|
| 1 | GPT-4o-mini (baseline) | **0.6962** | 0.647 | 0.753 | 0.803 |
| 2 | **Ensemble Union v2** (9B-v2+14B-v2) | **0.5831** | 0.433 | 0.891 | 0.634 |
| 3 | Ensemble Union v1 (9B+14B) | 0.5403 | 0.385 | 0.903 | 0.581 |
| 4 | DeepSeek-R1-14B v1 (SFT) | 0.4316 | 0.280 | 0.936 | 0.450 |
| 5 | Qwen3.5-9B v1 (SFT) | 0.4248 | 0.274 | 0.947 | 0.452 |
| 6 | Qwen3.5-9B v2 (SFT) | 0.4019 | 0.255 | 0.946 | 0.426 |
| 7 | Qwen2.5-14B v1 (SFT) | 0.3809 | 0.238 | 0.962 | 0.402 |
| 8 | Qwen2.5-14B v2 (SFT) | 0.3636 | 0.225 | 0.942 | 0.390 |
| 9 | Ensemble Vote2 v1 | 0.0904 | 0.047 | 0.999 | 0.090 |
| 10 | Qwen7B v1 (SFT) | 0.0043 | 0.002 | 0.455 | 0.002 |
| 11 | Qwen3-8B v1 (SFT) | 0.0029 | 0.001 | 0.714 | 0.002 |

> Ensemble uses `--cluster-threshold 0.98` (near-exact dedup). Lower thresholds cause over-clustering via connected components transitivity in SPECTER2 space.

**Key finding:** Precision is not the bottleneck (~89–96%). The main gap vs GPT-4o-mini is **recall** (0.27–0.43 vs 0.65). Root cause: training data was filtered to high-confidence concerns (93% eLife), misaligned with the benchmark evaluation target (all 5 journal sources, all non-figure concerns).

---

## Current Status (2026-03-13)

### Phase 0: Experimental contract frozen

- **Benchmark reference split** (current v3): 4,740 train / 838 val / 981 test
- Primary metric: `f1_micro`; secondary: `recall_overall`, `recall_major`
- Hard rule: do not compare new runs against legacy 982-article validation runs without labeling them as legacy

### Phase 1: Task-aligned corpora rebuilt

Two new corpora built from current v3 split:

| Corpus | Purpose | Train | Val | Avg concerns | Source coverage | Truncation |
|--------|---------|------:|----:|-------------:|-----------------|------------|
| **A: All non-figure** | Match benchmark target | 4,734 | 835 | 14.1–14.3 | All 5 sources | 52% |
| **B: High-confidence** | Curriculum warm-start | 700 | 118 | 6.7–6.9 | 93% eLife/Nature | 81% |

Corpus A restores all 5 sources (eLife, F1000, PLOS, PeerJ, Nature) and aligns concern density to evaluation-time ground truth.

### Phase 2: 9B experiment batch (in progress)

Three planned experiments on Qwen3.5-9B:
1. **Direct retrain** on Corpus A (all non-figure concerns)
2. **Curriculum retrain** (Corpus B stage 1 → Corpus A stage 2)
3. **Prompt-locked rerun** with frozen reviewer prompt

**Checkpoint probe results (Corpus A, 50-article probe):**

| Checkpoint | F1 | Recall | Precision | Model concerns |
|-----------|---:|-------:|----------:|---------------:|
| 100 | 0.0132 | 0.019 | 0.010 | 1300 |
| **200** | **0.0257** | **0.030** | **0.023** | **888** |
| 300 | 0.0221 | 0.025 | 0.020 | 868 |

Checkpoint-200 is the leading early candidate. Full validation eval submitted.

**Success gates:**
- Green: `recall ≥ 0.45` OR `f1 ≥ 0.58`
- Strong green: `recall ≥ 0.50` AND `precision ≥ 0.80`
- Stop: if recall stays < 0.35 → move to Phase 3 (pipeline changes)

### Phase 3–4 (contingency)

- **Phase 3A**: Section-wise inference (methods/results/discussion separately → merge)
- **Phase 3B**: Teacher distillation from GPT-4o-mini
- **Phase 4**: Scale to 14B only after 9B gates pass

See `results/next_steps_plan_2026-03-12.md` for full plan.

---

## Project Structure

```
BioReview_Training/
├── configs/                   # Training configurations
│   ├── qwen3.5_9b_qlora.yaml       # Qwen3.5-9B QLoRA (A100 80G)
│   ├── qwen3.5_9b_all_nonfig.yaml  # 9B on Corpus A (task-aligned)
│   ├── qwen3.5_9b_hi_conf.yaml     # 9B on Corpus B (curriculum)
│   ├── qwen2.5_14b_qlora.yaml      # Qwen2.5-14B QLoRA (A100 80G)
│   ├── deepseek_r1_14b_qlora.yaml  # DeepSeek-R1-14B QLoRA (A100 80G)
│   ├── qwen3_8b_qlora.yaml         # Qwen3-8B QLoRA
│   ├── qwen3_8b_all_nonfig.yaml    # 8B on Corpus A
│   ├── qwen3_8b_hi_conf.yaml       # 8B on Corpus B
│   ├── qwen7b_qlora.yaml           # Qwen7B QLoRA
│   └── sweep/                      # Hyperparameter sweep configs
│       ├── stage1_9b.yaml / stage1_14b.yaml
│       └── stage1_9b/ stage1_14b/  # Generated variant YAMLs
│
├── scripts/
│   ├── prepare_sft_data.py    # Convert benchmark splits → ShareGPT JSONL
│   ├── train_sft.py           # QLoRA SFT training (Unsloth or standard PEFT)
│   ├── run_sft_inference.py   # Inference + evaluation on val/test splits
│   ├── compare_models.py      # Side-by-side F1/Recall/Precision table
│   ├── generate_comparison_report.py  # Automated leaderboard report
│   ├── ensemble_concerns.py   # Multi-model union/vote ensemble
│   ├── reevaluate_ensemble.py # Re-run ensemble eval with SPECTER2 on HPC
│   ├── error_analysis.py      # Per-category P/R, failure modes (HPC + local)
│   ├── evaluate_by_source.py  # Per-source evaluation (eLife, Nature, PLOS, etc.)
│   ├── compare_step_probes.py # Checkpoint probe comparison
│   ├── reparse_inference.py   # Re-parse JSONL with updated parser (DeepSeek-R1 fix)
│   ├── postprocess_inference_output.py  # Post-inference output processing
│   ├── refresh_sft_summaries.py         # Refresh summary statistics
│   ├── audit_corpus_truncation.py       # Validate token truncation
│   ├── audit_output_split_alignment.py  # Verify split alignment across versions
│   ├── inspect_article_compare.py       # Detailed article-level inspection
│   ├── sweep_manager.py       # Generate sweep configs, log results
│   ├── run_baselines.py       # Evaluate GPT/Gemini baselines
│   ├── download_specter2.py   # Cache SPECTER2 model locally
│   └── build_phase1_corpora.sh  # Build Corpus A & B from v3 splits
│
├── slurm/
│   ├── train_sft.sh           # SLURM training job (A100 recommended)
│   ├── run_inference.sh       # SLURM inference job (supports RESUME=true)
│   ├── sweep_array.sh         # SLURM array job for hyperparameter sweeps
│   ├── submit_checkpoint_probe.sh       # Probe specific checkpoints
│   ├── submit_checkpoint_probe_when_ready.sh  # Auto-submit after training
│   ├── submit_final_eval.sh   # Final evaluation job
│   ├── submit_source_eval.sh  # Submit per-source evaluation
│   ├── run_source_eval.sh     # Per-source evaluation script
│   ├── sync_to_hpc.sh         # rsync: local → HPC (or --download)
│   └── setup_cayuga.sh        # One-time HPC environment setup
│
├── data/                      # SFT training data (gitignored)
│   ├── sft_train.jsonl        # 701 articles, ShareGPT format (v2, high-conf)
│   ├── sft_val.jsonl          # 121 articles
│   ├── sft_train_stats.json   # Category distribution stats
│   ├── corpus_all_nonfig/     # Corpus A: all non-figure concerns (4,734 train)
│   └── corpus_hi_conf/        # Corpus B: high-confidence subset (700 train)
│
├── models/                    # Model weights (gitignored)
│   └── specter2_base/         # Local SPECTER2 cache (required for evaluation)
│
├── results/
│   ├── sft_eval/              # Inference outputs + summary.json (gitignored)
│   ├── baseline_eval/         # Baseline evaluation outputs (gitignored)
│   ├── lessons_learned_*.md   # Per-iteration lessons
│   ├── next_steps_plan_2026-03-12.md       # Phase 0-4 plan
│   ├── phase0_contract_2026-03-12.md       # Frozen benchmark contract
│   ├── phase1_corpus_summary_2026-03-12.md # Corpus A/B statistics
│   ├── step_probe_progress_2026-03-12.md   # Checkpoint probe analysis
│   ├── checkpoint_probe_comparison_2026-03-13.md  # Probe comparison table
│   └── model_comparison_latest_2026-03-12.md      # Full leaderboard
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
# Corpus A: task-aligned (all non-figure concerns, all sources)
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

**Corpus statistics (v3 split, 2026-03-12):**

| Corpus | Split | Articles | Avg concerns | Source coverage |
|--------|-------|----------|--------------|----------------|
| A: All non-figure | train | 4,734 | 14.10 | eLife 1304, F1000 1933, PLOS 1255, PeerJ 176, Nature 66 |
| A: All non-figure | val | 835 | 14.27 | eLife 232, F1000 341, PLOS 221, PeerJ 31, Nature 10 |
| B: High-confidence | train | 700 | 6.87 | eLife 653, Nature 46, PLOS 1 |
| B: High-confidence | val | 118 | 6.74 | eLife 110, Nature 8 |

### 3. Train on HPC (Cornell Cayuga)

```bash
# Sync to HPC
bash slurm/sync_to_hpc.sh

# Submit training (use A100 for all models — A40 OOM at max_seq_length=16384)
/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch \
    --gres=gpu:a100:1 --mem=80G \
    --export=ALL,CONFIG=configs/qwen3.5_9b_all_nonfig.yaml \
    slurm/train_sft.sh

# Curriculum: train on Corpus B first, then resume on Corpus A
/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch \
    --gres=gpu:a100:1 --mem=80G \
    --export=ALL,CONFIG=configs/qwen3.5_9b_hi_conf.yaml \
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

# Checkpoint probing (submit probe on specific checkpoint)
bash slurm/submit_checkpoint_probe.sh

# Download results and compare locally
bash slurm/sync_to_hpc.sh --download

python scripts/compare_models.py \
    results/sft_eval/qwen3.5_9b_bioreview_v1_val.summary.json \
    results/sft_eval/qwen2.5_14b_bioreview_v1_val.summary.json

# Generate full leaderboard report
python scripts/generate_comparison_report.py
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

### Exploratory (not competitive)

| Model | F1 | Notes |
|-------|----|-------|
| Qwen7B v1 | 0.0043 | Near-zero recall |
| Qwen3-8B v1 | 0.0029 | Near-zero recall |

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

# Per-source breakdown (eLife, F1000, PLOS, PeerJ, Nature)
python scripts/evaluate_by_source.py

# Compare checkpoint probes
python scripts/compare_step_probes.py
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
| `scripts/prepare_sft_data.py` | Data preprocessing; `--min-resolution-confidence` / `--min-concerns` / `--drop-title-only` |
| `scripts/train_sft.py` | Training; auto-detects Unsloth vs standard PEFT |
| `scripts/run_sft_inference.py` | Inference with `--tag` suffix, `--resume`, per-article logging |
| `scripts/compare_models.py` | F1/R/P table from `.summary.json` / `.jsonl` paths |
| `scripts/generate_comparison_report.py` | Full leaderboard with gate status |
| `scripts/ensemble_concerns.py` | union / intersection / vote-k ensemble; requires SPECTER2 for `--evaluate` |
| `scripts/reevaluate_ensemble.py` | Re-run ensemble evaluation with SPECTER2 (use when original eval used Jaccard) |
| `scripts/error_analysis.py` | HPC mode (SPECTER2) + local mode (pre-computed JSON) |
| `scripts/evaluate_by_source.py` | Per-journal-source evaluation breakdown |
| `scripts/compare_step_probes.py` | Checkpoint probe comparison across training steps |
| `scripts/reparse_inference.py` | Re-parse existing JSONL with updated parser (DeepSeek-R1 fix) |
| `scripts/postprocess_inference_output.py` | Post-inference dedup + cap processing |
| `scripts/audit_corpus_truncation.py` | Validate token budget truncation |
| `scripts/audit_output_split_alignment.py` | Verify split version alignment |
| `scripts/sweep_manager.py` | `generate-configs` / `log-result` / `show-results` |
| `slurm/sync_to_hpc.sh` | rsync wrapper; `--download` to pull results |
| `slurm/submit_checkpoint_probe.sh` | Submit checkpoint probing jobs |
| `../peer-review-benchmark/bioreview_bench/baseline/reviewer.py` | `REVIEWER_SYSTEM` prompt used for SFT |

---

## Known Issues / Lessons Learned

### Data–Task Misalignment (primary issue)
- v1/v2 training used `resolution_confidence ≥ 0.8` filter → 93% eLife data, avg 6.9 concerns/article
- Benchmark evaluates all non-figure concerns → avg 14.2 concerns/article across all 5 sources
- **Fix**: Corpus A (all non-figure, conf ≥ 0.0) restores source balance and concern density

### Model Behavior
- **Qwen3.5-9B on A40**: OOM at step 0 with `max_seq_length=16384` (training data p99 ≈ 15K tokens). Use A100 (80GB).
- **14B figure repetition**: v1 model repeated the same concern template per figure. Fixed in v2 via anti-repetition rule in system prompt.
- **v2 individual models slightly worse than v1**: System prompt change (5–15 → 10–15) did not improve individual F1; v2 ensemble gains (0.54 → 0.58) come from more concerns to combine.
- **Qwen7B and Qwen3-8B**: Near-zero F1 (< 0.01). Not competitive.

### Evaluation Pitfalls
- **SPECTER2 fallback**: If `sentence-transformers` unavailable, evaluation silently uses Jaccard → misleadingly low scores (~F1=0.03). Always verify SPECTER2 before trusting metrics.
- **Ensemble cluster threshold**: SPECTER2 cosine similarity for short biomedical concerns is uniformly high (0.85–0.95+). Connected-components with threshold ≤ 0.90 causes transitivity chaining → all concerns merge into 1 per article. Use `--cluster-threshold 0.98`.

### Output Format
- **DeepSeek-R1 output format**: Model outputs bare JSON `{…}, {…}]` without leading `[`. Fixed in `parse_model_output()` step 5; use `scripts/reparse_inference.py` for existing files.

### Data Quality
- **writing_clarity imbalance**: 96% of writing_clarity concerns have `resolution_confidence=0.10` (no annotator response). Lowering confidence threshold causes 80%+ writing_clarity domination. Fix: Corpus A uses full v3 → 18.9% naturally.
- **Legacy split confusion**: v1/v2 results were evaluated on legacy 982-article val split. Current frozen v3 has 838 val articles. Do not compare across splits without labeling.

See `results/lessons_learned_2026-03-09.md` and `results/next_steps_plan_2026-03-12.md` for full analysis.

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
