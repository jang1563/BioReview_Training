# Confidence Sweep (2026-03-03)

## Setup

- Script: `scripts/prepare_sft_data.py`
- Dataset: `peer-review-benchmark/data/splits/v3/{train,val}.jsonl`
- Fixed options:
  - `token_budget=15000`
  - `min_concerns=3`
  - `include_figure_issues=False`
  - conservative token counter fallback (`chars/3`)
- Swept option:
  - `min_resolution_confidence`: `0.6`, `0.7`, `0.8`

## Size Comparison

| min_resolution_confidence | train kept | val kept | total kept |
|---|---:|---:|---:|
| 0.6 | 913 | 205 | 1118 |
| 0.7 | 843 | 186 | 1029 |
| 0.8 | 631 | 148 | 779 |

## Concern Density

| min_resolution_confidence | train avg concerns/article | val avg concerns/article |
|---|---:|---:|
| 0.6 | 6.81 | 7.21 |
| 0.7 | 6.48 | 6.92 |
| 0.8 | 6.06 | 6.30 |

## Category Snapshot (train concerns_after)

| Category | conf=0.6 | conf=0.7 | conf=0.8 |
|---|---:|---:|---:|
| missing_experiment | 2052 | 1784 | 1217 |
| interpretation | 1409 | 1192 | 743 |
| design_flaw | 927 | 819 | 551 |
| writing_clarity | 877 | 819 | 662 |
| statistical_methodology | 341 | 313 | 239 |
| prior_art_novelty | 377 | 329 | 244 |
| reagent_method_specificity | 224 | 196 | 157 |
| other | 10 | 10 | 8 |

## Recommendation

- **Default for next SFT run: `min_resolution_confidence=0.7`**
  - Keeps substantially more data than `0.8` (1029 vs 779; +32.1%).
  - Avoids the most permissive setting (`0.6`) while retaining broad category coverage.
  - Good balance for first training/evaluation loop.

- **Ablation plan**
  - Run main experiment at `0.7`.
  - Run one comparison at `0.8` for quality-focused contrast.
  - Promote `0.6` only if early training underfits due to data scarcity.
