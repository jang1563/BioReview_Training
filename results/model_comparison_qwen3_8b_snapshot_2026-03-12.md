# BioReview Eval Comparison

- Generated: 2026-03-12 20:26:10Z
- Baseline summary: `/Users/jak4013/Dropbox/Bioinformatics/Claude/BioReview_Training/results/baseline_eval/baseline_summary_val_20260304T022023Z.json`
- SFT source: `results/sft_eval/qwen3_8b_bioreview_v1_val.summary.json`
- Reference baseline: `gpt-4o-mini (baseline)`
- Gate: `f1_micro >= 0.58` and `recall_overall >= 0.45`

## Snapshot

- Best overall: `gpt-4o-mini (baseline)` (0.6962 F1)
- Best SFT: `qwen3_8b_bioreview_v1` (0.0029 F1, -0.6933 vs ref)
- SFT models meeting gate: none

## Leaderboard

| Rank | Model | Kind | F1 | Recall | Precision | Recall (major) | n_articles | dF1 vs ref | Gate |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | gpt-4o-mini (baseline) | baseline | 0.6962 | 0.6472 | 0.7531 | 0.8025 | 982 | +0.0000 | pass |
| 2 | qwen3_8b_bioreview_v1 | sft | 0.0029 | 0.0014 | 0.7143 | 0.0017 | 982 | -0.6933 | hold |

## Sources

- Baseline rows loaded from `baseline_summary_val_20260304T022023Z.json`
- `qwen3_8b_bioreview_v1`: `qwen3_8b_bioreview_v1_val.summary.json`
