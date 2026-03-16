# BioReview Eval Comparison

- Generated: 2026-03-12 20:27:00Z
- Baseline summary: `/Users/jak4013/Dropbox/Bioinformatics/Claude/BioReview_Training/results/baseline_eval/baseline_summary_val_20260304T022023Z.json`
- SFT source: `/Users/jak4013/Dropbox/Bioinformatics/Claude/BioReview_Training/results/sft_eval`
- Reference baseline: `gpt-4o-mini (baseline)`
- Gate: `f1_micro >= 0.58` and `recall_overall >= 0.45`

## Snapshot

- Best overall: `gpt-4o-mini (baseline)` (0.6962 F1)
- Best SFT: `ensemble_union [ensemble_union_v2_val]` (0.5831 F1, -0.1131 vs ref)
- SFT models meeting gate: none

## Leaderboard

| Rank | Model | Kind | F1 | Recall | Precision | Recall (major) | n_articles | dF1 vs ref | Gate |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | gpt-4o-mini (baseline) | baseline | 0.6962 | 0.6472 | 0.7531 | 0.8025 | 982 | +0.0000 | pass |
| 2 | ensemble_union [ensemble_union_v2_val] | sft | 0.5831 | 0.4332 | 0.8915 | 0.6343 | 982 | -0.1131 | hold |
| 3 | ensemble_union [ensemble_union_v1_val] | sft | 0.5403 | 0.3854 | 0.9034 | 0.5810 | 982 | -0.1559 | hold |
| 4 | deepseek_r1_14b_bioreview_v1 | sft | 0.4316 | 0.2804 | 0.9364 | 0.4497 | 982 | -0.2646 | hold |
| 5 | qwen3.5_9b_bioreview_v1 | sft | 0.4248 | 0.2738 | 0.9467 | 0.4523 | 982 | -0.2713 | hold |
| 6 | qwen3.5_9b_bioreview_v2 | sft | 0.4019 | 0.2551 | 0.9458 | 0.4260 | 982 | -0.2943 | hold |
| 7 | qwen2.5_14b_bioreview_v1 | sft | 0.3809 | 0.2375 | 0.9617 | 0.4018 | 982 | -0.3152 | hold |
| 8 | qwen2.5_14b_bioreview_v2 | sft | 0.3636 | 0.2253 | 0.9422 | 0.3903 | 982 | -0.3326 | hold |
| 9 | ensemble_vote2 | sft | 0.0904 | 0.0474 | 0.9985 | 0.0900 | 982 | -0.6057 | hold |
| 10 | qwen7b_bioreview_v1 | sft | 0.0043 | 0.0021 | 0.4545 | 0.0021 | 982 | -0.6919 | hold |
| 11 | qwen3_8b_bioreview_v1 | sft | 0.0029 | 0.0014 | 0.7143 | 0.0017 | 982 | -0.6933 | hold |

## Sources

- Baseline rows loaded from `baseline_summary_val_20260304T022023Z.json`
- `ensemble_union [ensemble_union_v2_val]`: `ensemble_union_v2_val.summary.json`
- `ensemble_union [ensemble_union_v1_val]`: `ensemble_union_v1_val.summary.json`
- `deepseek_r1_14b_bioreview_v1`: `deepseek_r1_14b_bioreview_v1_val.summary.json`
- `qwen3.5_9b_bioreview_v1`: `qwen3.5_9b_bioreview_v1_val.summary.json`
- `qwen3.5_9b_bioreview_v2`: `qwen3.5_9b_bioreview_v2_val.summary.json`
- `qwen2.5_14b_bioreview_v1`: `qwen2.5_14b_bioreview_v1_val.summary.json`
- `qwen2.5_14b_bioreview_v2`: `qwen2.5_14b_bioreview_v2_val.summary.json`
- `ensemble_vote2`: `ensemble_vote2_v1_val.summary.json`
- `qwen7b_bioreview_v1`: `qwen7b_bioreview_v1_val.summary.json`
- `qwen3_8b_bioreview_v1`: `qwen3_8b_bioreview_v1_val.summary.json`
