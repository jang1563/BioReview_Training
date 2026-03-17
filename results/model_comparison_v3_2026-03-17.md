# Model Comparison — v3 Split (838 val articles)

Updated: 2026-03-17

**Evaluation**: SPECTER2 embedding + Hungarian matching, threshold=0.65
**Gate**: F1 >= 0.58 or Recall >= 0.45

---

## Leaderboard

| Rank | Model | F1 | Recall | Precision | Recall (major) | Tool concerns | Gate |
|---:|---|---|---|---|---|---|---|
| -- | GPT-4o-mini (baseline) | 0.696 | 0.65 | -- | -- | -- | PASS |
| 1 | **8B merged+dedup+cap20** | **0.556** | 0.413 | 0.851 | -- | 5,794 | FAIL |
| 2 | 8B dedup+cap20 | 0.554 | 0.411 | 0.851 | 0.550 | 5,774 | FAIL |
| 3 | 8B dedup+cap15 | 0.548 | 0.397 | 0.883 | 0.542 | 5,381 | FAIL |
| 4 | 8B dedup+srcap15 | 0.547 | 0.397 | 0.879 | 0.542 | 5,405 | FAIL |
| 5 | 8B dedup+cap10 | 0.520 | 0.362 | 0.920 | 0.518 | 4,712 | FAIL |
| 6 | 8B dedup only | 0.519 | 0.418 | 0.685 | 0.553 | 7,301 | FAIL |
| 7 | 8B raw | 0.457 | 0.443 | 0.473 | 0.567 | 11,195 | FAIL |

**Human GT**: 11,955 concerns | **Best gap to gate**: -0.024 F1

---

## By-Source (Best variant: dedup+cap20)

| Source | N | F1 | Recall | Precision |
|---|---|---|---|---|
| PeerJ | 31 | **0.609** | 0.469 | 0.870 |
| F1000 | 341 | **0.595** | 0.469 | 0.815 |
| eLife | 232 | 0.565 | 0.419 | 0.866 |
| PLOS | 221 | 0.491 | 0.336 | 0.912 |
| Nature | 13 | 0.330 | 0.200 | 0.941 |

---

## Per-Category (Raw model)

| Category | F1 | Recall | Precision | N human |
|---|---|---|---|---|
| interpretation | 0.631 | 0.568 | 0.709 | 1,869 |
| missing_experiment | 0.627 | 0.564 | 0.706 | 1,924 |
| prior_art_novelty | 0.593 | 0.566 | 0.623 | 919 |
| writing_clarity | 0.570 | 0.492 | 0.677 | 4,386 |
| design_flaw | 0.553 | 0.519 | 0.591 | 1,311 |
| statistical_methodology | 0.463 | 0.448 | 0.479 | 641 |
| reagent_method_specificity | 0.453 | 0.428 | 0.480 | 866 |
| other | 0.450 | 0.451 | 0.449 | 39 |

---

## Postprocessing Experiments Summary

| Experiment | Result | Conclusion |
|---|---|---|
| Dedup (exact text) | +0.062 F1 (0.457 -> 0.519) | Essential — removes 35% duplicate concerns |
| Cap20 | +0.035 F1 (0.519 -> 0.554) | Optimal cap value |
| Cap15 | F1=0.548 | Marginal vs cap20 |
| Cap10 | F1=0.520 | Too aggressive |
| Source-adaptive cap | F1=0.547 | Worse than uniform cap20 |
| Category typo fix | F1 unchanged | 74 corrections, cosmetic only |
| Reparse (10 articles) | +0.002 F1 | 1/10 recovered |
| Threshold sensitivity | F1 identical (0.50-0.75) | SPECTER2 is bimodal |

---

## Pending Models

| Model | Status | ETA |
|---|---|---|
| Qwen3.5-9B all_nonfig | Training (job 2704886, 651/1773) | ~2026-03-19 |
| 8B+9B Ensemble | Pending 9B completion | -- |

---

## Legacy Reference (NOT comparable — different val split)

> These results use the old 982-article validation split, NOT the frozen v3 split.

| Model | Split | F1 | Recall | Precision |
|---|---|---|---|---|
| Ensemble Union v2 (9B+14B) | legacy-982 | 0.583 | 0.433 | 0.891 |
| Ensemble Union v1 (14B+9B) | legacy-982 | 0.540 | 0.385 | 0.903 |
| DeepSeek-R1-14B v1 | legacy-982 | 0.432 | 0.280 | 0.936 |
| Qwen3.5-9B v2 | legacy-982 | 0.402 | 0.255 | 0.946 |
| Qwen2.5-14B v2 | legacy-982 | 0.364 | 0.225 | 0.942 |

---

## Notes

- **Dedup**: Exact text deduplication (case-insensitive, whitespace normalized)
- **Cap**: Per-article concern count limit — keeps first N concerns in generation order
- **Merged**: Reparse of 10 failed articles, 1 recovered (37 concerns)
- **SPECTER2 fix (2026-03-15)**: Weight files re-downloaded; scripts patched with existence check
- **Category typos**: 74/11,195 concerns (0.7%) had misspelled categories; corrected
- **Threshold insensitive**: Matching threshold 0.50-0.75 all produce identical F1
