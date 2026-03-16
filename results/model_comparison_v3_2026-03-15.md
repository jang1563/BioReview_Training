# Model Comparison — v3 Split (838 val articles)

Generated: 2026-03-15

**Evaluation**: SPECTER2 embedding + Hungarian matching, threshold=0.65
**Gate**: F1 ≥ 0.58 or Recall ≥ 0.45

---

## Leaderboard

| Rank | Model | F1 | Recall | Precision | Recall (major) | Tool concerns | Gate |
|---:|---|---|---|---|---|---|---|
| — | GPT-4o-mini (baseline) | 0.696 | 0.65 | — | — | — | PASS |
| 1 | **8B all_nonfig dedup+cap20** | **0.554** | 0.411 | 0.851 | 0.550 | 5,774 | FAIL |
| 2 | 8B all_nonfig dedup+cap15 | 0.548 | 0.397 | 0.883 | 0.542 | 5,381 | FAIL |
| 3 | 8B all_nonfig dedup+cap10 | 0.520 | 0.362 | 0.920 | 0.518 | 4,712 | FAIL |
| 4 | 8B all_nonfig dedup | 0.519 | 0.418 | 0.685 | 0.553 | 7,301 | FAIL |
| 5 | 8B all_nonfig dedup+srcap15 | 0.547 | 0.397 | 0.879 | 0.542 | 5,405 | FAIL |
| 6 | 8B all_nonfig raw | 0.457 | 0.443 | 0.473 | 0.567 | 11,195 | FAIL |

**Human GT**: 11,955 concerns across 838 articles

---

## By-Source (Best variant: dedup+cap20)

| Source | N | F1 | Recall | Precision | Recall (major) |
|---|---|---|---|---|---|
| PeerJ | 31 | **0.609** | 0.469 | 0.870 | 0.606 |
| F1000 | 341 | **0.595** | 0.469 | 0.815 | 0.584 |
| eLife | 232 | 0.565 | 0.419 | 0.866 | 0.572 |
| PLOS | 221 | 0.491 | 0.336 | 0.912 | 0.478 |
| Nature | 13 | 0.330 | 0.200 | 0.941 | 0.332 |

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

## Pending Models

| Model | Status | Expected |
|---|---|---|
| Qwen3.5-9B all_nonfig | Training (job 2704323) | ~2026-03-21 |
| 8B+9B Ensemble | Pending 9B completion | — |

---

## Legacy Reference (NOT comparable — different val split)

> These results use the old 982-article validation split, NOT the frozen v3 split.
> Direct comparison with v3 results is invalid due to different article composition.

| Model | Split | F1 | Recall | Precision |
|---|---|---|---|---|
| Ensemble Union v2 (9B+14B) | legacy-982 | 0.583 | 0.433 | 0.891 |
| Ensemble Union v1 (14B+9B) | legacy-982 | 0.540 | 0.385 | 0.903 |
| DeepSeek-R1-14B v1 | legacy-982 | 0.432 | 0.280 | 0.936 |
| Qwen3.5-9B v2 | legacy-982 | 0.402 | 0.255 | 0.946 |
| Qwen2.5-14B v2 | legacy-982 | 0.364 | 0.225 | 0.942 |

---

## Notes

- **Dedup**: Exact text deduplication (case-insensitive, whitespace normalized) — removes ~35% of concerns
- **Cap**: Per-article concern count limit — keeps first N concerns in generation order
- **SPECTER2 fix (2026-03-15)**: Local model re-downloaded with weights; scripts patched with weight-existence check
- **Category typos**: 75/11,195 concerns (0.7%) have misspelled category names
