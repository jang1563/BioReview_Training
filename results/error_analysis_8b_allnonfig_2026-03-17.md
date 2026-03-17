# Error Analysis: Qwen3-8B all_nonfig (v3 val, 838 articles)

**Date**: 2026-03-17
**Models analyzed**: 8B-raw (11,195 concerns) vs 8B-dedup-cap20 (5,774 concerns)
**Matching**: SPECTER2 + Hungarian, threshold=0.65

---

## 1. Failure Mode Distribution

| Mode | Count | % | Impact |
|---|---|---|---|
| ok (valid JSON) | 828 | 98.8% | -- |
| parse_fail | 9 | 1.1% | 194 GT concerns lost |
| repetition_loop | 1 | 0.1% | 28 GT concerns lost |

**Total unrecoverable**: 10 articles, 222 GT concerns (1.9% of 11,955). These set a **recall ceiling of ~0.981**.

---

## 2. Zero Recall Articles (10)

| Article | Source | GT Concerns | Mode |
|---|---|---|---|
| plos:10.1371/journal.pmed.1004280 | PLOS | 51 | parse_fail |
| plos:10.1371/journal.pcbi.1012577 | PLOS | 30 | parse_fail |
| plos:10.1371/journal.pmed.1004461 | PLOS | 28 | repetition_loop |
| f1000:10.12688/f1000research.27123.2 | F1000 | 20 | parse_fail |
| plos:10.1371/journal.pmed.1004018 | PLOS | 16 | parse_fail |
| plos:10.1371/journal.pmed.1004422 | PLOS | 16 | parse_fail |
| f1000:10.12688/f1000research.165539.1 | F1000 | 13 | parse_fail |
| elife:90875 | eLife | 13 | parse_fail |
| elife:96699 | eLife | 5 | parse_fail |
| elife:88054 | eLife | 2 | parse_fail |

**Pattern**: PLOS dominates (5/10), especially `pmed` articles (complex/long papers). Reparse recovered only 1 (plos:1004461, repetition_loop). The 9 parse_fail articles cannot produce valid JSON.

---

## 3. Over-Generation Analysis (Raw)

### Top 10 concern generators

| Article | Source | Tool | GT | P | Ratio |
|---|---|---|---|---|---|
| f1000:52532.1 | F1000 | 141 | 12 | 0.085 | 11.8x |
| f1000:73900.1 | F1000 | 127 | 16 | 0.126 | 7.9x |
| f1000:170697.1 | F1000 | 127 | 7 | 0.055 | 18.1x |
| f1000:53560.1 | F1000 | 127 | 3 | 0.024 | 42.3x |
| peerj:9428 | PeerJ | 126 | 13 | 0.103 | 9.7x |
| f1000:28455.2 | F1000 | 125 | 8 | 0.064 | 15.6x |
| f1000:17893.2 | F1000 | 123 | 26 | 0.211 | 4.7x |
| f1000:156046.2 | F1000 | 122 | 6 | 0.049 | 20.3x |
| f1000:154866.2 | F1000 | 122 | 21 | 0.172 | 5.8x |
| f1000:138153.2 | F1000 | 120 | 41 | 0.342 | 2.9x |

**Key insight**: Over-generation is almost exclusively F1000 and PeerJ (multi-reviewer articles with long full texts). These articles tend to have ALL GT concerns matched (R=1.0) but with extremely low precision.

### Over-50-concern articles by source

| Source | N articles | >50 concerns |
|---|---|---|
| eLife | 232 | 0 (max=39) |
| F1000 | 341 | 28 |
| Nature | 13 | 0 (max=6) |
| PeerJ | 31 | 5 |
| PLOS | 221 | 30 |

**Why dedup+cap20 helps**: These 63 articles averaging ~85 concerns each contribute ~5,355 excess concerns. Dedup removes duplicates, cap20 limits the rest. This shifts the precision from 0.47 to 0.85 with only modest recall loss (0.44 to 0.41).

---

## 4. Recall Distribution

| Recall range | N articles (raw) |
|---|---|
| R = 0 | 10 |
| R in (0, 0.1] | 21 |
| R in (0.1, 0.2] | 129 |
| R in (0.2, 0.3] | 94 |
| R in (0.3, 0.4] | 93 |
| R in (0.4, 0.5] | 86 |
| R in (0.5, 0.6] | 46 |
| R in (0.6, 0.7] | 44 |
| R in (0.7, 0.8] | 34 |
| R in (0.8, 0.9] | 25 |
| R in (0.9, 1.0] | 256 |

**Bimodal pattern**: 252 articles have perfect recall (R=1.0), while 160 articles have R<=0.2. The model either finds everything or very little.

---

## 5. Per-Category Recall

| Category | R (raw) | R (dedup+cap20) | Delta | N GT |
|---|---|---|---|---|
| prior_art_novelty | 0.506 | 0.482 | -0.024 | 919 |
| design_flaw | 0.486 | 0.455 | -0.031 | 1,311 |
| interpretation | 0.476 | 0.451 | -0.025 | 1,869 |
| missing_experiment | 0.472 | 0.453 | -0.019 | 1,924 |
| statistical_methodology | 0.428 | 0.387 | -0.041 | 641 |
| writing_clarity | 0.404 | 0.367 | -0.037 | 4,386 |
| other | 0.385 | 0.359 | -0.026 | 39 |
| reagent_method_specificity | 0.382 | 0.331 | -0.051 | 866 |

**Weakest**: `reagent_method_specificity` (R=0.382) and `writing_clarity` (R=0.404). These are also the categories with highest article-level sparsity in training data (see training data analysis).

**Dedup+cap20 impact**: All categories lose some recall (-0.02 to -0.05), with `reagent_method_specificity` losing the most (-0.051). This suggests reagent/method concerns are more often generated as later (post-cap20) or duplicate entries.

---

## 6. Per-Severity Recall

| Severity | R (raw) | R (dedup+cap20) | Delta | N GT |
|---|---|---|---|---|
| major | 0.476 | 0.447 | -0.029 | 7,519 |
| minor | 0.389 | 0.354 | -0.035 | 4,177 |
| optional | 0.348 | 0.293 | -0.055 | 259 |

**Major concerns found best**: 12 percentage points higher recall than minor. This is good — the model prioritizes important issues. Optional concerns are hardest (R=0.348) and lose the most from cap20.

---

## 7. Per-Source Average Concern Generation (Raw)

| Source | N | Avg Tool | Max Tool | Avg GT | Over-gen pattern |
|---|---|---|---|---|---|
| eLife | 232 | 3.0 | 39 | ~9 | Under-generates |
| Nature | 13 | 3.9 | 6 | ~7 | Under-generates |
| F1000 | 341 | 17.8 | 141 | ~15 | Over-generates |
| PeerJ | 31 | 21.3 | 126 | ~15 | Over-generates |
| PLOS | 221 | 16.8 | 117 | ~14 | Over-generates |

**Two regimes**: eLife/Nature articles get conservative outputs (3-4 concerns), while F1000/PeerJ/PLOS get 17-21. This matches training data bias — 93% of training is eLife, so the model defaults to eLife-like behavior and over-generates when encountering other formats.

---

## 8. Summary of Key Findings

### Bottleneck: Recall, not precision

- After dedup+cap20, precision is 0.85 (excellent)
- Recall is 0.41 — the model misses 59% of GT concerns
- 160 articles (19%) have R<=0.2

### Three failure classes

1. **Parse failure** (10 articles, 1.2%): Model cannot produce valid JSON. Ceiling impact: -1.9% recall
2. **Under-generation** (eLife/Nature): Conservative outputs miss concerns. Model learned eLife's concise style
3. **Over-generation** (F1000/PLOS/PeerJ): 63 articles produce 50+ concerns, many are duplicates. Dedup+cap fixes most of this

### Category gaps

- `reagent_method_specificity`: Worst recall (0.382), loses most from cap (-0.051)
- `statistical_methodology`: Second worst (0.428), sparse in training data
- Both categories are absent in 60%+ of training articles — the model rarely learns to generate them

### Severity prioritization (positive)

- Model correctly prioritizes major concerns (R=0.476) over minor (R=0.389) and optional (R=0.348)

---

## 9. Implications for 9B Model

1. **Parse failures**: 9B (Mamba-hybrid) may have different parse patterns. Monitor this closely
2. **Category sparsity**: Same training data, so likely same weak categories. Consider category-weighted loss if 9B also struggles
3. **Under-generation on eLife**: May need post-processing to encourage more concerns for eLife articles
4. **Over-generation on F1000/PLOS**: Dedup+cap20 is the proven fix; apply same pipeline to 9B
5. **Ensemble potential**: If 9B has different error patterns (different articles at zero/low recall), union ensemble could improve recall significantly
