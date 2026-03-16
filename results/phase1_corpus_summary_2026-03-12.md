# Phase 1 Corpus Summary (2026-03-12)

## Build Outputs

- `data/corpus_all_nonfig`
- `data/corpus_hi_conf`

Built from:

- `../peer-review-benchmark/data/splits/v3`

Shared preprocessing:

- `token_budget=15000`
- `drop_title_only=true`
- `include_figure_issues=false`

---

## Corpus A: All Non-Figure Concerns

### Train

- rows: `4734`
- source mix:
  - eLife: `1304`
  - F1000: `1933`
  - PLOS: `1255`
  - PeerJ: `176`
  - Nature: `66`
- avg concerns/article: `14.10`
- truncation fraction: `0.5281`

### Val

- rows: `835`
- source mix:
  - eLife: `232`
  - F1000: `341`
  - PLOS: `221`
  - PeerJ: `31`
  - Nature: `10`
- avg concerns/article: `14.27`
- truncation fraction: `0.5174`

### Interpretation

- This corpus restores all 5 benchmark sources.
- Concern density is now close to evaluation-time ground truth.
- Truncation is still substantial, but much lower than the old high-confidence path.

---

## Corpus B: High-Confidence

### Train

- rows: `700`
- source mix:
  - eLife: `653`
  - Nature: `46`
  - PLOS: `1`
- avg concerns/article: `6.87`
- truncation fraction: `0.8471`

### Val

- rows: `118`
- source mix:
  - eLife: `110`
  - Nature: `8`
- avg concerns/article: `6.74`
- truncation fraction: `0.8051`

### Interpretation

- This path still collapses to an eLife/Nature-heavy subset.
- It remains useful only as a curriculum warm-start, not as the main task-aligned corpus.

---

## Key Takeaway

For the next training run:

1. Use `corpus_all_nonfig` as the primary SFT target.
2. If curriculum is desired, use `corpus_hi_conf` only as stage 1 warm-start.
3. Do not treat `corpus_hi_conf` as the main benchmark-representative training set.
