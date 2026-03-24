## Step Probe Progress — 2026-03-12

### Confirmed

- Remote Cayuga `v3` split is corrected to current local counts:
  - `val=838`
  - `test=981`
- Final queued jobs remain valid on current split:
  - `2702592` full val eval
  - `2702810` by-source eval

### Checkpoint-100 probe50

Source:
- [checkpoint-100 probe50 summary](./results/sft_eval/checkpoint-100_val_probe50_current_v3_checkpoint100.summary.json#L1)

Metrics:
- `F1=0.0132`
- `Recall=0.0194`
- `Precision=0.0100`
- `GT concerns=669`
- `Model concerns=1300`

Interpretation:
- Mixed-source `probe50` confirms checkpoint-100 is still badly over-generating.
- This is not only a duplicate problem. Exact dedup and `cap=15` improve precision but do not improve recall.

Postprocess ablations:
- [dedup summary](./results/sft_eval/checkpoint-100_val_probe50_current_v3_checkpoint100.dedup.summary.json#L1)
  - `F1=0.0250`
  - `Recall=0.0194`
  - `Precision=0.0351`
- [dedup+cap15 summary](./results/sft_eval/checkpoint-100_val_probe50_current_v3_checkpoint100.dedup_cap15.summary.json#L1)
  - `F1=0.0274`
  - `Recall=0.0194`
  - `Precision=0.0466`

### Checkpoint-200 early signal

Source:
- [checkpoint-200 probe50 summary](./results/sft_eval/checkpoint-200_val_probe50_current_v3_checkpoint200.summary.json#L1)

Full summary is now available.

Metrics:
- `F1=0.0257`
- `Recall=0.0299`
- `Precision=0.0225`
- `Recall major=0.0375`
- `GT concerns=669`
- `Model concerns=888`

Interpretation:
- Checkpoint-200 is materially better than checkpoint-100 on the same mixed-source `probe50`.
- The main gain is reduced over-generation: tool concerns dropped from `1300` to `888`.
- Recall also improved from `0.0194` to `0.0299`, so this is not just a precision-only change.

Early per-article evidence was consistent with the final summary: the first articles in the same `probe50` run already showed a strong reduction in over-generation.

First 8 articles, concern-count average:
- checkpoint-100: `13.0`
- checkpoint-200: `3.5`

First 13 articles, concern-count average:
- checkpoint-100: `23.85`
- checkpoint-200: `10.54`

Per-article counts for first 13:
- checkpoint-100: `[10, 4, 6, 4, 3, 67, 6, 4, 75, 49, 3, 3, 76]`
- checkpoint-200: `[8, 3, 3, 2, 3, 3, 3, 3, 87, 5, 2, 3, 12]`

Early-pattern note:
- Checkpoint-200 was clearly more controlled on most early articles.
- It was not stable yet. Large spikes still existed, for example the 9th article jumped to `87` concerns.

### Next checkpoints

- `checkpoint-200` probe50 is complete.
- `checkpoint-300` directory exists.
- `checkpoint-300` mixed-source `probe50` completed as job `2703079`.

### Checkpoint-300 result

Source:
- [checkpoint-300 probe50 summary](./results/sft_eval/checkpoint-300_val_probe50_current_v3_checkpoint300.summary.json#L1)

Metrics:
- `F1=0.0221`
- `Recall=0.0254`
- `Precision=0.0196`
- `Recall major=0.0415`
- `GT concerns=669`
- `Model concerns=868`

Interpretation:
- Checkpoint-300 stayed better than checkpoint-100, but it regressed versus checkpoint-200 on overall `F1`, `Recall`, and `Precision`.
- `Recall major` improved slightly from checkpoint-200 (`0.0375 -> 0.0415`), so the model may be trading some breadth for a small gain on high-severity items.
- Current read: for this direct run, checkpoint-200 is the best early checkpoint among `100/200/300`.

### Current recommendation

- Treat `checkpoint-200` as the leading candidate for this direct-training run.
- Run a full current-`v3` validation eval and by-source eval on checkpoint-200 instead of waiting only for the final adapter.
- Keep the final-training eval chain in place as a separate comparison, but do not assume the final checkpoint will beat checkpoint-200.

Submitted jobs:
- `2703138`: checkpoint-200 full current-`v3` validation eval
- `2703139`: checkpoint-200 by-source eval (dependency on `2703138`)
