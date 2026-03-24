# Corpus Truncation Audit: corpus_hi_conf

- Generated: 2026-03-12 20:58:50Z
- Corpus dir: `./data/corpus_hi_conf`
- Token counter: `heuristic:chars/3(conservative)`

## Split Summary

| Split | Articles | Marker trunc. | Near budget | Mean tokens | Median | P90 | Max | Avg concerns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 700 | 0.8471 | 0.8300 | 14517.4 | 15000.0 | 15000.0 | 15000 | 6.87 |
| val | 118 | 0.8051 | 0.7797 | 14429.7 | 15000.0 | 15000.0 | 15000 | 6.74 |

## Train By Source

| Source | Articles | Marker trunc. | Near budget | Mean tokens | P90 | Max | Avg concerns |
|---|---:|---:|---:|---:|---:|---:|---:|
| elife | 653 | 0.8560 | 0.8698 | 14591.8 | 15000.0 | 15000 | 6.06 |
| nature | 46 | 0.7174 | 0.2609 | 13452.3 | 14778.0 | 14888 | 18.07 |
| plos | 1 | 1.0000 | 1.0000 | 14892.0 | 14892.0 | 14892 | 20.00 |

## Train By Length Bin

| Bin | Articles | Marker trunc. | Near budget | Mean tokens | Top sources |
|---|---:|---:|---:|---:|---|
| <4k | 0 | 0.0000 | 0.0000 | 0.0 |  |
| 4k-8k | 8 | 0.0000 | 0.0000 | 6604.0 | elife:5, nature:3 |
| 8k-12k | 33 | 0.0000 | 0.0000 | 10500.5 | elife:29, nature:4 |
| 12k-14.5k | 72 | 0.2778 | 0.0000 | 13414.7 | elife:47, nature:25 |
| 14.5k+ | 587 | 0.9761 | 0.9898 | 14986.3 | elife:572, nature:14, plos:1 |

## Val By Source

| Source | Articles | Marker trunc. | Near budget | Mean tokens | P90 | Max | Avg concerns |
|---|---:|---:|---:|---:|---:|---:|---:|
| elife | 110 | 0.8000 | 0.8182 | 14439.2 | 15000.0 | 15000 | 6.12 |
| nature | 8 | 0.8750 | 0.2500 | 14299.4 | 14764.1 | 14769 | 15.25 |

## Val By Length Bin

| Bin | Articles | Marker trunc. | Near budget | Mean tokens | Top sources |
|---|---:|---:|---:|---:|---|
| <4k | 0 | 0.0000 | 0.0000 | 0.0 |  |
| 4k-8k | 2 | 0.0000 | 0.0000 | 7298.0 | elife:2 |
| 8k-12k | 7 | 0.0000 | 0.0000 | 10618.1 | elife:7 |
| 12k-14.5k | 17 | 0.2941 | 0.0000 | 13817.6 | elife:11, nature:6 |
| 14.5k+ | 92 | 0.9783 | 1.0000 | 14987.9 | elife:90, nature:2 |
