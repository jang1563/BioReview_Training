# Corpus Truncation Audit: corpus_all_nonfig

- Generated: 2026-03-12 20:58:56Z
- Corpus dir: `./data/corpus_all_nonfig`
- Token counter: `heuristic:chars/3(conservative)`

## Split Summary

| Split | Articles | Marker trunc. | Near budget | Mean tokens | Median | P90 | Max | Avg concerns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 4734 | 0.5281 | 0.5266 | 12412.5 | 14841.5 | 15000.0 | 15000 | 14.10 |
| val | 835 | 0.5174 | 0.5222 | 12433.7 | 14833.0 | 15000.0 | 15000 | 14.26 |

## Train By Source

| Source | Articles | Marker trunc. | Near budget | Mean tokens | P90 | Max | Avg concerns |
|---|---:|---:|---:|---:|---:|---:|---:|
| f1000 | 1933 | 0.1630 | 0.1614 | 9616.1 | 14887.0 | 15000 | 16.25 |
| elife | 1304 | 0.8482 | 0.8643 | 14585.4 | 15000.0 | 15000 | 6.38 |
| plos | 1255 | 0.7833 | 0.8048 | 14476.7 | 15000.0 | 15000 | 17.72 |
| peerj | 176 | 0.2614 | 0.1420 | 11831.0 | 14853.5 | 15000 | 19.31 |
| nature | 66 | 0.7576 | 0.2879 | 13683.6 | 14799.0 | 14964 | 21.00 |

## Train By Length Bin

| Bin | Articles | Marker trunc. | Near budget | Mean tokens | Top sources |
|---|---:|---:|---:|---:|---|
| <4k | 111 | 0.0000 | 0.0000 | 3221.5 | f1000:109, peerj:2 |
| 4k-8k | 660 | 0.0000 | 0.0000 | 6203.3 | f1000:624, peerj:17, elife:10 |
| 8k-12k | 838 | 0.0000 | 0.0000 | 10092.8 | f1000:626, plos:79, peerj:67 |
| 12k-14.5k | 559 | 0.0805 | 0.0000 | 13343.9 | f1000:228, plos:144, elife:100 |
| 14.5k+ | 2566 | 0.9567 | 0.9716 | 14961.9 | elife:1133, plos:1026, f1000:346 |

## Val By Source

| Source | Articles | Marker trunc. | Near budget | Mean tokens | P90 | Max | Avg concerns |
|---|---:|---:|---:|---:|---:|---:|---:|
| f1000 | 341 | 0.1466 | 0.1613 | 9564.0 | 14896.0 | 15000 | 16.26 |
| elife | 232 | 0.8233 | 0.8448 | 14623.5 | 15000.0 | 15000 | 6.06 |
| plos | 221 | 0.7873 | 0.8054 | 14516.7 | 15000.0 | 15000 | 18.96 |
| peerj | 31 | 0.2581 | 0.1613 | 12176.1 | 14776.0 | 15000 | 18.45 |
| nature | 10 | 0.9000 | 0.2000 | 14248.6 | 14762.7 | 14769 | 19.60 |

## Val By Length Bin

| Bin | Articles | Marker trunc. | Near budget | Mean tokens | Top sources |
|---|---:|---:|---:|---:|---|
| <4k | 13 | 0.0000 | 0.0000 | 3357.5 | f1000:13 |
| 4k-8k | 130 | 0.0000 | 0.0000 | 6219.7 | f1000:123, peerj:3, elife:2 |
| 8k-12k | 136 | 0.0000 | 0.0000 | 10087.4 | f1000:101, plos:14, peerj:11 |
| 12k-14.5k | 108 | 0.1019 | 0.0000 | 13474.4 | f1000:47, elife:22, plos:21 |
| 14.5k+ | 448 | 0.9397 | 0.9732 | 14961.5 | elife:198, plos:184, f1000:57 |
