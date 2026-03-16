# Error Analysis: Qwen3-8B all_nonfig — 2026-03-15

Source: `results/error_analysis/analysis_8b_allnonfig_2026-03-15.json`
Models: 8B-raw (원본), 8B-dedup-cap20 (best F1)

---

## 1. 핵심 요약

| | 8B-raw | 8B-dedup-cap20 |
|---|---|---|
| F1 | 0.457 | 0.554 |
| Recall | 0.443 | 0.411 |
| Precision | 0.473 | 0.851 |
| Matched / GT | 5,291 / 11,955 | 4,913 / 11,955 |
| OK articles | 828 (98.8%) | 828 (98.8%) |
| Parse failures | 9 (1.1%) | 9 (1.1%) |
| Repetition loops | 1 (0.1%) | 1 (0.1%) |
| Zero-recall articles | 10 | 10 |

---

## 2. Zero-Recall Articles 분석

10건 모두 **parse failure** 또는 **repetition loop** — 모델 생성 자체가 실패한 경우.
"의미적으로 가까운데 매칭 실패"가 아닌, 출력이 없거나 파싱 불가능한 케이스.

| Article ID | Source | GT Concerns | Failure Mode |
|---|---|---|---|
| plos:10.1371/journal.pmed.1004280 | PLOS | 51 | parse_fail |
| plos:10.1371/journal.pcbi.1012577 | PLOS | 30 | parse_fail |
| plos:10.1371/journal.pmed.1004461 | PLOS | 28 | repetition_loop |
| f1000:10.12688_f1000research.27123.2 | F1000 | 20 | parse_fail |
| plos:10.1371/journal.pmed.1004422 | PLOS | 16 | parse_fail |
| plos:10.1371/journal.pmed.1004018 | PLOS | 16 | parse_fail |
| f1000:10.12688_f1000research.165539.1 | F1000 | 13 | parse_fail |
| elife:90875 | eLife | 13 | parse_fail |
| elife:96699 | eLife | 5 | parse_fail |
| elife:88054 | eLife | 2 | parse_fail |

**패턴**: PLOS가 5/10건으로 가장 많고, 의학 논문(pmed)이 4건. 큰 논문(51, 30, 28 GT concerns)이 포함.

**영향**: 이 10건의 GT concerns 합계 = 194건. 전체 11,955건의 1.6%.
만약 이들만 수정해도 recall 상한 +1.6% 가능.

---

## 3. Per-Category Recall 분석

| Category | 8B-raw Recall | Matched / GT | 약점 여부 |
|---|---|---|---|
| prior_art_novelty | **0.506** | 465 / 919 | |
| design_flaw | 0.486 | 637 / 1,311 | |
| interpretation | 0.476 | 890 / 1,869 | |
| missing_experiment | 0.472 | 908 / 1,924 | |
| statistical_methodology | 0.427 | 274 / 641 | **약점** |
| writing_clarity | 0.404 | 1,771 / 4,386 | **약점** |
| other | 0.385 | 15 / 39 | (소표본) |
| reagent_method_specificity | **0.382** | 331 / 866 | **최약점** |

### 인사이트

- **reagent_method_specificity** (0.382): 실험 시약/방법의 구체성을 지적하는 카테고리.
  학습 데이터에서 이 카테고리가 상대적으로 적었을 가능성.

- **writing_clarity** (0.404): 가장 많은 GT concerns(4,386)을 가진 카테고리인데
  recall이 낮음. 모델이 writing 관련 concern을 충분히 생성하지 않는 경향.

  > 그러나 raw 모델의 tool concerns 중 41%가 writing_clarity (4,594/11,195).
  > 많이 생성하긴 하지만, 인간 리뷰어의 writing concern과 매칭이 안 되는 것.
  > → 모델이 다른 writing issue를 지적하고 있을 가능성.

- **prior_art_novelty** (0.506): 가장 높은 recall. 선행연구/참신성 지적은 모델이 잘함.

---

## 4. Per-Severity Recall 분석

| Severity | 8B-raw Recall | Matched / GT |
|---|---|---|
| **major** | **0.476** | 3,578 / 7,519 |
| minor | 0.389 | 1,623 / 4,177 |
| optional | 0.347 | 90 / 259 |

**Major concern 발견율이 가장 높음** (0.476) — 모델이 중요한 문제를 더 잘 포착.
Minor/optional은 상대적으로 낮아, 세부적/선택적 피드백에는 약함.

이는 실용적으로 긍정적: 실제 peer review에서 major concern이 가장 중요.

---

## 5. Failure Mode 분포

| Mode | Count | % |
|---|---|---|
| **ok** | 828 | 98.8% |
| parse_fail | 9 | 1.1% |
| repetition_loop | 1 | 0.1% |
| empty | 0 | 0% |
| missing | 0 | 0% |

모델 출력 품질은 매우 양호 (98.8% 정상 파싱).
9건 parse failure는 JSON 형식 불완전 출력으로 추정 (max_new_tokens 부족 가능성).

---

## 6. Dedup+Cap20 효과 분석

| Category | Raw Recall | Dedup+Cap20 Recall | Delta |
|---|---|---|---|
| prior_art_novelty | 0.506 | 0.482 | -0.024 |
| design_flaw | 0.486 | 0.455 | -0.031 |
| interpretation | 0.476 | 0.451 | -0.025 |
| missing_experiment | 0.472 | 0.453 | -0.019 |
| statistical_methodology | 0.427 | 0.387 | **-0.040** |
| writing_clarity | 0.404 | 0.367 | **-0.037** |
| reagent_method_specificity | 0.382 | 0.331 | **-0.051** |

**관찰**:
- 이미 약한 카테고리(reagent, writing, statistical)에서 recall 감소가 더 큼
- 이들 카테고리의 concern이 article 후반에 생성되어 cap에 의해 잘려나갈 가능성
- 반면 design_flaw, missing_experiment는 감소폭이 작음 (중요도 높아 먼저 생성)

---

## 7. Recall 분포 (8B-raw)

| 구간 | Articles | % |
|---|---|---|
| R = 0 (zero) | 10 | 1.2% |
| 0 < R < 0.5 (partial) | 375 | 44.7% |
| R ≥ 0.5 (decent) | 453 | 54.1% |

**54%의 articles에서 recall ≥ 0.5** — 과반수에서 양호한 성능.
문제는 나머지 45%의 partial recall articles에서 평균을 끌어내리는 것.

---

## 8. 개선 방향 제안

### 즉시 적용 가능 (재학습 없이)

1. **Parse failure 재시도**: 9건의 parse failure articles에 대해
   max_new_tokens 증가 또는 재시도 → recall +1.3% 가능

2. **Source-adaptive cap**: eLife/Nature는 cap 불필요(이미 3-4개),
   F1000/PLOS/PeerJ에만 cap 적용 → precision 유지하면서 eLife recall 보존

3. **Category-aware dedup**: 같은 카테고리 내에서만 dedup →
   cross-category 유사 concern 보존

### 9B 학습 시 반영 가능

4. **reagent_method_specificity 강화**: 학습 데이터에서 해당 카테고리
   concern을 augmentation 또는 가중치 부여

5. **Repetition penalty 조정**: repetition_penalty=1.05 →
   repetition loop 방지 위해 1.1-1.15 시도

6. **Longer generation**: max_new_tokens=4096 → 6144 or 8192로
   parse failure 감소 (truncation 방지)

### 장기적 개선

7. **Source-conditioned prompt**: 입력에 source 정보 포함하여
   source별 적절한 concern 수 생성 유도

8. **Writing clarity 매칭 개선**: 모델의 writing concern과 인간의 writing concern이
   다른 측면을 지적하는지 → 프롬프트 조정으로 alignment 개선
