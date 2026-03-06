# Input Quality Review (2026-03-03)

## Scope

- 대상 데이터:
  - `data/sweeps/conf_0_6/{sft_train.jsonl,sft_val.jsonl}`
  - `data/sweeps/conf_0_7/{sft_train.jsonl,sft_val.jsonl}`
  - `data/sweeps/conf_0_8/{sft_train.jsonl,sft_val.jsonl}`
- 검증 관점:
  - JSONL 파싱 가능 여부
  - ShareGPT 구조(`conversations` 길이/role 순서)
  - `gpt` 출력 JSON 파싱 및 필드 유효성(`text/category/severity`)
  - 토큰 예산 초과 여부(생성기와 동일한 보수 카운터)
  - 중복 row/article/concern 여부
  - 입력 텍스트 결손(제목만 존재하는 샘플)

## Structural Integrity

모든 conf/split에서 아래 항목은 **0건**:

- bad JSON line
- `conversations` 길이 오류
- role 순서 오류 (`system`,`human`,`gpt`)
- assistant JSON 파싱 실패
- assistant item 필드 누락/타입 오류
- 허용 범위 밖 category/severity
- article_id 중복 row
- row 내부 concern 중복
- 토큰 예산(15000) 초과

subset 일관성도 정상:

- train: `0.8 ⊂ 0.7 ⊂ 0.6`
- val: `0.8 ⊂ 0.7 ⊂ 0.6`

## Input Content Findings

### 1) Title-only 샘플 (주의)

- `conf_0_6 train`: 4건
- `conf_0_7 train`: 4건
- `conf_0_8 train`: 3건
- `val`: 0건

공통적으로 abstract/section이 비어 제목만 있는 케이스:

- `nature:s41556-024-01546-0`
- `nature:s41556-024-01550-4`
- `nature:s41588-024-02000-5`
- `nature:s41592-024-02557-3`

원본 split에서도 해당 article은 `abstract_len=0`, `paper_text_sections` 비어있음 확인.

### 2) 강한 truncation 비율 (정상 동작이지만 관찰 필요)

- `train` 기준 truncation marker 포함 비율:
  - `conf_0_6`: 767/913
  - `conf_0_7`: 711/843
  - `conf_0_8`: 529/631

이는 15K budget에 맞춘 설계 결과로 구조적 오류는 아님.

## Verdict by Confidence

- `0.6`: **적합** (구조 문제 없음, 단 title-only 4건 주의)
- `0.7`: **적합** (구조 문제 없음, 단 title-only 4건 주의)
- `0.8`: **적합** (구조 문제 없음, 단 title-only 3건 주의)

## Recommendation

- 바로 학습 진행 가능.
- 데이터 청결도를 더 올리려면 다음 한 줄 필터 추가 권장:
  - `user_input_token_estimate >= 500` 또는
  - `(abstract 없음) AND (section 없음)` article drop

영향 범위는 매우 작음(3~4건).
