---
license: apache-2.0
language:
- en
tags:
- peer-review
- biomedical
- scientific-critique
- qlora
- sft
- peft
datasets:
- jang1563/peer-review-benchmark
base_model: Qwen/Qwen3.5-9B
pipeline_tag: text-generation
model-index:
- name: bioreview-qwen3.5-9b-sft
  results:
  - task:
      type: text-generation
    dataset:
      name: peer-review-benchmark v3 (test split)
      type: jang1563/peer-review-benchmark
    metrics:
    - type: f1
      value: 0.621
      name: F1 (dedup+cap20)
    - type: recall
      value: 0.498
      name: Recall (dedup+cap20)
    - type: precision
      value: 0.827
      name: Precision (dedup+cap20)
---

# BioReview SFT — Qwen3.5-9B (all_nonfig)

A QLoRA-fine-tuned model for identifying scientific concerns in biomedical
research papers. It was trained on the access-controlled BioReview-Bench v3
corpus.

> **F1 = 0.621 · Recall = 0.498 · Precision = 0.827** (test set, dedup+cap20 postprocessing)
> Among the evaluated SFT configurations, the 8B+9B union ensemble has the
> highest test F1 (0.704) and recall (0.695).

---

## Model Description

This model was trained with supervised fine-tuning (SFT) and QLoRA on the
access-controlled BioReview-Bench v3 corpus.
Given the full text of a biomedical paper, it generates a structured list of scientific
concerns a peer reviewer might raise, each annotated with category and severity.

| | |
|---|---|
| **Base model** | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| **Model class** | `Qwen3_5ForConditionalGeneration` (vision-language; text-only fine-tuning) |
| **Training method** | QLoRA — 4-bit NF4 quantization, LoRA rank=16, alpha=32 |
| **Training data** | 4,734 articles · 5 journal sources (eLife, F1000Research, PLOS, PeerJ, Nature) |
| **Training duration** | ~35 h on 1× NVIDIA A100 80 GB · 3 epochs · 1,773 steps |
| **Framework** | Unsloth + TRL + PEFT |
| **Training code** | [jang1563/BioReview_Training](https://github.com/jang1563/BioReview_Training) |

---

## Performance

Evaluated on the **BioReview-Bench v3 test split (981 articles)** using
SPECTER2 semantic embeddings and the Hungarian algorithm (matching threshold =
0.65).

### Test Set

| Postprocessing | F1 | Recall | Precision |
|---|---:|---:|---:|
| **dedup+cap20** *(recommended)* | **0.621** | 0.498 | **0.827** |
| raw (no postprocessing) | 0.514 | 0.496 | 0.533 |

### Validation Set (838 articles)

| Postprocessing | F1 | Recall | Precision |
|---|---:|---:|---:|
| dedup+cap20 | **0.625** | 0.501 | 0.832 |

Validation and test F1 differ by 0.004.

### Comparison

| Model | Split | F1 | Recall | Precision | Gate |
|---|---|---:|---:|---:|:---:|
| GPT-4o-mini *(contextual baseline)* | validation* | 0.696 | 0.647 | 0.753 | – |
| **8B+9B Ensemble** (union, dedup+cap20) | test | **0.704** | **0.695** | 0.713 | ✓ |
| **This model** (Qwen3.5-9B, dedup+cap20) | test | **0.621** | 0.498 | **0.827** | ✓ |
| Qwen3-8B SFT (dedup+cap20) | test | 0.557 | 0.409 | 0.871 | ✗ |

> *GPT-4o-mini was evaluated on a different validation split. Its metrics are
> contextual only and are not directly comparable with the v3 test results.*
>
> **Precision note:** Qwen3-8B achieves higher postprocessed precision (0.871) but lower F1 and recall,
> and fails the evaluation threshold. The 9B model offers the best single-model balance
> of F1, recall, and precision. The 8B+9B union ensemble is recommended for highest F1.

---

## Output Format

The model returns a **JSON array of concern objects**, each with three fields:

```json
[
  {
    "text": "The sample size of n=12 per group is insufficient for the claimed statistical power.",
    "category": "statistical_methodology",
    "severity": "major"
  },
  {
    "text": "The authors do not compare their method to the current state-of-the-art baselines.",
    "category": "prior_art_novelty",
    "severity": "minor"
  }
]
```

**Valid categories:** `design_flaw` · `statistical_methodology` · `missing_experiment` ·
`prior_art_novelty` · `writing_clarity` · `reagent_method_specificity` · `interpretation` · `other`

**Valid severities:** `major` · `minor` · `optional`

---

## How to Use

> **Note:** Qwen3.5-9B is a Vision-Language model (`Qwen3_5ForConditionalGeneration`).
> Load via `AutoProcessor` and use its inner `.tokenizer` for text decoding.
> The adapter was saved via Unsloth — always pre-load the base model first (as shown below),
> rather than using `AutoPeftModelForCausalLM.from_pretrained` directly.

### Installation

```bash
pip install transformers peft bitsandbytes accelerate torch sentencepiece protobuf
```

### Inference

**Availability:** The evaluated adapter is access-controlled. Replace
`ADAPTER_REPO` below with an authorized Hub identifier or local adapter path.

```python
import json, re, torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

ADAPTER_REPO = "jang1563/bioreview-qwen3.5-9b-sft"
BASE_MODEL   = "Qwen/Qwen3.5-9B"

# --- Load model ---
# Qwen3.5-9B is a VL model: AutoProcessor returns a multimodal processor.
# Extract the inner text tokenizer for encoding/decoding.
processor = AutoProcessor.from_pretrained(BASE_MODEL)
tokenizer = processor.tokenizer   # text-only tokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True,            # requires bitsandbytes
)
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
model.eval()

# --- Prompts (must match training exactly) ---
# Note: the base sentence says "strings" but OUTPUT FORMAT overrides it with "objects".
# This apparent contradiction is intentional — it matches the training prompt exactly.
# Changing either part will degrade output quality.
SYSTEM_PROMPT = (
    "You are an expert peer reviewer for biomedical research papers published in "
    "high-impact journals. Identify specific scientific concerns, weaknesses, and "
    "issues. Return only a JSON array of concern strings.\n\n"
    "OUTPUT FORMAT: Return a JSON array of concern objects, nothing else.\n"
    'Each item must be: {"text": string, "category": one of '
    "[design_flaw, statistical_methodology, missing_experiment, "
    "prior_art_novelty, writing_clarity, reagent_method_specificity, "
    'interpretation, other], "severity": one of [major, minor, optional]}.\n'
    "Do NOT return a JSON array of strings."
)

USER_PREFIX = (
    "Review the following biomedical article.\n\n"
    "Return ONLY a JSON array of objects with keys: `text`, `category`, `severity`.\n"
    "Allowed category values: design_flaw, statistical_methodology, "
    "missing_experiment, prior_art_novelty, writing_clarity, "
    "reagent_method_specificity, interpretation, other.\n"
    "Allowed severity values: major, minor, optional.\n\n"
)


def review_paper(paper_text: str) -> list[dict]:
    """Generate peer-review concerns for a biomedical paper.

    Returns a list of concern dicts: {text, category, severity}.
    Apply postprocess_concerns() for best evaluation results.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PREFIX + paper_text},
    ]
    # Use processor.apply_chat_template (handles VL model chat format)
    input_ids = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=4096,
            temperature=0.1,
            repetition_penalty=1.05,
            do_sample=True,
        )

    # Use inner text tokenizer to decode (not the multimodal processor)
    raw = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract the first JSON array found in the output
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return []
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return []


concerns = review_paper(paper_text)
# [{"text": "...", "category": "design_flaw", "severity": "major"}, ...]
```

### Postprocessing — dedup+cap20 (Recommended)

Raw output typically contains 30–60 concerns, many overlapping.
Apply **dedup+cap20** to remove near-duplicates and cap at 20 concerns per article.
On the test set, this changes F1 from 0.514 to 0.621 (+0.107).

```bash
pip install sentence-transformers scikit-learn
```

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def postprocess_concerns(
    concerns: list[dict],
    cap: int = 20,
    dedup_threshold: float = 0.95,
) -> list[dict]:
    """Remove near-duplicate concerns and cap at `cap` items.

    Uses SPECTER2 embeddings on the `text` field.
    Requires: sentence-transformers, scikit-learn
    """
    if len(concerns) <= 1:
        return concerns

    texts = [c["text"] for c in concerns]
    embedder = SentenceTransformer("allenai/specter2_base")
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    sim = cosine_similarity(embeddings)

    keep: list[int] = []
    for i in range(len(concerns)):
        if all(sim[i][j] < dedup_threshold for j in keep):
            keep.append(i)

    return [concerns[i] for i in keep][:cap]


concerns = postprocess_concerns(review_paper(paper_text))
```

See the [BioReview Training repository](https://github.com/jang1563/BioReview_Training)
for the full inference + evaluation pipeline including SPECTER2-based scoring.

---

## Training Details

### Data

| Property | Value |
|---|---|
| Corpus | All non-figure concerns — peer-review-benchmark v3 train split |
| Total articles | 4,734 |
| Source breakdown | eLife 1,304 · F1000 1,933 · PLOS 1,255 · PeerJ 176 · Nature 66 |
| Mean concerns per article | 14.1 |
| Concern schema | `{text, category, severity}` objects |
| Format | ShareGPT (system / human / assistant turns) |
| Input truncation | 15,000-token budget · section priority: methods > results > intro > … |

### Hyperparameters

| Parameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantization | 4-bit NF4 (bitsandbytes) |
| Precision | bfloat16 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 (effective batch size = 8) |
| Max sequence length | 16,384 |
| Optimizer | AdamW 8-bit |
| Seed | 42 |

### Hardware

| | |
|---|---|
| GPU | 1× NVIDIA A100 80 GB |
| Cluster | Cornell Cayuga HPC |
| Wall time | ~35 hours |
| Steps | 1,773 (3 epochs) |

---

## Evaluation Methodology

Scientific concerns are matched semantically, not by exact string:

1. Encode all concerns (model-generated + human reviewer annotations) with **SPECTER2** (`allenai/specter2_base`)
2. Compute pairwise cosine similarity matrix
3. Apply **Hungarian algorithm** for optimal 1-to-1 matching
4. Threshold at **0.65** — pairs above this are counted as true positives
5. Compute micro-averaged **F1, Recall, Precision** across all articles

**Postprocessing (dedup+cap20):**
- Remove near-duplicate model outputs (cosine similarity > 0.95 → keep first)
- Cap at 20 concerns per article

> **Important:** SPECTER2 model weights must be downloaded for valid evaluation.
> Without them, the pipeline silently falls back to Jaccard similarity
> and produces scores near 0.03 instead of 0.62.

---

## Limitations

- **Recall gap:** Captures approximately 50% of human reviewer concerns on the v3 test split
- **No figure analysis:** Explicitly trained to skip figure/image-related concerns
- **Source bias:** Performance is weakest on Nature articles (n=66 in training)
- **Context truncation:** Papers exceeding 15K tokens are truncated; later sections may be missed
- **Parse failures:** ~1–2% of articles produce malformed JSON; fallback regex parsing is recommended
- **Best as ensemble:** Pair with the Qwen3-8B SFT model (union) to reach F1 = 0.704

---

## Citation

```bibtex
@software{bioreview_training_2026,
  title  = {BioReview Training: QLoRA SFT Pipeline for Biomedical Peer-Review LLMs},
  author = {Kim, JangKeun},
  year   = {2026},
  url    = {https://github.com/jang1563/BioReview_Training}
}
```
