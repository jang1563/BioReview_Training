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
datasets:
- jang1563/peer-review-benchmark
base_model: Qwen/Qwen3-8B
pipeline_tag: text-generation
---

# BioReview SFT — Qwen3-8B (all_nonfig)

QLoRA fine-tuned model for identifying scientific concerns in biomedical research papers.

## Model Description

This model was trained using supervised fine-tuning (SFT) with QLoRA on the [peer-review-benchmark](https://github.com/jang1563/peer-review-benchmark) dataset. It generates structured lists of scientific concerns (design flaws, statistical issues, missing experiments, etc.) given the full text of a biomedical paper.

- **Base model:** [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- **Training method:** QLoRA (4-bit quantization, LoRA rank=64, alpha=128)
- **Training data:** 4,734 articles from 5 journal sources (eLife, F1000Research, PLOS, PeerJ, Nature)
- **Training duration:** ~18 hours on 1x A100 80GB (3 epochs, 1,773 steps)
- **Framework:** Unsloth + TRL + PEFT

## Performance

Evaluated on the peer-review-benchmark v3 validation split (838 articles) using SPECTER2 semantic matching + Hungarian algorithm (threshold=0.65).

| Variant | F1 | Recall | Precision |
|---------|---:|-------:|----------:|
| dedup+cap20 (recommended) | **0.556** | 0.413 | 0.851 |
| raw | 0.457 | 0.443 | 0.473 |

**Comparison:**

| Model | F1 | Recall | Precision |
|-------|---:|-------:|----------:|
| GPT-4o-mini (baseline) | 0.696 | 0.647 | 0.753 |
| **This model** | **0.556** | 0.413 | 0.851 |
| Best previous SFT ensemble | 0.583 | 0.433 | 0.891 |

### By Source

| Source | N | F1 | Recall | Precision |
|--------|--:|---:|-------:|----------:|
| PeerJ | 31 | 0.609 | 0.469 | 0.870 |
| F1000 | 341 | 0.595 | 0.469 | 0.815 |
| eLife | 232 | 0.565 | 0.419 | 0.866 |
| PLOS | 221 | 0.491 | 0.336 | 0.912 |
| Nature | 13 | 0.330 | 0.200 | 0.941 |

### By Category

| Category | F1 | Recall | Precision |
|----------|---:|-------:|----------:|
| interpretation | 0.631 | 0.568 | 0.709 |
| missing_experiment | 0.627 | 0.564 | 0.706 |
| prior_art_novelty | 0.593 | 0.566 | 0.623 |
| writing_clarity | 0.570 | 0.492 | 0.677 |
| design_flaw | 0.553 | 0.519 | 0.591 |
| statistical_methodology | 0.463 | 0.448 | 0.479 |
| reagent_method_specificity | 0.453 | 0.428 | 0.480 |

## Intended Use

This model is designed for **automated scientific peer review assistance**. Given the full text of a biomedical research paper, it identifies potential scientific concerns that a human reviewer might raise.

**Intended users:** Researchers, journal editors, and peer review platforms seeking to augment (not replace) human review.

**Out of scope:**
- Final peer review decisions (this is an assistive tool)
- Non-biomedical papers
- Papers with heavy figure/image dependence (model is trained to skip figure-related concerns)

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "jang1563/bioreview-qwen3-8b-allnonfig")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Format input (system + paper text)
messages = [
    {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
    {"role": "user", "content": paper_full_text}
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

# Generate
output = model.generate(input_ids, max_new_tokens=4096, temperature=0.1)
response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
```

See the [BioReview Training repository](https://github.com/jang1563/BioReview_Training) for the full inference pipeline including postprocessing (dedup + cap).

## Training Details

### Training Data

- **Corpus:** All non-figure concerns from peer-review-benchmark v3 train split
- **Articles:** 4,734 (eLife 1304, F1000 1933, PLOS 1255, PeerJ 176, Nature 66)
- **Avg concerns/article:** 14.1
- **Format:** ShareGPT (system / human / assistant turns)
- **Input truncation:** 15,000 token budget with section priority (methods > results > intro > ...)

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA target modules | all linear layers |
| Quantization | 4-bit (NF4) |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Epochs | 3 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Max sequence length | 16,384 |
| Optimizer | AdamW (8-bit) |

### Hardware

- 1x NVIDIA A100 80GB (Cornell Cayuga HPC)
- Training time: ~18 hours

## Evaluation Methodology

Evaluation uses SPECTER2 semantic embeddings to match model-generated concerns against human reviewer annotations:

1. Encode all concerns (model + human) using SPECTER2
2. Compute pairwise cosine similarity
3. Apply Hungarian algorithm for optimal 1:1 matching
4. Threshold at 0.65 to determine true positives
5. Compute micro-averaged F1, Recall, Precision

**Critical:** SPECTER2 must be available for evaluation. Without it, the pipeline silently falls back to Jaccard similarity, producing incorrect scores (~0.03 instead of ~0.55).

## Limitations

- **Recall gap:** Model captures ~41% of human concerns (vs 65% for GPT-4o-mini)
- **Source bias:** Weaker on Nature (small sample) and PLOS articles
- **Category gaps:** `reagent_method_specificity` and `statistical_methodology` have lowest recall
- **Parse failures:** ~1.2% of articles produce invalid JSON output
- **No figure analysis:** Model explicitly skips figure-related concerns
- **Context window:** Long papers are truncated to 15K tokens, potentially missing content

## Citation

```bibtex
@software{bioreview_training_2026,
  title = {BioReview Training: QLoRA SFT Pipeline for Biomedical Peer-Review LLMs},
  author = {Jang, Andrew},
  year = {2026},
  url = {https://github.com/jang1563/BioReview_Training}
}
```
