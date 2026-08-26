#!/usr/bin/env python3
"""Run inference with a fine-tuned BioReview model and optionally evaluate.

Loads the trained LoRA adapter (or merged model), generates peer-review
concerns for each article in the specified split, and saves output in the
evaluation-compatible JSONL format used by bioreview_bench.

Usage:
  # Quick test (5 articles)
  python scripts/run_sft_inference.py \
      --model-dir models/qwen7b_bioreview_v1 \
      --max-articles 5

  # Full val run with evaluation
  python scripts/run_sft_inference.py \
      --model-dir models/qwen7b_bioreview_v1 \
      --evaluate

  # Custom split and output
  python scripts/run_sft_inference.py \
      --model-dir models/qwen7b_bioreview_v1 \
      --split val --output results/sft_eval/qwen7b_v1_val.jsonl \
      --evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

SYSTEMLESS_TEMPLATE_FAMILIES = {"gemma"}


# ---------------------------------------------------------------------------
# JSON output parsing
# ---------------------------------------------------------------------------


def _try_parse_array(text: str):
    """Try to parse text as JSON array and extract concerns.

    Returns (structured, texts) or None if parsing fails.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None

    structured: list[dict] = []
    texts: list[str] = []
    for item in parsed:
        if isinstance(item, dict):
            t = str(item.get("text", item.get("concern_text", ""))).strip()
            if t:
                structured.append(
                    {
                        "text": t,
                        "category": str(item.get("category", "unknown")),
                        "severity": str(item.get("severity", "unknown")),
                    }
                )
                texts.append(t)
        elif isinstance(item, str):
            s = item.strip()
            if s:
                structured.append(
                    {"text": s, "category": "unknown", "severity": "unknown"}
                )
                texts.append(s)
    return structured, texts


def _progressive_decode(fragment: str):
    """Salvage concerns from truncated JSON arrays."""
    decoder = json.JSONDecoder()
    structured: list[dict] = []
    texts: list[str] = []
    idx = 1  # skip '['
    while idx < len(fragment):
        while idx < len(fragment) and fragment[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(fragment) or fragment[idx] == "]":
            break
        try:
            value, consumed = decoder.raw_decode(fragment[idx:])
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            t = str(value.get("text", value.get("concern_text", ""))).strip()
            if t:
                structured.append(
                    {
                        "text": t,
                        "category": str(value.get("category", "unknown")),
                        "severity": str(value.get("severity", "unknown")),
                    }
                )
                texts.append(t)
        elif isinstance(value, str):
            s = value.strip()
            if s:
                structured.append(
                    {"text": s, "category": "unknown", "severity": "unknown"}
                )
                texts.append(s)
        idx += consumed
    if structured:
        return structured, texts
    return None


def _plain_text_concern_fallback(text: str):
    """Recover concern texts when a model ignores the requested JSON format."""
    chunks = re.split(r"\n\s*\n|(?:^|\n)\s*(?:[-*]|\d+[.)])\s+", text.strip())
    structured: list[dict] = []
    texts: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        concern = re.sub(r"\s+", " ", chunk).strip(" \t\r\n-*:;")
        if len(concern.split()) < 6:
            continue
        lowered = concern.lower()
        if lowered.startswith(("here are", "sure", "output", "json")):
            continue
        if concern in seen:
            continue
        seen.add(concern)
        structured.append(
            {"text": concern, "category": "unknown", "severity": "unknown"}
        )
        texts.append(concern)

    if texts:
        return structured, texts
    return None


def parse_model_output(text: str) -> tuple[list[dict], list[str]]:
    """Parse model output into (structured_concerns, text_concerns).

    Handles clean JSON, markdown fences, bracket extraction, and truncated JSON.
    Also handles DeepSeek-R1 format: bare objects without leading '['.
    """
    text = text.strip()
    if not text:
        return [], []

    # 1) Markdown code fences
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, flags=re.DOTALL)
    if fence:
        result = _try_parse_array(fence.group(1).strip())
        if result is not None:
            return result

    # 2) Full text as JSON
    result = _try_parse_array(text)
    if result is not None:
        return result

    # 3) Extract first [...] bracket pair
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        result = _try_parse_array(text[start : end + 1])
        if result is not None:
            return result

    # 4) Progressive decode (truncated JSON)
    if start != -1:
        result = _progressive_decode(text[start:])
        if result is not None:
            return result

    # 5) Bare objects without array wrapper (DeepSeek-R1 format: {...}, {...}])
    stripped = text.lstrip()
    if stripped.startswith("{"):
        tail = text.rstrip()
        if tail.endswith("]"):
            tail = tail[:-1]
        candidate = "[" + tail + "]"
        result = _try_parse_array(candidate)
        if result is not None:
            return result
        # Try progressive decode for truncated bare-object outputs
        result = _progressive_decode("[" + text)
        if result is not None:
            return result

    # 6) Plain paragraphs or bullets. Some adapters learn the concern content
    # but not the exact JSON wrapper; keep the text so evaluation can proceed.
    result = _plain_text_concern_fallback(text)
    if result is not None:
        return result

    return [], []


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def is_adapter_dir(model_dir: Path) -> bool:
    """Return True when `model_dir` looks like a LoRA adapter directory."""
    return (model_dir / "adapter_config.json").exists()


def has_model_weights(model_dir: Path) -> bool:
    """Return True when a full model directory contains serialized weights."""
    weight_patterns = (
        "*.safetensors",
        "pytorch_model*.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return any(any(model_dir.glob(pattern)) for pattern in weight_patterns)


def _patch_unsloth_gemma2_single_batch_mask_bug() -> bool:
    """Patch Unsloth Gemma-2 generation to normalize 2D masks for batch size 1."""
    try:
        import unsloth.models.gemma2 as gemma2_mod
    except Exception:
        return False

    if getattr(gemma2_mod, "_bioreview_single_batch_mask_patch", False):
        return False

    original = gemma2_mod.Gemma2Attention_fast_forward_inference

    def patched(
        self,
        hidden_states,
        past_key_value,
        position_ids,
        do_prefill=False,
        attention_mask=None,
        use_sliding_window=False,
        **kwargs,
    ):
        if (
            attention_mask is not None
            and isinstance(attention_mask, gemma2_mod.torch.Tensor)
            and attention_mask.dim() == 2
        ):
            bsz, q_len, _ = hidden_states.shape
            seq_len = past_key_value[0].shape[-2]
            sliding_window = self.config.sliding_window if use_sliding_window else None
            attention_mask = gemma2_mod._prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (bsz, q_len),
                hidden_states,
                seq_len,
                sliding_window=sliding_window,
            )

        return original(
            self,
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            position_ids=position_ids,
            do_prefill=do_prefill,
            attention_mask=attention_mask,
            use_sliding_window=use_sliding_window,
            **kwargs,
        )

    gemma2_mod.Gemma2Attention_fast_forward_inference = patched
    gemma2_mod._bioreview_original_gemma2_attention_fast_forward_inference = original
    gemma2_mod._bioreview_single_batch_mask_patch = True
    return True


def compute_max_input_tokens(max_seq_length: int, max_new_tokens: int) -> int:
    """Reserve room for generation inside the configured context window."""
    if max_new_tokens <= 0:
        return max_seq_length
    return max(1, max_seq_length - max_new_tokens)


def find_training_config_path(model_dir: Path) -> Path | None:
    """Find training_config.yaml for adapter roots or nested checkpoint dirs."""
    candidates = [
        model_dir / "training_config.yaml",
        model_dir.parent / "training_config.yaml",
        model_dir.parent.parent / "training_config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def infer_gemma2_training_prefill_cap(
    model_dir: Path, training_max_seq_length
):
    """Infer a safe Gemma-2 prefill cap from config or adapter naming."""
    if training_max_seq_length:
        return int(training_max_seq_length), "training_config"

    model_path = str(model_dir).lower()
    if "gemma2" not in model_path:
        return None, None
    if "8k" in model_path:
        return 8192, "model_dir_name"
    if "4k" in model_path:
        return 4096, "model_dir_name"
    return None, None


def maybe_cap_gemma2_unsloth_input_tokens(
    model_dir: Path,
    engine: str,
    requested_max_input_tokens: int,
    training_max_seq_length,
) -> tuple[int, str | None]:
    """Cap Gemma-2 Unsloth prompt length to the trained prefill length."""
    if engine != "unsloth":
        return requested_max_input_tokens, None
    if "gemma2" not in str(model_dir).lower():
        return requested_max_input_tokens, None

    inferred_cap, cap_source = infer_gemma2_training_prefill_cap(
        model_dir, training_max_seq_length
    )
    if not inferred_cap:
        return requested_max_input_tokens, None
    return min(requested_max_input_tokens, inferred_cap), cap_source


def maybe_cap_gemma2_hf_context_window(
    model_dir: Path,
    engine: str,
    requested_max_seq_length: int,
    training_max_seq_length,
) -> tuple[int, str | None]:
    """Cap Gemma-2 HF generation window to the model's native context length."""
    if engine != "hf":
        return requested_max_seq_length, None

    inferred_cap, cap_source = infer_gemma2_training_prefill_cap(
        model_dir, training_max_seq_length
    )
    if not inferred_cap:
        return requested_max_seq_length, None
    return min(requested_max_seq_length, inferred_cap), cap_source


def load_model_unsloth(model_dir: Path, max_seq_length: int, load_in_4bit: bool):
    """Load model with Unsloth (optimized inference).

    If ``from_pretrained`` fails on the adapter (e.g. PEFT doesn't support the
    target module type), fall back to manual adapter loading: base model →
    ``get_peft_model`` → load adapter weights.
    """
    from unsloth import FastLanguageModel

    if _patch_unsloth_gemma2_single_batch_mask_bug():
        print("  Applied Unsloth Gemma-2 single-batch mask patch.")

    adapter_config_path = model_dir / "adapter_config.json"

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_dir),
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
    except Exception as exc:
        if not adapter_config_path.exists():
            raise
        print(f"  Unsloth from_pretrained failed ({exc}), trying manual adapter load...")
        model, tokenizer = _load_adapter_manual(
            model_dir, adapter_config_path, max_seq_length, load_in_4bit,
        )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _load_adapter_manual(
    model_dir: Path,
    adapter_config_path: Path,
    max_seq_length: int,
    load_in_4bit: bool,
):
    """Load adapter by manually applying LoRA and loading weights.

    Bypasses PEFT ``from_pretrained`` which may not support custom layer types
    (e.g. ``Gemma4ClippableLinear``).
    """
    import safetensors.torch
    from unsloth import FastLanguageModel

    with adapter_config_path.open() as f:
        acfg = json.load(f)

    base_path = acfg["base_model_name_or_path"]
    print(f"  Manual adapter load — base model: {base_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=acfg["r"],
        lora_alpha=acfg["lora_alpha"],
        lora_dropout=acfg.get("lora_dropout", 0.0),
        target_modules=acfg["target_modules"],
        bias=acfg.get("bias", "none"),
        use_rslora=acfg.get("use_rslora", False),
    )

    # Load adapter weights with key remapping (saved vs model key formats may differ)
    adapter_path = model_dir / "adapter_model.safetensors"
    adapter_sd = safetensors.torch.load_file(str(adapter_path))
    model_sd = dict(model.named_parameters())

    loaded, skipped = 0, 0
    for ak, av in adapter_sd.items():
        matched = False
        # Try direct match
        if ak in model_sd and model_sd[ak].shape == av.shape:
            model_sd[ak].data.copy_(av)
            loaded += 1
            matched = True
        if not matched:
            # Try adding 'default.' (PEFT adapter naming convention)
            parts = ak.rsplit(".", 2)
            if len(parts) >= 3 and parts[-2] in ("lora_A", "lora_B"):
                remap = ".".join(parts[:-2] + [parts[-2], "default", parts[-1]])
                if remap in model_sd and model_sd[remap].shape == av.shape:
                    model_sd[remap].data.copy_(av)
                    loaded += 1
                    matched = True
        if not matched:
            skipped += 1
    print(f"  Adapter weights: {loaded} loaded, {skipped} skipped out of {len(adapter_sd)}")
    if loaded == 0:
        raise RuntimeError("No adapter weights matched — check key naming")

    return model, tokenizer


def _is_gemma4_adapter(adapter_cfg: dict) -> bool:
    """Return True for Gemma4 adapter configs saved by Unsloth/PEFT."""
    values = [
        adapter_cfg.get("base_model_name_or_path", ""),
        adapter_cfg.get("base_model_class", ""),
    ]
    auto_mapping = adapter_cfg.get("auto_mapping")
    if isinstance(auto_mapping, dict):
        values.extend(str(v) for v in auto_mapping.values())
    haystack = " ".join(str(v).lower() for v in values)
    return "gemma4" in haystack or "gemma-4" in haystack


def _load_hf_language_only_lora_adapter(model, model_dir: Path, adapter_cfg: dict):
    """Apply a Gemma4 text-only LoRA adapter with standard HF/PEFT.

    Gemma4 adapters trained through Unsloth may include zero-effect vision tower
    LoRA tensors. PEFT's default loader tries to inject those into
    ``Gemma4ClippableLinear`` wrappers and fails before text inference can run.
    For BioReview's text-only generation, load only language_model LoRA tensors.
    """
    import safetensors.torch
    from peft import LoraConfig, get_peft_model

    target_regex = (
        r".*language_model.*\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    )
    lora_config = LoraConfig(
        r=adapter_cfg["r"],
        lora_alpha=adapter_cfg["lora_alpha"],
        target_modules=target_regex,
        lora_dropout=adapter_cfg.get("lora_dropout", 0.0),
        bias=adapter_cfg.get("bias", "none"),
        task_type=adapter_cfg.get("task_type", "CAUSAL_LM"),
        use_rslora=adapter_cfg.get("use_rslora", False),
    )

    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)

    adapter_path = model_dir / "adapter_model.safetensors"
    adapter_sd = safetensors.torch.load_file(str(adapter_path))
    model_sd = dict(model.named_parameters())

    loaded, skipped, filtered = 0, 0, 0
    for ak, av in adapter_sd.items():
        if ".language_model." not in ak:
            filtered += 1
            continue

        candidates = [ak]
        parts = ak.rsplit(".", 2)
        if len(parts) >= 3 and parts[-2] in ("lora_A", "lora_B"):
            candidates.append(".".join(parts[:-2] + [parts[-2], "default", parts[-1]]))

        for mk in candidates:
            param = model_sd.get(mk)
            if param is not None and param.shape == av.shape:
                param.data.copy_(av)
                loaded += 1
                break
        else:
            skipped += 1

    print(
        "  Gemma4 language-only adapter weights: "
        f"{loaded} loaded, {skipped} skipped, {filtered} non-language skipped"
    )
    if loaded == 0:
        raise RuntimeError("No Gemma4 language LoRA weights matched the HF model")
    if skipped:
        raise RuntimeError(
            f"{skipped} Gemma4 language LoRA weights did not match the HF model"
        )

    return model


def load_model_hf(model_dir: Path, load_in_4bit: bool):
    """Fallback: load with standard HuggingFace + PEFT."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_config_path = model_dir / "adapter_config.json"

    if adapter_config_path.exists():
        # LoRA adapter — load base model then apply adapter
        from peft import PeftModel

        with adapter_config_path.open() as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path", "")
        if not base_model_name:
            print(
                "ERROR: adapter_config.json missing base_model_name_or_path",
                file=sys.stderr,
            )
            raise SystemExit(1)

        # Load tokenizer from adapter dir if present, else from base model
        tokenizer_config = model_dir / "tokenizer_config.json"
        tokenizer_source = (
            str(model_dir) if tokenizer_config.exists() else base_model_name
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        load_kwargs: dict = {"device_map": "auto", "torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
        if _is_gemma4_adapter(adapter_cfg):
            print("  Gemma4 adapter detected: using HF language-only manual LoRA load.")
            model = _load_hf_language_only_lora_adapter(model, model_dir, adapter_cfg)
        else:
            model = PeftModel.from_pretrained(model, str(model_dir))
    else:
        # Merged full model
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        load_kwargs = {"device_map": "auto", "torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)

    model.eval()
    return model, tokenizer


def _unwrap_processor(tokenizer_or_processor):
    """If tokenizer is actually a multimodal processor, extract the text tokenizer.

    Vision-language models (e.g. Qwen3.5-9B) return a processor from
    from_pretrained() that wraps the text tokenizer. Calling processor(text)
    triggers image parsing which fails for text-only inference. We keep the
    processor for apply_chat_template (handles string content correctly) but
    use the underlying text tokenizer for encoding.
    """
    # Check if it's a processor wrapping a tokenizer
    inner = getattr(tokenizer_or_processor, "tokenizer", None)
    if inner is not None and hasattr(inner, "encode") and inner is not tokenizer_or_processor:
        return tokenizer_or_processor, inner  # (processor, text_tokenizer)
    return tokenizer_or_processor, tokenizer_or_processor  # same object for both


def set_truncation_side(tokenizer, side: str) -> None:
    """Apply truncation side consistently to tokenizer wrappers and base tokenizers."""
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported truncation side: {side}")

    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    for candidate in (tokenizer, text_tokenizer):
        if hasattr(candidate, "truncation_side"):
            candidate.truncation_side = side


def load_model_vllm(model_dir: Path, max_seq_length: int):
    """Load model with vLLM for fast inference (requires merged weights)."""
    from vllm import LLM
    from transformers import AutoTokenizer

    if is_adapter_dir(model_dir):
        raise ValueError(
            "vLLM requires a merged full model directory, not a LoRA adapter. "
            "Merge the adapter first (for example with scripts/merge_lora_adapter.py)."
        )
    if not has_model_weights(model_dir):
        raise ValueError(
            "vLLM model directory is missing model weight files. "
            f"{model_dir} looks incomplete; rerun scripts/merge_lora_adapter.py."
        )

    model = LLM(
        model=str(model_dir),
        max_model_len=max_seq_length,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    return model, tokenizer


def load_model(
    model_dir: Path,
    max_seq_length: int,
    load_in_4bit: bool,
    engine_pref: str = "auto",
):
    """Load model, preferring Unsloth if available.

    Returns (model, tokenizer, text_tokenizer, engine_name).
    tokenizer: full processor/tokenizer (for apply_chat_template)
    text_tokenizer: text-only tokenizer (for encoding prompts)

    engine_pref: "auto" tries unsloth→hf, "vllm"/"unsloth"/"hf" forces that engine.
    """
    if engine_pref == "vllm":
        model, tokenizer = load_model_vllm(model_dir, max_seq_length)
        processor, text_tok = _unwrap_processor(tokenizer)
        return model, processor, text_tok, "vllm"

    if engine_pref in ("auto", "unsloth"):
        try:
            model, tokenizer = load_model_unsloth(
                model_dir, max_seq_length, load_in_4bit
            )
            processor, text_tok = _unwrap_processor(tokenizer)
            if text_tok is not processor:
                print(f"  VL model detected: using inner text tokenizer for encoding")
            return model, processor, text_tok, "unsloth"
        except ImportError:
            if engine_pref == "unsloth":
                raise
        except Exception as exc:
            if engine_pref == "unsloth":
                raise
            print(
                f"  Unsloth load failed ({type(exc).__name__}: {exc}), "
                "falling back to HuggingFace..."
            )

    model, tokenizer = load_model_hf(model_dir, load_in_4bit)
    processor, text_tok = _unwrap_processor(tokenizer)
    return model, processor, text_tok, "hf"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def detect_chat_template_family(tokenizer) -> str:
    """Auto-detect chat template family from tokenizer output."""
    test_messages = [
        {"role": "user", "content": "X"},
        {"role": "assistant", "content": "Y"},
    ]
    formatted = tokenizer.apply_chat_template(
        test_messages, tokenize=False, add_generation_prompt=False
    )
    if "<|im_start|>" in formatted:
        return "chatml"
    if "<｜User｜>" in formatted:
        return "deepseek"
    if "<|turn>" in formatted:
        return "gemma4"
    if "<start_of_turn>" in formatted:
        return "gemma"
    return "unknown"


def normalize_messages_for_template(
    messages: list[dict], template_family: str
) -> list[dict]:
    """Adapt messages for chat templates that do not support system turns."""
    if template_family not in SYSTEMLESS_TEMPLATE_FAMILIES:
        return messages

    normalized: list[dict] = []
    pending_system: list[str] = []

    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if not role or not content:
            continue

        if role == "system":
            pending_system.append(content)
            continue

        if role == "user" and pending_system:
            content = "\n\n".join([*pending_system, content])
            pending_system = []
        elif pending_system:
            normalized.append({"role": "user", "content": "\n\n".join(pending_system)})
            pending_system = []

        normalized.append({"role": role, "content": content})

    if pending_system:
        merged_system = "\n\n".join(pending_system)
        if normalized and normalized[0]["role"] == "user":
            normalized[0] = {
                "role": "user",
                "content": "\n\n".join([merged_system, normalized[0]["content"]]),
            }
        else:
            normalized.insert(0, {"role": "user", "content": merged_system})

    return normalized


def get_system_prompt(project_root: Path) -> str:
    """Read system prompt from SFT training data for exact match."""
    sft_train_path = project_root / "data" / "sft_train.jsonl"
    if sft_train_path.exists():
        try:
            with sft_train_path.open(encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    record = json.loads(first_line)
                    convs = record.get("conversations", [])
                    if convs and convs[0].get("from") == "system":
                        return convs[0]["value"]
                    print(
                        "WARNING: first SFT record has no system turn, "
                        "falling back to reconstruction",
                        file=sys.stderr,
                    )
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            print(
                f"WARNING: failed to read system prompt from SFT data ({exc}), "
                "falling back to reconstruction",
                file=sys.stderr,
            )

    # Fallback: reconstruct from prepare_sft_data
    from prepare_sft_data import build_sft_system_prompt, load_reviewer_constants

    workspace_root = project_root.parent
    reviewer_py = (
        workspace_root
        / "peer-review-benchmark"
        / "bioreview_bench"
        / "baseline"
        / "reviewer.py"
    )
    raw_sys, _ = load_reviewer_constants(reviewer_py)
    return build_sft_system_prompt(raw_sys)


def build_inference_prompt(
    entry: dict,
    system_prompt: str,
    tokenizer,
    template_family: str,
    token_budget: int,
    section_priority: list[str],
    count_tokens,
    enable_thinking: bool = True,
) -> str:
    """Build formatted inference prompt for a single article."""
    from prepare_sft_data import USER_PREFIX, build_paper_input

    prefix_tokens = count_tokens(USER_PREFIX)
    paper_budget = max(256, token_budget - prefix_tokens)

    paper_text = build_paper_input(
        entry=entry,
        token_budget=paper_budget,
        section_priority=section_priority,
        count_tokens=count_tokens,
    )
    user_message = USER_PREFIX + paper_text

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    messages = normalize_messages_for_template(messages, template_family)

    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if not enable_thinking:
        chat_kwargs["enable_thinking"] = False

    return tokenizer.apply_chat_template(messages, **chat_kwargs)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    max_seq_length: int = 16384,
    max_input_tokens=None,
    text_tokenizer=None,
    log_prefix: str | None = None,
) -> str:
    """Generate response for a single prompt.

    Args:
        text_tokenizer: text-only tokenizer for encoding (needed for VL models
            where `tokenizer` is a processor that triggers image parsing).
    """
    import torch

    enc = text_tokenizer if text_tokenizer is not None else tokenizer
    max_input_tokens = (
        max_input_tokens
        if max_input_tokens is not None
        else compute_max_input_tokens(max_seq_length, max_new_tokens)
    )
    inputs = enc(
        prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    allowed_new_tokens = max(1, max_seq_length - input_ids.shape[-1])
    final_max_new_tokens = min(max_new_tokens, allowed_new_tokens)

    if log_prefix:
        print(
            f"{log_prefix}: prompt_tokens={input_ids.shape[-1]}, "
            f"max_input_tokens={max_input_tokens}, "
            f"max_new_tokens={final_max_new_tokens}, "
            f"context_used={input_ids.shape[-1] + final_max_new_tokens}/{max_seq_length}"
        )

    gen_kwargs: dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": final_max_new_tokens,
        "pad_token_id": (
            enc.pad_token_id
            if enc.pad_token_id is not None
            else enc.eos_token_id
        ),
        "repetition_penalty": repetition_penalty,
    }

    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    # Decode only generated tokens
    generated_ids = outputs[0][input_ids.shape[-1] :]
    return enc.decode(generated_ids, skip_special_tokens=True)


def generate_one_vllm(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    max_seq_length: int,
    max_input_tokens=None,
    log_prefix: str | None = None,
) -> str:
    """Generate response using vLLM engine."""
    from vllm import SamplingParams

    max_input_tokens = (
        max_input_tokens
        if max_input_tokens is not None
        else compute_max_input_tokens(max_seq_length, max_new_tokens)
    )
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens
    )
    prompt = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
    allowed_new_tokens = max(1, max_seq_length - encoded["input_ids"].shape[-1])
    final_max_new_tokens = min(max_new_tokens, allowed_new_tokens)

    if log_prefix:
        print(
            f"{log_prefix}: prompt_tokens={encoded['input_ids'].shape[-1]}, "
            f"max_input_tokens={max_input_tokens}, "
            f"max_new_tokens={final_max_new_tokens}, "
            f"context_used={encoded['input_ids'].shape[-1] + final_max_new_tokens}/{max_seq_length}"
        )

    params = SamplingParams(
        max_tokens=final_max_new_tokens,
        temperature=temperature if temperature > 0 else 0,
        repetition_penalty=repetition_penalty,
        top_p=0.9 if temperature > 0 else 1.0,
    )
    outputs = model.generate([prompt], params, use_tqdm=False)
    return outputs[0].outputs[0].text


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split_articles(path: Path) -> list[dict]:
    """Load articles from a split JSONL file."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_existing_output(path: Path) -> tuple[list[dict], dict]:
    """Load existing inference JSONL and summarize current on-disk state."""
    rows: list[dict] = []
    stats = {
        "processed": 0,
        "failed_parse": 0,
        "total_concerns": 0,
        "total_time": 0.0,
    }

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
            stats["processed"] += 1
            stats["total_concerns"] += len(row.get("concerns", []))
            stats["total_time"] += float(row.get("generation_time_s", 0.0) or 0.0)
            if not row.get("parse_ok", False):
                stats["failed_parse"] += 1

    return rows, stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_evaluation_wrapper(
    output_path: Path,
    splits_dir: Path,
    split: str,
    threshold: float,
    algorithm: str,
    model_name: str,
) -> dict:
    """Run bioreview_bench evaluation and return metrics dict."""
    workspace_root = Path(__file__).resolve().parents[2]
    bench_root = workspace_root / "peer-review-benchmark"
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    # Prefer local SPECTER2 (only if weight files actually exist)
    local_specter2 = Path(__file__).resolve().parents[1] / "models" / "specter2_base"
    has_weights = any(local_specter2.glob("*.bin")) or any(local_specter2.glob("*.safetensors"))
    if local_specter2.exists() and has_weights and not os.getenv("BIOREVIEW_EMBED_MODEL"):
        os.environ["BIOREVIEW_EMBED_MODEL"] = str(local_specter2)

    from bioreview_bench.evaluate.runner import run_evaluation

    result, coverage = run_evaluation(
        tool_output=output_path,
        splits_dir=splits_dir,
        split=split,
        threshold=threshold,
        exclude_figure=True,
        use_embedding=True,
        algorithm=algorithm,
        bootstrap_n=200,
        tool_name=model_name,
        tool_version="sft-v1",
        extraction_manifest_id="em-v3",
        notes=f"SFT model inference from {model_name}",
    )

    metrics = {
        "recall_overall": float(result.recall_overall),
        "precision_overall": float(result.precision_overall),
        "f1_micro": float(result.f1_micro),
        "recall_major": float(result.recall_major),
        "n_articles": result.n_articles,
        "n_human_concerns": result.n_human_concerns,
        "n_tool_concerns": result.n_tool_concerns,
    }

    # Per-article coverage stats
    n_zero_recall = sum(1 for c in coverage if c["recall"] == 0.0)
    n_perfect = sum(1 for c in coverage if c["recall"] == 1.0)
    metrics["n_zero_recall_articles"] = n_zero_recall
    metrics["n_perfect_recall_articles"] = n_perfect

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    default_splits = workspace_root / "peer-review-benchmark" / "data" / "splits" / "v3"

    p = argparse.ArgumentParser(
        description="Run inference with a fine-tuned BioReview model."
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing the trained model (adapter or merged).",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to run inference on.",
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        default=default_splits,
        help="Directory containing split JSONL files.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: results/sft_eval/<model>_<split>.jsonl).",
    )
    p.add_argument(
        "--max-articles",
        type=int,
        default=0,
        help="If > 0, process only first N articles (for testing).",
    )
    p.add_argument(
        "--token-budget",
        type=int,
        default=15000,
        help="Token budget for paper input (must match training).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Max tokens to generate per article.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (0 for greedy).",
    )
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Repetition penalty.",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=16384,
        help="Max sequence length for model loading.",
    )
    p.add_argument(
        "--no-4bit",
        dest="load_in_4bit",
        action="store_false",
        default=True,
        help="Disable 4-bit quantization (load in full/half precision).",
    )
    p.add_argument(
        "--engine",
        choices=["auto", "vllm", "unsloth", "hf"],
        default="auto",
        help="Inference engine: auto (unsloth→hf), vllm (fast, needs merged model).",
    )
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after generation.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Concern matching threshold for evaluation.",
    )
    p.add_argument(
        "--algorithm",
        choices=["hungarian", "greedy"],
        default="hungarian",
        help="Matching algorithm for evaluation.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip articles already in output JSONL.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Tag appended to output filename (e.g., 'full', 'step50').",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log progress every N articles (default: every article).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]

    load_in_4bit = args.load_in_4bit

    # Resolve model dir (relative to project root if not absolute)
    model_dir = args.model_dir
    if not model_dir.is_absolute():
        model_dir = project_root / model_dir

    if not model_dir.exists():
        print(f"ERROR: model directory not found: {model_dir}", file=sys.stderr)
        raise SystemExit(1)

    # Output path
    if args.output:
        output_path = args.output
    else:
        model_name = model_dir.name
        tag_suffix = f"_{args.tag}" if args.tag else ""
        output_path = (
            project_root
            / "results"
            / "sft_eval"
            / f"{model_name}_{args.split}{tag_suffix}.jsonl"
        )

    print("=" * 60)
    print("BioReview SFT Inference")
    print("=" * 60)
    print(f"model_dir:      {model_dir}")
    print(f"split:          {args.split}")
    print(f"splits_dir:     {args.splits_dir}")
    print(f"output:         {output_path}")
    print(f"max_articles:   {args.max_articles if args.max_articles > 0 else '(all)'}")
    print(f"temperature:    {args.temperature}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"requested_max_seq_length: {args.max_seq_length}")
    print(f"load_in_4bit:   {load_in_4bit}")
    print(f"evaluate:       {args.evaluate}")
    if args.tag:
        print(f"tag:            {args.tag}")
    print(f"log_every:      {args.log_every}")
    print()

    # ── Add scripts dir to path for prepare_sft_data imports ────
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from prepare_sft_data import FALLBACK_SECTION_PRIORITY, get_token_counter

    # ── Load model ──────────────────────────────────────────────
    print("Loading model...")
    t0 = time.time()
    model, tokenizer, text_tokenizer, engine = load_model(
        model_dir, args.max_seq_length, load_in_4bit, engine_pref=args.engine
    )
    print(f"Model loaded in {time.time() - t0:.1f}s (engine={engine})")

    # Ensure pad token
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    # ── Read enable_thinking from training config ──────────────
    enable_thinking = True
    template_family = None
    training_max_seq_length = None
    truncation_side = getattr(text_tokenizer, "truncation_side", "right")
    training_config_path = find_training_config_path(model_dir)
    if training_config_path is not None:
        with training_config_path.open(encoding="utf-8") as f:
            tcfg = yaml.safe_load(f)
        enable_thinking = tcfg.get("model", {}).get("enable_thinking", True)
        template_family = tcfg.get("model", {}).get("chat_template")
        training_max_seq_length = tcfg.get("model", {}).get("max_seq_length")
        truncation_side = tcfg.get("model", {}).get("truncation_side", truncation_side)
        if not enable_thinking:
            print(f"enable_thinking: False (from training config)")
    set_truncation_side(tokenizer, truncation_side)
    print(f"truncation_side: {truncation_side}")
    if not template_family:
        template_family = detect_chat_template_family(tokenizer)
    print(f"chat template: {template_family}")

    effective_max_seq_length, max_seq_length_cap_source = (
        maybe_cap_gemma2_hf_context_window(
            model_dir=model_dir,
            engine=engine,
            requested_max_seq_length=args.max_seq_length,
            training_max_seq_length=training_max_seq_length,
        )
    )
    print(f"effective_max_seq_length: {effective_max_seq_length}")
    if effective_max_seq_length != args.max_seq_length:
        print(
            "max_seq_length capped for Gemma-2 HF stability: "
            f"{effective_max_seq_length}"
        )
        if max_seq_length_cap_source:
            print(f"max_seq_length cap source: {max_seq_length_cap_source}")

    requested_max_input_tokens = compute_max_input_tokens(
        effective_max_seq_length, args.max_new_tokens
    )
    print(f"requested_max_input_tokens: {requested_max_input_tokens}")

    effective_max_input_tokens, max_input_cap_source = (
        maybe_cap_gemma2_unsloth_input_tokens(
        model_dir=model_dir,
        engine=engine,
        requested_max_input_tokens=requested_max_input_tokens,
        training_max_seq_length=training_max_seq_length,
    )
    )
    print(f"effective_max_input_tokens: {effective_max_input_tokens}")
    if effective_max_input_tokens != requested_max_input_tokens:
        print(
            "max_input_tokens capped for Gemma-2 Unsloth stability: "
            f"{effective_max_input_tokens}"
        )
        if max_input_cap_source:
            print(f"max_input_tokens cap source: {max_input_cap_source}")

    # ── Load system prompt and token counter ────────────────────
    system_prompt = get_system_prompt(project_root)
    count_tokens, token_backend = get_token_counter()
    section_priority = FALLBACK_SECTION_PRIORITY
    print(f"token_counter: {token_backend}")

    # ── Load articles ───────────────────────────────────────────
    split_path = args.splits_dir / f"{args.split}.jsonl"
    if not split_path.exists():
        print(f"ERROR: split file not found: {split_path}", file=sys.stderr)
        raise SystemExit(1)

    articles = load_split_articles(split_path)
    usable = [a for a in articles if len(a.get("concerns", [])) > 0]
    if args.max_articles > 0:
        usable = usable[: args.max_articles]
    print(f"Articles: {len(usable)} usable (from {len(articles)} total)")

    # ── Resume support ──────────────────────────────────────────
    done_ids: set[str] = set()
    existing_stats = {
        "processed": 0,
        "failed_parse": 0,
        "total_concerns": 0,
        "total_time": 0.0,
    }
    if args.resume and output_path.exists():
        existing_rows, existing_stats = load_existing_output(output_path)
        kept_rows = []
        for row in existing_rows:
            art_id = row.get("article_id", "")
            if art_id and row.get("parse_ok", False):
                done_ids.add(art_id)
                kept_rows.append(row)

        # Drop failed rows before appending so retries do not create duplicates.
        if len(kept_rows) != len(existing_rows):
            with output_path.open("w", encoding="utf-8") as f:
                for row in kept_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing_stats = {
                "processed": len(kept_rows),
                "failed_parse": 0,
                "total_concerns": sum(len(r.get("concerns", [])) for r in kept_rows),
                "total_time": sum(
                    float(r.get("generation_time_s", 0.0) or 0.0) for r in kept_rows
                ),
            }

        print(
            f"Resuming: {len(done_ids)} successful articles already present"
            + (
                f", dropped {len(existing_rows) - len(kept_rows)} failed rows for retry"
                if len(kept_rows) != len(existing_rows)
                else ""
            )
        )

    to_process = [
        a for a in usable if a.get("id", a.get("article_id", "")) not in done_ids
    ]
    print(f"To process: {len(to_process)} articles")

    # ── Generate ────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and output_path.exists()) else "w"

    # Sanitize: ensure file ends with newline before appending (crash mid-write)
    if mode == "a" and output_path.stat().st_size > 0:
        with open(output_path, "rb") as check:
            check.seek(-1, 2)
            if check.read(1) != b"\n":
                with open(output_path, "a", encoding="utf-8") as fix:
                    fix.write("\n")

    stats = dict(existing_stats)

    with output_path.open(mode, encoding="utf-8") as fh:
        for i, entry in enumerate(to_process):
            art_id = entry.get("id", entry.get("article_id", ""))

            t_start = time.time()
            try:
                prompt = build_inference_prompt(
                    entry=entry,
                    system_prompt=system_prompt,
                    tokenizer=tokenizer,
                    template_family=template_family,
                    token_budget=args.token_budget,
                    section_priority=section_priority,
                    count_tokens=count_tokens,
                    enable_thinking=enable_thinking,
                )

                if engine == "vllm":
                    raw_output = generate_one_vllm(
                        model=model,
                        tokenizer=text_tokenizer,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty,
                        max_seq_length=effective_max_seq_length,
                        max_input_tokens=effective_max_input_tokens,
                        log_prefix=f"    generation window [{i + 1}/{len(to_process)}]",
                    )
                else:
                    raw_output = generate_one(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty,
                        max_seq_length=effective_max_seq_length,
                        max_input_tokens=effective_max_input_tokens,
                        text_tokenizer=text_tokenizer,
                        log_prefix=f"    generation window [{i + 1}/{len(to_process)}]",
                    )

                structured, texts = parse_model_output(raw_output)
                parse_ok = len(texts) > 0
            except Exception as exc:
                raw_output = ""
                structured = []
                texts = []
                parse_ok = False
                print(
                    f"  [{i + 1}/{len(to_process)}] {art_id}: ERROR {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

            t_elapsed = time.time() - t_start

            row = {
                "article_id": art_id,
                "concerns": texts,
                "structured_concerns": structured,
                "raw_output": raw_output,
                "parse_ok": parse_ok,
                "generation_time_s": round(t_elapsed, 2),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            stats["processed"] += 1
            stats["total_concerns"] += len(texts)
            stats["total_time"] += t_elapsed
            if not parse_ok:
                stats["failed_parse"] += 1

            if (i + 1) % args.log_every == 0 or (i + 1) == len(to_process):
                avg_time = stats["total_time"] / stats["processed"]
                remaining = (len(to_process) - i - 1) * avg_time
                print(
                    f"  [{i + 1}/{len(to_process)}] {art_id}: "
                    f"{len(texts)} concerns, {t_elapsed:.1f}s "
                    f"(avg {avg_time:.1f}s/article, ~{remaining/60:.0f}min remaining)",
                    flush=True,
                )

            # Flush JSONL periodically
            if stats["processed"] % 5 == 0:
                fh.flush()

    # ── Summary ─────────────────────────────────────────────────
    print(f"\nGeneration complete.")
    print(f"  processed:     {stats['processed']}")
    print(f"  failed_parse:  {stats['failed_parse']}")
    print(f"  total_concerns: {stats['total_concerns']}")
    if stats["processed"] > 0:
        print(f"  avg_concerns:  {stats['total_concerns'] / stats['processed']:.1f}")
        print(
            f"  avg_time:      {stats['total_time'] / stats['processed']:.1f}s/article"
        )
        print(f"  total_time:    {stats['total_time'] / 60:.1f} minutes")
    print(f"  output:        {output_path}")

    # Save generation summary
    summary_path = output_path.with_suffix(".summary.json")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {
        "generated_at_utc": ts,
        "model_dir": str(model_dir),
        "engine": engine,
        "split": args.split,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        **stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ── Evaluation ──────────────────────────────────────────────
    if args.evaluate:
        # Free GPU memory before loading SPECTER2 for evaluation
        del model
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        print("\nRunning evaluation...")
        try:
            metrics = run_evaluation_wrapper(
                output_path=output_path,
                splits_dir=args.splits_dir,
                split=args.split,
                threshold=args.threshold,
                algorithm=args.algorithm,
                model_name=model_dir.name,
            )
            print(f"\nEvaluation Results:")
            print(f"  recall:       {metrics['recall_overall']:.4f}")
            print(f"  precision:    {metrics['precision_overall']:.4f}")
            print(f"  f1_micro:     {metrics['f1_micro']:.4f}")
            print(f"  recall_major: {metrics['recall_major']:.4f}")
            print(f"  articles:     {metrics['n_articles']}")
            print(f"  GT concerns:  {metrics['n_human_concerns']}")
            print(f"  model concerns: {metrics['n_tool_concerns']}")

            # Baseline comparison
            print(f"\nBaseline Comparison:")
            print(f"  GPT-4o-mini:      F1=0.6962  Recall=0.6472  Precision=0.7531")
            print(f"  Gemini-2.5-Flash: F1=0.4489  Recall=0.3011  Precision=0.8820")
            print(
                f"  {model_dir.name}:{' ' * max(0, 16 - len(model_dir.name))}"
                f"F1={metrics['f1_micro']:.4f}  "
                f"Recall={metrics['recall_overall']:.4f}  "
                f"Precision={metrics['precision_overall']:.4f}"
            )

            # Save eval metrics
            summary["eval_metrics"] = metrics
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        except Exception as exc:
            print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
