#!/usr/bin/env python3
"""Fine-tune Qwen2.5-7B-Instruct on BioReview SFT data with QLoRA.

Primary path: Unsloth + TRL SFTTrainer (2-5x faster, 80% less VRAM).
Fallback path: Standard PEFT + bitsandbytes + TRL.

Usage:
  python scripts/train_sft.py --config configs/qwen7b_qlora.yaml
  python scripts/train_sft.py --config configs/qwen7b_qlora.yaml --dry-run
  python scripts/train_sft.py --config configs/qwen7b_qlora.yaml --max-steps 50

Requirements:
  pip install -r requirements-train.txt
  # For Unsloth (recommended): pip install unsloth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}

CHAT_TEMPLATES = {
    "chatml": {
        "instruction_template": "<|im_start|>user\n",
        "response_template": "<|im_start|>assistant\n",
    },
    "deepseek": {
        "instruction_template": "<\uff5cUser\uff5c>",
        "response_template": "<\uff5cAssistant\uff5c>",
    },
    "gemma": {
        "instruction_template": "<start_of_turn>user\n",
        "response_template": "<start_of_turn>model\n",
    },
}

SYSTEMLESS_TEMPLATE_FAMILIES = {"gemma"}


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
    if "\uff5cUser\uff5c" in formatted:
        return "deepseek"
    if "<start_of_turn>" in formatted:
        return "gemma"
    raise ValueError(
        f"Unknown chat template. Sample output:\n{formatted[:300]}"
    )


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


def get_text_tokenizer(tokenizer):
    """Normalize processor/tokenizer objects to a text tokenizer for .encode()."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def set_truncation_side(tokenizer, side: str) -> None:
    """Apply a truncation side consistently to tokenizer wrappers and base tokenizers."""
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported truncation side: {side}")

    text_tokenizer = get_text_tokenizer(tokenizer)
    for candidate in (tokenizer, text_tokenizer):
        if hasattr(candidate, "truncation_side"):
            candidate.truncation_side = side


def contains_token_subsequence(token_ids: list[int], needle: list[int]) -> bool:
    """Return True when `needle` appears contiguously within `token_ids`."""
    if not needle or len(needle) > len(token_ids):
        return False

    first = needle[0]
    window = len(needle)
    for idx, token_id in enumerate(token_ids[:-window + 1]):
        if token_id == first and token_ids[idx : idx + window] == needle:
            return True
    return False


def audit_response_truncation(
    dataset,
    text_tokenizer,
    max_seq_length: int,
    response_template: str,
    sample_size: int = 128,
) -> dict[str, int]:
    """Estimate how many samples retain the response marker after truncation."""
    total = min(sample_size, len(dataset))
    if total == 0:
        return {"sampled": 0, "over_limit": 0, "response_found": 0}

    response_ids = text_tokenizer.encode(
        response_template, add_special_tokens=False
    )
    over_limit = 0
    response_found = 0
    truncation_side = getattr(text_tokenizer, "truncation_side", "right")

    for idx in range(total):
        token_ids = text_tokenizer.encode(
            dataset[idx]["text"], add_special_tokens=False
        )
        if len(token_ids) > max_seq_length:
            over_limit += 1
            if truncation_side == "left":
                token_ids = token_ids[-max_seq_length:]
            else:
                token_ids = token_ids[:max_seq_length]
        if contains_token_subsequence(token_ids, response_ids):
            response_found += 1

    return {
        "sampled": total,
        "over_limit": over_limit,
        "response_found": response_found,
    }


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl_dataset(path: Path) -> list[dict]:
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


def sharegpt_to_messages(conversations: list[dict]) -> list[dict]:
    """Convert ShareGPT conversations to OpenAI-style messages."""
    return [
        {"role": ROLE_MAP.get(turn["from"], turn["from"]), "content": turn["value"]}
        for turn in conversations
    ]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune LLM on BioReview SFT data with QLoRA."
    )
    p.add_argument("--config", type=Path, required=True, help="YAML config path.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config, load data, show sample — then exit.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override max training steps (for quick test runs).",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a checkpoint directory.",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation during training (avoids eval OOM on large vocab models).",
    )
    p.add_argument(
        "--no-response-only",
        action="store_true",
        help="Train on full sequence instead of response-only loss masking.",
    )
    return p


def load_model_unsloth(mcfg: dict, lcfg: dict, seed: int):
    """Load model with Unsloth for optimized QLoRA training."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mcfg["name"],
        max_seq_length=mcfg["max_seq_length"],
        dtype=mcfg.get("dtype"),
        load_in_4bit=mcfg["load_in_4bit"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        target_modules=lcfg["target_modules"],
        bias=lcfg["bias"],
        use_gradient_checkpointing=lcfg["use_gradient_checkpointing"],
        use_rslora=lcfg.get("use_rslora", False),
        random_state=seed,
    )
    return model, tokenizer


def load_model_standard(mcfg: dict, lcfg: dict, use_bf16: bool = True):
    """Fallback: load model with standard PEFT + bitsandbytes."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model)
    if lcfg.get("use_gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        target_modules=lcfg["target_modules"],
        bias=lcfg["bias"],
        task_type="CAUSAL_LM",
        use_rslora=lcfg.get("use_rslora", False),
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / cfg["data"]["train_path"]
    val_path = project_root / cfg["data"]["val_path"]
    output_dir = project_root / cfg["output"]["dir"]

    mcfg = cfg["model"]
    lcfg = cfg["lora"]
    tcfg = cfg["training"]

    print("=" * 60)
    print("BioReview SFT Training")
    print("=" * 60)
    print(f"config:         {args.config}")
    print(f"model:          {mcfg['name']}")
    print(f"max_seq_length: {mcfg['max_seq_length']}")
    print(
        f"lora r={lcfg['r']} alpha={lcfg['lora_alpha']} dropout={lcfg['lora_dropout']}"
    )
    print(f"train_data:     {train_path} (exists={train_path.exists()})")
    print(f"val_data:       {val_path} (exists={val_path.exists()})")
    print(f"output_dir:     {output_dir}")
    print(f"epochs:         {tcfg['num_train_epochs']}")
    print(
        f"effective_batch: {tcfg['per_device_train_batch_size'] * tcfg['gradient_accumulation_steps']}"
    )
    print(f"lr:             {tcfg['learning_rate']}")
    print(f"max_steps:      {args.max_steps if args.max_steps > 0 else '(full)'}")
    print()

    if not train_path.exists():
        print(f"ERROR: train data not found: {train_path}", file=sys.stderr)
        raise SystemExit(1)
    if not val_path.exists():
        print(f"ERROR: val data not found: {val_path}", file=sys.stderr)
        raise SystemExit(1)

    # ── Load model ──────────────────────────────────────────────
    use_unsloth = False
    try:
        from unsloth import FastLanguageModel  # noqa: F401

        use_unsloth = True
    except ImportError:
        pass

    print(f"backend: {'unsloth' if use_unsloth else 'peft+bitsandbytes'}")

    if use_unsloth:
        model, tokenizer = load_model_unsloth(mcfg, lcfg, tcfg["seed"])
    else:
        model, tokenizer = load_model_standard(mcfg, lcfg, tcfg.get("bf16", False))
    text_tokenizer = get_text_tokenizer(tokenizer)
    response_only_enabled = not args.no_response_only
    truncation_side = mcfg.get("truncation_side")
    if truncation_side is None:
        truncation_side = "left" if response_only_enabled else getattr(
            text_tokenizer, "truncation_side", "right"
        )
    set_truncation_side(tokenizer, truncation_side)
    template_family = mcfg.get("chat_template") or detect_chat_template_family(
        tokenizer
    )

    model.print_trainable_parameters()

    # ── Load and format datasets ────────────────────────────────
    from datasets import Dataset

    raw_train = load_jsonl_dataset(train_path)
    raw_val = load_jsonl_dataset(val_path)
    print(f"loaded train={len(raw_train)}, val={len(raw_val)}")

    if len(raw_train) == 0:
        print("ERROR: train data loaded 0 valid rows", file=sys.stderr)
        raise SystemExit(1)
    if len(raw_val) == 0:
        print("ERROR: val data loaded 0 valid rows", file=sys.stderr)
        raise SystemExit(1)

    def format_example(example: dict) -> dict:
        messages = sharegpt_to_messages(example["conversations"])
        messages = normalize_messages_for_template(messages, template_family)
        chat_kwargs = {"tokenize": False, "add_generation_prompt": False}
        if mcfg.get("enable_thinking") is False:
            chat_kwargs["enable_thinking"] = False
        text = tokenizer.apply_chat_template(messages, **chat_kwargs)
        return {"text": text}

    train_ds = Dataset.from_list(raw_train).map(format_example)
    val_ds = Dataset.from_list(raw_val).map(format_example)

    # Remove original columns (keep only "text")
    drop_cols = [c for c in train_ds.column_names if c != "text"]
    train_ds = train_ds.remove_columns(drop_cols)
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c != "text"])

    # Sample verification
    sample_text = train_ds[0]["text"]
    sample_tokens = len(text_tokenizer.encode(sample_text))
    print(f"sample: {len(sample_text)} chars, ~{sample_tokens} tokens")
    print(f"preview:\n{sample_text[:300]}...")
    print(f"truncation_side: {truncation_side}")

    templates = CHAT_TEMPLATES[template_family]
    if response_only_enabled:
        response_audit = audit_response_truncation(
            train_ds,
            text_tokenizer,
            mcfg["max_seq_length"],
            templates["response_template"],
        )
        print(
            "response marker after truncation "
            f"(sampled {response_audit['sampled']}): "
            f"{response_audit['response_found']}/{response_audit['sampled']} retained"
        )
        print(
            f"  over max_seq_length in sample: {response_audit['over_limit']}/"
            f"{response_audit['sampled']}"
        )
        if (
            response_audit["over_limit"] > 0
            and response_audit["response_found"] < response_audit["sampled"]
        ):
            print(
                "WARNING: some sampled examples still lose the assistant response after "
                "truncation; consider reducing max_seq_length pressure or disabling "
                "response-only masking."
            )

    if args.dry_run:
        # Compute token length distribution
        lengths = [
            len(text_tokenizer.encode(train_ds[i]["text"])) for i in range(len(train_ds))
        ]
        lengths.sort()
        print(f"\nToken length distribution (train):")
        print(
            f"  min={lengths[0]}, median={lengths[len(lengths)//2]}, max={lengths[-1]}"
        )
        over = sum(1 for l in lengths if l > mcfg["max_seq_length"])
        print(
            f"  over max_seq_length ({mcfg['max_seq_length']}): {over}/{len(lengths)}"
        )
        print("\n[dry-run] Exiting without training.")
        return

    # ── Configure trainer ───────────────────────────────────────
    from trl import SFTConfig, SFTTrainer

    max_steps = args.max_steps if args.max_steps > 0 else -1

    from inspect import signature as _sig

    save_strategy = "steps" if max_steps > 0 else tcfg["save_strategy"]
    save_steps = max(1, max_steps // 2) if max_steps > 0 else tcfg.get("save_steps")
    eval_strategy = (
        "no" if args.no_eval else ("steps" if max_steps > 0 else tcfg.get("eval_strategy", "epoch"))
    )
    eval_steps = max(1, max_steps // 2) if max_steps > 0 else tcfg.get("eval_steps")

    sft_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": tcfg["num_train_epochs"] if max_steps < 0 else 1,
        "max_steps": max_steps,
        "per_device_train_batch_size": tcfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": tcfg["gradient_accumulation_steps"],
        "learning_rate": tcfg["learning_rate"],
        "lr_scheduler_type": tcfg["lr_scheduler_type"],
        "warmup_ratio": tcfg["warmup_ratio"],
        "weight_decay": tcfg["weight_decay"],
        "max_grad_norm": tcfg["max_grad_norm"],
        "bf16": tcfg.get("bf16", False),
        "fp16": tcfg.get("fp16", not tcfg.get("bf16", False)),
        "seed": tcfg["seed"],
        "logging_steps": tcfg["logging_steps"],
        "save_strategy": save_strategy,
        "save_total_limit": tcfg["save_total_limit"],
        "optim": tcfg["optim"],
        "dataset_text_field": "text",
        "packing": False,
        "eval_strategy": eval_strategy,
        "per_device_eval_batch_size": 1,
        "report_to": "none",
    }

    if save_strategy == "steps" and save_steps is not None:
        sft_kwargs["save_steps"] = save_steps
    if eval_strategy == "steps" and eval_steps is not None:
        sft_kwargs["eval_steps"] = eval_steps

    sft_config_params = _sig(SFTConfig.__init__).parameters
    if "max_seq_length" in sft_config_params:
        sft_kwargs["max_seq_length"] = mcfg["max_seq_length"]
    elif "max_length" in sft_config_params:
        sft_kwargs["max_length"] = mcfg["max_seq_length"]

    sft_config = SFTConfig(**sft_kwargs)

    # ── Build data collator for response-only loss masking ─────
    data_collator = None
    print(f"chat template: {template_family}")

    if not args.no_response_only and not use_unsloth:
        try:
            from trl import DataCollatorForCompletionOnlyLM

            response_template = templates["response_template"]
            instruction_template = templates["instruction_template"]
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                instruction_template=instruction_template,
                tokenizer=tokenizer,
            )
            # Verify the template is found in a sample
            sample_ids = text_tokenizer.encode(
                train_ds[0]["text"], add_special_tokens=False
            )
            template_ids = text_tokenizer.encode(
                response_template, add_special_tokens=False
            )
            found = any(
                sample_ids[j : j + len(template_ids)] == template_ids
                for j in range(len(sample_ids))
            )
            if found:
                print("loss masking: DataCollatorForCompletionOnlyLM (verified)")
            else:
                print(
                    "WARNING: response template tokens not found in sample, "
                    "falling back to full-sequence training"
                )
                data_collator = None
        except Exception as exc:
            print(
                f"WARNING: DataCollatorForCompletionOnlyLM failed ({exc}), "
                "training on full sequence"
            )
            data_collator = None

    # ── Create trainer (TRL version-compatible) ─────────────────
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "args": sft_config,
    }
    if data_collator is not None:
        trainer_kwargs["data_collator"] = data_collator

    # TRL >= 0.14 renamed tokenizer → processing_class
    try:
        from inspect import signature as _sig

        if "processing_class" in _sig(SFTTrainer.__init__).parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer
    except Exception:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    # ── Apply Unsloth response-only loss masking (post-init) ────
    if not args.no_response_only:
        if use_unsloth:
            try:
                from unsloth.chat_templates import train_on_responses_only

                trainer = train_on_responses_only(
                    trainer,
                    instruction_part=templates["instruction_template"],
                    response_part=templates["response_template"],
                )
                print("loss masking: unsloth train_on_responses_only")
            except Exception as exc:
                print(
                    f"WARNING: train_on_responses_only failed ({exc}), "
                    "training on full sequence"
                )
        elif data_collator is None:
            print("loss masking: disabled (training on full sequence)")
    else:
        print("loss masking: disabled (training on full sequence)")

    # ── Train ───────────────────────────────────────────────────
    print(f"\nStarting training...")
    t0 = time.time()

    resume_from = str(args.resume) if args.resume else None
    trainer.train(resume_from_checkpoint=resume_from)

    elapsed = time.time() - t0
    print(f"Training completed in {elapsed/60:.1f} minutes")

    # ── Save ────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training config for reproducibility
    config_save_path = output_dir / "training_config.yaml"
    with config_save_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Save training summary
    summary = {
        "model": mcfg["name"],
        "backend": "unsloth" if use_unsloth else "peft+bitsandbytes",
        "train_articles": len(raw_train),
        "val_articles": len(raw_val),
        "epochs": tcfg["num_train_epochs"],
        "max_steps": max_steps,
        "lora_r": lcfg["r"],
        "lora_alpha": lcfg["lora_alpha"],
        "lr": tcfg["learning_rate"],
        "effective_batch": tcfg["per_device_train_batch_size"]
        * tcfg["gradient_accumulation_steps"],
        "training_time_minutes": round(elapsed / 60, 1),
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nModel saved to:   {output_dir}")
    print(f"Config saved to:  {config_save_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
