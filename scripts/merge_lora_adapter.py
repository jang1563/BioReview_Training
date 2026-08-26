#!/usr/bin/env python3
"""Merge a trained LoRA adapter into a full model with Unsloth.

Supports adapters that Unsloth can load directly plus adapters that require
manual LoRA reconstruction (for example when PEFT does not recognize a custom
layer class such as Gemma 4's ``Gemma4ClippableLinear``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge a BioReview LoRA adapter into a full model directory."
    )
    p.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
        help="Directory containing adapter_config.json and adapter weights.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the merged full model into.",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=0,
        help="Override max sequence length. Defaults to training_config.yaml when present.",
    )
    p.add_argument(
        "--save-method",
        choices=["merged_16bit", "merged_4bit", "merged_4bit_forced"],
        default="merged_16bit",
        help="Unsloth save_pretrained_merged mode.",
    )
    p.add_argument(
        "--no-load-in-4bit",
        dest="load_in_4bit",
        action="store_false",
        default=True,
        help="Load the base model without 4-bit quantization before merging.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Write into an existing merged directory instead of skipping it.",
    )
    return p


def is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def has_model_weights(path: Path) -> bool:
    """Return True when a model directory contains serialized weights."""
    weight_patterns = (
        "*.safetensors",
        "pytorch_model*.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return any(any(path.glob(pattern)) for pattern in weight_patterns)


def read_adapter_config(adapter_dir: Path) -> dict:
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"Adapter config not found: {adapter_config_path}. "
            "This script expects a LoRA adapter directory."
        )
    with adapter_config_path.open(encoding="utf-8") as f:
        return json.load(f)


def infer_max_seq_length(adapter_dir: Path) -> int:
    training_config_path = adapter_dir / "training_config.yaml"
    if training_config_path.exists():
        with training_config_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        value = cfg.get("model", {}).get("max_seq_length")
        if value:
            return int(value)
    return 16384


def load_adapter_weights(model, adapter_dir: Path) -> tuple[int, int]:
    """Copy LoRA tensors from disk into an instantiated PEFT model."""
    import safetensors.torch

    adapter_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter weights not found: {adapter_path}")

    adapter_sd = safetensors.torch.load_file(str(adapter_path))
    model_params = dict(model.named_parameters())

    loaded = 0
    skipped = 0
    for adapter_key, value in adapter_sd.items():
        matched = False

        if adapter_key in model_params and model_params[adapter_key].shape == value.shape:
            model_params[adapter_key].data.copy_(value)
            loaded += 1
            matched = True

        if not matched:
            parts = adapter_key.rsplit(".", 2)
            if len(parts) >= 3 and parts[-2] in ("lora_A", "lora_B"):
                remapped = ".".join(parts[:-2] + [parts[-2], "default", parts[-1]])
                if remapped in model_params and model_params[remapped].shape == value.shape:
                    model_params[remapped].data.copy_(value)
                    loaded += 1
                    matched = True

        if not matched:
            skipped += 1

    return loaded, skipped


def load_unsloth_adapter(
    adapter_dir: Path,
    adapter_cfg: dict,
    max_seq_length: int,
    load_in_4bit: bool,
):
    """Load adapter with a direct path when possible, else rebuild LoRA manually."""
    from unsloth import FastLanguageModel

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir),
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        return model, tokenizer, "direct"
    except Exception as exc:
        print(
            f"Direct adapter load failed ({type(exc).__name__}: {exc}). "
            "Trying manual adapter reconstruction...",
            file=sys.stderr,
        )

    base_model_name = adapter_cfg["base_model_name_or_path"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=adapter_cfg["r"],
        lora_alpha=adapter_cfg["lora_alpha"],
        lora_dropout=adapter_cfg.get("lora_dropout", 0.0),
        target_modules=adapter_cfg["target_modules"],
        bias=adapter_cfg.get("bias", "none"),
        use_rslora=adapter_cfg.get("use_rslora", False),
    )
    loaded, skipped = load_adapter_weights(model, adapter_dir)
    if loaded == 0:
        raise RuntimeError(
            "No adapter tensors matched the reconstructed model. "
            "Check the adapter directory and target modules."
        )
    print(
        f"Manual adapter reconstruction loaded {loaded} tensors "
        f"and skipped {skipped}."
    )
    return model, tokenizer, "manual"


def main() -> None:
    args = build_parser().parse_args()

    adapter_dir = args.adapter_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not adapter_dir.exists():
        raise SystemExit(f"Adapter directory not found: {adapter_dir}")
    if not is_adapter_dir(adapter_dir):
        raise SystemExit(
            f"{adapter_dir} does not look like a LoRA adapter directory "
            "(missing adapter_config.json)."
        )

    if not args.overwrite and (output_dir / "config.json").exists():
        if has_model_weights(output_dir):
            print(f"Merged model already exists at {output_dir}; skipping.")
            return
        print(
            f"Found incomplete merged directory at {output_dir} "
            "(config present but no model weights). Rebuilding..."
        )

    max_seq_length = args.max_seq_length or infer_max_seq_length(adapter_dir)
    adapter_cfg = read_adapter_config(adapter_dir)

    print("=" * 60)
    print("Merge LoRA Adapter")
    print("=" * 60)
    print(f"adapter_dir:     {adapter_dir}")
    print(f"output_dir:      {output_dir}")
    print(f"base_model:      {adapter_cfg['base_model_name_or_path']}")
    print(f"max_seq_length:  {max_seq_length}")
    print(f"load_in_4bit:    {args.load_in_4bit}")
    print(f"save_method:     {args.save_method}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, load_mode = load_unsloth_adapter(
        adapter_dir=adapter_dir,
        adapter_cfg=adapter_cfg,
        max_seq_length=max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    print(f"adapter_load:    {load_mode}")
    print("Saving merged model...")
    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method=args.save_method,
    )

    if not (output_dir / "config.json").exists():
        raise RuntimeError(f"Merged model did not produce config.json at {output_dir}")
    if not has_model_weights(output_dir):
        raise RuntimeError(
            "Merged model did not produce any weight files. "
            f"{output_dir} still looks incomplete after save_pretrained_merged. "
            "If the adapter was trained from a 4-bit base, try loading a 16-bit base "
            "for the merge or use a 4-bit merge mode explicitly."
        )

    print("Merge complete.")
    print(f"merged_model:    {output_dir}")


if __name__ == "__main__":
    main()
