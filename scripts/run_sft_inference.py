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


def parse_model_output(text: str) -> tuple[list[dict], list[str]]:
    """Parse model output into (structured_concerns, text_concerns).

    Handles clean JSON, markdown fences, bracket extraction, and truncated JSON.
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

    return [], []


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_unsloth(model_dir: Path, max_seq_length: int, load_in_4bit: bool):
    """Load model with Unsloth (optimized inference)."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


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
        model = PeftModel.from_pretrained(model, str(model_dir))
    else:
        # Merged full model
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            device_map="auto",
            torch_dtype="auto",
        )

    model.eval()
    return model, tokenizer


def load_model(model_dir: Path, max_seq_length: int, load_in_4bit: bool):
    """Load model, preferring Unsloth if available.

    Returns (model, tokenizer, engine_name).
    """
    try:
        model, tokenizer = load_model_unsloth(model_dir, max_seq_length, load_in_4bit)
        return model, tokenizer, "unsloth"
    except ImportError:
        pass

    model, tokenizer = load_model_hf(model_dir, load_in_4bit)
    return model, tokenizer, "hf"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


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
) -> str:
    """Generate response for a single prompt."""
    import torch

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_seq_length
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    gen_kwargs: dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
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
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


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

    # Prefer local SPECTER2
    local_specter2 = Path(__file__).resolve().parents[1] / "models" / "specter2_base"
    if local_specter2.exists() and not os.getenv("BIOREVIEW_EMBED_MODEL"):
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
        output_path = (
            project_root / "results" / "sft_eval" / f"{model_name}_{args.split}.jsonl"
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
    print(f"load_in_4bit:   {load_in_4bit}")
    print(f"evaluate:       {args.evaluate}")
    print()

    # ── Add scripts dir to path for prepare_sft_data imports ────
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from prepare_sft_data import FALLBACK_SECTION_PRIORITY, get_token_counter

    # ── Load model ──────────────────────────────────────────────
    print("Loading model...")
    t0 = time.time()
    model, tokenizer, engine = load_model(model_dir, args.max_seq_length, load_in_4bit)
    print(f"Model loaded in {time.time() - t0:.1f}s (engine={engine})")

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Read enable_thinking from training config ──────────────
    enable_thinking = True
    training_config_path = model_dir / "training_config.yaml"
    if training_config_path.exists():
        with training_config_path.open(encoding="utf-8") as f:
            tcfg = yaml.safe_load(f)
        enable_thinking = tcfg.get("model", {}).get("enable_thinking", True)
        if not enable_thinking:
            print(f"enable_thinking: False (from training config)")

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
    if args.resume and output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    art_id = row.get("article_id", "")
                    if art_id and row.get("parse_ok", False):
                        done_ids.add(art_id)
                except json.JSONDecodeError:
                    continue
        print(f"Resuming: {len(done_ids)} articles already processed")

    to_process = [
        a for a in usable if a.get("id", a.get("article_id", "")) not in done_ids
    ]
    print(f"To process: {len(to_process)} articles")

    # ── Generate ────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and output_path.exists()) else "w"

    stats = {
        "processed": 0,
        "failed_parse": 0,
        "total_concerns": 0,
        "total_time": 0.0,
    }

    with output_path.open(mode, encoding="utf-8") as fh:
        for i, entry in enumerate(to_process):
            art_id = entry.get("id", entry.get("article_id", ""))

            t_start = time.time()
            try:
                prompt = build_inference_prompt(
                    entry=entry,
                    system_prompt=system_prompt,
                    tokenizer=tokenizer,
                    token_budget=args.token_budget,
                    section_priority=section_priority,
                    count_tokens=count_tokens,
                    enable_thinking=enable_thinking,
                )

                raw_output = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    max_seq_length=args.max_seq_length,
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

            if (i + 1) % 10 == 0 or (i + 1) == len(to_process):
                avg_time = stats["total_time"] / stats["processed"]
                remaining = (len(to_process) - i - 1) * avg_time
                print(
                    f"  [{i + 1}/{len(to_process)}] {art_id}: "
                    f"{len(texts)} concerns, {t_elapsed:.1f}s "
                    f"(avg {avg_time:.1f}s/article, ~{remaining/60:.0f}min remaining)"
                )

            # Flush periodically
            if stats["processed"] % 20 == 0:
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
