#!/usr/bin/env python3
"""Run baseline LLMs and evaluate them on bioreview-bench split.

Examples:
  python scripts/run_baselines.py --split val --max-articles 20 --dry-run
  python scripts/run_baselines.py --split val --model-spec gemini=google:gemini-2.5-flash
  python scripts/run_baselines.py --split val --model-spec haiku=anthropic:claude-haiku-4-5-20251001 --evaluate
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ModelSpec:
    alias: str
    provider: str
    model: str


DEFAULT_MODEL_SPECS = [
    "haiku=anthropic:claude-haiku-4-5-20251001",
    "gpt4omini=openai:gpt-4o-mini",
    "gemini_flash=google:gemini-2.5-flash",
]

SECTION_PRIORITY = [
    "methods",
    "materials and methods",
    "materials & methods",
    "method",
    "methodology",
    "results",
    "result",
    "introduction",
    "intro",
    "background",
    "discussion",
    "conclusion",
    "conclusions",
    "supplementary",
    "supplemental",
    "supporting",
]

REVIEWER_SYSTEM = """\
You are an expert peer reviewer for biomedical research papers published in \
high-impact journals such as eLife, Nature, and Science. Your task is to \
carefully read the manuscript and identify specific scientific concerns, \
weaknesses, and issues that a rigorous reviewer would raise.

CONCERN CATEGORIES (for your reference — output only concern text):
- design_flaw: Fundamental experimental design problems (missing controls, confounders, inappropriate comparisons)
- statistical_methodology: Statistical errors (wrong test, missing corrections, effect sizes, p-hacking)
- missing_experiment: Key experiments absent that are needed to support the main claim
- prior_art_novelty: Missing citations, overstated novelty, inadequate comparison to prior work
- writing_clarity: Unclear descriptions, undefined terms, missing methods details
- reagent_method_specificity: Missing reagent IDs, software versions, analysis parameters
- interpretation: Over-interpretation, correlation/causation confusion, unsupported generalizations
- other: Legitimate scientific concern not fitting above categories

RULES:
1. Generate 5-15 specific, actionable scientific concerns
2. Each concern should be a clear, self-contained statement (1-3 sentences)
3. Focus on scientific rigor: experimental design, methodology, statistics, interpretation
4. Do NOT generate concerns about figures — you cannot see them
5. Do NOT include general praise, vague criticism, or stylistic preferences
6. Each concern must be specific enough to be addressed by the authors
7. Prioritize major issues over minor ones

OUTPUT FORMAT: Return a JSON array of concern strings, nothing else:
["The study lacks a negative control for the X assay, making it impossible to ...", \
"The statistical analysis uses t-tests for multiple comparisons without correction ..."]"""


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", s)


def parse_model_spec(raw: str) -> ModelSpec:
    if "=" not in raw:
        raise ValueError(f"Invalid --model-spec '{raw}'. Expected alias=provider:model")
    alias, rhs = raw.split("=", 1)
    if ":" not in rhs:
        raise ValueError(f"Invalid --model-spec '{raw}'. Expected alias=provider:model")
    provider, model = rhs.split(":", 1)
    alias = alias.strip()
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in {"anthropic", "openai", "google"}:
        raise ValueError(f"Unsupported provider '{provider}' in --model-spec '{raw}'")
    if not alias or not model:
        raise ValueError(f"Invalid --model-spec '{raw}'. Empty alias/model.")
    return ModelSpec(alias=alias, provider=provider, model=model)


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve()
    workspace_root = here.parents[2]
    default_splits = workspace_root / "peer-review-benchmark" / "data" / "splits" / "v3"
    default_output = here.parents[1] / "results" / "baselines"
    default_eval = here.parents[1] / "results" / "baseline_eval"

    p = argparse.ArgumentParser(description="Run multi-model baselines and evaluate.")
    p.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to run.",
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        default=default_splits,
        help="Directory with split JSONL files.",
    )
    p.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Model spec: alias=provider:model. Can repeat.",
    )
    p.add_argument(
        "--max-articles",
        type=int,
        default=0,
        help="If > 0, run only first N usable articles.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory to write model output JSONL files.",
    )
    p.add_argument(
        "--eval-dir",
        type=Path,
        default=default_eval,
        help="Directory to write evaluation artifacts.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent API calls per model.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output JSONL.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate cost only. No API calls.",
    )
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluator after generation.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Concern matching threshold.",
    )
    p.add_argument(
        "--bootstrap-n",
        type=int,
        default=200,
        help="Bootstrap samples for CI during evaluation.",
    )
    p.add_argument(
        "--algorithm",
        choices=["hungarian", "greedy"],
        default="hungarian",
        help="Matching algorithm for evaluation.",
    )
    p.add_argument(
        "--no-embedding",
        action="store_true",
        help="Disable embedding and use lexical fallback.",
    )
    p.add_argument(
        "--include-figure-gt",
        action="store_true",
        help="Include figure concerns in GT (default excludes).",
    )
    p.add_argument(
        "--max-input-chars",
        type=int,
        default=80_000,
        help="Max chars for input paper text.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max output tokens per request.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    p.add_argument(
        "--allow-empty-concerns",
        action="store_true",
        help="Allow empty parsed concerns as successful output (default treats as failure).",
    )
    p.add_argument(
        "--gemini-free",
        action="store_true",
        help="Shortcut: run only Gemini free model via Google provider.",
    )
    p.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model id used with --gemini-free.",
    )
    return p


def _load_split(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _pricing_for(provider: str, model: str) -> dict[str, float]:
    # USD per 1M tokens, rough planning only.
    table = {
        ("anthropic", "claude-haiku-4-5-20251001"): {"input": 0.80, "output": 4.00},
        ("anthropic", "claude-sonnet-4-20250514"): {"input": 3.00, "output": 15.00},
        ("openai", "gpt-4o-mini"): {"input": 0.15, "output": 0.60},
        ("openai", "gpt-4o"): {"input": 2.50, "output": 10.00},
        ("google", "gemini-2.0-flash"): {"input": 0.10, "output": 0.40},
        ("google", "gemini-2.5-flash"): {"input": 0.30, "output": 2.50},
        ("google", "gemini-2.5-flash-lite"): {"input": 0.10, "output": 0.40},
    }
    if (provider, model) in table:
        return table[(provider, model)]
    if provider == "google":
        return {"input": 0.20, "output": 0.80}
    if provider == "openai":
        return {"input": 0.50, "output": 2.00}
    return {"input": 1.00, "output": 5.00}


def estimate_cost(articles: list[dict], provider: str, model: str, max_input_chars: int) -> dict[str, Any]:
    total_chars = 0
    for entry in articles:
        article_chars = len(entry.get("title", ""))
        article_chars += len(entry.get("abstract", ""))
        sections = entry.get("paper_text_sections") or entry.get("sections") or {}
        if isinstance(sections, dict):
            for text in sections.values():
                article_chars += len(str(text))
        total_chars += min(article_chars, max_input_chars)

    est_input_tokens = int(total_chars / 4)
    est_output_tokens = len(articles) * 1500
    p = _pricing_for(provider, model)
    est_cost = est_input_tokens * p["input"] / 1_000_000 + est_output_tokens * p["output"] / 1_000_000
    return {
        "n_articles": len(articles),
        "est_input_tokens": est_input_tokens,
        "est_output_tokens": est_output_tokens,
        "est_cost_usd": round(est_cost, 2),
        "provider": provider,
        "model": model,
    }


def _google_extract_text(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(resp, "candidates", None)
    if candidates:
        try:
            parts = candidates[0].content.parts
            out = []
            for part in parts:
                pt = getattr(part, "text", None)
                if isinstance(pt, str):
                    out.append(pt)
            joined = "".join(out).strip()
            if joined:
                return joined
        except Exception:
            pass
    return ""


def format_paper_input(entry: dict, max_input_chars: int) -> str:
    parts: list[str] = []
    budget = max_input_chars

    title = entry.get("title", "")
    if title:
        header = f"# {title}\n\n"
        parts.append(header)
        budget -= len(header)

    abstract = entry.get("abstract", "")
    if abstract and budget > 0:
        block = f"## Abstract\n{abstract}\n\n"
        if len(block) > budget:
            block = block[:budget]
        parts.append(block)
        budget -= len(block)

    sections = entry.get("paper_text_sections") or entry.get("sections") or {}
    if not isinstance(sections, dict) or not sections or budget <= 0:
        return "".join(parts)

    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for priority_name in SECTION_PRIORITY:
        for name, text in sections.items():
            lower = str(name).lower()
            if lower in seen:
                continue
            if priority_name in lower:
                ordered.append((str(name), str(text)))
                seen.add(lower)
    for name, text in sections.items():
        lower = str(name).lower()
        if lower not in seen:
            ordered.append((str(name), str(text)))
            seen.add(lower)

    for name, text in ordered:
        if budget <= 0:
            break
        block = f"## {name.title()}\n{text}\n\n"
        if len(block) > budget:
            parts.append(block[:budget])
            budget = 0
        else:
            parts.append(block)
            budget -= len(block)
    return "".join(parts)


def try_parse_string_array(text: str) -> list[str] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    out: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
        elif isinstance(item, dict):
            s = str(item.get("text", item.get("concern_text", ""))).strip()
            if s:
                out.append(s)
    return out


def parse_concerns(text: str) -> list[str]:
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, flags=re.DOTALL)
    if fence:
        result = try_parse_string_array(fence.group(1).strip())
        if result is not None:
            return result

    bracket = re.search(r"(\[.*?\])", text, flags=re.DOTALL)
    if bracket:
        result = try_parse_string_array(bracket.group(1))
        if result is not None:
            return result

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        result = try_parse_string_array(text[start : end + 1])
        if result is not None:
            return result

    # Recover concerns from truncated JSON arrays (common when model output is cut).
    # This progressively decodes complete JSON values after '[' and keeps only fully
    # parsed items, stopping at the first malformed/incomplete fragment.
    start = text.find("[")
    if start != -1:
        fragment = text[start:]
        decoder = json.JSONDecoder()
        out: list[str] = []
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
            if isinstance(value, str):
                s = value.strip()
                if s:
                    out.append(s)
            elif isinstance(value, dict):
                s = str(value.get("text", value.get("concern_text", ""))).strip()
                if s:
                    out.append(s)
            idx += consumed
        if out:
            return out
    return []


def build_provider_client(provider: str) -> Any:
    if provider == "anthropic":
        spec = importlib.util.find_spec("anthropic")
        if spec is None:
            raise ModuleNotFoundError(
                f"No module named 'anthropic' (python={sys.executable}, path0={sys.path[0] if sys.path else ''})"
            )
        anthropic = importlib.import_module("anthropic")  # type: ignore
        return anthropic.Anthropic()
    if provider == "openai":
        spec = importlib.util.find_spec("openai")
        if spec is None:
            raise ModuleNotFoundError(
                f"No module named 'openai' (python={sys.executable}, path0={sys.path[0] if sys.path else ''})"
            )
        openai = importlib.import_module("openai")  # type: ignore
        return openai.OpenAI()
    if provider == "google":
        return build_google_client()
    raise ValueError(f"Unsupported provider: {provider}")


def call_provider_llm(
    *,
    provider: str,
    client: Any,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> str:
    if provider == "anthropic":
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text  # type: ignore[union-attr]

    if provider == "openai":
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    mode, google_client = client
    if mode == "genai":
        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "system_instruction": system,
        }
        resp = google_client.models.generate_content(model=model, contents=user, config=config)
        return _google_extract_text(resp)

    if mode == "rest":
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={google_client}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini REST HTTP {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini REST connection error: {exc}") from exc

        data = json.loads(body)
        out: list[str] = []
        for cand in data.get("candidates", []):
            content = cand.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if isinstance(text, str):
                    out.append(text)
        return "".join(out).strip()

    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    model_obj = google_client.GenerativeModel(model_name=model, system_instruction=system)
    resp = model_obj.generate_content(user, generation_config=generation_config)
    return getattr(resp, "text", "") or _google_extract_text(resp)


def build_google_client() -> tuple[str, Any]:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()

    # Prefer SDKs if available.
    try:
        from google import genai  # type: ignore

        if api_key:
            return "genai", genai.Client(api_key=api_key)
        return "genai", genai.Client()
    except Exception:
        pass

    try:
        import google.generativeai as genai_legacy  # type: ignore

        if api_key:
            genai_legacy.configure(api_key=api_key)
        else:
            genai_legacy.configure()  # Reads GOOGLE_API_KEY from env if available.
        return "generativeai", genai_legacy
    except Exception as exc:
        if api_key:
            return "rest", api_key
        raise RuntimeError(
            "Google provider requested but no SDK and no GOOGLE_API_KEY/GEMINI_API_KEY found. "
            "Install `google-genai`/`google-generativeai` or set GOOGLE_API_KEY (or GEMINI_API_KEY) for REST fallback."
        ) from exc


def run_google_review(
    *,
    entry: dict,
    model: str,
    system: str,
    max_tokens: int,
    temperature: float,
    formatter: Any,
    parser: Any,
    client_bundle: tuple[str, Any],
) -> list[str]:
    mode, client = client_bundle
    user = formatter(entry)
    if not user.strip():
        return []

    if mode == "genai":
        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "system_instruction": system,
        }
        resp = client.models.generate_content(model=model, contents=user, config=config)
        raw = _google_extract_text(resp)
        return parser(raw)

    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    model_obj = client.GenerativeModel(model_name=model, system_instruction=system)
    resp = model_obj.generate_content(user, generation_config=generation_config)
    raw = getattr(resp, "text", "") or _google_extract_text(resp)
    return parser(raw)


def run_generation_for_model(
    *,
    spec: ModelSpec,
    articles: list[dict],
    output_path: Path,
    concurrency: int,
    resume: bool,
    allow_empty_concerns: bool,
    max_input_chars: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    resume_ids: set[str] = set()
    kept_rows: list[dict[str, Any]] = []
    if resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    art_id = row.get("article_id", "")
                    concerns = row.get("concerns", [])
                    if not isinstance(concerns, list):
                        concerns = []
                    if art_id and (allow_empty_concerns or len(concerns) > 0):
                        resume_ids.add(art_id)
                        kept_rows.append({"article_id": art_id, "concerns": concerns})
                except json.JSONDecodeError:
                    continue

    # If empty concern rows are not allowed, compact the output file so those
    # entries are retried on resume and we avoid duplicate article rows.
    if resume and output_path.exists() and not allow_empty_concerns:
        with output_path.open("w", encoding="utf-8") as f:
            for row in kept_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    to_process = [a for a in articles if a.get("id", a.get("article_id", "")) not in resume_ids]
    skipped = len(articles) - len(to_process)
    stats: dict[str, Any] = {
        "processed": 0,
        "skipped": skipped,
        "failed": 0,
        "total_concerns": 0,
        "error_examples": [],
    }
    client = build_provider_client(spec.provider)

    mode = "a" if resume_ids else "w"
    with output_path.open(mode, encoding="utf-8") as fh:
        def _one(entry: dict) -> tuple[str, list[str] | None, str | None]:
            art_id = entry.get("id", entry.get("article_id", ""))
            try:
                user = format_paper_input(entry, max_input_chars=max_input_chars)
                if not user.strip():
                    if allow_empty_concerns:
                        return art_id, [], None
                    return art_id, None, "EmptyInput: paper text empty after formatting."
                raw = call_provider_llm(
                    provider=spec.provider,
                    client=client,
                    model=spec.model,
                    system=REVIEWER_SYSTEM,
                    user=user,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                concerns = parse_concerns(raw)
                if not concerns and not allow_empty_concerns:
                    preview = re.sub(r"\s+", " ", raw).strip()[:240]
                    return art_id, None, f"EmptyConcerns: no parseable concerns. raw_preview={preview!r}"
                return art_id, concerns, None
            except Exception as exc:
                return art_id, None, f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(_one, e): e for e in to_process}
            for fut in as_completed(futures):
                art_id, concerns, err = fut.result()
                if concerns is None:
                    stats["failed"] += 1
                    if err and len(stats["error_examples"]) < 5:
                        stats["error_examples"].append(err)
                    continue
                row = {"article_id": art_id, "concerns": concerns}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                stats["processed"] += 1
                stats["total_concerns"] += len(concerns)

    return stats


def save_eval_artifacts(
    *,
    eval_dir: Path,
    split: str,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = eval_dir / f"baseline_summary_{split}_{ts}.json"
    md_path = eval_dir / f"baseline_summary_{split}_{ts}.md"

    payload = {
        "generated_at_utc": ts,
        "split": split,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    header = [
        f"# Baseline Summary ({split}, {ts})",
        "",
        "| alias | provider | model | processed | failed | est_cost_usd | recall | precision | f1_micro | recall_major |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    body = []
    for r in rows:
        body.append(
            "| {alias} | {provider} | `{model}` | {processed} | {failed} | {cost} | {recall} | {precision} | {f1} | {recall_major} |".format(
                alias=r["alias"],
                provider=r["provider"],
                model=r["model"],
                processed=r.get("processed", 0),
                failed=r.get("failed", 0),
                cost=r.get("est_cost_usd", "n/a"),
                recall=r.get("recall_overall", "n/a"),
                precision=r.get("precision_overall", "n/a"),
                f1=r.get("f1_micro", "n/a"),
                recall_major=r.get("recall_major", "n/a"),
            )
        )
    md_path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return json_path, md_path


def fmt_metric(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4f}"


def main() -> None:
    args = build_parser().parse_args()

    here = Path(__file__).resolve()
    workspace_root = here.parents[2]
    bench_root = workspace_root / "peer-review-benchmark"
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    # Prefer local SPECTER2 snapshot when available to avoid network flakiness.
    local_specter2 = here.parents[1] / "models" / "specter2_base"
    if local_specter2.exists() and not os.getenv("BIOREVIEW_EMBED_MODEL"):
        os.environ["BIOREVIEW_EMBED_MODEL"] = str(local_specter2)
        print(f"embedding_model_override={local_specter2}")

    split_path = args.splits_dir / f"{args.split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    if args.gemini_free:
        raw_specs = [f"gemini_free=google:{args.gemini_model}"]
    else:
        raw_specs = args.model_spec if args.model_spec else list(DEFAULT_MODEL_SPECS)
    specs = [parse_model_spec(s) for s in raw_specs]

    entries = _load_split(split_path)
    usable = [e for e in entries if len(e.get("concerns", [])) > 0]
    if args.max_articles and args.max_articles > 0:
        usable = usable[: args.max_articles]

    print(f"python={sys.executable}")
    print(f"split={args.split} usable_articles={len(usable)} from={split_path}")
    print("models:")
    for s in specs:
        print(f"  - {s.alias}: {s.provider}:{s.model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        out_name = f"{_safe_name(spec.alias)}_{args.split}.jsonl"
        output_path = args.output_dir / out_name
        cost = estimate_cost(usable, spec.provider, spec.model, args.max_input_chars)

        row: dict[str, Any] = {
            "alias": spec.alias,
            "provider": spec.provider,
            "model": spec.model,
            "output_path": str(output_path),
            "est_cost_usd": cost["est_cost_usd"],
            "est_input_tokens": cost["est_input_tokens"],
            "est_output_tokens": cost["est_output_tokens"],
            "n_articles": len(usable),
        }

        print(
            f"\n[{spec.alias}] provider={spec.provider} model={spec.model} "
            f"est_cost=${cost['est_cost_usd']:.2f} output={output_path}"
        )

        if args.dry_run:
            row["status"] = "dry_run"
            rows.append(row)
            continue

        try:
            stats = run_generation_for_model(
                spec=spec,
                articles=usable,
                output_path=output_path,
                concurrency=args.concurrency,
                resume=args.resume,
                allow_empty_concerns=args.allow_empty_concerns,
                max_input_chars=args.max_input_chars,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            row.update(stats)
            row["status"] = "generated"
            print(
                f"[{spec.alias}] processed={stats['processed']} skipped={stats['skipped']} "
                f"failed={stats['failed']} concerns={stats['total_concerns']}"
            )
            if stats.get("error_examples"):
                print(f"[{spec.alias}] error_examples:")
                for err in stats["error_examples"][:3]:
                    print(f"  - {err}")
        except Exception as exc:
            row["status"] = "generation_failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(f"[{spec.alias}] generation failed: {row['error']}")
            continue

        if args.evaluate:
            try:
                from bioreview_bench.evaluate.runner import run_evaluation

                result, _ = run_evaluation(
                    tool_output=output_path,
                    splits_dir=args.splits_dir,
                    split=args.split,
                    threshold=args.threshold,
                    exclude_figure=not args.include_figure_gt,
                    use_embedding=not args.no_embedding,
                    algorithm=args.algorithm,
                    bootstrap_n=args.bootstrap_n,
                    tool_name=spec.alias,
                    tool_version="baseline",
                    extraction_manifest_id="em-v3",
                    notes=f"provider={spec.provider}, model={spec.model}",
                )
                row["recall_overall"] = float(result.recall_overall)
                row["precision_overall"] = float(result.precision_overall)
                row["f1_micro"] = float(result.f1_micro)
                row["recall_major"] = float(result.recall_major)
                row["eval_status"] = "ok"
                print(
                    f"[{spec.alias}] eval recall={row['recall_overall']:.4f} "
                    f"precision={row['precision_overall']:.4f} f1={row['f1_micro']:.4f}"
                )
            except Exception as exc:
                row["eval_status"] = "failed"
                row["eval_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{spec.alias}] evaluation failed: {row['eval_error']}")

        rows.append(row)

    summary_json, summary_md = save_eval_artifacts(
        eval_dir=args.eval_dir,
        split=args.split,
        rows=rows,
    )

    print("\nSummary")
    for r in rows:
        print(
            f"- {r['alias']}: status={r.get('status')} "
            f"processed={r.get('processed', 0)} failed={r.get('failed', 0)} "
            f"cost=${r.get('est_cost_usd', 'n/a')} "
            f"f1={fmt_metric(r.get('f1_micro'))}"
        )
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")


if __name__ == "__main__":
    main()
