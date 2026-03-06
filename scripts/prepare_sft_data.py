#!/usr/bin/env python3
"""Convert bioreview-bench splits to Axolotl ShareGPT SFT JSONL format.

Default behavior (Phase 0):
- Convert train/val from peer-review-benchmark split v3.
- Filter concerns:
  - Exclude figure-dependent concerns
  - Keep concerns with resolution_confidence >= 0.8
  - Drop articles with < 3 remaining concerns
- Build ChatML-like ShareGPT conversations:
  - system: REVIEWER_SYSTEM (loaded from reviewer.py when available)
  - human: review instruction + prioritized, token-budgeted paper text
  - gpt: JSON array string of [{text, category, severity}, ...]
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

FALLBACK_SECTION_PRIORITY = [
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

FALLBACK_REVIEWER_SYSTEM = (
    "You are an expert peer reviewer for biomedical research papers published in "
    "high-impact journals. Identify specific scientific concerns, weaknesses, and "
    "issues. Return only a JSON array of concern strings."
)

USER_PREFIX = (
    "Review the following biomedical article.\n\n"
    "Return ONLY a JSON array of objects with keys: "
    "`text`, `category`, `severity`.\n"
    "Allowed category values: design_flaw, statistical_methodology, "
    "missing_experiment, prior_art_novelty, writing_clarity, "
    "reagent_method_specificity, interpretation, other.\n"
    "Allowed severity values: major, minor, optional.\n\n"
)

TRUNCATION_MARKER = "\n[…truncated]"

SFT_OUTPUT_SPEC = (
    "OUTPUT FORMAT: Return a JSON array of concern objects, nothing else.\n"
    "Each item must be: {\"text\": string, \"category\": one of "
    "[design_flaw, statistical_methodology, missing_experiment, "
    "prior_art_novelty, writing_clarity, reagent_method_specificity, "
    "interpretation, other], \"severity\": one of [major, minor, optional]}.\n"
    "Do NOT return a JSON array of strings."
)


@dataclass
class SplitStats:
    split: str
    total_articles: int = 0
    kept_articles: int = 0
    dropped_too_few_concerns: int = 0
    dropped_title_only: int = 0
    skipped_bad_json: int = 0
    concerns_before: int = 0
    concerns_after: int = 0
    concern_count_per_article: list[int] = field(default_factory=list)
    input_tokens: list[int] = field(default_factory=list)
    category_counter: Counter[str] = field(default_factory=Counter)
    severity_counter: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict:
        token_mean = statistics.mean(self.input_tokens) if self.input_tokens else 0.0
        token_median = statistics.median(self.input_tokens) if self.input_tokens else 0.0
        token_max = max(self.input_tokens) if self.input_tokens else 0
        concern_mean = (
            statistics.mean(self.concern_count_per_article)
            if self.concern_count_per_article
            else 0.0
        )
        return {
            "split": self.split,
            "total_articles": self.total_articles,
            "kept_articles": self.kept_articles,
            "dropped_too_few_concerns": self.dropped_too_few_concerns,
            "dropped_title_only": self.dropped_title_only,
            "skipped_bad_json": self.skipped_bad_json,
            "concerns_before": self.concerns_before,
            "concerns_after": self.concerns_after,
            "avg_concerns_per_article": concern_mean,
            "input_tokens": {
                "mean": token_mean,
                "median": token_median,
                "max": token_max,
            },
            "category_distribution": dict(self.category_counter),
            "severity_distribution": dict(self.severity_counter),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve()
    workspace_root = here.parents[2]
    default_splits_dir = workspace_root / "peer-review-benchmark" / "data" / "splits" / "v3"
    default_reviewer = (
        workspace_root
        / "peer-review-benchmark"
        / "bioreview_bench"
        / "baseline"
        / "reviewer.py"
    )
    default_output_dir = here.parents[1] / "data"

    parser = argparse.ArgumentParser(
        description="Prepare ShareGPT SFT data from bioreview-bench splits."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=default_splits_dir,
        help="Directory containing split JSONL files (train/val/test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Output directory for converted JSONL files and stats.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to process (default: train val).",
    )
    parser.add_argument(
        "--reviewer-py",
        type=Path,
        default=default_reviewer,
        help="Path to reviewer.py to reuse REVIEWER_SYSTEM and section priority.",
    )
    parser.add_argument(
        "--hf-tokenizer",
        type=str,
        default="",
        help=(
            "Optional HuggingFace tokenizer path/name for accurate token counting. "
            "If unavailable, falls back to tiktoken then conservative heuristic."
        ),
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=15_000,
        help="Approximate token budget for user prompt + paper text.",
    )
    parser.add_argument(
        "--min-concerns",
        type=int,
        default=3,
        help="Drop articles with fewer than this many filtered concerns.",
    )
    parser.add_argument(
        "--min-resolution-confidence",
        type=float,
        default=0.8,
        help="Minimum concern resolution_confidence to keep.",
    )
    parser.add_argument(
        "--include-figure-issues",
        action="store_true",
        help="Keep figure_issue/requires_figure_reading concerns (default: excluded).",
    )
    parser.add_argument(
        "--drop-title-only",
        action="store_true",
        help=(
            "Drop entries that have title but no abstract and no non-empty paper sections."
        ),
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=0,
        help="If > 0, process at most this many articles per split.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print first N converted examples per split.",
    )
    parser.add_argument(
        "--keep-nonascii-json",
        action="store_true",
        help="Preserve non-ASCII in JSON output (default writes escaped ASCII).",
    )
    return parser


def load_reviewer_constants(reviewer_py: Path) -> tuple[str, list[str]]:
    if not reviewer_py.exists():
        return FALLBACK_REVIEWER_SYSTEM, FALLBACK_SECTION_PRIORITY

    source = reviewer_py.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(reviewer_py))
    system = FALLBACK_REVIEWER_SYSTEM
    section_priority = FALLBACK_SECTION_PRIORITY

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target_names = [
            t.id for t in node.targets if isinstance(t, ast.Name)
        ]
        if "REVIEWER_SYSTEM" in target_names:
            try:
                value = ast.literal_eval(node.value)
                if isinstance(value, str) and value.strip():
                    system = value
            except (ValueError, SyntaxError):
                pass
        if "_SECTION_PRIORITY" in target_names:
            try:
                value = ast.literal_eval(node.value)
                if (
                    isinstance(value, list)
                    and value
                    and all(isinstance(x, str) for x in value)
                ):
                    section_priority = value
            except (ValueError, SyntaxError):
                pass

    return system, section_priority


def build_sft_system_prompt(base_prompt: str) -> str:
    prompt = base_prompt.strip()
    prompt = prompt.replace(
        "CONCERN CATEGORIES (for your reference — output only concern text):",
        "CONCERN CATEGORIES:",
    )
    prompt = prompt.replace(
        "CONCERN CATEGORIES (for your reference - output only concern text):",
        "CONCERN CATEGORIES:",
    )
    prompt = re.sub(r"OUTPUT FORMAT:.*\Z", "", prompt, flags=re.DOTALL).rstrip()
    return f"{prompt}\n\n{SFT_OUTPUT_SPEC}"


def get_token_counter(hf_tokenizer: str = "") -> tuple[Callable[[str], int], str]:
    if hf_tokenizer:
        try:
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(hf_tokenizer)

            def _count(text: str) -> int:
                if not text:
                    return 0
                return len(tok.encode(text, add_special_tokens=False))

            return _count, f"transformers:{hf_tokenizer}"
        except Exception:
            pass

    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")

        def _count(text: str) -> int:
            if not text:
                return 0
            return len(enc.encode(text))

        return _count, "tiktoken:cl100k_base"
    except Exception:
        # Conservative fallback to reduce overflow risk under long-context training.
        # (~3 chars/token upper-bound style estimate)
        def _count(text: str) -> int:
            if not text:
                return 0
            return max(1, math.ceil(len(text) / 3))

        return _count, "heuristic:chars/3(conservative)"


def parse_jsonl_line(line: str) -> tuple[dict | None, bool]:
    line = line.strip()
    if not line:
        return None, False
    try:
        return json.loads(line), False
    except json.JSONDecodeError:
        return None, True


def normalize_whitespace(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def truncate_text_to_tokens(
    text: str,
    budget_tokens: int,
    count_tokens: Callable[[str], int],
) -> str:
    text = normalize_whitespace(text)
    if budget_tokens <= 0 or not text:
        return ""
    if count_tokens(text) <= budget_tokens:
        return text

    lo = 0
    hi = len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid]
        if count_tokens(candidate) <= budget_tokens:
            lo = mid
        else:
            hi = mid - 1

    truncated = text[:lo].rstrip()
    if not truncated:
        return ""
    if truncated != text:
        marker_tokens = count_tokens(TRUNCATION_MARKER)
        text_budget = max(1, budget_tokens - marker_tokens)
        if count_tokens(truncated) > text_budget:
            lo2 = 0
            hi2 = len(truncated)
            while lo2 < hi2:
                mid2 = (lo2 + hi2 + 1) // 2
                candidate2 = truncated[:mid2]
                if count_tokens(candidate2) <= text_budget:
                    lo2 = mid2
                else:
                    hi2 = mid2 - 1
            truncated = truncated[:lo2].rstrip()
        if truncated:
            candidate_with_marker = truncated + TRUNCATION_MARKER
            if count_tokens(candidate_with_marker) <= budget_tokens:
                truncated = candidate_with_marker
    return truncated


def prioritize_sections(
    sections: dict[str, str],
    section_priority: list[str],
) -> list[tuple[str, str]]:
    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()

    for priority_name in section_priority:
        for name, text in sections.items():
            key = name.lower()
            if key in seen:
                continue
            if priority_name in key:
                ordered.append((name, text))
                seen.add(key)

    for name, text in sections.items():
        key = name.lower()
        if key not in seen:
            ordered.append((name, text))
            seen.add(key)

    return ordered


def allocate_section_budgets(
    section_blocks: list[tuple[str, int]],
    budget_tokens: int,
) -> dict[str, int]:
    if not section_blocks or budget_tokens <= 0:
        return {}

    full_sum = sum(tokens for _, tokens in section_blocks)
    if full_sum <= budget_tokens:
        return {name: tokens for name, tokens in section_blocks}

    n = len(section_blocks)
    min_each = min(128, max(16, budget_tokens // (n * 2)))
    base_total = min_each * n
    if base_total >= budget_tokens:
        even = max(1, budget_tokens // n)
        budgets = {name: even for name, _ in section_blocks}
        remainder = budget_tokens - (even * n)
        for i in range(remainder):
            budgets[section_blocks[i % n][0]] += 1
        return budgets

    remaining = budget_tokens - base_total
    shares: list[tuple[str, int, float]] = []
    for name, tok in section_blocks:
        frac = (tok / full_sum) if full_sum > 0 else 0.0
        shares.append((name, tok, frac))

    budgets = {name: min_each for name, _, _ in shares}
    used = base_total
    fractional_parts: list[tuple[str, float]] = []
    for name, _, frac in shares:
        extra_exact = remaining * frac
        extra_floor = int(extra_exact)
        budgets[name] += extra_floor
        used += extra_floor
        fractional_parts.append((name, extra_exact - extra_floor))

    remainder = max(0, budget_tokens - used)
    fractional_parts.sort(key=lambda x: x[1], reverse=True)
    for i in range(remainder):
        budgets[fractional_parts[i % len(fractional_parts)][0]] += 1

    return budgets


def build_paper_input(
    entry: dict,
    token_budget: int,
    section_priority: list[str],
    count_tokens: Callable[[str], int],
) -> str:
    def _join_parts(items: list[str]) -> str:
        return "\n\n".join(items).strip()

    parts: list[str] = []
    remaining = token_budget

    title = normalize_whitespace(str(entry.get("title", "")))
    if title:
        title_block = f"# {title}"
        title_block = truncate_text_to_tokens(title_block, remaining, count_tokens)
        if title_block:
            parts.append(title_block)
            remaining -= count_tokens(title_block)

    abstract = normalize_whitespace(str(entry.get("abstract", "")))
    if abstract and remaining > 0:
        abstract_block = f"## Abstract\n{abstract}"
        abstract_block = truncate_text_to_tokens(abstract_block, remaining, count_tokens)
        if abstract_block:
            parts.append(abstract_block)
            remaining -= count_tokens(abstract_block)

    if remaining <= 0:
        return _join_parts(parts)

    raw_sections = entry.get("paper_text_sections") or entry.get("sections") or {}
    if not isinstance(raw_sections, dict) or not raw_sections:
        return _join_parts(parts)

    cleaned_sections: dict[str, str] = {}
    for name, text in raw_sections.items():
        if text is None:
            continue
        name_str = str(name).strip()
        text_str = normalize_whitespace(str(text))
        if not name_str or not text_str:
            continue
        cleaned_sections[name_str] = text_str

    if not cleaned_sections:
        return _join_parts(parts)

    ordered = prioritize_sections(cleaned_sections, section_priority)
    blocks: list[tuple[str, str, int]] = []
    for name, text in ordered:
        block = f"## {name.title()}\n{text}"
        blocks.append((name, block, count_tokens(block)))

    budgets = allocate_section_budgets([(name, tok) for name, _, tok in blocks], remaining)

    for name, block, _ in blocks:
        sec_budget = budgets.get(name, 0)
        if sec_budget <= 0:
            continue
        clipped = truncate_text_to_tokens(block, sec_budget, count_tokens)
        if clipped:
            parts.append(clipped)

    return _join_parts(parts)


def filter_concerns(
    entry: dict,
    min_conf: float,
    include_figure_issues: bool,
) -> tuple[list[dict], int]:
    concerns = entry.get("concerns", [])
    if not isinstance(concerns, list):
        return [], 0

    kept: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for raw in concerns:
        if not isinstance(raw, dict):
            continue

        category = str(raw.get("category", "")).strip()
        text = normalize_whitespace(str(raw.get("concern_text", "")))
        severity = str(raw.get("severity", "")).strip()
        if not category or not text or not severity:
            continue

        if not include_figure_issues:
            if category == "figure_issue" or raw.get("requires_figure_reading", False):
                continue

        conf = raw.get("resolution_confidence", None)
        if conf is None:
            continue
        try:
            conf_value = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_value < min_conf:
            continue

        key = (text.lower(), category, severity)
        if key in seen:
            continue
        seen.add(key)
        kept.append({"text": text, "category": category, "severity": severity})

    return kept, len(concerns)


def is_title_only_entry(entry: dict) -> bool:
    abstract = normalize_whitespace(str(entry.get("abstract", "")))
    if abstract:
        return False

    sections = entry.get("paper_text_sections") or entry.get("sections") or {}
    if not isinstance(sections, dict):
        return True

    for _, text in sections.items():
        if text is None:
            continue
        if normalize_whitespace(str(text)):
            return False
    return True


def to_sharegpt_record(
    entry: dict,
    concerns: list[dict],
    system_prompt: str,
    token_budget: int,
    section_priority: list[str],
    count_tokens: Callable[[str], int],
) -> tuple[dict, int]:
    prefix_tokens = count_tokens(USER_PREFIX)
    paper_budget = max(256, token_budget - prefix_tokens)
    paper = build_paper_input(
        entry=entry,
        token_budget=paper_budget,
        section_priority=section_priority,
        count_tokens=count_tokens,
    )
    user_text = USER_PREFIX + paper
    if count_tokens(user_text) > token_budget:
        # Final hard cap in case section-level budgeting + marker effects
        # slightly overshoot the budget.
        lo = 0
        hi = len(paper)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = USER_PREFIX + paper[:mid].rstrip()
            if count_tokens(candidate) <= token_budget:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        user_text = best if best else USER_PREFIX
    assistant_text = json.dumps(concerns, ensure_ascii=False)

    record = {
        "id": entry.get("id", ""),
        "article_id": entry.get("id", ""),
        "source": entry.get("source", ""),
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": user_text},
            {"from": "gpt", "value": assistant_text},
        ],
    }
    return record, count_tokens(user_text)


def validate_jsonl(path: Path) -> tuple[bool, int]:
    lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            json.loads(stripped)
            lines += 1
    return True, lines


def process_split(
    split: str,
    input_path: Path,
    output_path: Path,
    stats_path: Path,
    system_prompt: str,
    section_priority: list[str],
    count_tokens: Callable[[str], int],
    token_budget: int,
    min_concerns: int,
    min_conf: float,
    include_figure_issues: bool,
    drop_title_only: bool,
    max_articles: int,
    preview_n: int,
    ensure_ascii: bool,
) -> tuple[SplitStats, list[dict]]:
    stats = SplitStats(split=split)
    previews: list[dict] = []

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if max_articles > 0 and stats.total_articles >= max_articles:
                break

            stats.total_articles += 1
            entry, is_bad_json = parse_jsonl_line(line)
            if entry is None:
                if is_bad_json:
                    stats.skipped_bad_json += 1
                continue

            if drop_title_only and is_title_only_entry(entry):
                stats.dropped_title_only += 1
                continue

            filtered_concerns, before_count = filter_concerns(
                entry=entry,
                min_conf=min_conf,
                include_figure_issues=include_figure_issues,
            )
            stats.concerns_before += before_count
            stats.concerns_after += len(filtered_concerns)

            if len(filtered_concerns) < min_concerns:
                stats.dropped_too_few_concerns += 1
                continue

            record, token_len = to_sharegpt_record(
                entry=entry,
                concerns=filtered_concerns,
                system_prompt=system_prompt,
                token_budget=token_budget,
                section_priority=section_priority,
                count_tokens=count_tokens,
            )
            fout.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")

            stats.kept_articles += 1
            stats.input_tokens.append(token_len)
            stats.concern_count_per_article.append(len(filtered_concerns))
            for c in filtered_concerns:
                stats.category_counter[c["category"]] += 1
                stats.severity_counter[c["severity"]] += 1

            if len(previews) < preview_n:
                previews.append(record)

    stats_path.write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
    return stats, previews


def print_split_summary(stats: SplitStats) -> None:
    summary = stats.as_dict()
    token_stats = summary["input_tokens"]
    print(f"\n[{stats.split}]")
    print(
        f"articles total={summary['total_articles']} kept={summary['kept_articles']} "
        f"dropped(title_only)={summary['dropped_title_only']} "
        f"dropped(<min_concerns)={summary['dropped_too_few_concerns']} "
        f"bad_json={summary['skipped_bad_json']}"
    )
    print(
        f"concerns before={summary['concerns_before']} after={summary['concerns_after']} "
        f"avg/article={summary['avg_concerns_per_article']:.2f}"
    )
    print(
        "input tokens "
        f"mean={token_stats['mean']:.1f} median={token_stats['median']:.1f} "
        f"max={token_stats['max']}"
    )
    print("top categories:")
    for cat, n in stats.category_counter.most_common(8):
        print(f"  - {cat}: {n}")


def print_previews(previews: list[dict], split: str) -> None:
    if not previews:
        return
    print(f"\nPreview ({split})")
    for i, rec in enumerate(previews, 1):
        conv = rec["conversations"]
        user = conv[1]["value"]
        gpt = conv[2]["value"]
        print(f"\n--- Example {i}: {rec.get('article_id', '')} ---")
        print("user:")
        print(user[:1200] + ("..." if len(user) > 1200 else ""))
        print("assistant:")
        print(gpt[:500] + ("..." if len(gpt) > 500 else ""))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_system_prompt, section_priority = load_reviewer_constants(args.reviewer_py)
    system_prompt = build_sft_system_prompt(raw_system_prompt)
    count_tokens, token_backend = get_token_counter(args.hf_tokenizer)

    print("prepare_sft_data configuration")
    print(f"- splits_dir: {args.splits_dir}")
    print(f"- output_dir: {args.output_dir}")
    print(f"- splits: {args.splits}")
    print(f"- token_budget: {args.token_budget}")
    print(f"- min_concerns: {args.min_concerns}")
    print(f"- min_resolution_confidence: {args.min_resolution_confidence}")
    print(f"- include_figure_issues: {args.include_figure_issues}")
    print(f"- drop_title_only: {args.drop_title_only}")
    print(f"- hf_tokenizer: {args.hf_tokenizer or '(none)'}")
    print(f"- token_counter: {token_backend}")
    print(f"- reviewer_py: {args.reviewer_py}")

    split_summaries: list[SplitStats] = []
    for split in args.splits:
        input_path = args.splits_dir / f"{split}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Split not found: {input_path}")

        output_path = args.output_dir / f"sft_{split}.jsonl"
        stats_path = args.output_dir / f"sft_{split}_stats.json"
        stats, previews = process_split(
            split=split,
            input_path=input_path,
            output_path=output_path,
            stats_path=stats_path,
            system_prompt=system_prompt,
            section_priority=section_priority,
            count_tokens=count_tokens,
            token_budget=args.token_budget,
            min_concerns=args.min_concerns,
            min_conf=args.min_resolution_confidence,
            include_figure_issues=args.include_figure_issues,
            drop_title_only=args.drop_title_only,
            max_articles=args.max_articles,
            preview_n=max(0, args.preview),
            ensure_ascii=not args.keep_nonascii_json,
        )
        validate_jsonl(output_path)
        split_summaries.append(stats)
        print_split_summary(stats)
        if args.preview > 0:
            print_previews(previews, split)

    total_in = sum(s.total_articles for s in split_summaries)
    total_kept = sum(s.kept_articles for s in split_summaries)
    print("\nDone.")
    print(f"total input articles={total_in}, total kept={total_kept}")


if __name__ == "__main__":
    main()
