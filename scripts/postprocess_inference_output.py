#!/usr/bin/env python3
"""Postprocess SFT inference JSONL outputs and optionally reevaluate them.

Useful for quick ablations such as:
- exact text deduplication
- capping per-article concern count

This keeps the original raw model output untouched and rewrites only the parsed
`concerns` / `structured_concerns` fields in a new JSONL file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


VALID_CATEGORIES = {
    "design_flaw",
    "interpretation",
    "missing_experiment",
    "prior_art_novelty",
    "reagent_method_specificity",
    "statistical_methodology",
    "writing_clarity",
    "other",
}

# Known typos from model outputs → canonical category
_CATEGORY_TYPO_MAP = {
    "interpretion": "interpretation",
    "interpret": "interpretation",
    "prior_art_novellicity": "prior_art_novelty",
    "writing_clary": "writing_clarity",
    "writing": "writing_clarity",
    "statoretical_methodology": "statistical_methodology",
}


def _fix_category(category: str) -> str:
    cat = category.strip().lower()
    if cat in VALID_CATEGORIES:
        return cat
    if cat in _CATEGORY_TYPO_MAP:
        return _CATEGORY_TYPO_MAP[cat]
    return cat


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _dedup_structured(
    structured: list[dict],
    texts: list[str],
    dedup_exact: bool,
    cap: int,
    fix_categories: bool = True,
) -> tuple[list[dict], list[str], int, int]:
    pairs: list[tuple[str, dict]] = []
    n_cat_fixed = 0

    if structured:
        for item in structured:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            category = str(item.get("category", "unknown"))
            if fix_categories:
                fixed = _fix_category(category)
                if fixed != category.strip().lower():
                    n_cat_fixed += 1
                category = fixed
            pairs.append(
                (
                    text,
                    {
                        "text": text,
                        "category": category,
                        "severity": str(item.get("severity", "unknown")),
                    },
                )
            )
    else:
        for text in texts:
            text = str(text).strip()
            if not text:
                continue
            pairs.append(
                (
                    text,
                    {"text": text, "category": "unknown", "severity": "unknown"},
                )
            )

    removed = 0
    if dedup_exact:
        seen: set[str] = set()
        deduped: list[tuple[str, dict]] = []
        for text, item in pairs:
            key = _normalize_text(text)
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            deduped.append((text, item))
        pairs = deduped

    if cap > 0 and len(pairs) > cap:
        removed += len(pairs) - cap
        pairs = pairs[:cap]

    out_structured = [item for _, item in pairs]
    out_texts = [text for text, _ in pairs]
    return out_structured, out_texts, removed, n_cat_fixed


def rewrite_output(
    input_path: Path,
    output_path: Path,
    dedup_exact: bool,
    cap: int,
    fix_categories: bool = True,
) -> dict:
    rows = 0
    total_before = 0
    total_after = 0
    total_removed = 0
    total_cat_fixed = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            structured = row.get("structured_concerns") or []
            texts = row.get("concerns") or []
            total_before += len(texts)
            new_structured, new_texts, removed, n_cat_fixed = _dedup_structured(
                structured=structured,
                texts=texts,
                dedup_exact=dedup_exact,
                cap=cap,
                fix_categories=fix_categories,
            )
            row["structured_concerns"] = new_structured
            row["concerns"] = new_texts
            row["postprocess"] = {
                "dedup_exact": dedup_exact,
                "cap": cap,
                "removed": removed,
                "categories_fixed": n_cat_fixed,
            }
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
            total_after += len(new_texts)
            total_removed += removed
            total_cat_fixed += n_cat_fixed

    return {
        "rows": rows,
        "total_concerns_before": total_before,
        "total_concerns_after": total_after,
        "total_removed": total_removed,
        "total_categories_fixed": total_cat_fixed,
    }


def run_evaluation_wrapper(
    output_path: Path,
    splits_dir: Path,
    split: str,
    threshold: float,
    algorithm: str,
    model_name: str,
) -> dict:
    workspace_root = Path(__file__).resolve().parents[2]
    bench_root = workspace_root / "peer-review-benchmark"
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    local_specter2 = Path(__file__).resolve().parents[1] / "models" / "specter2_base"
    has_weights = any(local_specter2.glob("*.bin")) or any(local_specter2.glob("*.safetensors"))
    if local_specter2.exists() and has_weights:
        import os

        os.environ.setdefault("BIOREVIEW_EMBED_MODEL", str(local_specter2))

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
        tool_version="postprocessed",
        extraction_manifest_id="em-v3",
        notes=f"Postprocessed output from {model_name}",
    )
    return {
        "recall_overall": float(result.recall_overall),
        "precision_overall": float(result.precision_overall),
        "f1_micro": float(result.f1_micro),
        "recall_major": float(result.recall_major),
        "n_articles": result.n_articles,
        "n_human_concerns": result.n_human_concerns,
        "n_tool_concerns": result.n_tool_concerns,
        "n_zero_recall_articles": sum(1 for c in coverage if c["recall"] == 0.0),
        "n_perfect_recall_articles": sum(1 for c in coverage if c["recall"] == 1.0),
    }


def build_parser() -> argparse.ArgumentParser:
    workspace_root = Path(__file__).resolve().parents[2]
    default_splits = workspace_root / "peer-review-benchmark" / "data" / "splits" / "v3"
    p = argparse.ArgumentParser(
        description="Postprocess inference JSONL and optionally reevaluate it."
    )
    p.add_argument("--input", type=Path, required=True, help="Input inference JSONL.")
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL for postprocessed rows.",
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        default=default_splits,
        help="Directory containing split JSONL files.",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Split to evaluate against.",
    )
    p.add_argument(
        "--no-dedup-exact",
        dest="dedup_exact",
        action="store_false",
        default=True,
        help="Disable exact text deduplication.",
    )
    p.add_argument(
        "--cap",
        type=int,
        default=0,
        help="Keep only the first N concerns per article (0 = no cap).",
    )
    p.add_argument(
        "--no-fix-categories",
        dest="fix_categories",
        action="store_false",
        default=True,
        help="Disable category typo correction.",
    )
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Run benchmark evaluation on the rewritten output.",
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
    return p


def main() -> None:
    args = build_parser().parse_args()
    stats = rewrite_output(
        input_path=args.input,
        output_path=args.output,
        dedup_exact=args.dedup_exact,
        cap=args.cap,
        fix_categories=args.fix_categories,
    )
    print(json.dumps(stats, indent=2))

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "dedup_exact": args.dedup_exact,
        "cap": args.cap,
        **stats,
    }

    if args.evaluate:
        metrics = run_evaluation_wrapper(
            output_path=args.output,
            splits_dir=args.splits_dir,
            split=args.split,
            threshold=args.threshold,
            algorithm=args.algorithm,
            model_name=args.output.stem,
        )
        summary["eval_metrics"] = metrics
        print(json.dumps(metrics, indent=2))

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
