#!/usr/bin/env python3
"""Inspect one article at a time: GT vs baseline model outputs.

This script mirrors the benchmark matcher behavior so we can manually review
model outputs per article with matched pairs and unmatched concerns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"[warn] malformed JSON at {path}:{lineno}: {exc}", file=sys.stderr
                )
    return rows


def normalize_tool_concerns(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
        elif isinstance(item, dict):
            s = str(item.get("text", item.get("concern_text", ""))).strip()
            if s:
                out.append(s)
    return out


def truncate(text: str, n: int) -> str:
    text = " ".join(text.split())
    if n <= 0 or len(text) <= n:
        return text
    return text[: n - 3] + "..."


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve()
    workspace_root = here.parents[2]
    bench_root = workspace_root / "peer-review-benchmark"
    bt_root = here.parents[1]

    p = argparse.ArgumentParser(
        description="Inspect one article: GT vs Gemini/GPT-4o-mini concerns."
    )
    p.add_argument(
        "--split-path",
        type=Path,
        default=bench_root / "data" / "splits" / "v3" / "val.jsonl",
        help="Path to benchmark split JSONL used for evaluation.",
    )
    p.add_argument(
        "--gemini-output",
        type=Path,
        default=bt_root / "results" / "baselines" / "gemini_free_val.jsonl",
        help="Path to Gemini output JSONL.",
    )
    p.add_argument(
        "--gpt-output",
        type=Path,
        default=bt_root / "results" / "baselines" / "gpt4omini_val.jsonl",
        help="Path to GPT-4o-mini output JSONL.",
    )
    p.add_argument(
        "--article-id",
        type=str,
        default="",
        help="Article id to inspect (e.g. elife:95952).",
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="Article index in split (used when --article-id is not set).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Matching threshold.",
    )
    p.add_argument(
        "--algorithm",
        choices=["hungarian", "greedy"],
        default="hungarian",
        help="Matching algorithm.",
    )
    p.add_argument(
        "--no-embedding",
        action="store_true",
        help="Disable embedding and use lexical fallback (Jaccard).",
    )
    p.add_argument(
        "--include-figure-gt",
        action="store_true",
        help="Include requires_figure_reading GT concerns.",
    )
    p.add_argument(
        "--max-text-chars",
        type=int,
        default=240,
        help="Max chars for each concern preview.",
    )
    p.add_argument(
        "--max-list-items",
        type=int,
        default=0,
        help="Max items shown per list (0 = show all).",
    )
    return p


def to_map(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        art_id = row.get("article_id", row.get("id", ""))
        if not isinstance(art_id, str) or not art_id:
            continue
        out[art_id] = normalize_tool_concerns(row.get("concerns", []))
    return out


def slice_items(items: list[Any], max_items: int) -> list[Any]:
    if max_items <= 0:
        return items
    return items[:max_items]


def main() -> None:
    args = build_parser().parse_args()

    here = Path(__file__).resolve()
    workspace_root = here.parents[2]
    bench_root = workspace_root / "peer-review-benchmark"
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    # Prefer local SPECTER2 snapshot when present.
    local_specter2 = here.parents[1] / "models" / "specter2_base"
    if local_specter2.exists() and not os.getenv("BIOREVIEW_EMBED_MODEL"):
        os.environ["BIOREVIEW_EMBED_MODEL"] = str(local_specter2)
        print(f"embedding_model_override={local_specter2}")

    from bioreview_bench.evaluate.metrics import ConcernMatcher

    split_rows = load_jsonl(args.split_path)
    if not split_rows:
        raise RuntimeError(f"No rows found in split file: {args.split_path}")

    idx_by_id = {str(r.get("id", "")): i for i, r in enumerate(split_rows)}
    if args.article_id:
        if args.article_id not in idx_by_id:
            raise KeyError(f"Article id not in split: {args.article_id}")
        idx = idx_by_id[args.article_id]
    else:
        idx = args.index
    if idx < 0 or idx >= len(split_rows):
        raise IndexError(f"Index out of range: {idx} (0..{len(split_rows)-1})")

    entry = split_rows[idx]
    art_id = str(entry.get("id", ""))
    title = str(entry.get("title", ""))
    source = str(entry.get("source", ""))
    doi = str(entry.get("doi", ""))
    gt_all = entry.get("concerns", [])
    if not isinstance(gt_all, list):
        gt_all = []
    if args.include_figure_gt:
        gt_active = gt_all
    else:
        gt_active = [c for c in gt_all if not c.get("requires_figure_reading", False)]
    n_excluded = len(gt_all) - len(gt_active)

    gemini_map = to_map(load_jsonl(args.gemini_output))
    gpt_map = to_map(load_jsonl(args.gpt_output))
    model_rows: list[tuple[str, list[str]]] = [
        ("gemini_free", gemini_map.get(art_id, [])),
        ("gpt4omini", gpt_map.get(art_id, [])),
    ]

    matcher = ConcernMatcher(
        threshold=args.threshold,
        exclude_figure=not args.include_figure_gt,
        use_embedding=not args.no_embedding,
        algorithm=args.algorithm,  # type: ignore[arg-type]
    )

    # Filter gt_active to only concerns with non-empty text, so indices stay aligned.
    gt_active = [c for c in gt_active if str(c.get("concern_text", "")).strip()]
    gt_texts = [str(c.get("concern_text", "")).strip() for c in gt_active]

    print(f"article_index={idx}")
    print(f"article_id={art_id}")
    print(f"source={source}")
    print(f"doi={doi}")
    print(f"title={title}")
    print(
        f"gt_concerns_active={len(gt_texts)} "
        f"figure_excluded={n_excluded} "
        f"gt_concerns_total={len(gt_all)}"
    )
    print()

    gt_display = slice_items(list(enumerate(gt_active)), args.max_list_items)
    print("GT concerns")
    for i, c in gt_display:
        text = str(c.get("concern_text", "")).strip()
        cat = str(c.get("category", "other"))
        sev = str(c.get("severity", "unknown"))
        print(f"  gt#{i:02d} [{cat}/{sev}] {truncate(text, args.max_text_chars)}")
    if args.max_list_items > 0 and len(gt_active) > args.max_list_items:
        print(f"  ... ({len(gt_active) - args.max_list_items} more GT concerns)")
    print()

    for alias, tool_concerns in model_rows:
        scores = matcher._compute_scores(tool_concerns, gt_texts)
        matches = matcher._match(scores)
        matched_gt_idxs = {m.gt_idx for m in matches}
        matched_tool_idxs = {m.tool_idx for m in matches}

        n_gt = len(gt_texts)
        n_tool = len(tool_concerns)
        n_matched = len(matches)
        recall = n_matched / n_gt if n_gt else 0.0
        precision = n_matched / n_tool if n_tool else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )

        print(f"[{alias}]")
        print(
            f"  method={scores.method} effective_threshold={scores.threshold:.3f} "
            f"n_tool={n_tool} n_gt={n_gt} n_matched={n_matched} "
            f"recall={recall:.3f} precision={precision:.3f} f1={f1:.3f}"
        )

        print("  matched_pairs")
        matched_sorted = sorted(matches, key=lambda m: m.score, reverse=True)
        for m in slice_items(matched_sorted, args.max_list_items):
            gt = gt_active[m.gt_idx]
            gt_cat = str(gt.get("category", "other"))
            gt_sev = str(gt.get("severity", "unknown"))
            print(
                f"    tool#{m.tool_idx:02d} -> gt#{m.gt_idx:02d} "
                f"score={m.score:.3f} [{gt_cat}/{gt_sev}]"
            )
            print(f"      GT: {truncate(gt_texts[m.gt_idx], args.max_text_chars)}")
            print(
                f"      TL: {truncate(tool_concerns[m.tool_idx], args.max_text_chars)}"
            )
        if not matched_sorted:
            print("    (none)")
        elif args.max_list_items > 0 and len(matched_sorted) > args.max_list_items:
            print(
                f"    ... ({len(matched_sorted) - args.max_list_items} more matched pairs)"
            )

        print("  unmatched_gt")
        unmatched_gt = [i for i in range(n_gt) if i not in matched_gt_idxs]
        for i in slice_items(unmatched_gt, args.max_list_items):
            c = gt_active[i]
            cat = str(c.get("category", "other"))
            sev = str(c.get("severity", "unknown"))
            print(
                f"    gt#{i:02d} [{cat}/{sev}] {truncate(gt_texts[i], args.max_text_chars)}"
            )
        if not unmatched_gt:
            print("    (none)")
        elif args.max_list_items > 0 and len(unmatched_gt) > args.max_list_items:
            print(
                f"    ... ({len(unmatched_gt) - args.max_list_items} more unmatched GT)"
            )

        print("  unmatched_tool")
        unmatched_tool = [i for i in range(n_tool) if i not in matched_tool_idxs]
        for i in slice_items(unmatched_tool, args.max_list_items):
            print(f"    tool#{i:02d} {truncate(tool_concerns[i], args.max_text_chars)}")
        if not unmatched_tool:
            print("    (none)")
        elif args.max_list_items > 0 and len(unmatched_tool) > args.max_list_items:
            print(
                f"    ... ({len(unmatched_tool) - args.max_list_items} more unmatched tool)"
            )
        print()


if __name__ == "__main__":
    main()
