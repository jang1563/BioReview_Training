#!/usr/bin/env python3
"""Merge reparse results back into the main inference JSONL.

Replaces articles in the original output with their reparsed versions
(only when the reparse produced more concerns than the original).

Usage:
    python scripts/merge_reparse.py \
        --original results/sft_eval/model_val.jsonl \
        --reparse  results/sft_eval/model_val_reparse.jsonl \
        --output   results/sft_eval/model_val_merged.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> dict[str, dict]:
    """Load JSONL keyed by article_id."""
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            aid = row.get("article_id", "")
            if aid:
                rows[aid] = row
    return rows


def merge(
    original: dict[str, dict],
    reparse: dict[str, dict],
    strategy: str = "more_concerns",
) -> tuple[dict[str, dict], dict]:
    """Merge reparse into original.

    strategy:
        'more_concerns': replace only if reparse has more concerns
        'always': always replace if reparse exists
    """
    merged = dict(original)
    stats = {"total_original": len(original), "total_reparse": len(reparse),
             "replaced": 0, "skipped": 0, "not_in_original": 0}

    for aid, reparse_row in reparse.items():
        if aid not in merged:
            stats["not_in_original"] += 1
            merged[aid] = reparse_row
            continue

        orig_concerns = len(merged[aid].get("concerns") or [])
        new_concerns = len(reparse_row.get("concerns") or [])

        if strategy == "always" or new_concerns > orig_concerns:
            reparse_row["merge_info"] = {
                "original_concerns": orig_concerns,
                "reparse_concerns": new_concerns,
                "action": "replaced",
            }
            merged[aid] = reparse_row
            stats["replaced"] += 1
        else:
            stats["skipped"] += 1

    return merged, stats


def main() -> None:
    p = argparse.ArgumentParser(description="Merge reparse results into main JSONL.")
    p.add_argument("--original", type=Path, required=True, help="Original inference JSONL.")
    p.add_argument("--reparse", type=Path, required=True, help="Reparse inference JSONL.")
    p.add_argument("--output", type=Path, required=True, help="Output merged JSONL.")
    p.add_argument(
        "--strategy",
        choices=["more_concerns", "always"],
        default="more_concerns",
        help="When to replace: 'more_concerns' (only if reparse has more) or 'always'.",
    )
    args = p.parse_args()

    original = load_jsonl(args.original)
    reparse = load_jsonl(args.reparse)
    merged, stats = merge(original, reparse, strategy=args.strategy)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in merged.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(stats, indent=2))
    print(f"\nMerged output: {args.output} ({len(merged)} articles)")


if __name__ == "__main__":
    main()
