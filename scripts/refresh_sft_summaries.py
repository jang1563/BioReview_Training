#!/usr/bin/env python3
"""Refresh SFT summary metadata from existing inference JSONL files.

This is useful for old runs whose `.summary.json` files were written before
resume accounting was fixed. The script recomputes processed/failed_parse/
total_concerns/total_time directly from the JSONL, while preserving any
existing evaluation metrics and model metadata already stored in the summary.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def summarize_jsonl(path: Path) -> dict:
    processed = 0
    failed_parse = 0
    total_concerns = 0
    total_time = 0.0

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            processed += 1
            total_concerns += len(row.get("concerns", []))
            total_time += float(row.get("generation_time_s", 0.0) or 0.0)
            if not row.get("parse_ok", False):
                failed_parse += 1

    return {
        "processed": processed,
        "failed_parse": failed_parse,
        "total_concerns": total_concerns,
        "total_time": total_time,
    }


def refresh_summary(jsonl_path: Path, dry_run: bool = False) -> None:
    summary_path = jsonl_path.with_suffix(".summary.json")
    if not summary_path.exists():
        print(f"SKIP {jsonl_path.name}: summary not found")
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fresh = summarize_jsonl(jsonl_path)

    updated = dict(summary)
    updated.update(fresh)
    updated["summary_refreshed_at_utc"] = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    updated["summary_refreshed_from_jsonl"] = jsonl_path.name

    changed = any(summary.get(k) != v for k, v in fresh.items())
    status = "UPDATE" if changed else "OK"
    print(
        f"{status} {jsonl_path.name}: processed={fresh['processed']} "
        f"failed_parse={fresh['failed_parse']} total_concerns={fresh['total_concerns']}"
    )

    if not dry_run:
        summary_path.write_text(
            json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Refresh SFT summary metadata from JSONL.")
    p.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="JSONL files to refresh. Default: all non-ensemble JSONL in results/sft_eval.",
    )
    p.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]

    if args.paths:
        paths = args.paths
    else:
        sft_eval_dir = project_root / "results" / "sft_eval"
        paths = sorted(
            p
            for p in sft_eval_dir.glob("*.jsonl")
            if not p.name.startswith("ensemble_")
        )

    if not paths:
        print("No JSONL files found.")
        return

    for path in paths:
        refresh_summary(path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
