#!/usr/bin/env python3
"""Compare SFT model evaluation results across multiple models.

Reads summary JSON files from results/sft_eval/ and prints a markdown
comparison table sorted by F1 score.

Usage:
  python scripts/compare_models.py
  python scripts/compare_models.py --results-dir results/sft_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASELINES = [
    {
        "model": "GPT-4o-mini (baseline)",
        "f1_micro": 0.6962,
        "recall_overall": 0.6472,
        "precision_overall": 0.7531,
        "recall_major": None,
    },
    {
        "model": "Gemini-2.5-Flash (baseline)",
        "f1_micro": 0.4489,
        "recall_overall": 0.3011,
        "precision_overall": 0.8820,
        "recall_major": None,
    },
]


def load_results(results_dir: Path) -> list[dict]:
    """Load all summary JSON files that contain eval_metrics."""
    rows = []
    for p in sorted(results_dir.glob("*.summary.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        metrics = data.get("eval_metrics")
        if not metrics:
            continue
        model_name = Path(data.get("model_dir", p.stem)).name
        rows.append(
            {
                "model": model_name,
                "f1_micro": metrics.get("f1_micro", 0),
                "recall_overall": metrics.get("recall_overall", 0),
                "precision_overall": metrics.get("precision_overall", 0),
                "recall_major": metrics.get("recall_major"),
                "n_articles": metrics.get("n_articles", 0),
                "file": p.name,
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    """Print markdown comparison table sorted by F1."""
    all_rows = BASELINES + rows
    all_rows.sort(key=lambda r: r.get("f1_micro", 0), reverse=True)

    print("| Model | F1 | Recall | Precision | Recall (major) |")
    print("|-------|---:|-------:|----------:|---------------:|")
    for r in all_rows:
        rmaj = f"{r['recall_major']:.4f}" if r.get("recall_major") is not None else "—"
        print(
            f"| {r['model']:<35} | {r['f1_micro']:.4f} | "
            f"{r['recall_overall']:.4f} | {r['precision_overall']:.4f} | {rmaj} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SFT model results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "sft_eval",
        help="Directory containing *.summary.json files.",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}")
        print("Run inference + evaluation first.")
        raise SystemExit(1)

    rows = load_results(args.results_dir)
    if not rows:
        print("No evaluation results found (summary files without eval_metrics).")
        print("Run inference with --evaluate flag first.")
        raise SystemExit(1)

    print(f"\nBioReview SFT Model Comparison ({len(rows)} models evaluated)")
    print(f"Results from: {args.results_dir}\n")
    print_table(rows)
    print()


if __name__ == "__main__":
    main()
