#!/usr/bin/env python3
"""Render a markdown comparison report for baseline and SFT eval summaries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"


def _fmt_gate(row: dict, target_f1: float, target_recall: float) -> str:
    f1 = row.get("f1_micro")
    recall = row.get("recall_overall")
    if f1 is None or recall is None:
        return "-"
    return "pass" if f1 >= target_f1 and recall >= target_recall else "hold"


def load_baseline_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for row in data.get("rows", []):
        if row.get("eval_status") != "ok":
            continue
        rows.append(
            {
                "model": f"{row.get('model', row.get('alias', path.stem))} (baseline)",
                "kind": "baseline",
                "alias": row.get("alias"),
                "f1_micro": row.get("f1_micro"),
                "recall_overall": row.get("recall_overall"),
                "precision_overall": row.get("precision_overall"),
                "recall_major": row.get("recall_major"),
                "n_articles": row.get("n_articles"),
                "source_file": path.name,
                "source_id": path.stem,
            }
        )
    return rows


def load_sft_row(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    metrics = data.get("eval_metrics")
    if not metrics:
        return None

    model_name = Path(data.get("model_dir", path.stem)).name
    return {
        "model": model_name,
        "kind": "sft",
        "alias": None,
        "f1_micro": metrics.get("f1_micro"),
        "recall_overall": metrics.get("recall_overall"),
        "precision_overall": metrics.get("precision_overall"),
        "recall_major": metrics.get("recall_major"),
        "n_articles": metrics.get("n_articles"),
        "source_file": path.name,
        "source_id": path.name.replace(".summary.json", ""),
    }


def load_sft_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*.summary.json")):
        row = load_sft_row(path)
        if row is not None:
            rows.append(row)
    return rows


def load_selected_sft_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        summary_path = path
        if path.suffix == ".jsonl":
            summary_path = path.with_suffix(".summary.json")
        row = load_sft_row(summary_path)
        if row is not None:
            rows.append(row)
    return rows


def find_latest_baseline(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("baseline_summary_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No baseline summaries found under {results_dir}")
    return candidates[-1]


def choose_reference(rows: list[dict], preferred_alias: str) -> dict:
    for row in rows:
        if row.get("alias") == preferred_alias:
            return row
    baseline_rows = [row for row in rows if row.get("kind") == "baseline"]
    if not baseline_rows:
        raise ValueError("Need at least one baseline row to compute deltas.")
    return max(baseline_rows, key=lambda row: row.get("f1_micro") or -1.0)


def disambiguate_model_names(rows: list[dict]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["model"]] = counts.get(row["model"], 0) + 1
    for row in rows:
        if counts[row["model"]] > 1:
            row["model"] = f"{row['model']} [{row['source_id']}]"


def render_report(
    rows: list[dict],
    baseline_path: Path,
    sft_source_label: str,
    reference: dict,
    target_f1: float,
    target_recall: float,
) -> str:
    ranked = sorted(rows, key=lambda row: row.get("f1_micro") or -1.0, reverse=True)
    best_overall = ranked[0]
    sft_rows = [row for row in ranked if row["kind"] == "sft"]
    best_sft = sft_rows[0] if sft_rows else None
    passing = [
        row["model"]
        for row in sft_rows
        if _fmt_gate(row, target_f1=target_f1, target_recall=target_recall) == "pass"
    ]

    lines = [
        "# BioReview Eval Comparison",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        f"- Baseline summary: `{baseline_path}`",
        f"- SFT source: `{sft_source_label}`",
        f"- Reference baseline: `{reference['model']}`",
        f"- Gate: `f1_micro >= {target_f1:.2f}` and `recall_overall >= {target_recall:.2f}`",
        "",
        "## Snapshot",
        "",
        f"- Best overall: `{best_overall['model']}` ({_fmt_metric(best_overall.get('f1_micro'))} F1)",
    ]
    if best_sft is not None:
        lines.append(
            f"- Best SFT: `{best_sft['model']}` ({_fmt_metric(best_sft.get('f1_micro'))} F1, {_fmt_delta((best_sft.get('f1_micro') or 0.0) - (reference.get('f1_micro') or 0.0))} vs ref)"
        )
    if passing:
        lines.append(f"- SFT models meeting gate: {', '.join(f'`{name}`' for name in passing)}")
    else:
        lines.append("- SFT models meeting gate: none")

    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| Rank | Model | Kind | F1 | Recall | Precision | Recall (major) | n_articles | dF1 vs ref | Gate |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    ref_f1 = reference.get("f1_micro")
    for idx, row in enumerate(ranked, start=1):
        delta_f1 = None
        if row.get("f1_micro") is not None and ref_f1 is not None:
            delta_f1 = row["f1_micro"] - ref_f1
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    row["model"],
                    row["kind"],
                    _fmt_metric(row.get("f1_micro")),
                    _fmt_metric(row.get("recall_overall")),
                    _fmt_metric(row.get("precision_overall")),
                    _fmt_metric(row.get("recall_major")),
                    str(row.get("n_articles", "-")),
                    _fmt_delta(delta_f1),
                    _fmt_gate(row, target_f1=target_f1, target_recall=target_recall),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Sources",
            "",
            f"- Baseline rows loaded from `{baseline_path.name}`",
        ]
    )
    for row in ranked:
        if row["kind"] == "sft":
            lines.append(f"- `{row['model']}`: `{row['source_file']}`")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a markdown eval comparison report.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional explicit SFT summary/jsonl files to include.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "sft_eval",
        help="Directory containing SFT *.summary.json files.",
    )
    parser.add_argument(
        "--baseline-results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "baseline_eval",
        help="Directory containing baseline_summary_*.json files.",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=None,
        help="Optional explicit baseline summary JSON.",
    )
    parser.add_argument(
        "--reference-alias",
        default="gpt4omini",
        help="Preferred baseline alias for delta columns.",
    )
    parser.add_argument(
        "--target-f1",
        type=float,
        default=0.58,
        help="Gate threshold for f1_micro.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.45,
        help="Gate threshold for recall_overall.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown output path. Prints to stdout if omitted.",
    )
    args = parser.parse_args()

    baseline_path = args.baseline_summary or find_latest_baseline(args.baseline_results_dir)
    baseline_rows = load_baseline_rows(baseline_path)
    if not baseline_rows:
        raise SystemExit(f"No evaluated baseline rows found in {baseline_path}")

    if args.paths:
        sft_rows = load_selected_sft_rows(args.paths)
        sft_source_label = ", ".join(str(path) for path in args.paths)
    else:
        if not args.results_dir.exists():
            raise SystemExit(f"Results directory not found: {args.results_dir}")
        sft_rows = load_sft_rows(args.results_dir)
        sft_source_label = str(args.results_dir)

    if not sft_rows:
        raise SystemExit("No evaluated SFT summaries found.")

    all_rows = baseline_rows + sft_rows
    disambiguate_model_names(all_rows)
    reference = choose_reference(baseline_rows, args.reference_alias)
    report = render_report(
        all_rows,
        baseline_path=baseline_path,
        sft_source_label=sft_source_label,
        reference=reference,
        target_f1=args.target_f1,
        target_recall=args.target_recall,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
