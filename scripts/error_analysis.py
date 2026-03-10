#!/usr/bin/env python3
"""Error analysis for BioReview SFT model outputs.

Two modes:
  HPC mode: Runs SPECTER2 matching, saves detailed analysis JSON
  Local mode: Loads pre-computed JSON, renders markdown report

Usage:
  # HPC mode (requires bioreview_bench + SPECTER2):
  python scripts/error_analysis.py \
      --models results/sft_eval/qwen3.5_9b_bioreview_v1_val.jsonl \
               results/sft_eval/qwen2.5_14b_bioreview_v1_val.jsonl \
      --model-labels "Qwen3.5-9B" "Qwen2.5-14B" \
      --splits-dir /athena/.../peer-review-benchmark/data/splits/v3 \
      --output-json results/error_analysis/analysis.json

  # Local mode (no dependencies beyond stdlib):
  python scripts/error_analysis.py \
      --load-json results/error_analysis/analysis.json \
      --format markdown
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Failure mode detection
# ---------------------------------------------------------------------------


def detect_repetition_loop(raw_output: str, threshold: float = 10.0) -> bool:
    """Detect if output contains repetition loops.

    Heuristic: if any non-trivial line (>20 chars) appears >= threshold times,
    it is likely a repetition loop.
    """
    if len(raw_output) < 200:
        return False
    lines = raw_output.split("\n")
    if len(lines) < 5:
        return False
    counts = Counter(line.strip() for line in lines if len(line.strip()) > 20)
    if not counts:
        return False
    return max(counts.values()) >= threshold


def classify_failure_mode(row: dict) -> str:
    """Classify an inference output row into a failure mode category."""
    if not row.get("parse_ok", False):
        raw = row.get("raw_output", "")
        if detect_repetition_loop(raw):
            return "repetition_loop"
        if not raw.strip():
            return "empty"
        return "parse_fail"
    if len(row.get("concerns", [])) == 0:
        return "empty"
    return "ok"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_inference_outputs(path: Path) -> dict[str, dict]:
    """Load inference JSONL → {article_id: row_dict}."""
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            art_id = row.get("article_id", "")
            if art_id:
                rows[art_id] = row
    return rows


def load_gt_entries(splits_dir: Path, split: str) -> dict[str, dict]:
    """Load ground truth split → {article_id: entry_dict}."""
    gt_path = splits_dir / f"{split}.jsonl"
    if not gt_path.exists():
        print(f"ERROR: split file not found: {gt_path}", file=sys.stderr)
        raise SystemExit(1)
    entries: dict[str, dict] = {}
    with gt_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            art_id = entry.get("id", "")
            if art_id:
                entries[art_id] = entry
    return entries


# ---------------------------------------------------------------------------
# HPC mode: per-article matching with ConcernMatcher
# ---------------------------------------------------------------------------


def analyze_model_hpc(
    outputs: dict[str, dict],
    gt_entries: dict[str, dict],
    matcher,
    label: str,
    source_file: str,
) -> dict:
    """Run detailed per-article matching for one model (HPC mode)."""
    per_article = []
    cat_stats: dict[str, dict] = defaultdict(lambda: {"n_gt": 0, "n_matched": 0})
    sev_stats: dict[str, dict] = defaultdict(lambda: {"n_gt": 0, "n_matched": 0})
    failure_counts: dict[str, int] = Counter()
    recall_bins = {"zero": 0, "partial": 0, "decent": 0}

    overall_n_gt = 0
    overall_n_tool = 0
    overall_n_matched = 0

    for art_id, gt_entry in gt_entries.items():
        gt_concerns = gt_entry.get("concerns", [])

        # Filter figure-dependent concerns (same as ConcernMatcher.score_article)
        active_gt = [
            c for c in gt_concerns if not c.get("requires_figure_reading", False)
        ]
        gt_texts = [c["concern_text"] for c in active_gt]

        # Get model output
        row = outputs.get(art_id, {})
        tool_texts = row.get("concerns", [])
        failure_mode = classify_failure_mode(row) if row else "missing"
        failure_counts[failure_mode] += 1

        # Run matching
        matched_gt_indices: set[int] = set()
        if tool_texts and gt_texts:
            try:
                scores = matcher._compute_scores(tool_texts, gt_texts)
                matches = matcher._match(scores)
                matched_gt_indices = {m.gt_idx for m in matches}
            except Exception:
                pass

        n_matched = len(matched_gt_indices)
        n_gt = len(active_gt)
        n_tool = len(tool_texts)

        overall_n_gt += n_gt
        overall_n_tool += n_tool
        overall_n_matched += n_matched

        # Per-article recall
        recall = n_matched / n_gt if n_gt > 0 else 1.0
        precision = n_matched / n_tool if n_tool > 0 else 1.0
        f1 = (
            2 * recall * precision / (recall + precision)
            if (recall + precision) > 0
            else 0.0
        )

        # Recall bins
        if n_gt > 0:
            if recall == 0:
                recall_bins["zero"] += 1
            elif recall < 0.5:
                recall_bins["partial"] += 1
            else:
                recall_bins["decent"] += 1

        # Per-concern GT details
        gt_details = []
        for idx, c in enumerate(active_gt):
            matched = idx in matched_gt_indices
            cat = c.get("category", "unknown")
            sev = c.get("severity", "unknown")
            gt_details.append(
                {
                    "concern_text": c["concern_text"][:200],  # truncate for JSON size
                    "category": cat,
                    "severity": sev,
                    "matched": matched,
                }
            )
            cat_stats[cat]["n_gt"] += 1
            sev_stats[sev]["n_gt"] += 1
            if matched:
                cat_stats[cat]["n_matched"] += 1
                sev_stats[sev]["n_matched"] += 1

        per_article.append(
            {
                "article_id": art_id,
                "n_gt": n_gt,
                "n_tool": n_tool,
                "n_matched": n_matched,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "f1": round(f1, 4),
                "failure_mode": failure_mode,
                "gt_concerns": gt_details,
            }
        )

    # Compute per-category and per-severity recall
    per_category = {}
    for cat, s in sorted(cat_stats.items()):
        per_category[cat] = {
            "n_gt": s["n_gt"],
            "n_matched": s["n_matched"],
            "recall": round(s["n_matched"] / s["n_gt"], 4) if s["n_gt"] > 0 else 0.0,
        }

    per_severity = {}
    for sev, s in sorted(sev_stats.items()):
        per_severity[sev] = {
            "n_gt": s["n_gt"],
            "n_matched": s["n_matched"],
            "recall": round(s["n_matched"] / s["n_gt"], 4) if s["n_gt"] > 0 else 0.0,
        }

    overall_recall = (
        overall_n_matched / overall_n_gt if overall_n_gt > 0 else 0.0
    )
    overall_precision = (
        overall_n_matched / overall_n_tool if overall_n_tool > 0 else 0.0
    )
    overall_f1 = (
        2 * overall_recall * overall_precision / (overall_recall + overall_precision)
        if (overall_recall + overall_precision) > 0
        else 0.0
    )

    return {
        "label": label,
        "source_file": source_file,
        "overall": {
            "recall": round(overall_recall, 4),
            "precision": round(overall_precision, 4),
            "f1": round(overall_f1, 4),
            "n_articles": len(gt_entries),
            "n_gt_concerns": overall_n_gt,
            "n_tool_concerns": overall_n_tool,
            "n_matched": overall_n_matched,
        },
        "per_category": per_category,
        "per_severity": per_severity,
        "failure_modes": dict(failure_counts),
        "zero_recall_breakdown": recall_bins,
        "per_article": per_article,
    }


# ---------------------------------------------------------------------------
# Model agreement analysis
# ---------------------------------------------------------------------------


def compute_agreement(model_results: dict[str, dict]) -> dict:
    """Find articles where all models fail vs. model-exclusive successes."""
    # Collect per-article recall for each model
    model_recalls: dict[str, dict[str, float]] = {}
    for label, data in model_results.items():
        model_recalls[label] = {}
        for art in data["per_article"]:
            model_recalls[label][art["article_id"]] = art["recall"]

    # All article IDs
    all_ids = set()
    for recalls in model_recalls.values():
        all_ids.update(recalls.keys())

    all_zero = []
    exclusive_hits: dict[str, list[str]] = {label: [] for label in model_results}

    for art_id in sorted(all_ids):
        recalls = {
            label: model_recalls[label].get(art_id, 0.0)
            for label in model_results
        }
        if all(r == 0.0 for r in recalls.values()):
            all_zero.append(art_id)
        else:
            # Check for exclusive hits (only one model has recall > 0)
            positive_models = [label for label, r in recalls.items() if r > 0]
            if len(positive_models) == 1:
                exclusive_hits[positive_models[0]].append(art_id)

    return {
        "all_zero_recall": all_zero,
        "n_all_zero": len(all_zero),
        "model_exclusive_hits": {
            label: {"articles": arts, "count": len(arts)}
            for label, arts in exclusive_hits.items()
        },
    }


# ---------------------------------------------------------------------------
# Markdown report rendering (Local mode)
# ---------------------------------------------------------------------------


def render_markdown(data: dict) -> str:
    """Render analysis JSON as markdown report."""
    lines: list[str] = []
    lines.append("# BioReview SFT Error Analysis Report")
    lines.append(f"\nGenerated: {data.get('generated_at', 'unknown')}")
    cfg = data.get("config", {})
    lines.append(
        f"Config: threshold={cfg.get('threshold', '?')}, "
        f"algorithm={cfg.get('algorithm', '?')}, "
        f"exclude_figure={cfg.get('exclude_figure', '?')}"
    )

    models = data.get("models", {})

    # ── Section 1: Summary table ────────────────────────────
    lines.append("\n## 1. Summary Table\n")
    lines.append(
        "| Model | F1 | Recall | Precision | GT | Tool | Matched | "
        "Parse Fail | Empty | Repetition |"
    )
    lines.append(
        "|-------|----|--------|-----------|----|------|---------|"
        "------------|-------|------------|"
    )
    for label, m in models.items():
        o = m["overall"]
        fm = m.get("failure_modes", {})
        lines.append(
            f"| {label} | {o['f1']:.4f} | {o['recall']:.4f} | "
            f"{o['precision']:.4f} | {o['n_gt_concerns']} | "
            f"{o['n_tool_concerns']} | {o['n_matched']} | "
            f"{fm.get('parse_fail', 0)} | {fm.get('empty', 0)} | "
            f"{fm.get('repetition_loop', 0)} |"
        )

    # ── Section 2: Zero-recall breakdown ────────────────────
    lines.append("\n## 2. Zero-Recall Breakdown\n")
    lines.append("| Model | Zero (R=0) | Partial (0<R<0.5) | Decent (R>=0.5) |")
    lines.append("|-------|------------|--------------------|--------------------|")
    for label, m in models.items():
        zr = m.get("zero_recall_breakdown", {})
        lines.append(
            f"| {label} | {zr.get('zero', 0)} | "
            f"{zr.get('partial', 0)} | {zr.get('decent', 0)} |"
        )

    # ── Section 3: Per-category recall ──────────────────────
    lines.append("\n## 3. Per-Category Recall\n")
    # Collect all categories
    all_cats: set[str] = set()
    for m in models.values():
        all_cats.update(m.get("per_category", {}).keys())
    all_cats_sorted = sorted(all_cats)

    header = "| Category |"
    sep = "|----------|"
    for label in models:
        header += f" {label} (n/N) |"
        sep += "------------|"
    lines.append(header)
    lines.append(sep)

    for cat in all_cats_sorted:
        row_str = f"| {cat} |"
        for label, m in models.items():
            cs = m.get("per_category", {}).get(cat, {})
            n_m = cs.get("n_matched", 0)
            n_gt = cs.get("n_gt", 0)
            recall = cs.get("recall", 0.0)
            row_str += f" {recall:.3f} ({n_m}/{n_gt}) |"
        lines.append(row_str)

    # ── Section 4: Per-severity recall ──────────────────────
    lines.append("\n## 4. Per-Severity Recall\n")
    header = "| Severity |"
    sep = "|----------|"
    for label in models:
        header += f" {label} (n/N) |"
        sep += "------------|"
    lines.append(header)
    lines.append(sep)

    all_sevs = set()
    for m in models.values():
        all_sevs.update(m.get("per_severity", {}).keys())
    for sev in sorted(all_sevs):
        row_str = f"| {sev} |"
        for label, m in models.items():
            ss = m.get("per_severity", {}).get(sev, {})
            n_m = ss.get("n_matched", 0)
            n_gt = ss.get("n_gt", 0)
            recall = ss.get("recall", 0.0)
            row_str += f" {recall:.3f} ({n_m}/{n_gt}) |"
        lines.append(row_str)

    # ── Section 5: Model agreement ──────────────────────────
    agreement = data.get("agreement", {})
    if agreement:
        lines.append("\n## 5. Model Agreement\n")
        lines.append(
            f"- All models zero recall: **{agreement.get('n_all_zero', 0)}** articles"
        )
        excl = agreement.get("model_exclusive_hits", {})
        for label, info in excl.items():
            lines.append(
                f"- {label} exclusive hits: **{info.get('count', 0)}** articles"
            )

    # ── Section 6: Failure modes detail ─────────────────────
    lines.append("\n## 6. Failure Mode Breakdown\n")
    for label, m in models.items():
        fm = m.get("failure_modes", {})
        total = sum(fm.values())
        lines.append(f"### {label}")
        for mode_name in ["ok", "parse_fail", "empty", "repetition_loop", "missing"]:
            count = fm.get(mode_name, 0)
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"- {mode_name}: {count} ({pct:.1f}%)")
        lines.append("")

    # ── Section 7: Sample zero-recall articles ──────────────
    lines.append("\n## 7. Sample Zero-Recall Articles (Top 10 by GT count)\n")
    for label, m in models.items():
        lines.append(f"### {label}\n")
        zero_arts = [
            a for a in m.get("per_article", []) if a["recall"] == 0 and a["n_gt"] > 0
        ]
        zero_arts.sort(key=lambda a: a["n_gt"], reverse=True)
        lines.append("| Article ID | GT Concerns | Model Concerns | Failure Mode |")
        lines.append("|------------|-------------|----------------|--------------|")
        for a in zero_arts[:10]:
            lines.append(
                f"| {a['article_id']} | {a['n_gt']} | {a['n_tool']} | "
                f"{a['failure_mode']} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Error analysis for BioReview SFT model outputs."
    )

    # HPC mode arguments
    p.add_argument(
        "--models",
        nargs="+",
        type=Path,
        help="Inference JSONL files (HPC mode).",
    )
    p.add_argument(
        "--model-labels",
        nargs="+",
        help="Labels for each model (same order as --models).",
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        help="Directory containing split JSONL files.",
    )
    p.add_argument(
        "--split",
        default="val",
        help="Which split to evaluate against (default: val).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Concern matching threshold (default: 0.65).",
    )
    p.add_argument(
        "--algorithm",
        choices=["hungarian", "greedy"],
        default="hungarian",
        help="Matching algorithm (default: hungarian).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        help="Output JSON path for analysis data (HPC mode).",
    )

    # Local mode arguments
    p.add_argument(
        "--load-json",
        type=Path,
        help="Load pre-computed analysis JSON (Local mode).",
    )
    p.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format for local mode (default: markdown).",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── Local mode ──────────────────────────────────────────
    if args.load_json:
        if not args.load_json.exists():
            print(f"ERROR: JSON not found: {args.load_json}", file=sys.stderr)
            raise SystemExit(1)
        data = json.loads(args.load_json.read_text(encoding="utf-8"))
        if args.format == "markdown":
            print(render_markdown(data))
        else:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    # ── HPC mode ────────────────────────────────────────────
    if not args.models:
        print("ERROR: provide --models (HPC mode) or --load-json (Local mode).")
        raise SystemExit(1)

    if not args.splits_dir:
        print("ERROR: --splits-dir is required in HPC mode.", file=sys.stderr)
        raise SystemExit(1)

    labels = args.model_labels or [p.stem for p in args.models]
    if len(labels) != len(args.models):
        print("ERROR: --model-labels count must match --models count.", file=sys.stderr)
        raise SystemExit(1)

    # Setup bioreview_bench imports
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    bench_root = workspace_root / "peer-review-benchmark"
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    # Setup SPECTER2
    local_specter2 = project_root / "models" / "specter2_base"
    if local_specter2.exists() and not os.getenv("BIOREVIEW_EMBED_MODEL"):
        os.environ["BIOREVIEW_EMBED_MODEL"] = str(local_specter2)

    from bioreview_bench.evaluate.metrics import ConcernMatcher

    matcher = ConcernMatcher(
        threshold=args.threshold,
        exclude_figure=True,
        use_embedding=True,
        algorithm=args.algorithm,
    )

    # Load ground truth
    print(f"Loading ground truth from {args.splits_dir}/{args.split}.jsonl...")
    gt_entries = load_gt_entries(args.splits_dir, args.split)
    print(f"  {len(gt_entries)} articles with concerns")

    # Analyze each model
    model_results: dict[str, dict] = {}
    for model_path, label in zip(args.models, labels):
        print(f"\nAnalyzing {label} from {model_path}...")
        if not model_path.exists():
            print(f"  WARNING: file not found, skipping.", file=sys.stderr)
            continue
        outputs = load_inference_outputs(model_path)
        print(f"  {len(outputs)} articles in inference output")
        result = analyze_model_hpc(
            outputs, gt_entries, matcher, label, str(model_path)
        )
        model_results[label] = result
        o = result["overall"]
        print(
            f"  F1={o['f1']:.4f}  Recall={o['recall']:.4f}  "
            f"Precision={o['precision']:.4f}"
        )

    # Agreement analysis
    agreement = compute_agreement(model_results) if len(model_results) > 1 else {}

    # Build output
    analysis = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "threshold": args.threshold,
            "algorithm": args.algorithm,
            "exclude_figure": True,
            "split": args.split,
        },
        "models": model_results,
        "agreement": agreement,
    }

    # Save
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nAnalysis saved to {args.output_json}")
        print(f"  View locally: python scripts/error_analysis.py "
              f"--load-json {args.output_json}")
    else:
        # Print markdown directly
        print("\n" + render_markdown(analysis))


if __name__ == "__main__":
    main()
