#!/usr/bin/env python3
"""Evaluate a tool output overall and per source_db/article source."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    default_splits = workspace_root / "peer-review-benchmark" / "data" / "splits" / "v3"
    default_output_dir = project_root / "results" / "source_eval"

    parser = argparse.ArgumentParser(
        description="Evaluate one tool output overall and per article source."
    )
    parser.add_argument("tool_output", type=Path, help="Path to tool output JSONL.")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=default_splits,
        help="Directory containing benchmark split JSONL files.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Benchmark split to evaluate.",
    )
    parser.add_argument(
        "--tool-name",
        default="",
        help="Optional display name. Defaults to the tool_output stem.",
    )
    parser.add_argument(
        "--tool-version",
        default="slice-eval",
        help="Version label recorded in the result JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Concern matching threshold.",
    )
    parser.add_argument(
        "--algorithm",
        default="hungarian",
        choices=["hungarian", "greedy"],
        help="Matching algorithm.",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=0,
        help="Bootstrap resamples per slice (0 disables CIs).",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Disable SPECTER2 embeddings and force Jaccard fallback.",
    )
    parser.add_argument(
        "--embed-model",
        type=Path,
        default=None,
        help="Optional local sentence-transformers model path.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix for .json and .md files.",
    )
    return parser


def configure_imports() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    bench_root = project_root.parent / "peer-review-benchmark"
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))
    return project_root


def maybe_configure_embedding(project_root: Path, embed_model: Path | None) -> tuple[str, bool]:
    chosen = embed_model or (project_root / "models" / "specter2_base")
    if not chosen.exists():
        try:
            import sentence_transformers  # type: ignore  # noqa: F401
            return "default:allenai/specter2_base", True
        except Exception:
            return "unavailable:jaccard_fallback", False

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        from bioreview_bench.evaluate import metrics as eval_metrics

        eval_metrics._EMBED_MODEL = SentenceTransformer(str(chosen))
        eval_metrics._EMBED_AVAILABLE = True
        return f"local:{chosen}", True
    except Exception:
        try:
            import sentence_transformers  # type: ignore  # noqa: F401
            return "default:allenai/specter2_base", True
        except Exception:
            return "unavailable:jaccard_fallback", False


def infer_source(entry: dict) -> str:
    source = str(entry.get("source", "")).strip().lower()
    if source:
        return source
    art_id = str(entry.get("id", entry.get("article_id", ""))).strip().lower()
    if ":" in art_id:
        return art_id.split(":", 1)[0]
    return "unknown"


def result_to_dict(result, coverage_log: list[dict], article_results: list) -> dict:
    n_covered = sum(1 for row in coverage_log if row["in_tool_output"])
    n_zero = sum(1 for row in coverage_log if row["recall"] == 0.0)
    n_perfect = sum(1 for row in coverage_log if row["recall"] == 1.0)
    return {
        "n_articles": int(result.n_articles),
        "n_articles_in_output": n_covered,
        "coverage_rate": (n_covered / result.n_articles) if result.n_articles else 0.0,
        "n_human_concerns": int(result.n_human_concerns),
        "n_tool_concerns": int(result.n_tool_concerns),
        "n_zero_recall_articles": n_zero,
        "n_perfect_recall_articles": n_perfect,
        "recall_overall": float(result.recall_overall),
        "precision_overall": float(result.precision_overall),
        "f1_micro": float(result.f1_micro),
        "recall_major": float(result.recall_major),
        "soft_recall_overall": float(result.soft_recall_overall),
        "soft_precision_overall": float(result.soft_precision_overall),
        "soft_f1": float(result.soft_f1),
        "matching_method": (
            article_results[0].matching_method if article_results else "unknown"
        ),
        "matching_algorithm": (
            result.matching_stats.algorithm if result.matching_stats else "unknown"
        ),
        "per_category": {
            cat: {
                "recall": float(metrics.recall),
                "precision": float(metrics.precision),
                "f1_micro": float(metrics.f1_micro),
                "n_human_concerns": int(metrics.n_human_concerns),
                "n_matched": int(metrics.n_matched),
            }
            for cat, metrics in sorted(result.per_category.items())
        },
    }


def render_markdown(
    tool_name: str,
    split: str,
    tool_output: Path,
    overall: dict,
    by_source: dict[str, dict],
    threshold: float,
    algorithm: str,
    use_embedding: bool,
    embed_label: str,
) -> str:
    lines = [
        f"# Source-Sliced Evaluation: {tool_name}",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        f"- Tool output: `{tool_output}`",
        f"- Split: `{split}`",
        f"- Threshold: `{threshold}`",
        f"- Algorithm: `{algorithm}`",
        f"- Embedding: `{'on' if use_embedding else 'off'}`",
        f"- Embed model: `{embed_label}`",
        "",
        "## Overall",
        "",
        "| Articles | Coverage | GT concerns | Tool concerns | Recall | Precision | F1 | Recall (major) | Zero recall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {overall['n_articles']} | {overall['coverage_rate']:.4f} | "
            f"{overall['n_human_concerns']} | {overall['n_tool_concerns']} | "
            f"{overall['recall_overall']:.4f} | {overall['precision_overall']:.4f} | "
            f"{overall['f1_micro']:.4f} | {overall['recall_major']:.4f} | "
            f"{overall['n_zero_recall_articles']} |"
        ),
        "",
        "## By Source",
        "",
        "| Source | Articles | Coverage | GT concerns | Tool concerns | Recall | Precision | F1 | Recall (major) | Zero recall | Perfect recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    rows = sorted(
        by_source.items(),
        key=lambda item: item[1]["n_articles"],
        reverse=True,
    )
    for source, row in rows:
        lines.append(
            f"| {source} | {row['n_articles']} | {row['coverage_rate']:.4f} | "
            f"{row['n_human_concerns']} | {row['n_tool_concerns']} | "
            f"{row['recall_overall']:.4f} | {row['precision_overall']:.4f} | "
            f"{row['f1_micro']:.4f} | {row['recall_major']:.4f} | "
            f"{row['n_zero_recall_articles']} | {row['n_perfect_recall_articles']} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    project_root = configure_imports()
    embed_label, embedding_ready = maybe_configure_embedding(project_root, args.embed_model)

    from bioreview_bench.evaluate.metrics import ConcernMatcher
    from bioreview_bench.evaluate.runner import (
        aggregate_results,
        build_tool_map,
        evaluate_articles,
        load_split,
    )

    tool_name = args.tool_name or args.tool_output.stem
    tool_map = build_tool_map(args.tool_output)
    gt_entries = load_split(args.splits_dir, args.split)

    matcher = ConcernMatcher(
        threshold=args.threshold,
        exclude_figure=True,
        use_embedding=not args.no_embedding,
        algorithm=args.algorithm,
    )

    article_results, coverage_log = evaluate_articles(tool_map, gt_entries, matcher)
    overall_result = aggregate_results(
        article_results=article_results,
        n_bootstrap=args.bootstrap_n,
        tool_name=tool_name,
        tool_version=args.tool_version,
        git_hash="",
        split=args.split,
        extraction_manifest_id="em-v3",
        n_articles=len(article_results),
        n_human_concerns=sum(r.n_gt_total for r in article_results),
        n_tool_concerns=sum(r.n_tool_total for r in article_results),
        n_figure_excluded=sum(r.n_gt_figure_excluded for r in article_results),
        notes=f"Source-sliced evaluation for {tool_name}",
    )

    grouped: dict[str, dict[str, list]] = {}
    for entry, article_result, coverage in zip(gt_entries, article_results, coverage_log):
        source = infer_source(entry)
        if source not in grouped:
            grouped[source] = {"results": [], "coverage": []}
        grouped[source]["results"].append(article_result)
        grouped[source]["coverage"].append(coverage)

    by_source_results: dict[str, dict] = {}
    for source, payload in sorted(grouped.items()):
        src_results = payload["results"]
        src_coverage = payload["coverage"]
        src_result = aggregate_results(
            article_results=src_results,
            n_bootstrap=args.bootstrap_n,
            tool_name=tool_name,
            tool_version=args.tool_version,
            git_hash="",
            split=args.split,
            extraction_manifest_id="em-v3",
            n_articles=len(src_results),
            n_human_concerns=sum(r.n_gt_total for r in src_results),
            n_tool_concerns=sum(r.n_tool_total for r in src_results),
            n_figure_excluded=sum(r.n_gt_figure_excluded for r in src_results),
            notes=f"Source-sliced evaluation for {tool_name} [{source}]",
        )
        by_source_results[source] = result_to_dict(src_result, src_coverage, src_results)

    overall = result_to_dict(overall_result, coverage_log, article_results)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "tool_name": tool_name,
        "tool_output": str(args.tool_output.resolve()),
        "split": args.split,
        "splits_dir": str(args.splits_dir.resolve()),
        "threshold": args.threshold,
        "algorithm": args.algorithm,
        "use_embedding": not args.no_embedding,
        "embedding_ready": embedding_ready,
        "embed_model": embed_label,
        "overall": overall,
        "by_source": by_source_results,
    }

    if args.output_prefix is None:
        output_prefix = (
            project_root
            / "results"
            / "source_eval"
            / f"{args.tool_output.stem}_{args.split}_by_source"
        )
    else:
        output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        render_markdown(
            tool_name=tool_name,
            split=args.split,
            tool_output=args.tool_output.resolve(),
            overall=overall,
            by_source=by_source_results,
            threshold=args.threshold,
            algorithm=args.algorithm,
            use_embedding=(not args.no_embedding and embedding_ready),
            embed_label=embed_label,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
