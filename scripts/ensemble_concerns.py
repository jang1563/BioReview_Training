#!/usr/bin/env python3
"""Ensemble concerns from multiple SFT model outputs.

Takes multiple model JSONL files, deduplicates concerns per article using
SPECTER2 embeddings, applies a combination strategy (union/intersection/vote),
and outputs a standard JSONL compatible with the evaluation pipeline.

Usage:
  # Union of all concerns (max recall):
  python scripts/ensemble_concerns.py \
      --models results/sft_eval/model1_val.jsonl results/sft_eval/model2_val.jsonl \
      --labels "Model1" "Model2" \
      --strategy union \
      --output results/sft_eval/ensemble_union_val.jsonl

  # Vote: concerns from >= 2 models:
  python scripts/ensemble_concerns.py \
      --models model1.jsonl model2.jsonl model3.jsonl \
      --labels "M1" "M2" "M3" \
      --strategy vote --min-votes 2 \
      --output ensemble_vote2_val.jsonl

  # With evaluation:
  python scripts/ensemble_concerns.py \
      --models model1.jsonl model2.jsonl \
      --labels "M1" "M2" \
      --strategy union \
      --output ensemble_union_val.jsonl \
      --evaluate --splits-dir /path/to/splits/v3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Concern clustering
# ---------------------------------------------------------------------------


def _load_embedder():
    """Load SPECTER2 embedding model using the project convention."""
    # Same pattern as run_sft_inference.py:414-416
    local_specter2 = Path(__file__).resolve().parents[1] / "models" / "specter2_base"
    if local_specter2.exists() and not os.getenv("BIOREVIEW_EMBED_MODEL"):
        os.environ["BIOREVIEW_EMBED_MODEL"] = str(local_specter2)

    try:
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("BIOREVIEW_EMBED_MODEL", "allenai/specter2_base").strip()
        local_only_str = os.getenv("BIOREVIEW_EMBED_LOCAL_ONLY", "").strip().lower()
        kwargs = (
            {"local_files_only": True}
            if local_only_str in {"1", "true", "yes", "on"}
            else {}
        )
        return SentenceTransformer(model_name, **kwargs)
    except Exception as exc:
        print(
            f"WARNING: SPECTER2 not available ({exc}). Falling back to Jaccard.",
            file=sys.stderr,
        )
        return None


def _jaccard(a: str, b: str) -> float:
    """Simple word-level Jaccard similarity."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cluster_concerns(
    all_concerns: list[dict],
    embedder,
    threshold: float = 0.70,
) -> list[list[int]]:
    """Cluster concerns by semantic similarity using connected components.

    Args:
        all_concerns: List of concern dicts with 'text' field.
        embedder: SentenceTransformer model (or None for Jaccard fallback).
        threshold: Similarity threshold for clustering.

    Returns:
        List of clusters, each a list of concern indices.
    """
    n = len(all_concerns)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    texts = [c["text"] for c in all_concerns]

    # Build similarity matrix
    if embedder is not None:
        embs = embedder.encode(texts, normalize_embeddings=True)
        sim_matrix = (embs @ embs.T).tolist()
    else:
        # Jaccard fallback with scaled threshold
        sim_matrix = [[_jaccard(a, b) for b in texts] for a in texts]
        threshold = threshold * 0.3  # Same scale as bioreview_bench

    # Connected components via union-find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= threshold:
                union(i, j)

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values())


# ---------------------------------------------------------------------------
# Ensemble strategies
# ---------------------------------------------------------------------------


def apply_strategy(
    clusters: list[list[int]],
    all_concerns: list[dict],
    strategy: str,
    min_votes: int,
    n_models: int,
) -> list[dict]:
    """Apply ensemble strategy to select concerns from clusters.

    Each concern in all_concerns has a 'source_model' field.

    Returns selected concerns (one representative per selected cluster).
    """
    selected = []

    for cluster in clusters:
        # Count unique models contributing to this cluster
        models_in_cluster = {all_concerns[i]["source_model"] for i in cluster}
        n_models_hit = len(models_in_cluster)

        keep = False
        if strategy == "union":
            keep = True
        elif strategy == "intersection":
            keep = n_models_hit == n_models
        elif strategy == "vote":
            keep = n_models_hit >= min_votes
        else:
            keep = True  # default to union

        if keep:
            # Pick representative: prefer the concern with most detail
            best_idx = max(cluster, key=lambda i: len(all_concerns[i]["text"]))
            selected.append(all_concerns[best_idx])

    return selected


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_model_outputs(paths: list[Path], labels: list[str]) -> dict[str, dict]:
    """Load multiple model JSONL files.

    Returns: {article_id: {label: row_dict}}.
    """
    combined: dict[str, dict] = {}
    for path, label in zip(paths, labels):
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
                    combined.setdefault(art_id, {})[label] = row
    return combined


# ---------------------------------------------------------------------------
# Main ensemble logic
# ---------------------------------------------------------------------------


def ensemble_article(
    article_data: dict[str, dict],
    embedder,
    strategy: str,
    min_votes: int,
    n_models: int,
    cluster_threshold: float,
) -> list[dict]:
    """Ensemble concerns for a single article across models."""
    all_concerns: list[dict] = []

    for label, row in article_data.items():
        structured = row.get("structured_concerns", [])
        texts = row.get("concerns", [])

        if structured:
            for sc in structured:
                all_concerns.append(
                    {
                        "text": sc.get("text", ""),
                        "category": sc.get("category", "unknown"),
                        "severity": sc.get("severity", "unknown"),
                        "source_model": label,
                    }
                )
        elif texts:
            for t in texts:
                all_concerns.append(
                    {
                        "text": t,
                        "category": "unknown",
                        "severity": "unknown",
                        "source_model": label,
                    }
                )

    # Filter empty texts
    all_concerns = [c for c in all_concerns if c["text"].strip()]

    if not all_concerns:
        return []

    clusters = cluster_concerns(all_concerns, embedder, cluster_threshold)
    return apply_strategy(clusters, all_concerns, strategy, min_votes, n_models)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    default_splits = workspace_root / "peer-review-benchmark" / "data" / "splits" / "v3"

    p = argparse.ArgumentParser(
        description="Ensemble concerns from multiple SFT model outputs."
    )
    p.add_argument(
        "--models",
        nargs="+",
        type=Path,
        required=True,
        help="Inference JSONL files from different models.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        help="Labels for each model (same order as --models).",
    )
    p.add_argument(
        "--strategy",
        choices=["union", "intersection", "vote"],
        default="union",
        help="Ensemble strategy (default: union).",
    )
    p.add_argument(
        "--min-votes",
        type=int,
        default=2,
        help="Min models supporting a concern for 'vote' strategy (default: 2).",
    )
    p.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.70,
        help="Similarity threshold for clustering concerns (default: 0.70).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path.",
    )
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after ensemble.",
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        default=default_splits,
        help="Splits directory for evaluation.",
    )
    p.add_argument(
        "--split",
        default="val",
        help="Dataset split for evaluation (default: val).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    labels = args.labels or [p.stem for p in args.models]
    if len(labels) != len(args.models):
        print("ERROR: --labels count must match --models count.", file=sys.stderr)
        raise SystemExit(1)

    n_models = len(args.models)

    # Validate inputs
    for p in args.models:
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            raise SystemExit(1)

    print(f"Ensemble Strategy: {args.strategy}")
    if args.strategy == "vote":
        print(f"  min_votes: {args.min_votes}")
    print(f"  cluster_threshold: {args.cluster_threshold}")
    print(f"  models: {', '.join(labels)}")
    print(f"  output: {args.output}")

    # Load embedder
    print("\nLoading embedder...", flush=True)
    embedder = _load_embedder()

    # Load all model outputs
    print("Loading model outputs...", flush=True)
    combined = load_model_outputs(args.models, labels)
    print(f"  {len(combined)} unique articles across {n_models} models")

    # Ensemble per article
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_concerns = 0
    n_empty = 0

    with args.output.open("w", encoding="utf-8") as fh:
        for i, (art_id, article_data) in enumerate(sorted(combined.items())):
            selected = ensemble_article(
                article_data,
                embedder,
                args.strategy,
                args.min_votes,
                n_models,
                args.cluster_threshold,
            )

            # Output format: all 6 fields (Bug fix #7)
            row = {
                "article_id": art_id,
                "concerns": [c["text"] for c in selected],
                "structured_concerns": [
                    {
                        "text": c["text"],
                        "category": c["category"],
                        "severity": c["severity"],
                    }
                    for c in selected
                ],
                "raw_output": f"[ensemble:{args.strategy}]",
                "parse_ok": len(selected) > 0,
                "generation_time_s": 0.0,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            total_concerns += len(selected)
            if len(selected) == 0:
                n_empty += 1

            if (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{len(combined)}] processed...", flush=True)

    print(f"\nEnsemble complete.")
    print(f"  articles:   {len(combined)}")
    print(f"  concerns:   {total_concerns}")
    print(f"  avg/article: {total_concerns / len(combined):.1f}" if combined else "")
    print(f"  empty:      {n_empty}")
    print(f"  output:     {args.output}")

    # Evaluation
    if args.evaluate:
        print("\nRunning evaluation...", flush=True)
        project_root = Path(__file__).resolve().parents[1]
        workspace_root = project_root.parent
        bench_root = workspace_root / "peer-review-benchmark"
        if str(bench_root) not in sys.path:
            sys.path.insert(0, str(bench_root))

        # Verify SPECTER2 is loadable before evaluation (Jaccard gives misleadingly low scores)
        try:
            from sentence_transformers import SentenceTransformer as _ST
            _model_path = os.getenv("BIOREVIEW_EMBED_MODEL", "allenai/specter2_base")
            _local = Path(_model_path).exists() if _model_path != "allenai/specter2_base" else False
            _m = _ST(_model_path, local_files_only=_local)
            _m.encode(["test"], normalize_embeddings=True)
            del _m
            print("  SPECTER2 verified OK", flush=True)
        except Exception as _e:
            print(
                f"ERROR: SPECTER2 not available for evaluation ({_e}).\n"
                "Evaluation requires SPECTER2. Skipping to avoid misleading Jaccard metrics.\n"
                "Run scripts/reevaluate_ensemble.py on HPC where SPECTER2 is accessible.",
                file=sys.stderr,
            )
            return

        try:
            from bioreview_bench.evaluate.runner import run_evaluation

            result, _ = run_evaluation(
                tool_output=args.output,
                splits_dir=args.splits_dir,
                split=args.split,
                threshold=0.65,
                exclude_figure=True,
                use_embedding=True,
                algorithm="hungarian",
                tool_name=f"ensemble_{args.strategy}",
                tool_version="v1",
            )
            print(f"\nEvaluation Results:")
            print(f"  recall:    {result.recall_overall:.4f}")
            print(f"  precision: {result.precision_overall:.4f}")
            print(f"  f1_micro:  {result.f1_micro:.4f}")

            # Save summary
            summary_path = args.output.with_suffix(".summary.json")
            # model_dir field is read by compare_models.py:load_results() to derive model name
            ensemble_tag = f"ensemble_{args.strategy}"
            if args.strategy == "vote":
                ensemble_tag += f"{args.min_votes}"
            summary = {
                "model_dir": str(args.output.parent / ensemble_tag),
                "strategy": args.strategy,
                "min_votes": args.min_votes,
                "cluster_threshold": args.cluster_threshold,
                "models": [str(p) for p in args.models],
                "labels": labels,
                "eval_metrics": {
                    "recall_overall": float(result.recall_overall),
                    "precision_overall": float(result.precision_overall),
                    "f1_micro": float(result.f1_micro),
                    "recall_major": float(result.recall_major),
                    "n_articles": result.n_articles,
                },
            }
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  summary:   {summary_path}")

        except Exception as exc:
            print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
