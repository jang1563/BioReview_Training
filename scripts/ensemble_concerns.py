#!/usr/bin/env python3
"""Ensemble concerns from two SFT model inference outputs.

Combines inference JSONL outputs from two models (e.g., 8B and 9B) using
union or intersection strategy, with SPECTER2-based semantic deduplication.

For union strategy:
  - Collect all concerns from both models.
  - Deduplicate using SPECTER2 cosine similarity >= cluster-threshold.
  - When two concerns cluster together, prefer the one from model-a.

For intersection strategy:
  - Only keep concerns that appear in both models (cross-model similarity
    >= cluster-threshold).

Usage:
  # Union (max recall, dedup at 0.98):
  python scripts/ensemble_concerns.py \\
      --model-a results/sft_eval/qwen3.5_9b_val.jsonl \\
      --model-b results/sft_eval/deepseek_r1_14b_val.jsonl \\
      --output results/sft_eval/ensemble_union_val.jsonl \\
      --strategy union

  # Intersection (high precision):
  python scripts/ensemble_concerns.py \\
      --model-a results/sft_eval/qwen3.5_9b_val.jsonl \\
      --model-b results/sft_eval/deepseek_r1_14b_val.jsonl \\
      --output results/sft_eval/ensemble_intersection_val.jsonl \\
      --strategy intersection --cluster-threshold 0.85

  # With evaluation:
  python scripts/ensemble_concerns.py \\
      --model-a results/sft_eval/model_a_val.jsonl \\
      --model-b results/sft_eval/model_b_val.jsonl \\
      --output results/sft_eval/ensemble_union_val.jsonl \\
      --strategy union --evaluate --splits-dir /path/to/splits/v3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# SPECTER2 loading
# ---------------------------------------------------------------------------

_EMBEDDER_CACHE: Any = None


def _check_local_weights(model_dir: Path) -> bool:
    """Check whether a local model directory contains actual weight files."""
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    has_bin = any(model_dir.glob("*.bin"))
    has_safetensors = any(model_dir.glob("*.safetensors"))
    return has_bin or has_safetensors


def _load_embedder(embed_model: str | None = None) -> Any:
    """Load SPECTER2 embedding model.

    Resolution order:
      1. --embed-model CLI argument (explicit path or HF repo)
      2. BIOREVIEW_EMBED_MODEL env var
      3. Local models/specter2_base/ (if weight files exist)
      4. Hub download: allenai/specter2_base

    Returns a SentenceTransformer instance or None on failure.
    """
    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE is not None:
        return _EMBEDDER_CACHE

    # Determine model path
    local_specter2 = Path(__file__).resolve().parents[1] / "models" / "specter2_base"

    if embed_model:
        model_name = embed_model
    elif os.getenv("BIOREVIEW_EMBED_MODEL"):
        model_name = os.environ["BIOREVIEW_EMBED_MODEL"].strip()
    elif _check_local_weights(local_specter2):
        model_name = str(local_specter2)
        os.environ["BIOREVIEW_EMBED_MODEL"] = model_name
    else:
        model_name = "allenai/specter2_base"

    # Determine if local-only loading is appropriate
    local_only = False
    model_path = Path(model_name)
    if model_path.exists() and model_path.is_dir():
        if _check_local_weights(model_path):
            local_only = True
        else:
            print(
                f"WARNING: directory {model_name} exists but has no weight files "
                f"(.bin/.safetensors). Attempting hub download.",
                file=sys.stderr,
            )
    env_flag = os.getenv("BIOREVIEW_EMBED_LOCAL_ONLY", "").strip().lower()
    if env_flag in {"1", "true", "yes", "on"}:
        local_only = True

    try:
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {}
        if local_only:
            kwargs["local_files_only"] = True
        embedder = SentenceTransformer(model_name, **kwargs)
        _EMBEDDER_CACHE = embedder
        print(f"  SPECTER2 loaded: {model_name} (local_only={local_only})")
        return embedder
    except Exception as exc:
        print(
            f"WARNING: SPECTER2 not available ({exc}). Falling back to Jaccard.",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


def _jaccard(a: str, b: str) -> float:
    """Simple word-level Jaccard similarity."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _compute_similarity_matrix(
    texts: list[str], embedder: Any
) -> list[list[float]]:
    """Compute pairwise cosine similarity matrix.

    Uses SPECTER2 embeddings when available, Jaccard fallback otherwise.
    """
    n = len(texts)
    if embedder is not None:
        embs = embedder.encode(texts, normalize_embeddings=True)
        return (embs @ embs.T).tolist()
    else:
        return [[_jaccard(texts[i], texts[j]) for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# Union deduplication
# ---------------------------------------------------------------------------


def _union_dedup(
    concerns_a: list[dict],
    concerns_b: list[dict],
    embedder: Any,
    cluster_threshold: float,
) -> list[dict]:
    """Union strategy: keep all unique concerns, dedup across models.

    For each pair of concerns from different models with cosine similarity
    >= cluster_threshold, keep only one (prefer model-a's version).
    """
    all_concerns = concerns_a + concerns_b
    if not all_concerns:
        return []

    n = len(all_concerns)
    n_a = len(concerns_a)
    texts = [c["text"] for c in all_concerns]

    # Compute similarity matrix
    sim = _compute_similarity_matrix(texts, embedder)

    # Adjust threshold for Jaccard fallback
    effective_threshold = cluster_threshold
    if embedder is None:
        effective_threshold = cluster_threshold * 0.3

    # Connected-component clustering via union-find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i][j] >= effective_threshold:
                union(i, j)

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # For each cluster, pick representative: prefer model-a, then longest text
    selected: list[dict] = []
    for cluster_indices in groups.values():
        # Determine sources present in this cluster
        sources_in_cluster = set()
        for idx in cluster_indices:
            sources_in_cluster.add(all_concerns[idx]["source"])

        # Prefer model-a concerns, then longest text
        a_indices = [i for i in cluster_indices if i < n_a]
        if a_indices:
            best_idx = max(a_indices, key=lambda i: len(all_concerns[i]["text"]))
        else:
            best_idx = max(cluster_indices, key=lambda i: len(all_concerns[i]["text"]))

        concern = dict(all_concerns[best_idx])
        # Record which model(s) contributed to this cluster
        concern["source"] = "+".join(sorted(sources_in_cluster))
        selected.append(concern)

    return selected


# ---------------------------------------------------------------------------
# Intersection
# ---------------------------------------------------------------------------


def _intersection(
    concerns_a: list[dict],
    concerns_b: list[dict],
    embedder: Any,
    cluster_threshold: float,
) -> list[dict]:
    """Intersection strategy: keep only concerns with a cross-model match.

    A concern from model-a is kept if there exists at least one concern
    from model-b with similarity >= cluster_threshold (and vice versa).
    We prefer the model-a version.
    """
    if not concerns_a or not concerns_b:
        return []

    texts_a = [c["text"] for c in concerns_a]
    texts_b = [c["text"] for c in concerns_b]

    # Compute cross-model similarities
    if embedder is not None:
        embs_a = embedder.encode(texts_a, normalize_embeddings=True)
        embs_b = embedder.encode(texts_b, normalize_embeddings=True)
        cross_sim = (embs_a @ embs_b.T).tolist()
    else:
        cross_sim = [
            [_jaccard(a, b) for b in texts_b] for a in texts_a
        ]
        cluster_threshold = cluster_threshold * 0.3

    # For each concern in A, find best match in B
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    for i in range(len(concerns_a)):
        for j in range(len(concerns_b)):
            if cross_sim[i][j] >= cluster_threshold:
                matched_a.add(i)
                matched_b.add(j)

    # Build output: keep model-a's version of matched concerns, plus any
    # model-b concerns that matched but have no model-a counterpart
    selected: list[dict] = []
    used_b: set[int] = set()

    for i in sorted(matched_a):
        concern = dict(concerns_a[i])
        # Find best match in B for source annotation
        best_j = max(range(len(concerns_b)), key=lambda j: cross_sim[i][j])
        if cross_sim[i][best_j] >= cluster_threshold:
            used_b.add(best_j)
        source_a = concerns_a[i]["source"]
        source_b = concerns_b[best_j]["source"] if cross_sim[i][best_j] >= cluster_threshold else ""
        if source_b and source_a != source_b:
            concern["source"] = f"{source_a}+{source_b}"
        selected.append(concern)

    return selected


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _group_by_article(rows: list[dict]) -> dict[str, dict]:
    """Group rows by article_id. Returns {article_id: row}."""
    grouped: dict[str, dict] = {}
    for row in rows:
        art_id = row.get("article_id", "")
        if art_id:
            grouped[art_id] = row
    return grouped


def _extract_concerns(row: dict, source_label: str) -> list[dict]:
    """Extract concerns from a row, normalizing to structured format.

    Each returned dict has: text, category, severity, source.
    """
    concerns: list[dict] = []
    structured = row.get("structured_concerns") or []
    texts = row.get("concerns") or []

    if structured:
        for sc in structured:
            text = str(sc.get("text", "")).strip()
            if not text:
                continue
            concerns.append(
                {
                    "text": text,
                    "category": str(sc.get("category", "unknown")),
                    "severity": str(sc.get("severity", "unknown")),
                    "source": source_label,
                }
            )
    elif texts:
        for t in texts:
            t = str(t).strip()
            if not t:
                continue
            concerns.append(
                {
                    "text": t,
                    "category": "unknown",
                    "severity": "unknown",
                    "source": source_label,
                }
            )

    return concerns


# ---------------------------------------------------------------------------
# Per-article ensemble
# ---------------------------------------------------------------------------


def ensemble_article(
    row_a: dict | None,
    row_b: dict | None,
    label_a: str,
    label_b: str,
    embedder: Any,
    strategy: str,
    cluster_threshold: float,
) -> list[dict]:
    """Ensemble concerns for a single article."""
    concerns_a = _extract_concerns(row_a, label_a) if row_a else []
    concerns_b = _extract_concerns(row_b, label_b) if row_b else []

    if strategy == "union":
        return _union_dedup(concerns_a, concerns_b, embedder, cluster_threshold)
    elif strategy == "intersection":
        return _intersection(concerns_a, concerns_b, embedder, cluster_threshold)
    else:
        # Fallback to union
        return _union_dedup(concerns_a, concerns_b, embedder, cluster_threshold)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_evaluation_wrapper(
    output_path: Path,
    splits_dir: Path,
    split: str,
    threshold: float,
    algorithm: str,
    tool_name: str,
) -> dict:
    """Run bioreview_bench evaluation and return metrics dict.

    Follows the same pattern as postprocess_inference_output.py and
    run_sft_inference.py.
    """
    workspace_root = Path(__file__).resolve().parents[2]
    bench_root = workspace_root / "peer-review-benchmark"
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    # Verify SPECTER2 is available for evaluation
    local_specter2 = Path(__file__).resolve().parents[1] / "models" / "specter2_base"
    has_weights = _check_local_weights(local_specter2)
    if local_specter2.exists() and has_weights:
        os.environ.setdefault("BIOREVIEW_EMBED_MODEL", str(local_specter2))

    # Pre-flight SPECTER2 check: evaluation with Jaccard gives misleadingly
    # low scores, so we refuse to proceed without real embeddings.
    try:
        from sentence_transformers import SentenceTransformer as _ST

        _model_path = os.getenv("BIOREVIEW_EMBED_MODEL", "allenai/specter2_base")
        _local = Path(_model_path).exists() and _check_local_weights(Path(_model_path))
        _m = _ST(_model_path, local_files_only=_local)
        _m.encode(["test"], normalize_embeddings=True)
        del _m
        print("  SPECTER2 verified OK for evaluation", flush=True)
    except Exception as _e:
        print(
            f"ERROR: SPECTER2 not available for evaluation ({_e}).\n"
            "Evaluation requires SPECTER2. Skipping to avoid misleading Jaccard metrics.\n"
            "Run scripts/reevaluate_ensemble.py on HPC where SPECTER2 is accessible.",
            file=sys.stderr,
        )
        return {}

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
        tool_name=tool_name,
        tool_version="ensemble-v2",
        extraction_manifest_id="em-v3",
        notes=f"Ensemble output from {tool_name}",
    )

    metrics = {
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
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    default_splits = workspace_root / "peer-review-benchmark" / "data" / "splits" / "v3"

    p = argparse.ArgumentParser(
        description=(
            "Ensemble concerns from two SFT model outputs using union or "
            "intersection strategy with SPECTER2 deduplication."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/ensemble_concerns.py \\\n"
            "      --model-a results/sft_eval/9b_val.jsonl \\\n"
            "      --model-b results/sft_eval/14b_val.jsonl \\\n"
            "      --output results/sft_eval/ensemble_union_val.jsonl\n"
            "\n"
            "  python scripts/ensemble_concerns.py \\\n"
            "      --model-a results/sft_eval/9b_val.jsonl \\\n"
            "      --model-b results/sft_eval/14b_val.jsonl \\\n"
            "      --output results/sft_eval/ensemble_intersection_val.jsonl \\\n"
            "      --strategy intersection --cluster-threshold 0.85\n"
        ),
    )

    # Input / output
    p.add_argument(
        "--model-a",
        type=Path,
        required=True,
        help="Inference JSONL from model A (preferred for dedup representative).",
    )
    p.add_argument(
        "--model-b",
        type=Path,
        required=True,
        help="Inference JSONL from model B.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output merged JSONL path.",
    )

    # Strategy
    p.add_argument(
        "--strategy",
        choices=["union", "intersection"],
        default="union",
        help="Ensemble strategy (default: union).",
    )
    p.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.98,
        help=(
            "SPECTER2 cosine similarity threshold for deduplication. "
            "0.98 is the empirically tuned value from v1/v2 experiments "
            "(default: 0.98)."
        ),
    )

    # Embedding model
    p.add_argument(
        "--embed-model",
        type=str,
        default=None,
        help=(
            "Path to a local SPECTER2 model directory, or HuggingFace repo ID. "
            "Default: auto-detect from models/specter2_base/ or "
            "BIOREVIEW_EMBED_MODEL env var."
        ),
    )

    # Evaluation
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Run bioreview_bench evaluation after merging.",
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        default=default_splits,
        help="Splits directory for evaluation (default: peer-review-benchmark/data/splits/v3).",
    )
    p.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split for evaluation (default: val).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Concern matching threshold for evaluation (default: 0.65).",
    )
    p.add_argument(
        "--algorithm",
        choices=["hungarian", "greedy"],
        default="hungarian",
        help="Matching algorithm for evaluation (default: hungarian).",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    # Validate input files
    for label, path in [("model-a", args.model_a), ("model-b", args.model_b)]:
        if not path.exists():
            print(f"ERROR: --{label} file not found: {path}", file=sys.stderr)
            raise SystemExit(1)

    # Derive labels from filenames
    label_a = args.model_a.stem
    label_b = args.model_b.stem

    print("=" * 60)
    print("Ensemble Concerns")
    print("=" * 60)
    print(f"  strategy:          {args.strategy}")
    print(f"  cluster-threshold: {args.cluster_threshold}")
    print(f"  model-a:           {args.model_a}  ({label_a})")
    print(f"  model-b:           {args.model_b}  ({label_b})")
    print(f"  output:            {args.output}")
    if args.embed_model:
        print(f"  embed-model:       {args.embed_model}")
    print()

    # Load embedder
    print("Loading SPECTER2 embedder...", flush=True)
    t0 = time.time()
    embedder = _load_embedder(embed_model=args.embed_model)
    print(f"  embedder ready in {time.time() - t0:.1f}s", flush=True)
    print()

    # Load model outputs
    print("Loading model outputs...", flush=True)
    rows_a = _load_jsonl(args.model_a)
    rows_b = _load_jsonl(args.model_b)
    grouped_a = _group_by_article(rows_a)
    grouped_b = _group_by_article(rows_b)

    # Determine the full set of article IDs
    all_article_ids = sorted(set(grouped_a.keys()) | set(grouped_b.keys()))
    articles_only_a = set(grouped_a.keys()) - set(grouped_b.keys())
    articles_only_b = set(grouped_b.keys()) - set(grouped_a.keys())
    articles_both = set(grouped_a.keys()) & set(grouped_b.keys())

    print(f"  model-a articles:  {len(grouped_a)}")
    print(f"  model-b articles:  {len(grouped_b)}")
    print(f"  shared articles:   {len(articles_both)}")
    print(f"  only in A:         {len(articles_only_a)}")
    print(f"  only in B:         {len(articles_only_b)}")
    print(f"  total unique:      {len(all_article_ids)}")
    print()

    # Per-model concern counts (before ensemble)
    total_concerns_a = sum(
        len(row.get("concerns", [])) for row in grouped_a.values()
    )
    total_concerns_b = sum(
        len(row.get("concerns", [])) for row in grouped_b.values()
    )
    print(f"  total concerns A:  {total_concerns_a}")
    print(f"  total concerns B:  {total_concerns_b}")
    print()

    # Ensemble per article
    print(f"Running {args.strategy} ensemble...", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_after = 0
    n_empty = 0
    t_start = time.time()

    with args.output.open("w", encoding="utf-8") as fh:
        for i, art_id in enumerate(all_article_ids):
            row_a = grouped_a.get(art_id)
            row_b = grouped_b.get(art_id)

            selected = ensemble_article(
                row_a=row_a,
                row_b=row_b,
                label_a=label_a,
                label_b=label_b,
                embedder=embedder,
                strategy=args.strategy,
                cluster_threshold=args.cluster_threshold,
            )

            # Build output row with all standard fields
            output_row = {
                "article_id": art_id,
                "concerns": [c["text"] for c in selected],
                "structured_concerns": [
                    {
                        "text": c["text"],
                        "category": c["category"],
                        "severity": c["severity"],
                        "source": c["source"],
                    }
                    for c in selected
                ],
                "raw_output": f"[ensemble:{args.strategy}]",
                "parse_ok": len(selected) > 0,
                "generation_time_s": 0.0,
            }
            fh.write(json.dumps(output_row, ensure_ascii=False) + "\n")

            total_after += len(selected)
            if len(selected) == 0:
                n_empty += 1

            if (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{len(all_article_ids)}] processed...", flush=True)

    elapsed = time.time() - t_start

    # Summary stats
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  articles processed: {len(all_article_ids)}")
    print(f"  concerns from A:    {total_concerns_a}")
    print(f"  concerns from B:    {total_concerns_b}")
    print(f"  concerns combined:  {total_concerns_a + total_concerns_b}")
    print(f"  concerns after {args.strategy}: {total_after}")
    if total_concerns_a + total_concerns_b > 0:
        dedup_pct = (
            1.0 - total_after / (total_concerns_a + total_concerns_b)
        ) * 100
        print(f"  dedup reduction:    {dedup_pct:.1f}%")
    if len(all_article_ids) > 0:
        print(f"  avg concerns/article: {total_after / len(all_article_ids):.1f}")
    print(f"  empty articles:     {n_empty}")
    print(f"  elapsed:            {elapsed:.1f}s")
    print(f"  output:             {args.output}")

    # Save summary JSON
    summary = {
        "model_dir": str(args.output.parent / f"ensemble_{args.strategy}"),
        "strategy": args.strategy,
        "cluster_threshold": args.cluster_threshold,
        "model_a": str(args.model_a),
        "model_b": str(args.model_b),
        "label_a": label_a,
        "label_b": label_b,
        "stats": {
            "articles_processed": len(all_article_ids),
            "articles_shared": len(articles_both),
            "articles_only_a": len(articles_only_a),
            "articles_only_b": len(articles_only_b),
            "concerns_from_a": total_concerns_a,
            "concerns_from_b": total_concerns_b,
            "concerns_combined": total_concerns_a + total_concerns_b,
            "concerns_after_merge": total_after,
            "empty_articles": n_empty,
            "elapsed_s": round(elapsed, 2),
        },
    }

    # Evaluation
    if args.evaluate:
        print()
        print("Running evaluation...", flush=True)
        tool_name = f"ensemble_{args.strategy}"
        metrics = run_evaluation_wrapper(
            output_path=args.output,
            splits_dir=args.splits_dir,
            split=args.split,
            threshold=args.threshold,
            algorithm=args.algorithm,
            tool_name=tool_name,
        )
        if metrics:
            summary["eval_metrics"] = metrics
            print()
            print("Evaluation Results:")
            print(f"  recall:    {metrics['recall_overall']:.4f}")
            print(f"  precision: {metrics['precision_overall']:.4f}")
            print(f"  f1_micro:  {metrics['f1_micro']:.4f}")
            if metrics.get("recall_major") is not None:
                print(f"  recall_major: {metrics['recall_major']:.4f}")
            print(f"  n_articles:  {metrics['n_articles']}")

    # Write summary
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  summary:   {summary_path}")


if __name__ == "__main__":
    main()
