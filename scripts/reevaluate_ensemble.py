#!/usr/bin/env python3
"""Re-evaluate ensemble JSONL files with SPECTER2 (fixes Jaccard fallback issue).

Run this on HPC where peer-review-benchmark and models/specter2_base/ are accessible.

Usage:
    python scripts/reevaluate_ensemble.py
    python scripts/reevaluate_ensemble.py --jsonl results/sft_eval/ensemble_union_v1_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    default_splits = workspace_root / "peer-review-benchmark" / "data" / "splits" / "v3"

    p = argparse.ArgumentParser(description="Re-evaluate ensemble JSONL with SPECTER2.")
    p.add_argument(
        "--jsonl",
        nargs="+",
        type=Path,
        default=None,
        help="JSONL files to re-evaluate. Default: all ensemble_*_val.jsonl in results/sft_eval/",
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        default=default_splits,
    )
    p.add_argument("--split", default="val")
    p.add_argument("--threshold", type=float, default=0.65)
    return p


def main() -> None:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    bench_root = workspace_root / "peer-review-benchmark"

    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    # Set SPECTER2 path
    local_specter2 = project_root / "models" / "specter2_base"
    if local_specter2.exists():
        os.environ["BIOREVIEW_EMBED_MODEL"] = str(local_specter2)
        print(f"SPECTER2 path: {local_specter2}", flush=True)
    else:
        print(f"WARNING: {local_specter2} not found — will try to download", flush=True)

    # Verify SPECTER2 loads
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get("BIOREVIEW_EMBED_MODEL", "allenai/specter2_base")
        model = SentenceTransformer(model_name, local_files_only=local_specter2.exists())
        test_emb = model.encode(["test concern"], normalize_embeddings=True)
        print(f"SPECTER2 loaded OK (dim={test_emb.shape[1]})", flush=True)
        del model  # free before eval (will be re-loaded by metrics.py)
    except Exception as exc:
        print(f"ERROR: SPECTER2 failed to load: {exc}", file=sys.stderr)
        print("Cannot proceed without SPECTER2. Aborting.", file=sys.stderr)
        sys.exit(1)

    from bioreview_bench.evaluate.runner import run_evaluation

    # Find files to evaluate
    if args.jsonl:
        files = args.jsonl
    else:
        sft_eval_dir = project_root / "results" / "sft_eval"
        files = sorted(sft_eval_dir.glob("ensemble_*_val.jsonl"))
        print(f"Auto-detected {len(files)} ensemble files: {[f.name for f in files]}", flush=True)

    if not files:
        print("No files to evaluate.", file=sys.stderr)
        sys.exit(1)

    for tool_path in files:
        if not tool_path.exists():
            print(f"SKIP: {tool_path} not found", file=sys.stderr)
            continue

        tag = tool_path.stem  # e.g. ensemble_union_v1_val
        print(f"\n{'='*60}")
        print(f"Evaluating: {tool_path.name}", flush=True)

        try:
            result, _ = run_evaluation(
                tool_output=tool_path,
                splits_dir=args.splits_dir,
                split=args.split,
                threshold=args.threshold,
                exclude_figure=True,
                use_embedding=True,
                algorithm="hungarian",
                tool_name=tag,
                tool_version="v1",
            )

            print(f"\nResults for {tag}:")
            print(f"  recall:    {result.recall_overall:.4f}")
            print(f"  precision: {result.precision_overall:.4f}")
            print(f"  f1_micro:  {result.f1_micro:.4f}")
            print(f"  n_articles: {result.n_articles}")
            print(f"  n_human:   {result.n_human_concerns}")
            print(f"  n_tool:    {result.n_tool_concerns}")

            # Update summary JSON
            summary_path = tool_path.with_suffix(".summary.json")
            if summary_path.exists():
                with summary_path.open() as f:
                    summary = json.load(f)
            else:
                summary = {"model_dir": str(tool_path.parent / tag)}

            summary["eval_metrics"] = {
                "recall_overall": float(result.recall_overall),
                "precision_overall": float(result.precision_overall),
                "f1_micro": float(result.f1_micro),
                "recall_major": float(result.recall_major),
                "n_articles": result.n_articles,
                "n_human_concerns": result.n_human_concerns,
                "n_tool_concerns": result.n_tool_concerns,
                "eval_method": "specter2",
            }
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            print(f"  summary saved: {summary_path}")

        except Exception as exc:
            print(f"ERROR evaluating {tool_path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
