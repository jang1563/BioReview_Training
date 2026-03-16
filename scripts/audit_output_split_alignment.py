#!/usr/bin/env python3
"""Audit article-id overlap between tool outputs and benchmark split files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    default_split_paths = [
        workspace_root / "peer-review-benchmark" / "data" / "splits" / "val.jsonl",
        workspace_root / "peer-review-benchmark" / "data" / "splits" / "v2" / "val.jsonl",
        workspace_root / "peer-review-benchmark" / "data" / "splits" / "v3" / "val.jsonl",
    ]
    parser = argparse.ArgumentParser(
        description="Compare tool-output article IDs against one or more split files."
    )
    parser.add_argument(
        "tool_outputs",
        nargs="+",
        type=Path,
        help="Tool output JSONL files to audit.",
    )
    parser.add_argument(
        "--split-path",
        dest="split_paths",
        action="append",
        type=Path,
        default=list(default_split_paths),
        help="Candidate split JSONL path. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix for .json and .md files.",
    )
    return parser


def load_ids(path: Path, field_names: tuple[str, ...]) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            for field in field_names:
                value = row.get(field)
                if value:
                    ids.add(str(value))
                    break
    return ids


def compare_sets(output_ids: set[str], split_ids: set[str]) -> dict:
    inter = output_ids & split_ids
    union = output_ids | split_ids
    return {
        "rows_output": len(output_ids),
        "rows_split": len(split_ids),
        "intersection": len(inter),
        "output_not_in_split": len(output_ids - split_ids),
        "split_not_in_output": len(split_ids - output_ids),
        "output_coverage": (len(inter) / len(output_ids)) if output_ids else 0.0,
        "split_coverage": (len(inter) / len(split_ids)) if split_ids else 0.0,
        "jaccard": (len(inter) / len(union)) if union else 0.0,
        "exact_match": output_ids == split_ids,
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Output/Split Alignment Audit",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        "",
    ]

    for tool_name, rows in payload["comparisons"].items():
        best = max(rows, key=lambda row: (row["intersection"], row["split_coverage"]))
        lines.extend(
            [
                f"## {tool_name}",
                "",
                f"- Best candidate split: `{best['split_path']}`",
                f"- Exact match: `{'yes' if best['exact_match'] else 'no'}`",
                "",
                "| Split | Output rows | Split rows | Intersection | Output cov. | Split cov. | Output-only | Split-only | Jaccard | Exact |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {Path(row['split_path']).parent.name}/{Path(row['split_path']).name} | "
                f"{row['rows_output']} | {row['rows_split']} | {row['intersection']} | "
                f"{row['output_coverage']:.4f} | {row['split_coverage']:.4f} | "
                f"{row['output_not_in_split']} | {row['split_not_in_output']} | "
                f"{row['jaccard']:.4f} | {'yes' if row['exact_match'] else 'no'} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()

    comparisons: dict[str, list[dict]] = {}
    for tool_output in args.tool_outputs:
        output_ids = load_ids(tool_output, ("article_id", "id"))
        rows: list[dict] = []
        for split_path in args.split_paths:
            split_ids = load_ids(split_path, ("id", "article_id"))
            row = compare_sets(output_ids, split_ids)
            row["split_path"] = str(split_path.resolve())
            rows.append(row)
        comparisons[tool_output.name] = rows

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "comparisons": comparisons,
    }

    if args.output_prefix is None:
        output_prefix = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "source_eval"
            / "output_split_alignment"
        )
    else:
        output_prefix = args.output_prefix

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
