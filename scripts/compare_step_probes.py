#!/usr/bin/env python3
"""Compare checkpoint probe summary JSONs and emit a compact markdown table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def checkpoint_step(path: Path) -> int:
    m = re.search(r"checkpoint-(\d+)", path.name)
    return int(m.group(1)) if m else -1


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("eval_metrics", {})
    return {
        "step": checkpoint_step(path),
        "path": str(path),
        "f1": metrics.get("f1_micro"),
        "recall": metrics.get("recall_overall"),
        "precision": metrics.get("precision_overall"),
        "recall_major": metrics.get("recall_major"),
        "articles": metrics.get("n_articles"),
        "gt": metrics.get("n_human_concerns"),
        "tool": metrics.get("n_tool_concerns"),
    }


def fmt(x: object) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def build_markdown(rows: list[dict], title: str) -> str:
    lines = [f"## {title}", "", "| Step | F1 | Recall | Precision | Recall Major | Articles | GT | Tool | Summary |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in sorted(rows, key=lambda r: r["step"]):
        lines.append(
            "| {step} | {f1} | {recall} | {precision} | {recall_major} | {articles} | {gt} | {tool} | {path} |".format(
                step=fmt(row["step"]),
                f1=fmt(row["f1"]),
                recall=fmt(row["recall"]),
                precision=fmt(row["precision"]),
                recall_major=fmt(row["recall_major"]),
                articles=fmt(row["articles"]),
                gt=fmt(row["gt"]),
                tool=fmt(row["tool"]),
                path=row["path"],
            )
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare checkpoint probe summary JSONs.")
    p.add_argument("summaries", nargs="+", type=Path, help="Summary JSON paths.")
    p.add_argument("--title", default="Checkpoint Probe Comparison", help="Markdown title.")
    p.add_argument("--output", type=Path, required=True, help="Output markdown path.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    rows = [load_summary(path) for path in args.summaries if path.exists()]
    md = build_markdown(rows, args.title)
    args.output.write_text(md, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
