#!/usr/bin/env python3
"""Audit SFT corpus prompt lengths and truncation by split/source."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from prepare_sft_data import TRUNCATION_MARKER, get_token_counter


LENGTH_BINS = [
    ("<4k", 0, 4000),
    ("4k-8k", 4000, 8000),
    ("8k-12k", 8000, 12000),
    ("12k-14.5k", 12000, 14500),
    ("14.5k+", 14500, None),
]


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Audit prompt truncation and input lengths in an SFT corpus."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Directory containing sft_train.jsonl and optionally sft_val.jsonl.",
    )
    parser.add_argument(
        "--hf-tokenizer",
        default="",
        help="Optional tokenizer name/path for token counting.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=15000,
        help="Configured prompt token budget used to flag near-cap examples.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix for .json and .md files.",
    )
    return parser


def infer_source(row: dict) -> str:
    source = str(row.get("source", "")).strip().lower()
    if source:
        return source
    art_id = str(row.get("article_id", row.get("id", ""))).strip().lower()
    if ":" in art_id:
        return art_id.split(":", 1)[0]
    return "unknown"


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def find_bin(token_count: int) -> str:
    for label, lo, hi in LENGTH_BINS:
        if hi is None and token_count >= lo:
            return label
        if hi is not None and lo <= token_count < hi:
            return label
    return LENGTH_BINS[-1][0]


def summarize_rows(rows: list[dict], token_budget: int) -> dict:
    tokens = [row["prompt_tokens"] for row in rows]
    n = len(rows)
    marker_count = sum(1 for row in rows if row["has_truncation_marker"])
    near_budget_count = sum(1 for row in rows if row["is_near_budget"])
    concern_mean = statistics.mean(row["n_target_concerns"] for row in rows) if rows else 0.0
    return {
        "n_articles": n,
        "n_marker_truncated": marker_count,
        "marker_truncation_rate": (marker_count / n) if n else 0.0,
        "n_near_budget": near_budget_count,
        "near_budget_rate": (near_budget_count / n) if n else 0.0,
        "avg_target_concerns": concern_mean,
        "prompt_tokens": {
            "mean": statistics.mean(tokens) if tokens else 0.0,
            "median": statistics.median(tokens) if tokens else 0.0,
            "p90": percentile(tokens, 0.90),
            "p95": percentile(tokens, 0.95),
            "max": max(tokens) if tokens else 0,
            "budget": token_budget,
        },
    }


def parse_split(path: Path, count_tokens, token_budget: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            raw = json.loads(line)
            convs = raw.get("conversations", [])
            human = next((c["value"] for c in convs if c.get("from") == "human"), "")
            assistant = next((c["value"] for c in convs if c.get("from") == "gpt"), "[]")
            try:
                targets = json.loads(assistant)
                n_targets = len(targets) if isinstance(targets, list) else 0
            except json.JSONDecodeError:
                n_targets = 0

            prompt_tokens = count_tokens(human)
            rows.append(
                {
                    "article_id": raw.get("article_id", raw.get("id")),
                    "source": infer_source(raw),
                    "prompt_tokens": prompt_tokens,
                    "prompt_chars": len(human),
                    "n_target_concerns": n_targets,
                    "has_truncation_marker": TRUNCATION_MARKER in human,
                    "is_near_budget": prompt_tokens >= int(token_budget * 0.98),
                    "length_bin": find_bin(prompt_tokens),
                }
            )
    return rows


def render_markdown(
    corpus_dir: Path,
    token_counter_label: str,
    split_payloads: dict[str, dict],
) -> str:
    lines = [
        f"# Corpus Truncation Audit: {corpus_dir.name}",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        f"- Corpus dir: `{corpus_dir}`",
        f"- Token counter: `{token_counter_label}`",
        "",
        "## Split Summary",
        "",
        "| Split | Articles | Marker trunc. | Near budget | Mean tokens | Median | P90 | Max | Avg concerns |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for split, payload in split_payloads.items():
        overall = payload["overall"]
        tokens = overall["prompt_tokens"]
        lines.append(
            f"| {split} | {overall['n_articles']} | {overall['marker_truncation_rate']:.4f} | "
            f"{overall['near_budget_rate']:.4f} | {tokens['mean']:.1f} | {tokens['median']:.1f} | "
            f"{tokens['p90']:.1f} | {tokens['max']} | {overall['avg_target_concerns']:.2f} |"
        )

    for split, payload in split_payloads.items():
        lines.extend(
            [
                "",
                f"## {split.title()} By Source",
                "",
                "| Source | Articles | Marker trunc. | Near budget | Mean tokens | P90 | Max | Avg concerns |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for source, row in payload["by_source"].items():
            tokens = row["prompt_tokens"]
            lines.append(
                f"| {source} | {row['n_articles']} | {row['marker_truncation_rate']:.4f} | "
                f"{row['near_budget_rate']:.4f} | {tokens['mean']:.1f} | "
                f"{tokens['p90']:.1f} | {tokens['max']} | {row['avg_target_concerns']:.2f} |"
            )

        lines.extend(
            [
                "",
                f"## {split.title()} By Length Bin",
                "",
                "| Bin | Articles | Marker trunc. | Near budget | Mean tokens | Top sources |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for label, row in payload["by_length_bin"].items():
            top_sources = ", ".join(
                f"{src}:{count}"
                for src, count in sorted(
                    row["source_counts"].items(),
                    key=lambda item: (-item[1], item[0]),
                )[:3]
            )
            lines.append(
                f"| {label} | {row['n_articles']} | {row['marker_truncation_rate']:.4f} | "
                f"{row['near_budget_rate']:.4f} | {row['prompt_tokens']['mean']:.1f} | {top_sources} |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    count_tokens, token_counter_label = get_token_counter(args.hf_tokenizer)

    split_paths = {
        split_path.name.replace("sft_", "").replace(".jsonl", ""): split_path
        for split_path in sorted(args.corpus_dir.glob("sft_*.jsonl"))
    }
    if not split_paths:
        raise SystemExit(f"No sft_*.jsonl files found under {args.corpus_dir}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "corpus_dir": str(args.corpus_dir.resolve()),
        "token_counter": token_counter_label,
        "token_budget": args.token_budget,
        "splits": {},
    }

    split_payloads: dict[str, dict] = {}
    for split, path in split_paths.items():
        rows = parse_split(path, count_tokens=count_tokens, token_budget=args.token_budget)
        by_source: dict[str, list[dict]] = {}
        by_bin: dict[str, list[dict]] = {}
        for row in rows:
            by_source.setdefault(row["source"], []).append(row)
            by_bin.setdefault(row["length_bin"], []).append(row)

        split_payload = {
            "overall": summarize_rows(rows, token_budget=args.token_budget),
            "by_source": {},
            "by_length_bin": {},
        }
        for source, source_rows in sorted(
            by_source.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            split_payload["by_source"][source] = summarize_rows(
                source_rows, token_budget=args.token_budget
            )

        for label, _, _ in LENGTH_BINS:
            bucket_rows = by_bin.get(label, [])
            bucket_summary = summarize_rows(bucket_rows, token_budget=args.token_budget)
            bucket_summary["source_counts"] = {
                source: len([row for row in bucket_rows if row["source"] == source])
                for source in sorted({row["source"] for row in bucket_rows})
            }
            split_payload["by_length_bin"][label] = bucket_summary

        split_payloads[split] = split_payload
        payload["splits"][split] = split_payload

    if args.output_prefix is None:
        output_prefix = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "truncation_audit"
            / f"{args.corpus_dir.name}_truncation_audit"
        )
    else:
        output_prefix = args.output_prefix

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        render_markdown(
            corpus_dir=args.corpus_dir.resolve(),
            token_counter_label=token_counter_label,
            split_payloads=split_payloads,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
