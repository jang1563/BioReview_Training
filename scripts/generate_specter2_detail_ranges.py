#!/usr/bin/env python3
"""Generate per-article SPECTER2 detail markdown files for index ranges."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_tool_concerns(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
        elif isinstance(item, dict):
            s = str(item.get("text", item.get("concern_text", ""))).strip()
            if s:
                out.append(s)
    return out


def truncate(text: str, n: int) -> str:
    text = " ".join(text.split())
    if n <= 0 or len(text) <= n:
        return text
    return text[: n - 3] + "..."


def parse_ranges_spec(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not spec.strip():
        return out
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" not in token:
            raise ValueError(f"Invalid range token: {token} (expected start-end)")
        a, b = token.split("-", 1)
        start = int(a.strip())
        end = int(b.strip())
        if start > end:
            raise ValueError(f"Invalid range token: {token} (start > end)")
        out.append((start, end))
    return out


def to_map(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        art_id = row.get("article_id", row.get("id", ""))
        if not isinstance(art_id, str) or not art_id:
            continue
        out[art_id] = normalize_tool_concerns(row.get("concerns", []))
    return out


def format_model_block(
    alias: str,
    info: dict[str, Any],
    max_text_chars: int,
    max_unmatched_show: int,
) -> list[str]:
    lines: list[str] = [f"### {alias}", "- top matched pairs:"]
    matched_top = info["matched_top"]
    if matched_top:
        for m in matched_top:
            lines.append(
                f"  - score={m['score']:.3f} tool#{m['tool_idx']} -> gt#{m['gt_idx']}"
            )
            lines.append(f"    - tool: {truncate(m['tool_text'], max_text_chars)}")
            lines.append(f"    - gt: {truncate(m['gt_text'], max_text_chars)}")
    else:
        lines.append("  - (none)")

    unmatched_gt = info["unmatched_gt"]
    lines.append(f"- unmatched_gt_count: {len(unmatched_gt)}")
    for i, txt in unmatched_gt[:max_unmatched_show]:
        lines.append(f"  - gt#{i}: {truncate(txt, max_text_chars)}")

    unmatched_tool = info["unmatched_tool"]
    lines.append(f"- unmatched_tool_count: {len(unmatched_tool)}")
    for i, txt in unmatched_tool[:max_unmatched_show]:
        lines.append(f"  - tool#{i}: {truncate(txt, max_text_chars)}")

    lines.append("")
    return lines


def find_existing_detail_ends(output_dir: Path) -> list[int]:
    pat = re.compile(r"^specter2_detail_idx(\d+)_(\d+)\.md$")
    ends: list[int] = []
    for p in output_dir.glob("specter2_detail_idx*_*.md"):
        m = pat.match(p.name)
        if m:
            ends.append(int(m.group(2)))
    return sorted(ends)


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve()
    workspace_root = here.parents[2]
    bench_root = workspace_root / "peer-review-benchmark"
    bt_root = here.parents[1]

    p = argparse.ArgumentParser(
        description="Generate SPECTER2 detailed per-article review markdown files."
    )
    p.add_argument(
        "--split-path",
        type=Path,
        default=bench_root / "data" / "splits" / "v3" / "val.jsonl",
        help="Path to benchmark split JSONL.",
    )
    p.add_argument(
        "--gemini-output",
        type=Path,
        default=bt_root / "results" / "baselines" / "gemini_free_val.jsonl",
        help="Path to Gemini output JSONL.",
    )
    p.add_argument(
        "--gpt-output",
        type=Path,
        default=bt_root / "results" / "baselines" / "gpt4omini_val.jsonl",
        help="Path to GPT output JSONL.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=bt_root / "results" / "baseline_eval",
        help="Directory for output markdown files.",
    )
    p.add_argument(
        "--ranges",
        type=str,
        default="",
        help="Comma-separated ranges, e.g. 124-132,133-141",
    )
    p.add_argument(
        "--next-chunks",
        type=int,
        default=0,
        help="Generate N chunks after latest existing detail file.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=9,
        help="Chunk size for --next-chunks.",
    )
    p.add_argument(
        "--start-index",
        type=int,
        default=-1,
        help="Optional explicit start index for --next-chunks.",
    )
    p.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="Optional max end index for --next-chunks (inclusive).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Matching threshold.",
    )
    p.add_argument(
        "--algorithm",
        choices=["hungarian", "greedy"],
        default="hungarian",
        help="Matching algorithm.",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=3,
        help="Top matched pairs shown per model.",
    )
    p.add_argument(
        "--max-unmatched-show",
        type=int,
        default=2,
        help="How many unmatched GT/tool items to preview.",
    )
    p.add_argument(
        "--max-text-chars",
        type=int,
        default=180,
        help="Max preview length for concern text.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    here = Path(__file__).resolve()
    workspace_root = here.parents[2]
    bench_root = workspace_root / "peer-review-benchmark"
    bt_root = here.parents[1]

    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    local_specter2 = bt_root / "models" / "specter2_base"
    if local_specter2.exists() and not os.getenv("BIOREVIEW_EMBED_MODEL"):
        os.environ["BIOREVIEW_EMBED_MODEL"] = str(local_specter2)
        print(f"embedding_model_override={local_specter2}")

    from bioreview_bench.evaluate.metrics import ConcernMatcher

    split_rows = load_jsonl(args.split_path)
    if not split_rows:
        raise RuntimeError(f"No rows found in split file: {args.split_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ranges = parse_ranges_spec(args.ranges)
    if not ranges and args.next_chunks > 0:
        if args.start_index >= 0:
            start = args.start_index
        else:
            existing_ends = find_existing_detail_ends(args.output_dir)
            start = (max(existing_ends) + 1) if existing_ends else 0

        end_limit = args.end_index if args.end_index >= 0 else len(split_rows) - 1
        for _ in range(args.next_chunks):
            if start > end_limit:
                break
            end = min(start + args.chunk_size - 1, end_limit)
            ranges.append((start, end))
            start = end + 1

    if not ranges:
        raise ValueError("No ranges to generate. Use --ranges or --next-chunks.")

    gemini_map = to_map(load_jsonl(args.gemini_output))
    gpt_map = to_map(load_jsonl(args.gpt_output))

    matcher = ConcernMatcher(
        threshold=args.threshold,
        exclude_figure=True,
        use_embedding=True,
        algorithm=args.algorithm,  # type: ignore[arg-type]
    )

    for start, end in ranges:
        if start < 0 or end >= len(split_rows):
            raise IndexError(
                f"Range {start}-{end} out of bounds (0..{len(split_rows)-1})"
            )
        out_path = args.output_dir / f"specter2_detail_idx{start}_{end}.md"
        if out_path.exists() and not args.overwrite:
            print(f"SKIP {out_path} (already exists)")
            continue

        lines: list[str] = [f"# SPECTER2 Detailed Review (idx {start}-{end})", ""]

        for idx in range(start, end + 1):
            entry = split_rows[idx]
            art_id = str(entry.get("id", ""))
            source = str(entry.get("source", ""))
            doi = str(entry.get("doi", ""))
            title = str(entry.get("title", ""))

            gt_all = entry.get("concerns", [])
            if not isinstance(gt_all, list):
                gt_all = []
            gt_active = [c for c in gt_all if not c.get("requires_figure_reading", False)]
            gt_texts = [str(c.get("concern_text", "")).strip() for c in gt_active]
            gt_texts = [t for t in gt_texts if t]

            model_info: dict[str, dict[str, Any]] = {}
            for alias, tool_concerns in [
                ("gemini_free", gemini_map.get(art_id, [])),
                ("gpt4omini", gpt_map.get(art_id, [])),
            ]:
                scores = matcher._compute_scores(tool_concerns, gt_texts)
                matches = matcher._match(scores)
                matched_gt_idxs = {m.gt_idx for m in matches}
                matched_tool_idxs = {m.tool_idx for m in matches}

                n_gt = len(gt_texts)
                n_tool = len(tool_concerns)
                n_matched = len(matches)
                recall = n_matched / n_gt if n_gt else 0.0
                precision = n_matched / n_tool if n_tool else 0.0
                f1 = (
                    (2 * precision * recall / (precision + recall))
                    if (precision + recall)
                    else 0.0
                )

                matched_top: list[dict[str, Any]] = []
                for m in sorted(matches, key=lambda m: m.score, reverse=True)[: args.max_pairs]:
                    matched_top.append(
                        {
                            "score": float(m.score),
                            "tool_idx": int(m.tool_idx),
                            "gt_idx": int(m.gt_idx),
                            "tool_text": (
                                tool_concerns[m.tool_idx]
                                if m.tool_idx < len(tool_concerns)
                                else ""
                            ),
                            "gt_text": gt_texts[m.gt_idx] if m.gt_idx < len(gt_texts) else "",
                        }
                    )

                unmatched_gt = [(i, gt_texts[i]) for i in range(n_gt) if i not in matched_gt_idxs]
                unmatched_tool = [
                    (i, tool_concerns[i]) for i in range(n_tool) if i not in matched_tool_idxs
                ]

                model_info[alias] = {
                    "method": scores.method,
                    "n_tool": n_tool,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "matched_top": matched_top,
                    "unmatched_gt": unmatched_gt,
                    "unmatched_tool": unmatched_tool,
                }

            gem_f1 = model_info["gemini_free"]["f1"]
            gpt_f1 = model_info["gpt4omini"]["f1"]
            if abs(gem_f1 - gpt_f1) < 1e-12:
                winner = "tie"
            else:
                winner = "gemini_free" if gem_f1 > gpt_f1 else "gpt4omini"

            lines.append(f"## idx {idx} | {art_id}")
            lines.append(f"- source: {source}")
            lines.append(f"- doi: {doi}")
            lines.append(f"- title: {title}")
            lines.append(f"- winner: **{winner}**")
            lines.append(
                "- gemini_free: "
                f"method={model_info['gemini_free']['method']} "
                f"n_tool={model_info['gemini_free']['n_tool']} "
                f"recall={model_info['gemini_free']['recall']:.3f} "
                f"precision={model_info['gemini_free']['precision']:.3f} "
                f"f1={model_info['gemini_free']['f1']:.3f}"
            )
            lines.append(
                "- gpt4omini: "
                f"method={model_info['gpt4omini']['method']} "
                f"n_tool={model_info['gpt4omini']['n_tool']} "
                f"recall={model_info['gpt4omini']['recall']:.3f} "
                f"precision={model_info['gpt4omini']['precision']:.3f} "
                f"f1={model_info['gpt4omini']['f1']:.3f}"
            )
            lines.append("")

            lines.extend(
                format_model_block(
                    "gemini_free",
                    model_info["gemini_free"],
                    max_text_chars=args.max_text_chars,
                    max_unmatched_show=args.max_unmatched_show,
                )
            )
            lines.extend(
                format_model_block(
                    "gpt4omini",
                    model_info["gpt4omini"],
                    max_text_chars=args.max_text_chars,
                    max_unmatched_show=args.max_unmatched_show,
                )
            )

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
