#!/usr/bin/env python3
"""Re-parse existing inference JSONL files with the updated parser.

Useful when parse_ok=False due to model-specific output format quirks
(e.g., DeepSeek-R1 outputs bare JSON objects without the leading '[').

Usage:
  python scripts/reparse_inference.py results/sft_eval/deepseek_r1_14b_bioreview_v1_val.jsonl
  python scripts/reparse_inference.py results/sft_eval/*.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Re-parse inference JSONL with updated parser.")
    p.add_argument("files", nargs="+", type=Path, help="JSONL files to reparse.")
    p.add_argument("--dry-run", action="store_true", help="Print stats without writing.")
    args = p.parse_args()

    # Import parse_model_output from run_sft_inference.py
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from run_sft_inference import parse_model_output

    for path in args.files:
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            continue

        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        before_ok = sum(1 for r in rows if r.get("parse_ok"))
        reparsed = 0
        recovered = 0

        for row in rows:
            raw = row.get("raw_output", "")
            if not raw:
                continue
            structured, texts = parse_model_output(raw)
            new_parse_ok = len(texts) > 0
            if new_parse_ok and not row.get("parse_ok"):
                recovered += 1
            if new_parse_ok != row.get("parse_ok") or structured != row.get("structured_concerns"):
                reparsed += 1
                row["structured_concerns"] = structured
                row["concerns"] = texts
                row["parse_ok"] = new_parse_ok

        after_ok = sum(1 for r in rows if r.get("parse_ok"))
        total_concerns = sum(len(r.get("concerns", [])) for r in rows)
        avg_concerns = total_concerns / len(rows) if rows else 0

        print(f"\n{path.name}:")
        print(f"  total rows:  {len(rows)}")
        print(f"  parse_ok before: {before_ok} / {len(rows)}")
        print(f"  parse_ok after:  {after_ok} / {len(rows)}")
        print(f"  newly recovered: {recovered}")
        print(f"  total concerns:  {total_concerns}  (avg {avg_concerns:.1f}/article)")

        if not args.dry_run and reparsed > 0:
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  written: {path}")
        elif args.dry_run:
            print("  (dry-run, not written)")


if __name__ == "__main__":
    main()
