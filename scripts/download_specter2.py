#!/usr/bin/env python3
"""Download SPECTER2 to a local project path for stable embedding evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve()
    default_dir = here.parents[1] / "models" / "specter2_base"
    p = argparse.ArgumentParser(description="Download allenai/specter2_base locally.")
    p.add_argument(
        "--repo-id",
        type=str,
        default="allenai/specter2_base",
        help="Hugging Face repo id.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=default_dir,
        help="Local directory where model snapshot is stored.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(args.output_dir),
    )
    print(f"downloaded_to={path}")
    print(f"set BIOREVIEW_EMBED_MODEL={args.output_dir}")
    print("optional_offline=export BIOREVIEW_EMBED_LOCAL_ONLY=1")


if __name__ == "__main__":
    main()
