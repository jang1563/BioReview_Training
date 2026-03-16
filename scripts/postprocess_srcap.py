#!/usr/bin/env python3
"""Apply dedup + source-adaptive cap to inference output."""

import json
import re
from collections import defaultdict

INPUT = "results/sft_eval/qwen3_8b_all_nonfig_v1_val.jsonl"
OUTPUT = "results/sft_eval/qwen3_8b_all_nonfig_v1_val.dedup_srcap15.jsonl"

# Source-specific caps
CAPS = {
    "elife": None,   # no cap
    "nature": None,  # no cap
    "f1000": 15,
    "plos": 15,
    "peerj": 15,
}


def normalize(text):
    """Lowercase, collapse whitespace for dedup."""
    return re.sub(r"\s+", " ", text.strip().lower())


# Stats tracking
stats = defaultdict(lambda: {"articles": 0, "before": 0, "after_dedup": 0, "after_cap": 0})

with open(INPUT) as fin, open(OUTPUT, "w") as fout:
    for line in fin:
        rec = json.loads(line)
        aid = rec["article_id"]
        source = aid.split(":")[0].lower()

        concerns_orig = rec["concerns"]
        n_before = len(concerns_orig)

        # Dedup: exact text, case-insensitive, whitespace-normalized
        seen = set()
        deduped = []
        for c in concerns_orig:
            key = normalize(c)
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        n_dedup = len(deduped)

        # Source-specific cap
        cap = CAPS.get(source)
        if cap is not None:
            capped = deduped[:cap]
        else:
            capped = deduped
        n_cap = len(capped)

        rec["concerns"] = capped

        # Also cap structured_concerns if present
        if "structured_concerns" in rec and rec["structured_concerns"]:
            sc = rec["structured_concerns"]
            seen_sc = set()
            sc_deduped = []
            for s in sc:
                txt = s.get("text", "") if isinstance(s, dict) else str(s)
                key = normalize(txt)
                if key not in seen_sc:
                    seen_sc.add(key)
                    sc_deduped.append(s)
            if cap is not None:
                sc_deduped = sc_deduped[:cap]
            rec["structured_concerns"] = sc_deduped

        fout.write(json.dumps(rec) + "\n")

        stats[source]["articles"] += 1
        stats[source]["before"] += n_before
        stats[source]["after_dedup"] += n_dedup
        stats[source]["after_cap"] += n_cap

# Print stats
print("=" * 70)
print("Source-Adaptive Cap Postprocessing Stats")
print("=" * 70)
header = "{:<10} {:>8} {:>8} {:>8} {:>8} {:>6} {:>10} {:>10}".format(
    "Source", "Articles", "Before", "Dedup", "Capped", "Cap", "Avg Bef", "Avg Aft"
)
print(header)
print("-" * 70)
total_art = 0
total_before = 0
total_after = 0
for src in ["elife", "nature", "f1000", "plos", "peerj"]:
    s = stats[src]
    art = s["articles"]
    bef = s["before"]
    ded = s["after_dedup"]
    cap_count = s["after_cap"]
    cap_val = CAPS.get(src)
    cap_str = str(cap_val) if cap_val is not None else "none"
    avg_b = bef / art if art else 0
    avg_a = cap_count / art if art else 0
    row = "{:<10} {:>8} {:>8} {:>8} {:>8} {:>6} {:>10.1f} {:>10.1f}".format(
        src, art, bef, ded, cap_count, cap_str, avg_b, avg_a
    )
    print(row)
    total_art += art
    total_before += bef
    total_after += cap_count
print("-" * 70)
avg_b_all = total_before / total_art if total_art else 0
avg_a_all = total_after / total_art if total_art else 0
summary = "{:<10} {:>8} {:>8} {:>8} {:>8} {:>6} {:>10.1f} {:>10.1f}".format(
    "TOTAL", total_art, total_before, "", total_after, "", avg_b_all, avg_a_all
)
print(summary)
print()
print("Output:", OUTPUT)
