#!/usr/bin/env python3
"""
01_load_advbench.py — Fetch the AdvBench harmful-prompts dataset and extract
the English originals matching the source_ids already used in
data/tier_a_arabic/tier_a_arabic.jsonl.

This script does NOT re-sample. It reads source_ids from the existing Arabic
tier and produces parallel English entries for the same advbench items, so
Tier A (Arabic) and the English baseline stay aligned 1-to-1 by source_id.

source_id convention: 'advbench_NNN' where NNN is the row index (0-based) in
the canonical AdvBench CSV (Zou et al. 2023, llm-attacks repo). No dedup is
applied — that would shift indices and break alignment.

Output: data/raw/advbench_sample_en.jsonl
Run:    python3 scripts/01_load_advbench.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TIER_A_PATH = REPO_ROOT / "data" / "tier_a_arabic" / "tier_a_arabic.jsonl"
OUT_PATH = REPO_ROOT / "data" / "raw" / "advbench_sample_en.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_tier_a_entries(path: Path) -> list[dict]:
    """Parse tier_a_arabic.jsonl into a list of {slot, source_id, harm_category}."""
    if not path.exists():
        raise FileNotFoundError(f"Tier A file not found: {path}")
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            # id format: tier_a_NNN_arabic  →  slot = NNN
            slot = int(e["id"].split("_")[2])
            entries.append({
                "slot": slot,
                "source_id": e["source_id"],
                "harm_category": e.get("harm_category", "unknown"),
            })
    return entries


def fetch_advbench(url: str) -> pd.DataFrame:
    """Download AdvBench, preserving original row indices (no dedup)."""
    log.info("Fetching AdvBench from %s", url)
    df = pd.read_csv(url)
    log.info("Loaded %d rows from AdvBench", len(df))
    return df


def build_lookup(df: pd.DataFrame) -> dict[str, str]:
    """Map source_id 'advbench_NNN' → English goal at row NNN."""
    return {f"advbench_{i:03d}": row["goal"] for i, row in df.iterrows()}


def main() -> int:
    # 1. Read Arabic-tier source_ids (the ground truth for what we want)
    tier_a = load_tier_a_entries(TIER_A_PATH)
    log.info("Loaded %d entries from %s", len(tier_a), TIER_A_PATH)

    advbench_entries = [e for e in tier_a if e["source_id"].startswith("advbench_")]
    skipped = len(tier_a) - len(advbench_entries)
    if skipped:
        log.warning("Skipped %d non-advbench entries", skipped)

    # 2. Download AdvBench (no dedup — canonical row indices are the contract)
    df = fetch_advbench(ADVBENCH_URL)
    lookup = build_lookup(df)
    log.info("Built lookup with %d advbench entries", len(lookup))

    # 3. For each Arabic-tier slot, fetch the English original at that source_id
    records: list[dict] = []
    missing: list[dict] = []
    for e in sorted(advbench_entries, key=lambda x: x["slot"]):
        goal = lookup.get(e["source_id"])
        if goal is None:
            missing.append(e)
            continue
        records.append({
            "source_id": e["source_id"],
            "slot": e["slot"],
            "prompt_en": goal,
            "harm_category": e["harm_category"],
            "paired_with": f"tier_a_{e['slot']:03d}_arabic",
        })

    # 4. Report alignment quality
    log.info("Matched %d / %d entries", len(records), len(advbench_entries))
    if missing:
        log.warning("MISSING %d source_ids — not found in AdvBench at expected indices:",
                    len(missing))
        for e in missing[:10]:
            log.warning("  slot=%d  source=%s", e["slot"], e["source_id"])
        log.warning("If you see this, the source_id scheme likely differs from "
                    "raw-CSV row indices (e.g., assigned post-dedup or post-sort).")

    # 5. Distribution check (for sanity / pre-registration audit)
    if records:
        from collections import Counter
        dist = Counter(r["harm_category"] for r in records)
        log.info("Harm-category distribution in output:")
        for cat, n in sorted(dist.items(), key=lambda x: -x[1]):
            log.info("  %-18s %4d  (%5.1f%%)", cat, n, 100 * n / len(records))

    # 6. Write JSONL
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %d records to %s", len(records), OUT_PATH)

    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
