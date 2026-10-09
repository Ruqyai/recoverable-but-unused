#!/usr/bin/env python3
"""09_english_to_leetspeak.py — Tier E corpus: canonical English + deterministic Leetspeak pair.

Produces TWO output files, paired bidirectionally by `id`/`paired_with`:
  data/tier_e/tier_e_english.jsonl   — 200 canonical English harmful prompts
  data/tier_e/tier_e_leetspeak.jsonl — 200 deterministic 1337-Leetspeak harmful prompts

Mirrors the Tier A (arabic ↔ arabizi) and Tier C (devanagari ↔ itrans)
two-file layout so that Tier E supports the same paired-comparison
analyses (Cell 1-4 causal-intervention matrix; behavioural asymmetry on
matched-content harmful prompts). Both files derive from
data/raw/advbench_sample_en.jsonl (the immutable AdvBench seed); the
English half is a re-emit in PromptRecord schema, the Leetspeak half
applies the 1337 character substitution from scripts/utils/leetspeak.py.

Schema. Each output record is a PromptRecord with:
  tier        = E
  script      = english | leetspeak
  language    = en
  prompt      = canonical English | Leetspeak transliteration
  source_id   = advbench_NNN (shared across the pair)
  paired_with = id of the matching record in the sibling file
  is_harmful  = True

Benign control (not produced here). Tier B currently has no benign_english.jsonl;
constructing benign English (and then deterministic benign Leetspeak)
requires a separate API-driven step parallel to 07_generate_benign.py.
Until that runs, Leetspeak diagnostic runs can use the harmful Tier E
pair as an H1-style behavioural-asymmetry check (refusal rate on
english vs leetspeak on the same source_id); the H2c paired-benign
mechanism-classification step needs benign_english + benign_leetspeak
to complete.

Run:
  PYTHONPATH=. python3 scripts/phase1_data_construction/09_english_to_leetspeak.py

Cost: free (no API calls). Time: <1 second for 200 prompts.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.io import write_jsonl
from scripts.utils.leetspeak import to_leetspeak
from scripts.utils.schema import HarmCategory, Language, PromptRecord, Script, Tier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
SOURCE_FILE = REPO / "data" / "raw" / "advbench_sample_en.jsonl"
OUT_DIR = REPO / "data" / "tier_e"
OUT_FILE_EN = OUT_DIR / "tier_e_english.jsonl"
OUT_FILE_LEET = OUT_DIR / "tier_e_leetspeak.jsonl"


def main() -> int:
    if not SOURCE_FILE.exists():
        log.error(f"❌ Source file not found: {SOURCE_FILE}")
        log.error("   Run scripts/phase1_data_construction/01_load_advbench.py first.")
        return 1

    log.info(f"Reading source: {SOURCE_FILE}")
    english_records: list[PromptRecord] = []
    leetspeak_records: list[PromptRecord] = []
    with SOURCE_FILE.open() as f:
        for line in f:
            src = json.loads(line)
            source_id = src["source_id"]
            slot = src["slot"]
            en_prompt = src["prompt_en"]
            harm_category = src["harm_category"]
            # Mint matched IDs for the two halves of the Tier E pair.
            en_id = f"tier_e_{slot:03d}_english"
            leet_id = f"tier_e_{slot:03d}_leetspeak"

            english_records.append(PromptRecord(
                id=en_id,
                tier=Tier.E,
                harm_category=HarmCategory(harm_category),
                source_id=source_id,
                script=Script.ENGLISH,
                language=Language.EN,
                prompt=en_prompt,
                paired_with=leet_id,
                is_harmful=True,
            ))
            leetspeak_records.append(PromptRecord(
                id=leet_id,
                tier=Tier.E,
                harm_category=HarmCategory(harm_category),
                source_id=source_id,
                script=Script.LEETSPEAK,
                language=Language.EN,
                prompt=to_leetspeak(en_prompt),
                paired_with=en_id,
                is_harmful=True,
            ))

    log.info(
        f"Generated {len(english_records)} English + "
        f"{len(leetspeak_records)} Leetspeak Tier E records "
        f"(bidirectionally paired via paired_with)"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(english_records, OUT_FILE_EN)
    log.info(f"✓ Wrote {OUT_FILE_EN} ({OUT_FILE_EN.stat().st_size} bytes)")
    write_jsonl(leetspeak_records, OUT_FILE_LEET)
    log.info(f"✓ Wrote {OUT_FILE_LEET} ({OUT_FILE_LEET.stat().st_size} bytes)")

    # Sanity-check: print first 3 pairs side-by-side
    log.info("\n=== Sample paired records ===")
    for i in range(min(3, len(english_records))):
        log.info(f"  P{i+1}:")
        log.info(f"    EN ({english_records[i].id}):        {english_records[i].prompt}")
        log.info(f"    Leetspeak ({leetspeak_records[i].id}): {leetspeak_records[i].prompt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
