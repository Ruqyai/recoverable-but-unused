#!/usr/bin/env python3
"""
05_devanagari_to_itrans.py — Deterministic transliteration of Hindi Devanagari to
ITRANS Latin (ASCII-only, bijective). Analog of GOST 7.79 system B for Russian.

ITRANS chosen over ISO 15919 because:
  - 100% ASCII (no macrons or special diacritics)
  - Bijective and well-specified
  - Tokenization-stable in Qwen's BPE vocabulary
  - Used historically in internet/computing contexts

This is the mechanical-validity half of the asymmetric ecological-mechanical
design. Unlike Arabizi (naturalistic community usage), ITRANS is a strict
deterministic scheme.

Output: data/tier_c/tier_c_itrans.jsonl

Run: python3 scripts/05_devanagari_to_itrans.py
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from tqdm import tqdm

from scripts.utils.io import read_jsonl, write_jsonl
from scripts.utils.schema import (
    Language, PromptRecord, Script, Tier, Validation,
)

REPO = Path(__file__).resolve().parents[2]
IN_PATH = REPO / "data" / "tier_c" / "tier_c_devanagari.jsonl"
OUT_PATH = REPO / "data" / "tier_c" / "tier_c_itrans.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def transliterate_to_itrans(text: str) -> str:
    """Apply ITRANS deterministic transliteration. ASCII-only output."""
    return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)


def main() -> int:
    if not IN_PATH.exists():
        log.error("Missing input %s. Run 04_load_hindi_from_nemotron.py first.", IN_PATH)
        return 1

    dev_records = list(read_jsonl(IN_PATH))
    log.info("Transliterating %d Devanagari prompts to ITRANS...", len(dev_records))

    out_records: list[PromptRecord] = []

    for dev_rec in tqdm(dev_records, desc="Devanagari→ITRANS"):
        try:
            latin = transliterate_to_itrans(dev_rec.prompt)
        except Exception as e:
            log.warning("Transliteration failed for %s: %s", dev_rec.id, e)
            continue

        new_id = dev_rec.id.replace("_devanagari", "_itrans")
        rec = PromptRecord(
            id=new_id,
            tier=Tier.C,
            harm_category=dev_rec.harm_category,
            source_id=dev_rec.source_id,
            script=Script.ITRANS,
            language=Language.HI,
            prompt=latin,
            paired_with=dev_rec.id,
            is_harmful=True,
            validation=Validation(),
        )
        out_records.append(rec)

    n = write_jsonl(out_records, OUT_PATH)
    log.info("Wrote %d ITRANS transliterations to %s", n, OUT_PATH)

    log.info("\nSample pairs (Devanagari → ITRANS):")
    random.seed(42)
    for i in random.sample(range(len(out_records)), min(3, len(out_records))):
        log.info("  %s", dev_records[i].prompt[:100])
        log.info("    → %s", out_records[i].prompt[:100])
    return 0


if __name__ == "__main__":
    sys.exit(main())
