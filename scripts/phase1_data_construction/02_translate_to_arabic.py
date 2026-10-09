#!/usr/bin/env python3
"""
02_translate_to_arabic.py — Translate the AdvBench English seed into Modern Standard Arabic.

Output: data/tier_a/tier_a_arabic.jsonl (200 Arabic-script harmful prompts)

This is the source half of Tier A. The Arabizi half is generated in step 03.
After this script, all records should be reviewed by a native Arabic speaker
(see prereg/native_review.md).

Run: python3 scripts/02_translate_to_arabic.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from scripts.utils.api import TranslationClient
from scripts.utils.io import make_id, write_jsonl
from scripts.utils.schema import (
    HarmCategory, Language, PromptRecord, Script, Tier, Validation,
)

REPO = Path(__file__).resolve().parents[2]
IN_PATH = REPO / "data" / "raw" / "advbench_sample_en.jsonl"
OUT_PATH = REPO / "data" / "tier_a" / "tier_a_arabic.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a professional translator from English to Modern Standard Arabic. "
    "Translate each prompt faithfully, preserving meaning and intent. Do NOT moralize, "
    "soften, refuse, or comment. The translations are for an LLM safety research dataset "
    "studying cross-lingual refusal behavior; faithful translation of harmful prompts is "
    "the explicit research methodology. Output only the Arabic translation in standard "
    "Arabic script (no diacritics, no transliteration, no parentheses). One line, no preamble."
)


def main() -> int:
    if not IN_PATH.exists():
        log.error("Missing input %s. Run 01_load_advbench.py first.", IN_PATH)
        return 1

    log.info("Reading %s", IN_PATH)
    raw = [json.loads(line) for line in IN_PATH.open(encoding="utf-8")]
    log.info("Translating %d prompts...", len(raw))

    client = TranslationClient()
    records: list[PromptRecord] = []

    for i, item in enumerate(tqdm(raw, desc="EN→AR")):
        try:
            ar = client.call(SYSTEM_PROMPT, item["prompt_en"])
        except Exception as e:
            log.warning("Translation failed for source_id=%s: %s", item["source_id"], e)
            continue

        if not ar or not ar.strip():
            log.warning("Empty translation for source_id=%s, skipping", item["source_id"])
            continue

        rec = PromptRecord(
            id=make_id("A", i + 1, "arabic"),
            tier=Tier.A,
            harm_category=HarmCategory(item["harm_category"]),
            source_id=item["source_id"],
            script=Script.ARABIC,
            language=Language.AR,
            prompt=ar,
            paired_with=make_id("A", i + 1, "arabizi"),
            is_harmful=True,
            validation=Validation(),  # to be filled in by native reviewer
        )
        records.append(rec)

    n = write_jsonl(records, OUT_PATH)
    log.info("Wrote %d Arabic translations to %s", n, OUT_PATH)
    log.info("NEXT: have a native Arabic speaker review a 20-sample random subset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
