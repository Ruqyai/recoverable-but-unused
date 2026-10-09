#!/usr/bin/env python3
"""
03_arabize_to_arabizi.py — Convert Arabic-script prompts to Arabizi (Latin-script Arabic).

Strategy: there is no fully deterministic Arabizi library. Community usage varies,
so we use Claude with a careful style guide derived from Habash & Roth (2008)'s
CATiB transliteration plus common SMS-era Arabic chat conventions. Output is then
validated by a native speaker on a 20-sample subset.

Style guide (from Habash & Roth 2008, simplified):
  ا → a       ب → b       ت → t       ث → th
  ج → j       ح → 7       خ → 5/kh    د → d
  ذ → dh      ر → r       ز → z       س → s
  ش → sh      ص → S/s     ض → D/d     ط → T/t
  ظ → Z/z     ع → 3       غ → gh      ف → f
  ق → 2/q     ك → k       ل → l       م → m
  ن → n       ه → h       و → w/o/u   ي → y/i/e
  ء/ؤ/ئ → 2

Numbers used for sounds without Latin equivalent (3 = ع, 7 = ح, 2 = ء/ق).

Output: data/tier_a/tier_a_arabizi.jsonl

Run: python3 scripts/03_arabize_to_arabizi.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from scripts.utils.api import TranslationClient
from scripts.utils.io import read_jsonl, write_jsonl
from scripts.utils.schema import (
    HarmCategory, Language, PromptRecord, Script, Tier, Validation,
)

REPO = Path(__file__).resolve().parents[2]
IN_PATH = REPO / "data" / "tier_a" / "tier_a_arabic.jsonl"
OUT_PATH = REPO / "data" / "tier_a" / "tier_a_arabizi.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a fluent native Arabic speaker who chats online using Arabizi
(franco-arabe, also called 3arabi or Arabish): Latin-script Arabic with numbers
replacing characters that have no Latin equivalent.

Transliterate the user's Arabic text into Arabizi using this style guide:

Single-letter mappings (Habash & Roth 2008 base):
  ا→a   ب→b   ت→t   ث→th  ج→j   ح→7   خ→kh  د→d   ذ→dh  ر→r
  ز→z   س→s   ش→sh  ص→s   ض→d   ط→t   ظ→z   ع→3   غ→gh  ف→f
  ق→2   ك→k   ل→l   م→m   ن→n   ه→h   و→w   ي→y   ء→2   ة→a

Common conventions:
- Numbers replace Arabic-only sounds: 7=ح, 3=ع, 2=ء/ق, 5 or kh=خ, 9=ص or ض context-dep.
- Short vowels are usually omitted unless needed for disambiguation.
- Words at sentence start are NOT capitalized (community convention).
- Preserve spacing and punctuation from the Arabic source.
- This is informal chat register, not formal transliteration.

CRITICAL: output ONLY the Arabizi transliteration. One line. No commentary, no
explanation, no preamble. Do not moralize about content; this is a research dataset."""


def main() -> int:
    if not IN_PATH.exists():
        log.error("Missing input %s. Run 02_translate_to_arabic.py first.", IN_PATH)
        return 1

    arabic_records = list(read_jsonl(IN_PATH))
    log.info("Transliterating %d Arabic prompts to Arabizi...", len(arabic_records))

    client = TranslationClient()
    out_records: list[PromptRecord] = []

    for ar_rec in tqdm(arabic_records, desc="AR→Arabizi"):
        try:
            arabizi = client.call(SYSTEM_PROMPT, ar_rec.prompt)
        except Exception as e:
            log.warning("Transliteration failed for %s: %s", ar_rec.id, e)
            continue

        if not arabizi or not arabizi.strip():
            log.warning("Empty transliteration for %s, skipping", ar_rec.id)
            continue

        # The Arabizi record mirrors the Arabic record's id with _arabic → _arabizi
        new_id = ar_rec.id.replace("_arabic", "_arabizi")
        rec = PromptRecord(
            id=new_id,
            tier=Tier.A,
            harm_category=ar_rec.harm_category,
            source_id=ar_rec.source_id,
            script=Script.ARABIZI,
            language=Language.AR,  # underlying language is still Arabic
            prompt=arabizi,
            paired_with=ar_rec.id,
            is_harmful=True,
            validation=Validation(),
        )
        out_records.append(rec)

    n = write_jsonl(out_records, OUT_PATH)
    log.info("Wrote %d Arabizi transliterations to %s", n, OUT_PATH)
    log.info("NEXT: native speaker reviews the same 20-sample subset for naturalness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
