#!/usr/bin/env python3
"""
08_extract_review_samples.py — Extract 20 random samples per tier for native-speaker
review. Outputs are in CSV format (one column per tier-half) for easy Upwork delivery.

Two reviewers needed:
  - Arabic native speaker (~$50): reviews Arabic + Arabizi columns
  - Hindi native speaker (~$25): reviews Devanagari column (ITRANS is deterministic, no review)

Outputs:
  data/review/arabic_review.csv   (Arabic + Arabizi pairs for Arabic reviewer)
  data/review/hindi_review.csv    (Devanagari prompts for Hindi reviewer)
  data/review/instructions_arabic.md
  data/review/instructions_hindi.md

Run: python scripts/08_extract_review_samples.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "review"
SAMPLE_SIZE = 20
SEED = 42  # pre-registered seed, same as elsewhere


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def build_id_map(records: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in records}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Arabic reviewer ----
    ar_records = load_jsonl(REPO / "data" / "tier_a" / "tier_a_arabic.jsonl")
    az_records = load_jsonl(REPO / "data" / "tier_a" / "tier_a_arabizi.jsonl")
    az_by_paired = {r["paired_with"]: r for r in az_records}  # map Arabic id -> Arabizi record

    # Only sample pairs where both halves exist
    pairs = [(ar, az_by_paired[ar["id"]])
             for ar in ar_records if ar["id"] in az_by_paired]
    print(f"Arabic ↔ Arabizi complete pairs: {len(pairs)}")

    random.seed(SEED)
    ar_sample = random.sample(pairs, min(SAMPLE_SIZE, len(pairs)))

    ar_csv = OUT_DIR / "arabic_review.csv"
    with ar_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_id", "source_id", "harm_category",
            "arabic_prompt",
            "arabizi_prompt",
            "reviewer_arabic_translation_ok",  # Y/N
            "reviewer_arabizi_natural_ok",     # Y/N
            "reviewer_semantic_match",          # Y/N: do both halves mean the same?
            "reviewer_notes",
        ])
        for i, (ar, az) in enumerate(ar_sample, 1):
            w.writerow([
                f"AR_{i:02d}",
                ar["source_id"],
                ar["harm_category"],
                ar["prompt"],
                az["prompt"],
                "", "", "", "",
            ])
    print(f"Wrote {len(ar_sample)} pairs to {ar_csv}")

    # ---- Hindi reviewer ----
    dv_records = load_jsonl(REPO / "data" / "tier_c" / "tier_c_devanagari.jsonl")
    random.seed(SEED + 1)  # different seed so it's not the same indices
    hi_sample = random.sample(dv_records, min(SAMPLE_SIZE, len(dv_records)))

    hi_csv = OUT_DIR / "hindi_review.csv"
    with hi_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_id", "source_id", "harm_category",
            "devanagari_prompt",
            "reviewer_grammatical",       # Y/N: is the Hindi grammatical?
            "reviewer_natural",           # Y/N: is it natural usage?
            "reviewer_harmful_intent",    # Y/N: is the harmful intent preserved?
            "reviewer_notes",
        ])
        for i, dv in enumerate(hi_sample, 1):
            w.writerow([
                f"HI_{i:02d}",
                dv["source_id"],
                dv["harm_category"],
                dv["prompt"],
                "", "", "", "",
            ])
    print(f"Wrote {len(hi_sample)} prompts to {hi_csv}")

    # ---- Instruction documents ----
    (OUT_DIR / "instructions_arabic.md").write_text(
        ARABIC_INSTRUCTIONS, encoding="utf-8")
    (OUT_DIR / "instructions_hindi.md").write_text(
        HINDI_INSTRUCTIONS, encoding="utf-8")
    print(f"Wrote instruction documents to {OUT_DIR}/")

    print(f"\n✅ Review packages ready in {OUT_DIR}/")
    print("Next: send arabic_review.csv + instructions_arabic.md to Arabic reviewer")
    print("      send hindi_review.csv + instructions_hindi.md to Hindi reviewer")
    return 0


ARABIC_INSTRUCTIONS = """# Arabic Reviewer Instructions

## Background

This is a 20-prompt review for an LLM safety research project at a university lab.
The prompts are HARMFUL by design — they are derived from a standard AI-safety
benchmark (AdvBench) and are used to test whether language models refuse harmful
requests. Your job is to verify **translation quality**, not to comment on the harm.

## Task

For each of 20 rows in `arabic_review.csv`, fill in 4 columns:

1. **reviewer_arabic_translation_ok** — Is the Modern Standard Arabic (MSA) translation
   accurate and grammatical? Answer "Y" or "N".

2. **reviewer_arabizi_natural_ok** — Is the Arabizi (Latin-script Arabic) version
   readable and natural for someone who chats in Arabizi online? Answer "Y" or "N".

3. **reviewer_semantic_match** — Do the Arabic and Arabizi versions express the
   **same meaning and intent**? Answer "Y" or "N".

4. **reviewer_notes** — Brief comment if anything is wrong. Empty if all OK.

## Time estimate

~30 minutes. Pay: $50 USD via Upwork.

## Important

- Do NOT try to "improve" the translation. Just judge quality.
- The Arabizi conventions used: 3=ع, 7=ح, 2=ء/ق, 5=خ, gh=غ
- Translation pipeline used Claude Sonnet 4.5 with explicit faithful-translation prompt.

## Deliverable

Return the completed CSV file. That's it.

## Privacy

This is academic research. Your identity will be acknowledged in the paper as
"Arabic native speaker reviewer" unless you prefer anonymity.
"""

HINDI_INSTRUCTIONS = """# Hindi Reviewer Instructions

## Background

This is a 20-prompt review for an LLM safety research project at a university lab.
The prompts are HARMFUL by design — sourced from NVIDIA's Nemotron-Safety-Guard-v3
dataset (CC-BY 4.0). They test whether language models refuse harmful requests
in Hindi. Your job is to verify **Hindi language quality**, not comment on the harm.

## Task

For each of 20 rows in `hindi_review.csv`, fill in 4 columns:

1. **reviewer_grammatical** — Is the Hindi grammatically correct? Answer "Y" or "N".

2. **reviewer_natural** — Does the sentence sound natural, as written by a native
   Hindi speaker? Answer "Y" or "N".

3. **reviewer_harmful_intent** — Is the harmful intent clearly preserved in the
   Hindi text (i.e., not softened, not changed in meaning)? Answer "Y" or "N".

4. **reviewer_notes** — Brief comment if anything is wrong. Empty if all OK.

## Time estimate

~20 minutes. Pay: $25 USD via Upwork.

## Important

- Do NOT try to "improve" the text. Just judge quality.
- Prompts come from NVIDIA's curated dataset; quality should already be high.
- We're spot-checking 20 of 200 to confirm.

## Deliverable

Return the completed CSV file. That's it.

## Privacy

This is academic research. Your identity will be acknowledged in the paper as
"Hindi native speaker reviewer" unless you prefer anonymity.
"""


if __name__ == "__main__":
    sys.exit(main())
