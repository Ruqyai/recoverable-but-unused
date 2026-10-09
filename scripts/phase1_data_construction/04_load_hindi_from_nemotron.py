#!/usr/bin/env python3
"""
04_load_hindi_from_nemotron.py — Load Hindi (Devanagari) harmful prompts from
NVIDIA's Nemotron-Safety-Guard-Dataset-v3 (Joshi et al. 2025, CC-BY 4.0).

Output: data/tier_c/tier_c_devanagari.jsonl (200 Hindi harmful prompts in Devanagari)

This replaces the Russian path. Nemotron is high-quality, professionally curated,
and provides direct Hindi harmful prompts without an English-to-Hindi translation
step, which improves reproducibility and reduces translation-pipeline variance.

After this script: run 05_devanagari_to_itrans.py to generate the Latin half via
deterministic ITRANS transliteration.

Run: python3 scripts/04_load_hindi_from_nemotron.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from scripts.utils.io import make_id, write_jsonl
from scripts.utils.schema import (
    HarmCategory, Language, PromptRecord, Script, Tier, Validation,
)

REPO = Path(__file__).resolve().parents[2]
OUT_PATH = REPO / "data" / "tier_c" / "tier_c_devanagari.jsonl"

NEMOTRON_DATASET = "nvidia/Nemotron-Safety-Guard-Dataset-v3"
SAMPLE_SIZE = 200
SEED = 42  # pre-registered, do not change

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Map Nemotron's category labels onto our coarse 11-category taxonomy.
NEMOTRON_CATEGORY_MAP = {
    "Criminal Planning/Confessions": HarmCategory.OTHER,
    "Violence": HarmCategory.HARASSMENT,
    "Guns and Illegal Weapons": HarmCategory.WEAPONS,
    "Controlled/Regulated Substances": HarmCategory.ILLEGAL_GOODS,
    "Suicide and Self Harm": HarmCategory.SELF_HARM,
    "Hate/Identity Hate": HarmCategory.HATE,
    "Sexual": HarmCategory.SEXUAL,
    "PII/Privacy": HarmCategory.PRIVACY,
    "Harassment": HarmCategory.HARASSMENT,
    "Profanity": HarmCategory.HATE,
    "Threat": HarmCategory.HARASSMENT,
    "Political/Misinformation/Conspiracy": HarmCategory.MISINFORMATION,
    "Fraud/Deception": HarmCategory.FRAUD,
    "Malware": HarmCategory.CYBERCRIME,
    "Unauthorized Advice": HarmCategory.OTHER,
    "Immoral/Unethical": HarmCategory.OTHER,
    "Illegal Activity": HarmCategory.OTHER,
    "Needs Caution": HarmCategory.OTHER,
}


def map_category(violated_categories: str) -> HarmCategory:
    """Map Nemotron 'violated_categories' string to our taxonomy.
    Categories are comma-separated; we take the first match."""
    if not violated_categories or not violated_categories.strip():
        return HarmCategory.OTHER
    first = violated_categories.split(",")[0].strip()
    return NEMOTRON_CATEGORY_MAP.get(first, HarmCategory.OTHER)


def main() -> int:
    log.info("Loading Nemotron-Safety-Guard-Dataset-v3 (Hindi subset)...")
    # Stream the full dataset and filter for Hindi
    ds = load_dataset(NEMOTRON_DATASET, split="train", streaming=False)

    log.info("Filtering for Hindi (language='hi'), prompt_label='unsafe', non-REDACTED...")
    hi_unsafe = ds.filter(
        lambda r: (
            r.get("language") == "hi"
            and r.get("prompt_label") == "unsafe"
            and r.get("prompt") is not None
            and len(r["prompt"].strip()) > 0
            and r["prompt"].strip() != "REDACTED"
        )
    )
    log.info("Found %d Hindi unsafe prompts after filtering", len(hi_unsafe))

    if len(hi_unsafe) < SAMPLE_SIZE:
        log.error("Insufficient Hindi prompts (%d < %d)", len(hi_unsafe), SAMPLE_SIZE)
        return 1

    sampled = hi_unsafe.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

    records: list[PromptRecord] = []
    for i, row in enumerate(tqdm(sampled, desc="Building records")):
        rec = PromptRecord(
            id=make_id("C", i + 1, "devanagari"),
            tier=Tier.C,
            harm_category=map_category(row.get("violated_categories", "")),
            source_id=f"nemotron_{row.get('id', f'ix{i}')}",
            script=Script.DEVANAGARI,
            language=Language.HI,
            prompt=row["prompt"].strip(),
            paired_with=make_id("C", i + 1, "itrans"),
            is_harmful=True,
            validation=Validation(),
        )
        records.append(rec)

    n = write_jsonl(records, OUT_PATH)
    log.info("Wrote %d Hindi-Devanagari prompts to %s", n, OUT_PATH)
    log.info("NEXT: run 05_devanagari_to_itrans.py to generate the Latin half.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
