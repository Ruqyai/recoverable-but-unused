#!/usr/bin/env python3
"""11_complete_missing_benigns.py — Manual completion for Tier B benign English+Leetspeak.

The Claude-driven generation in 10_english_to_benign_leetspeak.py dropped 20/200
source_ids on the most severe harm categories (bombs, poisons, viruses, CSAM)
because the API refused to produce a benign reframing for them. This script
provides a manual-completion workflow to bring the corpus to 200/200.

Two modes, auto-detected:

  1. **Template mode** (when data/tier_b/benign_english_pending.jsonl does NOT exist):
     Identify the source_ids present in tier_e_english.jsonl but missing from
     benign_english.jsonl. Write a template file with one record per missing
     source_id, containing the original harmful English prompt for reference
     and an empty `benign_english` field to fill manually.

  2. **Finalize mode** (when data/tier_b/benign_english_pending.jsonl EXISTS):
     Read the template file, validate that every `benign_english` field has
     been filled (non-empty, no "TODO" placeholder, and not a refusal),
     append the filled records to benign_english.jsonl as proper PromptRecord
     entries, then deterministically regenerate benign_leetspeak.jsonl from
     the merged English file.

Run:
  # First pass — emit the template
  PYTHONPATH=. python3 scripts/phase1_data_construction/11_complete_missing_benigns.py

  # ... fill data/tier_b/benign_english_pending.jsonl manually ...

  # Second pass — merge manual completions back into the corpus
  PYTHONPATH=. python3 scripts/phase1_data_construction/11_complete_missing_benigns.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.io import make_id, read_jsonl
from scripts.utils.leetspeak import to_leetspeak
from scripts.utils.schema import (
    HarmCategory, Language, PromptRecord, Script, Tier, Validation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
TIER_E_EN = REPO / "data" / "tier_e" / "tier_e_english.jsonl"
BENIGN_EN = REPO / "data" / "tier_b" / "benign_english.jsonl"
BENIGN_LEET = REPO / "data" / "tier_b" / "benign_leetspeak.jsonl"
PENDING = REPO / "data" / "tier_b" / "benign_english_pending.jsonl"

TODO_MARKERS = {"", "TODO", "todo", "TODO: fill", "<fill manually>"}


def find_missing_source_ids() -> list[dict]:
    """Return harmful-Tier-E records whose source_id is missing from benign_english."""
    harmful = list(read_jsonl(TIER_E_EN))
    if BENIGN_EN.exists():
        benign_srcs = {r.source_id for r in read_jsonl(BENIGN_EN)}
    else:
        benign_srcs = set()
    missing = [h for h in harmful if h.source_id not in benign_srcs]
    return missing


def emit_template(missing: list) -> None:
    """Write the pending template file with placeholder records for manual fill."""
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    with PENDING.open("w", encoding="utf-8") as f:
        for rec in missing:
            entry = {
                "source_id": rec.source_id,
                "harm_category": rec.harm_category.value,
                "harmful_english": rec.prompt,
                "benign_english": "",  # ← fill this manually
                "_paired_with_tier_e_id": rec.id,
                "_idx_hint": int(rec.id.split("_")[2]),  # slot number from tier_e_NNN_english
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info(f"✓ Template written to {PENDING} with {len(missing)} placeholder records")
    log.info(
        f"  Fill the `benign_english` field for each record, then re-run this "
        f"script to merge into {BENIGN_EN.name} + {BENIGN_LEET.name}."
    )


def is_filled(value: str) -> bool:
    """Validate a manual fill: non-empty, not TODO, not obviously a refusal."""
    if value.strip() in TODO_MARKERS:
        return False
    if not value or len(value.strip()) < 3:
        return False
    lower = value.lower()
    refusal_markers = [
        "i cannot", "i can't", "i'm unable", "as an ai",
        "i won't help", "i will not",
    ]
    if any(m in lower for m in refusal_markers):
        return False
    return True


def finalize(pending_records: list[dict]) -> int:
    """Merge filled-in template records into benign_english.jsonl + regenerate benign_leetspeak.jsonl."""
    appended = 0
    skipped = 0
    with BENIGN_EN.open("a", encoding="utf-8") as out:
        for p in pending_records:
            benign_text = p.get("benign_english", "").strip()
            if not is_filled(benign_text):
                skipped += 1
                continue
            slot = p["_idx_hint"]
            rec = PromptRecord(
                id=make_id("BENIGN", slot, Script.ENGLISH.value),
                tier=Tier.B,
                harm_category=HarmCategory(p["harm_category"]),
                source_id=p["source_id"],
                script=Script.ENGLISH,
                language=Language.EN,
                prompt=benign_text,
                paired_with=p["_paired_with_tier_e_id"],
                is_harmful=False,
                validation=Validation(),
            )
            out.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")
            appended += 1

    log.info(
        f"✓ Appended {appended} manual completions to {BENIGN_EN.name} "
        f"({skipped} unfilled or refused, left in pending file)"
    )

    # Regenerate benign_leetspeak.jsonl deterministically from the merged English file.
    log.info("Regenerating benign_leetspeak.jsonl deterministically from merged benign_english.jsonl...")
    if BENIGN_LEET.exists():
        BENIGN_LEET.unlink()
    n = 0
    for en_rec in read_jsonl(BENIGN_EN):
        slot = int(en_rec.id.split("_")[-1])  # benign_english_NNN → NNN
        leet_id = make_id("BENIGN", slot, Script.LEETSPEAK.value)
        rec = PromptRecord(
            id=leet_id,
            tier=Tier.B,
            harm_category=HarmCategory.OTHER,
            source_id=en_rec.source_id,
            script=Script.LEETSPEAK,
            language=Language.EN,
            prompt=to_leetspeak(en_rec.prompt),
            paired_with=en_rec.id,
            is_harmful=False,
            validation=Validation(),
        )
        with BENIGN_LEET.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")
        n += 1
    log.info(f"✓ Wrote {n} records to {BENIGN_LEET.name}")
    return appended


def main() -> int:
    if not TIER_E_EN.exists():
        log.error(f"❌ {TIER_E_EN} not found. Run 09_english_to_leetspeak.py first.")
        return 1

    # Mode 2 (finalize): pending file exists → ingest filled records
    if PENDING.exists():
        log.info(f"Found {PENDING} — running in FINALIZE mode")
        pending_records = [json.loads(l) for l in PENDING.open()]
        log.info(f"  {len(pending_records)} pending records loaded")

        filled = sum(1 for p in pending_records if is_filled(p.get("benign_english", "")))
        unfilled = len(pending_records) - filled
        log.info(f"  {filled} filled, {unfilled} still unfilled")

        if filled == 0:
            log.error(
                "❌ No filled records found in pending file. Fill the `benign_english` "
                "field for each record, then re-run."
            )
            return 1

        finalize(pending_records)

        # Final summary
        log.info("\nFinal benign Tier E counts:")
        for fpath in [BENIGN_EN, BENIGN_LEET]:
            n = sum(1 for _ in fpath.open()) if fpath.exists() else 0
            log.info(f"  {fpath.name:30} {n} records")

        if unfilled > 0:
            log.warning(
                f"\n⚠️  {unfilled} records still unfilled in {PENDING.name}. "
                f"Re-run after filling them to reach 200/200."
            )
        else:
            log.info(f"\n✓ All pending records ingested. Pending file can be deleted.")
        return 0

    # Mode 1 (template): pending file does not exist → emit template
    log.info(f"{PENDING.name} not found — running in TEMPLATE mode")
    missing = find_missing_source_ids()
    log.info(f"Found {len(missing)} source_ids in {TIER_E_EN.name} missing from {BENIGN_EN.name}")

    if not missing:
        log.info("✓ benign_english.jsonl is already complete (no missing source_ids).")
        return 0

    emit_template(missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
