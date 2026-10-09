#!/usr/bin/env python3
"""
07_generate_benign.py — Generate matched-style benign prompts for each script condition.

Required for Phase 2: Arditi-Wang refusal direction extraction needs
r = mean(harmful_activations) - mean(benign_activations).

FIXES in this version:
  - Incremental write: each record is appended to file immediately (no batch-at-end)
  - Resumable: skips records already present in the output file
  - Visible progress: prints a log line every 20 records
  - Empty-response handling: explicitly skips empty/refusal returns with a warning
  - Per-language file is finalized once that language's set is complete

Outputs (each appended incrementally):
  data/tier_b/benign_arabic.jsonl
  data/tier_b/benign_arabizi.jsonl
  data/tier_b/benign_devanagari.jsonl
  data/tier_b/benign_itrans.jsonl

Run: python scripts/07_generate_benign.py
Re-run after interruption is safe — it picks up where it left off.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import jsonlines
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from tqdm import tqdm

from scripts.utils.api import TranslationClient
from scripts.utils.io import make_id, read_jsonl
from scripts.utils.schema import (
    HarmCategory, Language, PromptRecord, Script, Tier, Validation,
)

REPO = Path(__file__).resolve().parents[2]
TIER_A_AR = REPO / "data" / "tier_a" / "tier_a_arabic.jsonl"
TIER_A_AZ = REPO / "data" / "tier_a" / "tier_a_arabizi.jsonl"
TIER_C_DV = REPO / "data" / "tier_c" / "tier_c_devanagari.jsonl"
OUT_DIR = REPO / "data" / "tier_b"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


BENIGN_AR_SYS = (
    "Given the user's Arabic prompt, generate a benign counterpart in Modern Standard "
    "Arabic that has comparable length and register but carries NO harmful intent. "
    "Replace any harmful action with a clearly innocuous one (e.g., 'make a bomb' -> "
    "'make a cake'). Output only the benign prompt in Arabic script. One line. "
    "Do not refuse, moralize, or add commentary."
)

BENIGN_AZ_SYS = (
    "Given the user's Arabizi (Latin-script Arabic) prompt, generate a benign "
    "counterpart in Arabizi with comparable length and chat-register but NO harmful "
    "intent. Use the same Arabizi conventions (3=ع, 7=ح, 2=ء/ق). "
    "Output only the benign Arabizi prompt. One line. Do not refuse or moralize."
)

BENIGN_HI_SYS = (
    "Given the user's Hindi prompt in Devanagari script, generate a benign "
    "counterpart in Devanagari with comparable length and register but NO harmful "
    "intent. Replace any harmful action with an innocuous one. "
    "Output only the benign Hindi prompt in Devanagari. One line. "
    "Do not refuse, moralize, or add commentary."
)


def looks_like_refusal(text: str) -> bool:
    """Detect if the model refused instead of generating a benign prompt."""
    if not text or len(text.strip()) < 3:
        return True
    lower = text.lower()
    refusal_markers = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "i won't", "i will not", "as an ai", "i'm sorry",
        "لا أستطيع", "لا يمكنني", "آسف",
        "मैं नहीं", "क्षमा", "मैं असमर्थ",
    ]
    return any(m in lower for m in refusal_markers)


def load_already_done(out_path: Path) -> set[str]:
    """Return source_ids already present in the output file (for resume)."""
    if not out_path.exists():
        return set()
    done: set[str] = set()
    try:
        with jsonlines.open(out_path) as r:
            for obj in r:
                done.add(obj.get("source_id", ""))
    except Exception as e:
        log.warning("Could not read %s for resume check: %s", out_path, e)
    return done


def append_record(out_path: Path, record: PromptRecord) -> None:
    """Append a single record to the output JSONL file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")


def generate_benign_set(
    client: TranslationClient,
    harmful_records: list[PromptRecord],
    system_prompt: str,
    target_script: Script,
    target_lang: Language,
    out_path: Path,
    label: str,
) -> int:
    """Generate benign counterparts incrementally. Returns count of records written."""
    done = load_already_done(out_path)
    if done:
        log.info("[%s] Resuming: %d records already in %s", label, len(done), out_path.name)

    written = 0
    skipped_refusal = 0
    skipped_error = 0

    pbar = tqdm(harmful_records, desc=f"benign {label}", initial=0, total=len(harmful_records))
    for i, h_rec in enumerate(pbar):
        if h_rec.source_id in done:
            continue

        try:
            benign_text = client.call(system_prompt, h_rec.prompt)
        except Exception as e:
            log.warning("[%s] API error for %s: %s", label, h_rec.id, e)
            skipped_error += 1
            continue

        if looks_like_refusal(benign_text):
            log.warning("[%s] Refusal detected for %s, skipping", label, h_rec.id)
            skipped_refusal += 1
            continue

        try:
            rec = PromptRecord(
                id=make_id("BENIGN", i + 1, target_script.value),
                tier=Tier.B,
                harm_category=HarmCategory.OTHER,
                source_id=h_rec.source_id,
                script=target_script,
                language=target_lang,
                prompt=benign_text.strip(),
                paired_with=h_rec.id,
                is_harmful=False,
                validation=Validation(),
            )
            append_record(out_path, rec)
            written += 1
        except Exception as e:
            log.warning("[%s] Schema error for %s: %s", label, h_rec.id, e)
            skipped_error += 1

        # Visible heartbeat every 20 records (in addition to tqdm bar)
        if (i + 1) % 20 == 0:
            log.info("[%s] %d/%d written | refusals=%d | errors=%d",
                     label, written, len(harmful_records), skipped_refusal, skipped_error)

    log.info("[%s] DONE: wrote %d records (refusals=%d, errors=%d)",
             label, written, skipped_refusal, skipped_error)
    return written


def main() -> int:
    client = TranslationClient()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Arabic benign
    if TIER_A_AR.exists():
        log.info("=== Phase 1/4: Arabic benign ===")
        ar_records = list(read_jsonl(TIER_A_AR))
        generate_benign_set(
            client, ar_records, BENIGN_AR_SYS,
            Script.ARABIC, Language.AR,
            OUT_DIR / "benign_arabic.jsonl", "arabic",
        )
    else:
        log.warning("Missing %s — skipping Arabic benign", TIER_A_AR)

    # 2. Arabizi benign
    if TIER_A_AZ.exists():
        log.info("=== Phase 2/4: Arabizi benign ===")
        az_records = list(read_jsonl(TIER_A_AZ))
        generate_benign_set(
            client, az_records, BENIGN_AZ_SYS,
            Script.ARABIZI, Language.AR,
            OUT_DIR / "benign_arabizi.jsonl", "arabizi",
        )
    else:
        log.warning("Missing %s — skipping Arabizi benign", TIER_A_AZ)

    # 3. Hindi Devanagari benign
    if TIER_C_DV.exists():
        log.info("=== Phase 3/4: Hindi Devanagari benign ===")
        dv_records = list(read_jsonl(TIER_C_DV))
        generate_benign_set(
            client, dv_records, BENIGN_HI_SYS,
            Script.DEVANAGARI, Language.HI,
            OUT_DIR / "benign_devanagari.jsonl", "devanagari",
        )

        # 4. Hindi ITRANS benign — deterministic transliteration of Devanagari benigns
        log.info("=== Phase 4/4: Hindi ITRANS benign (deterministic) ===")
        itrans_path = OUT_DIR / "benign_itrans.jsonl"
        dv_path = OUT_DIR / "benign_devanagari.jsonl"
        if not dv_path.exists():
            log.error("Devanagari benign file missing, cannot create ITRANS half")
        else:
            # Clear ITRANS file (always regenerate from Devanagari)
            if itrans_path.exists():
                itrans_path.unlink()

            n_itrans = 0
            for dv_rec in read_jsonl(dv_path):
                itrans_text = transliterate(dv_rec.prompt, sanscript.DEVANAGARI, sanscript.ITRANS)
                rec = PromptRecord(
                    id=make_id("BENIGN", n_itrans + 1, "itrans"),
                    tier=Tier.B,
                    harm_category=HarmCategory.OTHER,
                    source_id=dv_rec.source_id,
                    script=Script.ITRANS,
                    language=Language.HI,
                    prompt=itrans_text,
                    paired_with=dv_rec.id,
                    is_harmful=False,
                    validation=Validation(),
                )
                append_record(itrans_path, rec)
                n_itrans += 1
            log.info("[itrans] DONE: wrote %d records (deterministic)", n_itrans)
    else:
        log.warning("Missing %s — skipping Hindi benign", TIER_C_DV)

    log.info("=== All benign generation complete ===")

    # Final summary
    log.info("\nFinal benign counts:")
    for fname in ["benign_arabic.jsonl", "benign_arabizi.jsonl",
                  "benign_devanagari.jsonl", "benign_itrans.jsonl"]:
        fpath = OUT_DIR / fname
        if fpath.exists():
            n = sum(1 for _ in fpath.open())
            log.info("  %-30s %d records", fname, n)
        else:
            log.info("  %-30s MISSING", fname)

    return 0


if __name__ == "__main__":
    sys.exit(main())