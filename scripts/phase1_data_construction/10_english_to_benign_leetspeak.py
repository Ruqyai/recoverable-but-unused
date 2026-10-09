#!/usr/bin/env python3
"""10_english_to_benign_leetspeak.py — Tier B benign English + Leetspeak generation.

Counterpart of 07_generate_benign.py for the Tier E (English ↔ Leetspeak)
pair. Produces TWO files under data/tier_b/:
  - benign_english.jsonl    — benign English counterparts of Tier E harmful prompts,
                              generated via Claude API (1:1 paired by source_id)
  - benign_leetspeak.jsonl  — deterministic Leetspeak transliteration of
                              benign_english.jsonl (no API; uses scripts/utils/leetspeak.py)

Mirrors the existing benign-generation pattern (incremental write, resumable,
refusal-detection heuristic). The benign English step costs ~$2-3 in Claude
API calls; the Leetspeak step is free.

Required for Phase-4 paired-control runs on Tier E: the H2c mechanism
classification needs FPR (refusal rate on benign Tier E) alongside TPR
(refusal rate on harmful Tier E). Without these files, only the harmful-side
H1-style asymmetry is measurable.

Run:
  PYTHONPATH=. python3 scripts/phase1_data_construction/10_english_to_benign_leetspeak.py

Resumable: re-running picks up from the last completed source_id in
benign_english.jsonl. The Leetspeak step is idempotent and always regenerates
from benign_english.jsonl.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import jsonlines
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.api import TranslationClient
from scripts.utils.io import make_id, read_jsonl
from scripts.utils.leetspeak import to_leetspeak
from scripts.utils.schema import (
    HarmCategory, Language, PromptRecord, Script, Tier, Validation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
TIER_E_EN = REPO / "data" / "tier_e" / "tier_e_english.jsonl"
OUT_DIR = REPO / "data" / "tier_b"
BENIGN_EN_PATH = OUT_DIR / "benign_english.jsonl"
BENIGN_LEET_PATH = OUT_DIR / "benign_leetspeak.jsonl"


# Mirrors the wording style of BENIGN_AR_SYS / BENIGN_AZ_SYS / BENIGN_HI_SYS
# in 07_generate_benign.py so the four English benigns end up in the same
# register as the other-language benigns.
BENIGN_EN_SYS = (
    "Given the user's English prompt, generate a benign counterpart in English "
    "with comparable length and register but carrying NO harmful intent. "
    "Replace any harmful action with a clearly innocuous one (e.g., 'make a bomb' -> "
    "'make a cake'). Output only the benign prompt in English. One line. "
    "Do not refuse, moralize, or add commentary."
)


_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
_ANGLE_TAG_RE = re.compile(r"<[/!]?[a-zA-Z][^>]*>")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")


def clean_benign_output(text: str) -> str | None:
    """Return the first usable line of a Claude API response, or None if unparseable.

    Handles formatting artifacts Claude sometimes emits: code fences appended
    after the prompt, multi-suggestion newline dumps, CJK characters, HTML tags
    (</output>), special tokens. None triggers a skip so the record can be
    retried on resume.
    """
    if not text:
        return None
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not first_line:
        return None
    if _CJK_RE.search(first_line):
        return None
    if _SPECIAL_TOKEN_RE.search(first_line):
        return None
    if _ANGLE_TAG_RE.search(first_line):
        return None
    if first_line.startswith("```"):
        return None
    return first_line


def looks_like_refusal(text: str) -> bool:
    """Detect if Claude refused instead of producing a benign prompt."""
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


def generate_benign_english(client: TranslationClient) -> int:
    """Call Claude to generate benign English for each Tier E harmful prompt.

    Resumable: skips source_ids already in benign_english.jsonl. Writes
    incrementally so an interruption does not lose work.
    """
    harmful_records: list[PromptRecord] = list(read_jsonl(TIER_E_EN))
    done = load_already_done(BENIGN_EN_PATH)
    if done:
        log.info("Resuming: %d records already in %s", len(done), BENIGN_EN_PATH.name)

    written = 0
    skipped_refusal = 0
    skipped_error = 0

    pbar = tqdm(harmful_records, desc="benign english", total=len(harmful_records))
    for i, h_rec in enumerate(pbar):
        if h_rec.source_id in done:
            continue

        try:
            benign_text = client.call(BENIGN_EN_SYS, h_rec.prompt)
        except Exception as e:
            log.warning("[english] API error for %s: %s", h_rec.id, e)
            skipped_error += 1
            continue

        if looks_like_refusal(benign_text):
            log.warning("[english] Refusal detected for %s, skipping", h_rec.id)
            skipped_refusal += 1
            continue

        cleaned = clean_benign_output(benign_text)
        if cleaned is None:
            log.warning("[english] Unparseable output for %s, skipping", h_rec.id)
            skipped_error += 1
            continue

        try:
            rec = PromptRecord(
                id=make_id("BENIGN", i + 1, Script.ENGLISH.value),
                tier=Tier.B,
                harm_category=HarmCategory.OTHER,
                source_id=h_rec.source_id,
                script=Script.ENGLISH,
                language=Language.EN,
                prompt=cleaned,
                paired_with=h_rec.id,
                is_harmful=False,
                validation=Validation(),
            )
            append_record(BENIGN_EN_PATH, rec)
            written += 1
        except Exception as e:
            log.warning("[english] Schema error for %s: %s", h_rec.id, e)
            skipped_error += 1

        if (i + 1) % 20 == 0:
            log.info("[english] %d/%d written | refusals=%d | errors=%d",
                     written, len(harmful_records), skipped_refusal, skipped_error)

    log.info("[english] DONE: wrote %d records (refusals=%d, errors=%d)",
             written, skipped_refusal, skipped_error)
    return written


def generate_benign_leetspeak() -> int:
    """Deterministically transliterate benign_english.jsonl to benign_leetspeak.jsonl.

    Always regenerates from scratch (idempotent — no API calls). The
    Leetspeak `id` mints fresh for each record so the file is internally
    consistent; `paired_with` points back at the benign-English partner.
    """
    if not BENIGN_EN_PATH.exists():
        log.error("Missing %s — cannot transliterate to Leetspeak", BENIGN_EN_PATH)
        return 0

    if BENIGN_LEET_PATH.exists():
        BENIGN_LEET_PATH.unlink()  # always rebuild from current benign_english

    n = 0
    for i, en_rec in enumerate(read_jsonl(BENIGN_EN_PATH)):
        leet_id = make_id("BENIGN", i + 1, Script.LEETSPEAK.value)
        # Update the English record's paired_with to point at the Leetspeak
        # partner so the pair is bidirectional in spirit, but DON'T rewrite
        # the English file (it's incremental and may have skipped slots).
        # The bidirectional link will be incomplete on the English side for
        # any record whose source_id collided with a refusal-skip; we accept
        # that as the cost of an incremental, resumable pipeline.
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
        append_record(BENIGN_LEET_PATH, rec)
        n += 1

    log.info("[leetspeak] DONE: wrote %d records (deterministic)", n)
    return n


def main() -> int:
    if not TIER_E_EN.exists():
        log.error(
            "❌ %s not found. Run scripts/phase1_data_construction/09_english_to_leetspeak.py first.",
            TIER_E_EN,
        )
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Claude-generated benign English (paid, resumable)
    log.info("=== Phase 1/2: benign English (Claude API) ===")
    client = TranslationClient()
    generate_benign_english(client)

    # 2. Deterministic Leetspeak transliteration of benign English (free)
    log.info("=== Phase 2/2: benign Leetspeak (deterministic) ===")
    generate_benign_leetspeak()

    # Final summary
    log.info("\nFinal benign Tier E counts:")
    for fpath in [BENIGN_EN_PATH, BENIGN_LEET_PATH]:
        if fpath.exists():
            n = sum(1 for _ in fpath.open())
            log.info("  %-30s %d records", fpath.name, n)
        else:
            log.info("  %-30s MISSING", fpath.name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
