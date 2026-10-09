#!/usr/bin/env python3
"""
20_deep_review.py — Deep, cross-language integrity audit of the whole dataset.

Emits ONE CSV (data/review/deep_review_findings.csv) listing every damaged /
mismatched cell across ALL tiers and scripts, with a machine-checkable reason.

This script covers the FULLY AUTOMATABLE checks (no human/LLM judgment needed):

  REFUSAL_LEAK     prompt cell contains a model refusal instead of content
  LEET_MISMATCH    to_leetspeak(paired English) != the leetspeak cell  (deterministic)
  ITRANS_MISMATCH  transliterate(devanagari) != the itrans cell        (deterministic)
  PAIR_BROKEN      paired_with points to an id that doesn't exist in the sibling file
  DUP_PROMPT       prompt text is an exact duplicate of another row in the same file
  SCRIPT_IMPURITY  cell is in the wrong script (e.g. Latin text in an Arabic-script file)
  EN_NOT_CANONICAL English prompt != canonical AdvBench goal at that source_id
  EMPTY_SHORT      prompt empty or implausibly short

Semantic mis-key checks (translation describes a DIFFERENT source_id than its
English) require reading and are merged in from 21_semantic_findings.csv.

Run: python3 scripts/phase1_data_construction/20_deep_review.py
"""
from __future__ import annotations

import csv
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.leetspeak import to_leetspeak  # noqa: E402

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAVE_ITRANS = True
except Exception:
    HAVE_ITRANS = False

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT = DATA / "review" / "deep_review_findings.csv"
ADVBENCH_CSV = Path("/tmp/advbench.csv")

REFUSAL_MARKERS = [
    "ana mish", "ma2dar", "ma2der", "ma b3mil", "mish 7a3mal", "ma2dar a3teek",
    "i can't", "i cannot", "i won't", "i'm sorry", "as an ai", "ghayr 2anoni",
    "ghayr 2anooni", "can't help", "cannot help", "नहीं कर सकता", "माफ कर",
    "لا أستطيع", "لا يمكنني", "آسف", "لن أقوم", "لا أقدر", "(fake news)",
]

# Unicode script ranges
def script_of(ch: str) -> str:
    o = ord(ch)
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F: return "arabic"
    if 0x0900 <= o <= 0x097F: return "devanagari"
    if (0x41 <= o <= 0x5A) or (0x61 <= o <= 0x7A): return "latin"
    return "other"


def script_profile(text: str) -> Counter:
    c = Counter()
    for ch in text:
        if ch.isalpha():
            c[script_of(ch)] += 1
    return c


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def looks_refusal(text: str) -> str | None:
    low = text.lower()
    for m in REFUSAL_MARKERS:
        if m in low:
            return m
    return None


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    findings: list[dict] = []

    def add(file, rec, ptype, severity, detail):
        findings.append({
            "file": file,
            "tier": rec.get("tier", ""),
            "source_id": rec.get("source_id", ""),
            "id": rec.get("id", ""),
            "language": rec.get("language", ""),
            "script": rec.get("script", ""),
            "problem_type": ptype,
            "severity": severity,
            "detail": detail,
            "prompt": rec.get("prompt", rec.get("prompt_en", "")),
        })

    # canonical advbench
    canon = {}
    if ADVBENCH_CSV.exists():
        import pandas as pd
        df = pd.read_csv(ADVBENCH_CSV)
        canon = {f"advbench_{i:03d}": row["goal"] for i, row in df.iterrows()}

    files = sorted(DATA.glob("**/*.jsonl"))
    data = {f.name: load(f) for f in files}
    by_path = {str(f.relative_to(REPO)): load(f) for f in files}

    # index id -> exists, per file (for pair integrity)
    all_ids = set()
    for rows in by_path.values():
        for r in rows:
            all_ids.add(r.get("id"))

    for relpath, rows in by_path.items():
        fname = Path(relpath).name
        is_arabic_script = "arabic" in fname and "arabizi" not in fname
        is_devanagari = "devanagari" in fname
        is_arabizi = "arabizi" in fname
        is_itrans = "itrans" in fname
        is_leet = "leetspeak" in fname
        is_english = fname.endswith("english.jsonl") or fname == "advbench_sample_en.jsonl"

        # duplicate prompt clusters within this file
        prompts = [r.get("prompt", r.get("prompt_en", "")) for r in rows]
        dup_counts = Counter(p.strip() for p in prompts if p.strip())

        for r in rows:
            p = r.get("prompt", r.get("prompt_en", "")) or ""
            ps = p.strip()
            sid = r.get("source_id", "")

            # empty / short
            if len(ps) < 4:
                add(relpath, r, "EMPTY_SHORT", "high", f"len={len(ps)}")
                continue

            # refusal leak
            m = looks_refusal(p)
            if m:
                add(relpath, r, "REFUSAL_LEAK", "high", f"marker={m!r}")

            # duplicate prompt within file
            if dup_counts[ps] > 1:
                add(relpath, r, "DUP_PROMPT", "medium",
                    f"prompt appears {dup_counts[ps]}x in this file")

            # pair integrity
            pw = r.get("paired_with")
            if pw and pw not in all_ids:
                add(relpath, r, "PAIR_BROKEN", "medium", f"paired_with={pw} not found")

            # script purity
            prof = script_profile(p)
            tot = sum(prof.values()) or 1
            if is_arabic_script and prof.get("latin", 0) / tot > 0.5:
                add(relpath, r, "SCRIPT_IMPURITY", "high", f"latin-heavy in arabic file: {dict(prof)}")
            if is_devanagari and prof.get("devanagari", 0) / tot < 0.3 and tot > 5:
                add(relpath, r, "SCRIPT_IMPURITY", "high", f"low devanagari ratio: {dict(prof)}")
            if is_arabic_script and prof.get("arabic", 0) / tot < 0.3 and tot > 5:
                add(relpath, r, "SCRIPT_IMPURITY", "high", f"low arabic ratio: {dict(prof)}")

            # english canonical check
            if is_english and canon and sid in canon:
                if ps != canon[sid].strip() and "benign" not in fname:
                    add(relpath, r, "EN_NOT_CANONICAL", "high",
                        f"canonical={canon[sid]!r}")

    # ---- deterministic LEETSPEAK check (tier_e + tier_b) ----
    for leet_file, en_file in [
        ("data/tier_e/tier_e_leetspeak.jsonl", "data/tier_e/tier_e_english.jsonl"),
        ("data/tier_b/benign_leetspeak.jsonl", "data/tier_b/benign_english.jsonl"),
    ]:
        leet = {r["source_id"]: r for r in load(REPO / leet_file)}
        eng = {r["source_id"]: r for r in load(REPO / en_file)}
        for sid, lr in leet.items():
            er = eng.get(sid)
            if not er:
                continue
            expect = to_leetspeak(er["prompt"])
            if lr["prompt"].strip() != expect.strip():
                add(leet_file, lr, "LEET_MISMATCH", "high",
                    f"decoded English would be {er['prompt'][:60]!r}; leet not a faithful encode")

    # ---- deterministic ITRANS check (tier_c + tier_b) ----
    if HAVE_ITRANS:
        for itr_file, dev_file in [
            ("data/tier_c/tier_c_itrans.jsonl", "data/tier_c/tier_c_devanagari.jsonl"),
            ("data/tier_b/benign_itrans.jsonl", "data/tier_b/benign_devanagari.jsonl"),
        ]:
            itr = {r["source_id"]: r for r in load(REPO / itr_file)}
            dev = {r["source_id"]: r for r in load(REPO / dev_file)}
            for sid, ir in itr.items():
                dr = dev.get(sid)
                if not dr:
                    continue
                expect = transliterate(dr["prompt"], sanscript.DEVANAGARI, sanscript.ITRANS)
                # normalize whitespace; ITRANS case can vary, compare case-insensitively loosely
                if ir["prompt"].strip() != expect.strip():
                    add(itr_file, ir, "ITRANS_MISMATCH", "medium",
                        f"itrans != transliterate(devanagari); expected≈{expect[:50]!r}")

    # write CSV
    cols = ["file", "tier", "source_id", "id", "language", "script",
            "problem_type", "severity", "detail", "prompt"]
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in findings:
            w.writerow(row)

    # summary
    bytype = Counter(f["problem_type"] for f in findings)
    byfile = Counter(f["file"] for f in findings)
    print(f"✅ {len(findings)} automatable findings -> {OUT}")
    print("\nBy problem_type:")
    for k, v in bytype.most_common():
        print(f"  {k:18s} {v}")
    print("\nBy file:")
    for k, v in byfile.most_common():
        print(f"  {k:48s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
