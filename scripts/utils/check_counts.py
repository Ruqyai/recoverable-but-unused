#!/usr/bin/env python3
"""
check_counts.py — Diagnostic tool to report data counts and analyzable N per tier.

Reports:
  1. Raw count per file
  2. Pair intersection per tier (the actual analyzable N)
  3. Whether each tier meets the statistical threshold (N >= 150)

Run: PYTHONPATH=. python3 scripts/utils/check_counts.py
Moved 2026-05-29 from scripts/check_counts.py to scripts/utils/ (diagnostic utility, lives with other utils).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
THRESHOLD = 150  # minimum analyzable N for statistical power


def load_ids(path: Path) -> tuple[set[str], dict[str, str]]:
    """Return (set of record ids, map of id -> source_id)."""
    if not path.exists():
        return set(), {}
    ids, src = set(), {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids.add(r["id"])
            src[r["id"]] = r.get("source_id", "")
    return ids, src


def main() -> int:
    print("=" * 78)
    print("RAW FILE COUNTS")
    print("=" * 78)

    counts = {}
    for f in sorted(DATA.rglob("*.jsonl")):
        n = sum(1 for _ in f.open())
        rel = f.relative_to(REPO)
        print(f"  {str(rel):55s} {n:>5d} records")
        counts[str(rel)] = n

    print()
    print("=" * 78)
    print("ANALYZABLE N PER TIER (intersection of paired conditions)")
    print("=" * 78)

    def status(n: int) -> str:
        if n >= 200: return f"✅ {n}  (full)"
        if n >= THRESHOLD: return f"✅ {n}  (above threshold {THRESHOLD})"
        return f"⚠️  {n}  (BELOW threshold {THRESHOLD})"

    # Tier A: need Arabic + Arabizi + benign_Arabic + benign_Arabizi all paired by source_id
    ar_a, src_ar_a = load_ids(DATA / "tier_a" / "tier_a_arabic.jsonl")
    az_a, src_az_a = load_ids(DATA / "tier_a" / "tier_a_arabizi.jsonl")
    bn_ar, src_bn_ar = load_ids(DATA / "tier_b" / "benign_arabic.jsonl")
    bn_az, src_bn_az = load_ids(DATA / "tier_b" / "benign_arabizi.jsonl")

    # Take intersection on source_id
    src_intersect_a = (
        set(src_ar_a.values()) & set(src_az_a.values()) &
        set(src_bn_ar.values()) & set(src_bn_az.values())
    )
    print(f"\n[Tier A — Arabic ↔ Arabizi]")
    print(f"  arabic       : {len(ar_a)}")
    print(f"  arabizi      : {len(az_a)}")
    print(f"  benign_AR    : {len(bn_ar)}")
    print(f"  benign_AZ    : {len(bn_az)}")
    print(f"  Analyzable N : {status(len(src_intersect_a))}")

    # Tier E: English + Leetspeak + benign for both (v6 within-language control)
    en_e, src_en_e = load_ids(DATA / "tier_e" / "tier_e_english.jsonl")
    leet_e, src_leet_e = load_ids(DATA / "tier_e" / "tier_e_leetspeak.jsonl")
    bn_en, src_bn_en = load_ids(DATA / "tier_b" / "benign_english.jsonl")
    bn_leet, src_bn_leet = load_ids(DATA / "tier_b" / "benign_leetspeak.jsonl")
    src_intersect_e = (
        set(src_en_e.values()) & set(src_leet_e.values()) &
        set(src_bn_en.values()) & set(src_bn_leet.values())
    )
    print(f"\n[Tier E — English ↔ Leetspeak (v6 within-language control)]")
    print(f"  english      : {len(en_e)}")
    print(f"  leetspeak    : {len(leet_e)}")
    print(f"  benign_EN    : {len(bn_en)}")
    print(f"  benign_LEET  : {len(bn_leet)}")
    print(f"  Analyzable N : {status(len(src_intersect_e))}")

    # Tier C: Devanagari + ITRANS + benign for both
    dv, src_dv = load_ids(DATA / "tier_c" / "tier_c_devanagari.jsonl")
    it, src_it = load_ids(DATA / "tier_c" / "tier_c_itrans.jsonl")
    bn_dv, src_bn_dv = load_ids(DATA / "tier_b" / "benign_devanagari.jsonl")
    bn_it, src_bn_it = load_ids(DATA / "tier_b" / "benign_itrans.jsonl")
    src_intersect_c = (
        set(src_dv.values()) & set(src_it.values()) &
        set(src_bn_dv.values()) & set(src_bn_it.values())
    )
    print(f"\n[Tier C — Devanagari ↔ ITRANS]")
    print(f"  devanagari   : {len(dv)}")
    print(f"  itrans       : {len(it)}")
    print(f"  benign_DV    : {len(bn_dv)}")
    print(f"  benign_IT    : {len(bn_it)}")
    print(f"  Analyzable N : {status(len(src_intersect_c))}")

    # Tier D: within-script paraphrase, 3 languages (Arabic, English, Devanagari)
    d_ar, _ = load_ids(DATA / "tier_d" / "tier_d_arabic_paraphrased.jsonl")
    d_en, _ = load_ids(DATA / "tier_d" / "tier_d_english_paraphrased.jsonl")
    d_dv, _ = load_ids(DATA / "tier_d" / "tier_d_devanagari_paraphrased.jsonl")
    d_total = len(d_ar) + len(d_en) + len(d_dv)
    print(f"\n[Tier D — Within-script paraphrase (3 languages, variation control)]")
    print(f"  arabic       : {len(d_ar)}")
    print(f"  english      : {len(d_en)}")
    print(f"  devanagari   : {len(d_dv)}")
    print(f"  Total Tier D : {d_total}  (expect 600)")

    # Tier B: matched benign controls only (6 conditions; benign-only since 2026-05-29)
    print(f"\n[Tier B — Matched benign controls (6 conditions, benign-only per 2026-05-29)]")
    benign_total = 0
    for name in ("benign_arabic", "benign_arabizi", "benign_devanagari",
                 "benign_itrans", "benign_english", "benign_leetspeak"):
        ids, _ = load_ids(DATA / "tier_b" / f"{name}.jsonl")
        benign_total += len(ids)
        print(f"  {name:18s}: {len(ids)}")
    print(f"  Total benign : {benign_total}  (expect 1,200)")

    # Grand total — corpus records (exclude the immutable raw AdvBench seed)
    harmful_total = sum(
        n for rel, n in counts.items()
        if not rel.startswith("data/raw/") and "tier_b/" not in rel
    )
    grand_total = harmful_total + benign_total
    print()
    print("=" * 78)
    print("CORPUS TOTAL (excludes data/raw/ immutable seed)")
    print("=" * 78)
    print(f"  Harmful (A/C/D/E): {harmful_total}  (expect 1,800)")
    print(f"  Benign  (Tier B) : {benign_total}  (expect 1,200)")
    print(f"  GRAND TOTAL      : {grand_total}  (expect 3,000)")

    print()
    print("=" * 78)
    print("OVERALL ASSESSMENT")
    print("=" * 78)
    all_ok = (
        len(src_intersect_a) >= THRESHOLD and
        len(src_intersect_c) >= THRESHOLD and
        len(src_intersect_e) >= THRESHOLD and
        grand_total == 3000
    )
    if all_ok:
        print(f"\n  ✅ Confirmatory pairs A/C/E ≥ {THRESHOLD} and corpus = 3,000. Phase 2 can proceed.")
    else:
        print(f"\n  ⚠️  Check the per-tier output above against the expected counts.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
