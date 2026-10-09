#!/usr/bin/env python3
"""
15_self_validate.py — Author self-validation of translation quality.

Walks through the 20 samples in data/review/arabic_review.csv, displays each
Arabic ↔ Arabizi pair, and asks for Y/N on three criteria. Saves results to
data/review/arabic_self_validated.csv.

This avoids exposing third-party reviewers to harmful prompt content. The
author (native Arabic speaker, project lead) self-validates with full
awareness of the research context.

Limitations are reported transparently in the paper.

Positioned near the end of the sequence to reflect its role as a final
corpus self-check.

Run: PYTHONPATH=. python3 scripts/phase1_data_construction/15_self_validate.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INPUT = REPO / "data" / "review" / "arabic_review.csv"
OUTPUT = REPO / "data" / "review" / "arabic_self_validated.csv"


def prompt_yn(question: str) -> str:
    """Prompt user for Y/N answer, return 'Y' or 'N'."""
    while True:
        ans = input(f"  {question} [Y/N/s=skip]: ").strip().upper()
        if ans in ("Y", "N", "S"):
            return ans
        print("    Please enter Y, N, or s")


def main() -> int:
    if not INPUT.exists():
        print(f"❌ Input file not found: {INPUT}")
        return 1

    rows = list(csv.DictReader(INPUT.open(encoding="utf-8")))
    print(f"\n{'='*70}")
    print(f"  SELF-VALIDATION OF ARABIC ↔ ARABIZI TRANSLATION QUALITY")
    print(f"  Total samples: {len(rows)}")
    print(f"{'='*70}\n")

    results = []
    for i, row in enumerate(rows, 1):
        print(f"\n--- Sample {i}/{len(rows)} (ID: {row['sample_id']}) ---")
        print(f"Category: {row['harm_category']}")
        print(f"\nArabic   : {row['arabic_prompt']}")
        print(f"Arabizi  : {row['arabizi_prompt']}")
        print()

        q1 = prompt_yn("Q1. Arabic translation grammatically correct?")
        q2 = prompt_yn("Q2. Arabizi reads naturally for online chat?")
        q3 = prompt_yn("Q3. Both versions express the same meaning?")
        notes = input("  Notes (or Enter to skip): ").strip()

        results.append({
            **row,
            "reviewer_arabic_translation_ok": q1,
            "reviewer_arabizi_natural_ok": q2,
            "reviewer_semantic_match": q3,
            "reviewer_notes": notes,
        })

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    # Summary
    n = len(results)
    q1_yes = sum(1 for r in results if r["reviewer_arabic_translation_ok"] == "Y")
    q2_yes = sum(1 for r in results if r["reviewer_arabizi_natural_ok"] == "Y")
    q3_yes = sum(1 for r in results if r["reviewer_semantic_match"] == "Y")

    print(f"\n{'='*70}")
    print(f"  SELF-VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Arabic OK:       {q1_yes}/{n} ({q1_yes/n*100:.0f}%)")
    print(f"  Arabizi natural: {q2_yes}/{n} ({q2_yes/n*100:.0f}%)")
    print(f"  Semantic match:  {q3_yes}/{n} ({q3_yes/n*100:.0f}%)")
    print(f"\n  Saved to: {OUTPUT}")

    # Recommendation
    worst = min(q1_yes, q2_yes, q3_yes) / n
    print()
    if worst >= 0.85:
        print("  ✅ Translation quality PASS — proceed to Phase 2")
    elif worst >= 0.70:
        print("  🟡 Acceptable but report flagged records in Limitations")
    else:
        print("  ❌ Quality issues detected — review the pipeline before Phase 2")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
