#!/usr/bin/env python3
"""
14_validate.py — Walk every JSONL under data/ and verify schema compliance.

Checks:
- Every line parses as a PromptRecord
- IDs are unique within a file
- Paired records reference back to existing IDs
- Script choice matches language
- All harmful records have is_harmful=True (and vice versa for benign)

Run: python3 scripts/14_validate.py
"""
from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scripts.utils.io import read_jsonl
from scripts.utils.schema import PromptRecord

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def validate_file(path: Path) -> tuple[int, list[str]]:
    """Return (record_count, errors)."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    n = 0

    try:
        for rec in read_jsonl(path):
            n += 1
            if rec.id in seen_ids:
                errors.append(f"{path.name}: duplicate id {rec.id!r}")
            seen_ids.add(rec.id)

            if not rec.script_consistent_with_language():
                errors.append(
                    f"{path.name}: {rec.id} script={rec.script.value} not valid "
                    f"for language={rec.language.value}"
                )

            if "benign" in path.name and rec.is_harmful:
                errors.append(f"{path.name}: {rec.id} marked is_harmful in benign file")
            if "benign" not in path.name and not rec.is_harmful:
                errors.append(f"{path.name}: {rec.id} marked NOT harmful in harmful file")
    except Exception as e:
        errors.append(f"{path.name}: parse failed at row {n+1}: {e}")

    return n, errors


def check_pairings(all_records: dict[str, PromptRecord]) -> list[str]:
    """Verify that paired_with references point to existing records.
    - Benign→harmful are one-directional (benign is derived, not paired).
    - Tier D→Tier A are one-directional (Tier D is derived control).
    - Tier A within-script pairs (arabic↔arabizi) should be bidirectional
      but missing partners are allowed (generation failures).
    - Any record with is_harmful=False is treated as one-directional
      (handles legacy benign files that use tier="B")."""
    errors: list[str] = []
    for rec_id, rec in all_records.items():
        if rec.paired_with is None:
            continue
        if rec.paired_with not in all_records:
            # Allow missing partners for Tier A (generation failures are expected)
            if rec.tier.value not in ["A"]:
                errors.append(f"{rec_id}: paired_with {rec.paired_with!r} not found")
            continue

        # Benign (is_harmful=False) and Tier D are derived; one-directional pairings OK.
        # Accepts legacy tier="B" on benign_* files (is_harmful=False).
        if rec.tier.value in ["BENIGN", "D"] or not rec.is_harmful:
            continue

        # Within-tier Tier A pairs should be bidirectional (if both exist)
        partner = all_records[rec.paired_with]
        if partner.paired_with != rec_id:
            errors.append(
                f"{rec_id}: paired_with {rec.paired_with!r}, "
                f"but partner points to {partner.paired_with!r}"
            )
    return errors


def main() -> int:
    files = sorted(f for f in DATA.rglob("*.jsonl") if "raw" not in f.parts)
    if not files:
        log.warning("No JSONL files under %s", DATA)
        return 0

    all_records: dict[str, PromptRecord] = {}
    all_errors: list[str] = []
    counts: dict[str, int] = {}

    for path in files:
        n, errors = validate_file(path)
        counts[path.relative_to(REPO).as_posix()] = n
        all_errors.extend(errors)
        for rec in read_jsonl(path):
            all_records[rec.id] = rec

    pair_errors = check_pairings(all_records)
    all_errors.extend(pair_errors)

    log.info("\n=== File sizes ===")
    for f, n in counts.items():
        log.info("  %4d records — %s", n, f)
    log.info("Total: %d records across %d files", sum(counts.values()), len(files))

    if all_errors:
        log.error("\n=== %d ERRORS ===", len(all_errors))
        for e in all_errors[:30]:
            log.error("  %s", e)
        if len(all_errors) > 30:
            log.error("  ... and %d more", len(all_errors) - 30)
        return 1

    # Summary by (tier, script)
    by_tier_script: Counter = Counter()
    for rec in all_records.values():
        by_tier_script[(rec.tier.value, rec.script.value)] += 1
    log.info("\n=== Distribution by (tier, script) ===")
    for key, count in sorted(by_tier_script.items()):
        log.info("  %s × %s: %d", key[0], key[1], count)

    log.info("\nAll validation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
