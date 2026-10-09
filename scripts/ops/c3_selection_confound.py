#!/usr/bin/env python3
"""C3 selection-confound check: is the comprehended-vs-non-comprehended split
confounded with harm category? (Reviewer point: comprehended items might be an
independently more-refused harm-category sample, not merely more decodable.)

For each model, pool the three romanized harmful conditions, join per-item Gemini
comprehension labels to AdvBench harm_category, and test category x comprehension
independence (chi-square). A non-significant result means comprehension does not
preferentially select certain harm categories -> the confound is not supported.

Reads live from results/phase5/comprehension_validation/*/items_gemini.jsonl and
data/tier_*/*.jsonl. No GPU. Run: python3 scripts/ops/c3_selection_confound.py
"""
from __future__ import annotations
import glob
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPREHENDED = {"ON_TOPIC", "REFUSAL"}
ROMANIZED = ["arabizi_harmful", "leetspeak_harmful", "itrans_harmful"]


def load_id2cat() -> dict[str, str]:
    m: dict[str, str] = {}
    for f in glob.glob(os.path.join(ROOT, "data", "tier_*", "*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in r and "harm_category" in r:
                m[r["id"]] = r["harm_category"]
    return m


def chi2_sf(x: float, k: int) -> float:
    """Upper tail of chi-square; scipy if present, else Wilson-Hilferty."""
    try:
        from scipy.stats import chi2

        return float(chi2.sf(x, k))
    except Exception:
        t = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
        return 0.5 * math.erfc(t / math.sqrt(2))


def chi2_table(tab: dict[str, list[int]]) -> tuple[float, int, int]:
    rows = list(tab.values())
    col = [sum(r[j] for r in rows) for j in range(2)]
    n = sum(col)
    exp = [[sum(rows[i]) * col[j] / n for j in range(2)] for i in range(len(rows))]
    chi = sum(
        (rows[i][j] - exp[i][j]) ** 2 / exp[i][j]
        for i in range(len(rows))
        for j in range(2)
        if exp[i][j] > 0
    )
    return chi, len(rows), n


def main() -> None:
    id2cat = load_id2cat()
    print(f"corpus ids with harm_category: {len(id2cat)}")
    print(f"{'model':22s} {'N':>4} {'df':>3} {'chi2':>6} {'p':>7}  category-independent?")
    n_sig = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "results/phase5/comprehension_validation/*/items_gemini.jsonl"))):
        model = os.path.basename(os.path.dirname(f))
        tab: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for line in open(f):
            r = json.loads(line)
            if r.get("condition") in ROMANIZED and r["id"] in id2cat:
                comp = 0 if r.get("label", "").upper() in COMPREHENDED else 1
                tab[id2cat[r["id"]]][comp] += 1
        n = sum(sum(v) for v in tab.values())
        if n < 10:
            continue
        chi, rows, _ = chi2_table(tab)
        df = rows - 1
        p = chi2_sf(chi, df)
        sig = p <= 0.05
        n_sig += sig
        print(f"{model:22s} {n:4d} {df:3d} {chi:6.1f} {p:7.3f}  {'NO (SIG)' if sig else 'yes'}")
    print(f"\nModels with significant category x comprehension association: {n_sig}/10")
    print("-> 0 significant => comprehension does not select by harm category; confound not supported.")


if __name__ == "__main__":
    main()
