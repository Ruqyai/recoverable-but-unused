#!/usr/bin/env python3
"""Stage 03 — agreement statistics across judges and humans.

Merges the per-item labels row-by-row and reports the numbers the reviewer asked
for. Everything is reported BOTH on the 4-label scheme and on the binary W1 gate
(COMPREHENDED = ON_TOPIC+REFUSAL vs NOT) — the binary collapse is the measure the
paper actually uses, so it is the headline kappa.

Computed when the inputs exist (each is skipped + noted if missing):
  * gemini ↔ claude     — the "second judge" robustness number (all judged items)
  * human  ↔ gemini      — grounds the Gemini judge against a fluent reader
  * human  ↔ claude      — grounds the Claude judge
  * english ↔ hindi      — INTER-HUMAN kappa on the shared English overlap (the
                           conflict-of-interest-neutralizing number)
  * Fleiss(gemini,claude,human) — joint agreement where all three labeled an item

Each block is reported overall, per-script (condition), and binary-vs-4-class.

Inputs:
  results/phase5/comprehension_validation/<model>/items_gemini.jsonl   (Stage 01 --provider gemini)
  results/phase5/comprehension_validation/<model>/items_claude.jsonl   (Stage 01 --provider anthropic)
  results/phase5/comprehension_validation/human_sheets/sheet_{arabic,english,hindi}.csv  (filled in)
  results/phase5/comprehension_validation/human_sheets/sheet_{arabic,english,hindi}_KEY.csv

Output:
  results/phase5/comprehension_validation/agreement_report.json  (+ console banner)

Usage
-----
  PYTHONPATH=. python3 scripts/phase5_sae_features/comprehension_validation/03_compute_agreement.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.phase5_sae_features.comprehension_validation import _common as C

SHEET_DIR = C.OUT_BASE / "human_sheets"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Agreement (Cohen/Fleiss kappa) across judges and humans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--models", nargs="*", default=None, help="Default: all with item files.")
    return ap.parse_args()


# ── load per-item LLM labels keyed by (model, id) ────────────────────────────
def load_llm(models: list[str], tag: str) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for m in models:
        for r in C.read_items(C.items_path(m, tag)):
            if r.get("label") and r["label"] != "ERROR":
                out[(r["model"], r["id"])] = r
    return out


# ── derive the single label from a filled row (two accepted sheet layouts) ────
def _row_label(row: dict, single_field: str | None) -> str | None:
    """Return the row's label, or None if unfilled/invalid.

    Two layouts are supported:
      (a) single column ``human_label*`` holding one of the four labels;
      (b) four Yes/No columns (ON_TOPIC/REFUSAL/OFF_TOPIC/GIBBERISH) — the label
          is the ONE marked 'yes'. Zero or >1 'yes' ⇒ invalid (skipped) so a
          multi-select or an unfilled row never silently miscounts.
    """
    if single_field is not None:
        lab = (row.get(single_field) or "").strip().upper()
        return lab if lab in C.VALID_LABELS else None
    yes = [lab for lab in C.VALID_LABELS if (row.get(lab) or "").strip().lower() == "yes"]
    return yes[0] if len(yes) == 1 else None


# ── load a filled human sheet: annot_id → {label, model, id, condition} ──────
def load_human(name: str) -> dict[str, dict]:
    sheet = SHEET_DIR / f"sheet_{name}.csv"
    key = SHEET_DIR / f"sheet_{name}_KEY.csv"
    if not sheet.exists() or not key.exists():
        return {}
    keymap: dict[str, dict] = {}
    with key.open() as fh:
        for row in csv.DictReader(fh):
            keymap[row["annot_id"]] = row
    out: dict[str, dict] = {}
    with sheet.open() as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        # layout (a): a single human_label column; else layout (b): four Yes/No columns
        single_field = next((f for f in fields if f.startswith("human_label")), None)
        if single_field is None and not (C.VALID_LABELS <= set(fields)):
            return {}  # neither layout present ⇒ sheet not filled yet
        for row in reader:
            lab = _row_label(row, single_field)
            if lab and row.get("annot_id") in keymap:
                k = keymap[row["annot_id"]]
                out[row["annot_id"]] = {
                    "label": lab, "model": k["model"], "id": k["id"], "condition": k["condition"],
                }
    return out


def _kappa_block(pairs: list[tuple[str, str, str]]) -> dict:
    """pairs = [(label_a, label_b, condition)]. Return 4-class + binary kappa, overall + per-script."""
    if not pairs:
        return {"n": 0}
    a4 = [p[0] for p in pairs]
    b4 = [p[1] for p in pairs]
    ab = [C.to_binary(p[0]) for p in pairs]
    bb = [C.to_binary(p[1]) for p in pairs]
    k4, po4 = C.cohen_kappa(a4, b4)
    kb, pob = C.cohen_kappa(ab, bb)
    per_script = {}
    by_cond: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for p in pairs:
        by_cond[p[2]].append(p)
    for cond, ps in sorted(by_cond.items()):
        kb_c, pob_c = C.cohen_kappa(
            [C.to_binary(x[0]) for x in ps], [C.to_binary(x[1]) for x in ps]
        )
        per_script[cond] = {"n": len(ps), "binary_kappa": _r(kb_c), "binary_agreement": _r(pob_c)}
    return {
        "n": len(pairs),
        "kappa_4class": _r(k4), "agreement_4class": _r(po4),
        "kappa_binary": _r(kb), "agreement_binary": _r(pob),
        "per_script": per_script,
    }


def _r(x: float) -> float | None:
    return None if x != x else round(x, 4)  # x!=x catches NaN


def main() -> int:
    args = parse_args()
    models = args.models or C.models_with_generations()
    gem = load_llm(models, "gemini")
    cla = load_llm(models, "claude")
    arabic = load_human("arabic")
    english = load_human("english")
    hindi = load_human("hindi")
    human_sets = (("arabic", arabic), ("english", english), ("hindi", hindi))

    report: dict = {
        "models": models,
        "counts": {
            "gemini_items": len(gem), "claude_items": len(cla),
            "human_arabic_labeled": len(arabic), "human_english_labeled": len(english),
            "human_hindi_labeled": len(hindi),
        },
        "comparisons": {},
        "note": (
            "Robustness check on the exploratory comprehension-gating measure. "
            "Binary W1 gate = COMPREHENDED (ON_TOPIC+REFUSAL) vs NOT. kappa_binary is the "
            "headline number. Inter-human kappa (english vs hindi on the shared English "
            "overlap) is an independent-annotator cross-check; report it alongside human↔LLM."
        ),
    }
    cmp = report["comparisons"]

    # 1) gemini ↔ claude (all items both judged)
    keys = set(gem) & set(cla)
    cmp["gemini_vs_claude"] = _kappa_block(
        [(gem[k]["label"], cla[k]["label"], gem[k]["condition"]) for k in sorted(keys)]
    )

    # 2/3) human ↔ each LLM, per language sheet
    for hname, hmap in human_sets:
        if not hmap:
            continue
        for jtag, jmap in (("gemini", gem), ("claude", cla)):
            pairs = [
                (h["label"], jmap[(h["model"], h["id"])]["label"], h["condition"])
                for h in hmap.values()
                if (h["model"], h["id"]) in jmap
            ]
            cmp[f"human_{hname}_vs_{jtag}"] = _kappa_block(pairs)

    # 4) inter-human (english ↔ hindi annotators on the shared English overlap)
    shared = set(english) & set(hindi)
    cmp["english_vs_hindi"] = _kappa_block(
        [(english[a]["label"], hindi[a]["label"], english[a]["condition"]) for a in sorted(shared)]
    )

    # 5) Fleiss(gemini, claude, human) on items all three labeled (per human set)
    for hname, hmap in human_sets:
        if not hmap:
            continue
        aligned_g, aligned_c, aligned_h = [], [], []
        for h in hmap.values():
            k = (h["model"], h["id"])
            if k in gem and k in cla:
                aligned_g.append(C.to_binary(gem[k]["label"]))
                aligned_c.append(C.to_binary(cla[k]["label"]))
                aligned_h.append(C.to_binary(h["label"]))
        cmp[f"fleiss_binary_{hname}"] = {
            "n": len(aligned_h),
            "fleiss_kappa": _r(C.fleiss_kappa([aligned_g, aligned_c, aligned_h])),
            "raters": ["gemini", "claude", f"human_{hname}"],
        }

    C.OUT_BASE.mkdir(parents=True, exist_ok=True)
    out_path = C.OUT_BASE / "agreement_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # ── console banner ───────────────────────────────────────────────────────
    print("=" * 72)
    print("COMPREHENSION-JUDGE AGREEMENT  (binary W1 gate = headline)")
    print(f"  items: gemini={len(gem)}  claude={len(cla)}  "
          f"human: arabic={len(arabic)} english={len(english)} hindi={len(hindi)}")
    print("-" * 72)
    print(f"  {'comparison':<26} {'n':>5} {'κ_bin':>8} {'agree':>7} {'κ_4cls':>8}")
    for name, blk in cmp.items():
        if "fleiss" in name:
            print(f"  {name:<26} {blk.get('n', 0):>5} {str(blk.get('fleiss_kappa')):>8}  (Fleiss)")
            continue
        if blk.get("n", 0) == 0:
            print(f"  {name:<26} {'—':>5}   (no data yet)")
            continue
        print(f"  {name:<26} {blk['n']:>5} {str(blk['kappa_binary']):>8} "
              f"{str(blk['agreement_binary']):>7} {str(blk['kappa_4class']):>8}")
    print("=" * 72)
    print(f"wrote {out_path}")
    if not arabic and not english and not hindi:
        print("NOTE: no filled human sheets found — only gemini↔claude is populated.")
        print("      Fill sheet_arabic.csv / sheet_english.csv / sheet_hindi.csv and re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
