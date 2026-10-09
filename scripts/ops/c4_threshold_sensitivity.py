#!/usr/bin/env python3
"""C4 threshold sensitivity (reviewer dL1G): how do the paper's verdicts move when each threshold moves?

Reads existing results only (no model, no API). Three thresholds:

1. Comprehension gate (judge labels, items_gemini.jsonl). A romanized cell passes when its
   comprehension (ON_TOPIC + REFUSAL) is >= FLOOR and its drop from the native cell is <= DROP.
   Sweep FLOOR in {50, 60, 70} and DROP in {10, 15, 20}. Report which cells pass, and whether the
   three anchoring cells named in the paper survive.

2. Activation-patching verdict (patching_results.json, latest run per model x pair).
   matched-minus-random gap in pp, computed two ways:
     - at the layer where matched recovery peaks (the definition the paper's table used)
     - the maximum over all layers (a stricter reading)
   Classified with the registered three-way rule in 03_plot_recover_curves.py
   (REJECT < 10, INDETERMINATE 10 to 25, CONFIRM >= 25), and with single bands 10/15/20/25/30.
   Also split by room >= 20 vs < 20, since small-room cells are noisy.

3. Concept-probe anchor floor (logit_lens.json, English-anchored Leetspeak N=200; and the
   native-anchor runs). Sweep the native-anchor validity floor {0.20, 0.30, 0.40} and the collapse
   rule (native >= {0.8, 0.9} and romanized < {0.4, 0.5, 0.6}).

Output: results/sensitivity/threshold_sensitivity.json and .md
Usage: PYTHONPATH=. python3 scripts/ops/c4_threshold_sensitivity.py
"""
from __future__ import annotations

import collections
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "sensitivity"
CV = ROOT / "results" / "phase5" / "comprehension_validation"
NATIVE = {"arabizi_harmful": "arabic_harmful", "leetspeak_harmful": "english_harmful", "itrans_harmful": "devanagari_harmful"}
PAPER_ANCHOR_CELLS = {("gemma_2_9b_it", "leetspeak_harmful"), ("llama_3_2_3b_Instruct", "leetspeak_harmful"),
                      ("gemma_3_27b_it", "itrans_harmful")}
ANCHOR_NATIVE_FILTER = 84.0  # the paper's added native-comprehension filter for clean anchoring cells


def comprehension_table():
    tab = {}
    for f in sorted(CV.glob("*/items_gemini.jsonl")):
        model = f.parent.name
        by = collections.defaultdict(list)
        for line in open(f):
            r = json.loads(line)
            by[r["condition"]].append(r["label"].upper())
        for cond, labs in by.items():
            n = len(labs)
            tab[(model, cond)] = round(100 * sum(l in ("ON_TOPIC", "REFUSAL") for l in labs) / n, 1)
    return tab


def gate_sweep(comp):
    out = []
    for floor in (50, 60, 70):
        for drop in (10, 15, 20):
            passing = []
            for (model, cond), c in comp.items():
                if cond not in NATIVE:
                    continue
                nat = comp.get((model, NATIVE[cond]))
                if nat is None:
                    continue
                if c >= floor and (nat - c) <= drop:
                    passing.append((model, cond, c, nat))
            clean = {(m, c) for m, c, _, nat in passing if nat >= ANCHOR_NATIVE_FILTER}
            out.append({
                "floor": floor, "max_drop_pp": drop, "n_pass": len(passing),
                "passing": [f"{m} x {c.split('_')[0]} ({cc:.0f}% vs native {nn:.0f}%)" for m, c, cc, nn in sorted(passing)],
                "paper_anchor_cells_surviving": sorted(f"{m} x {c.split('_')[0]}" for m, c in PAPER_ANCHOR_CELLS & clean),
            })
    return out


def patching_sweep():
    latest = {}
    for f in sorted(glob.glob(str(ROOT / "results/phase4_patching/*/run_*/patching_results.json"))):
        d = json.load(open(f))
        k = (d["model_key"], d["pair"])
        if k not in latest or f > latest[k][0]:
            latest[k] = (f, d)
    cells = []
    for (model, pair), (f, d) in sorted(latest.items()):
        m, r = d.get("recover_curve_A2B") or {}, d.get("controls_random") or {}
        if not m:
            continue
        pk = str(d["recover_peak_layer"])
        at_peak = round((m[pk] - r.get(pk, 0.0)) * 100, 1)
        per_layer = {L: (m[L] - r.get(L, 0.0)) * 100 for L in m}
        bestL = max(per_layer, key=per_layer.get)
        cells.append({"model": model, "pair": pair, "room": d.get("n_room"), "peak_layer": int(pk),
                      "gap_at_peak_pp": at_peak, "max_gap_layer": int(bestL),
                      "max_gap_pp": round(per_layer[bestL], 1), "run": os.path.basename(os.path.dirname(f))})

    def rule(g):
        return "REJECT" if g < 10 else ("INDETERMINATE" if g < 25 else "CONFIRM")

    summary = {}
    for metric in ("gap_at_peak_pp", "max_gap_pp"):
        for subset, sel in (("all", lambda c: True), ("room>=20", lambda c: (c["room"] or 0) >= 20),
                            ("room<20", lambda c: (c["room"] or 0) < 20)):
            cs = [c for c in cells if sel(c)]
            summary[f"{metric}|{subset}"] = {
                "n": len(cs),
                "registered_rule": dict(collections.Counter(rule(c[metric]) for c in cs)),
                "cells_at_or_above": {b: sum(c[metric] >= b for c in cs) for b in (10, 15, 20, 25, 30)},
                "max": max((c[metric] for c in cs), default=None),
            }
    return cells, summary


def lens_sweep():
    """English-anchored Leetspeak N=200 and native-anchor runs: validity floor and collapse rule."""
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results/phase4_patching/*/logitlens_*/logit_lens.json"))):
        d = json.load(open(f))
        if d.get("n_pairs") != 200:
            continue
        s = d.get("summary", {})
        n = len(d.get("records", [])) or 1
        nat = int(str(s.get("native_concept_emerged_count", "0/1")).split("/")[0]) / n
        var = s.get("concept_present_in_variant_frac_anchored", s.get("concept_present_in_variant_frac"))
        rows.append({"model": d["model_key"], "pair": d["pair"], "anchor": "english", "native": round(nat, 3),
                     "romanized": var, "file": os.path.relpath(f, ROOT)})
    for f in sorted(glob.glob(str(ROOT / "results/phase4_patching/*/nativeanchor_*/logit_lens.json"))):
        d = json.load(open(f))
        s = d.get("summary", {})
        n = len(d.get("records", [])) or 1
        nat = int(str(s.get("native_concept_emerged_count", "0/1")).split("/")[0]) / n
        var = s.get("concept_present_in_variant_frac_anchored", s.get("concept_present_in_variant_frac"))
        rows.append({"model": d["model_key"], "pair": d["pair"], "anchor": "native", "native": round(nat, 3),
                     "romanized": var, "file": os.path.relpath(f, ROOT)})
    # keep latest per (model, pair, anchor)
    latest = {}
    for r in rows:
        k = (r["model"], r["pair"], r["anchor"])
        if k not in latest or r["file"] > latest[k]["file"]:
            latest[k] = r
    rows = list(latest.values())
    floors = {}
    for fl in (0.20, 0.30, 0.40):
        floors[fl] = sorted(f"{r['model']}" for r in rows if r["anchor"] == "english" and r["pair"] == "tier_e" and r["native"] >= fl)
    collapse = {}
    for nf in (0.8, 0.9):
        for rc in (0.4, 0.5, 0.6):
            cs = [r for r in rows if r["anchor"] == "native" and r["romanized"] is not None and r["native"] >= nf and r["romanized"] < rc]
            collapse[f"native>={nf},romanized<{rc}"] = {"n_cells": len(cs), "n_models": len({r['model'] for r in cs}),
                                                       "cells": sorted(f"{r['model']} x {r['pair']}" for r in cs)}
    return rows, floors, collapse


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comp = comprehension_table()
    gate = gate_sweep(comp)
    cells, psum = patching_sweep()
    lens_rows, floors, collapse = lens_sweep()
    res = {"gate": gate, "patching_cells": cells, "patching_summary": psum,
           "lens_rows": lens_rows, "lens_validity_floor": floors, "native_anchor_collapse": collapse,
           "notes": ["Patching thresholds: registered three-way rule from 03_plot_recover_curves.py "
                     "(CONFIRM_PP=25, REJECT_PP=10). The submitted paper describes only the 25 pp bound.",
                     "Gate: the paper's rule is floor 60 and drop <= 15; the anchoring cells also need native >= 84."]}
    (OUT / "threshold_sensitivity.json").write_text(json.dumps(res, indent=1))

    md = ["# Threshold sensitivity (C4)", "", "Generated by `scripts/ops/c4_threshold_sensitivity.py` from existing results. No new runs.", ""]
    md += ["## 1. Comprehension gate", "", "| floor | max drop | cells passing | paper anchoring cells that survive (of 3) |", "|---|---|---|---|"]
    for g in gate:
        md.append(f"| {g['floor']}% | {g['max_drop_pp']} pp | {g['n_pass']} | {len(g['paper_anchor_cells_surviving'])}: {', '.join(g['paper_anchor_cells_surviving']) or 'none'} |")
    md += ["", "## 2. Activation patching, matched minus random", "",
           "Registered rule: REJECT < 10 pp, INDETERMINATE 10 to 25, CONFIRM >= 25.", "",
           "| gap measured | cells | n | REJECT | INDETERMINATE | CONFIRM | max gap |", "|---|---|---|---|---|---|---|"]
    for key, v in psum.items():
        metric, subset = key.split("|")
        rr = v["registered_rule"]
        md.append(f"| {'at peak layer' if metric == 'gap_at_peak_pp' else 'max over layers'} | {subset} | {v['n']} | {rr.get('REJECT', 0)} | {rr.get('INDETERMINATE', 0)} | {rr.get('CONFIRM', 0)} | {v['max']} |")
    md += ["", "Cells not REJECT under either measure:", "", "| model | pair | room | gap at peak | max gap (layer) |", "|---|---|---|---|---|"]
    for c in cells:
        if c["gap_at_peak_pp"] >= 10 or c["max_gap_pp"] >= 10:
            md.append(f"| {c['model']} | {c['pair']} | {c['room']} | {c['gap_at_peak_pp']} | {c['max_gap_pp']} (L{c['max_gap_layer']}) |")
    md += ["", "## 3. Concept probe", "", "English-anchored Leetspeak (N = 200): models whose native anchor passes the validity floor.", "",
           "| floor | models valid |", "|---|---|"]
    for fl, ms in floors.items():
        md.append(f"| {fl:.2f} | {len(ms)}: {', '.join(ms)} |")
    md += ["", "Native-anchored probe: cells counted as collapse.", "", "| rule | cells | models |", "|---|---|---|"]
    for k, v in collapse.items():
        md.append(f"| {k} | {v['n_cells']} | {v['n_models']} |")
    (OUT / "threshold_sensitivity.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
