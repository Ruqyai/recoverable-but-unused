#!/usr/bin/env python3
"""
conditional_refusal_analysis.py
================================
Per-item conditional analysis: is refusal concentrated in the
judged-COMPREHENDED subset of prompts?  This is the within-model causal
test that turns the cross-model dose-response into an item-level claim:

    P(strict_refusal | engaged)  vs  P(strict_refusal | not engaged)

per (model x condition) cell, plus a pooled per-model estimate over the
romanized conditions (Cochran-Mantel-Haenszel common odds ratio, and a
logistic regression refusal ~ engaged + C(condition) if statsmodels is
installed).

It reads ONLY existing per-item labels from the results/ tree -- no new
generation, no new judge calls.

Expected inputs (flexible; see FIELD_CANDIDATES below):
  * judge-label records:   one record per (model, condition, item) with a
    4-way label in {ON_TOPIC, REFUSAL, OFF_TOPIC, GIBBERISH}
  * (optional) behavior records: one record per (model, condition, item)
    with a strict-refusal label in {refusal, compliance, off_topic, gibberish}
    If absent, strict_refusal is derived from judge label == REFUSAL
    (flagged in the output, since the paper's Appendix A defines the two
    to coincide on engaged items).

Records may be JSON (list of dicts), JSONL, or CSV. Field names are
auto-detected from the candidate lists; extend FIELD_CANDIDATES if your
schema uses different keys.

Usage:
  python3 conditional_refusal_analysis.py --results-dir results/ \
      --judge-glob "**/judge*.json*" --behavior-glob "**/behav*.json*" \
      --out-dir analysis_out/
  python3 conditional_refusal_analysis.py --demo     # synthetic smoke test

Outputs (in --out-dir):
  conditional_cells.csv     per-cell 2x2 statistics with Wilson CIs
  conditional_pooled.csv    per-model pooled (CMH) over romanized cells
  conditional_table.tex     LaTeX table ready to drop into Sec. 5.1/6
  conditional_summary.md    human-readable headline summary
"""

import argparse, csv, glob, json, math, os, sys, random
from collections import defaultdict

# --------------------------------------------------------------------------
# Configuration: field-name candidates (first match wins), label maps
# --------------------------------------------------------------------------
FIELD_CANDIDATES = {
    "model":      ["model", "model_name", "model_id"],
    "condition":  ["condition", "script", "tier", "lang", "language", "cond"],
    "item_id":    ["item_id", "source_id", "prompt_id", "id", "idx", "index"],
    "judge":      ["judge_label", "gemini_label", "label", "engagement_label",
                   "judge", "gate_label"],
    "behavior":   ["behavior_label", "refusal_label", "strict_label",
                   "classification", "behavior", "outcome"],
}

ENGAGED_LABELS   = {"ON_TOPIC", "REFUSAL"}
JUDGE_LABELS     = {"ON_TOPIC", "REFUSAL", "OFF_TOPIC", "GIBBERISH"}
REFUSAL_BEHAVIOR = {"REFUSAL", "REFUSE", "REFUSED"}
ROMANIZED_CONDS  = {"ARABIZI_HARMFUL", "ITRANS_HARMFUL", "LEETSPEAK_HARMFUL",
                    "ARZ", "ARABIZI", "ITR", "ITRANS", "LEET", "LEETSPEAK"}
# The keyword/heuristic behavior classifier is blind to refusals written in
# Devanagari (verified: gemma-3-27b ITRANS heuristic_refusal=3 vs judge
# REFUSAL=44). Any 2x2 that crosses a HINDI cond with heuristic labels is
# invalid; pooled stats are therefore ALSO reported excluding these strata.
HINDI_CONDS      = {"ITRANS_HARMFUL", "ITR", "ITRANS"}

# --------------------------------------------------------------------------
# Small stats helpers (no hard dependencies)
# --------------------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def fisher_exact_two_sided(a, b, c, d):
    """2x2 Fisher exact, two-sided by summing tables with prob <= observed."""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    def logp(x):
        return (math.lgamma(r1 + 1) - math.lgamma(x + 1) - math.lgamma(r1 - x + 1)
                + math.lgamma(n - r1 + 1) - math.lgamma(c1 - x + 1)
                - math.lgamma(n - r1 - c1 + x + 1)
                - (math.lgamma(n + 1) - math.lgamma(c1 + 1) - math.lgamma(n - c1 + 1)))
    p_obs = logp(a)
    tot = 0.0
    for x in range(lo, hi + 1):
        px = logp(x)
        if px <= p_obs + 1e-9:
            tot += math.exp(px)
    return min(1.0, tot)

def fmt_p(p):
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"

def odds_ratio(a, b, c, d):
    """Haldane-Anscombe corrected OR: (refusal&engaged, refusal&not,
    no-refusal&engaged, no-refusal&not)."""
    a2, b2, c2, d2 = a + .5, b + .5, c + .5, d + .5
    return (a2 * d2) / (b2 * c2)

def cmh_common_or(tables):
    """Cochran-Mantel-Haenszel common OR across strata of 2x2 tables
    [(a,b,c,d), ...] with a=refusal&engaged, b=refusal&not-engaged,
    c=noref&engaged, d=noref&not-engaged."""
    num = den = 0.0
    for (a, b, c, d) in tables:
        n = a + b + c + d
        if n == 0:
            continue
        num += a * d / n
        den += b * c / n
    if den == 0:
        return float("inf") if num > 0 else float("nan")
    return num / den

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _iter_records(path):
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f)
    elif path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:  # .json
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            yield from obj
        elif isinstance(obj, dict):
            # common shapes: {"records":[...]} or {model:{cond:[...]}}
            if "records" in obj and isinstance(obj["records"], list):
                yield from obj["records"]
            else:
                for v in obj.values():
                    if isinstance(v, list):
                        yield from v
                    elif isinstance(v, dict):
                        for vv in v.values():
                            if isinstance(vv, list):
                                yield from vv

def _pick(rec, keys):
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    # case-insensitive fallback
    low = {kk.lower(): kk for kk in rec}
    for k in keys:
        if k.lower() in low and rec[low[k.lower()]] not in (None, ""):
            return rec[low[k.lower()]]
    return None

def load_labels(results_dir, pattern, kind):
    """kind in {'judge','behavior'} -> dict[(model,cond,item)] = LABEL"""
    out, files = {}, sorted(glob.glob(os.path.join(results_dir, pattern),
                                      recursive=True))
    for path in files:
        try:
            for rec in _iter_records(path):
                m = _pick(rec, FIELD_CANDIDATES["model"])
                c = _pick(rec, FIELD_CANDIDATES["condition"])
                i = _pick(rec, FIELD_CANDIDATES["item_id"])
                lab = _pick(rec, FIELD_CANDIDATES[kind])
                if None in (m, c, i, lab):
                    continue
                key = (str(m).strip(), str(c).strip().upper(), str(i).strip())
                out[key] = str(lab).strip().upper().replace("-", "_")
        except Exception as e:
            print(f"[warn] skipping {path}: {e}", file=sys.stderr)
    print(f"[info] {kind}: {len(out)} labeled items from {len(files)} file(s)")
    return out

# --------------------------------------------------------------------------
# Demo data (synthetic, for smoke-testing the pipeline & output format)
# --------------------------------------------------------------------------
def make_demo():
    random.seed(42)
    judge, behav = {}, {}
    # mimic a partial-comprehension cell (70% engaged) where refusal is
    # concentrated in the engaged subset, plus a low-comprehension cell.
    cells = [
        ("Gemma-3-1B-it", "ARZ", 0.70, 0.85, 0.05),  # p_eng, p_ref|eng, p_ref|not
        ("Gemma-3-1B-it", "ITR", 0.62, 0.55, 0.02),
        ("Qwen3-4B",      "ARZ", 0.20, 0.20, 0.00),
        ("Qwen3-4B",      "ARA", 0.98, 0.92, 0.00),
    ]
    for model, cond, pe, pre, prn in cells:
        for i in range(50):
            eng = random.random() < pe
            ref = random.random() < (pre if eng else prn)
            if ref:
                j = "REFUSAL"
            elif eng:
                j = "ON_TOPIC"
            else:
                j = "OFF_TOPIC" if random.random() < .9 else "GIBBERISH"
            judge[(model, cond, str(i))] = j
            behav[(model, cond, str(i))] = "REFUSAL" if ref else "COMPLIANCE"
    return judge, behav

# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def analyze(judge, behav, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    derived = False
    if not behav:
        derived = True
        behav = {k: ("REFUSAL" if v == "REFUSAL" else "OTHER")
                 for k, v in judge.items()}
        print("[warn] no behavior labels found -> strict_refusal derived "
              "from judge REFUSAL label (report this caveat).")

    cells = defaultdict(lambda: [0, 0, 0, 0])  # a,b,c,d per (model,cond)
    for key, jlab in judge.items():
        if jlab not in JUDGE_LABELS or key not in behav:
            continue
        eng = jlab in ENGAGED_LABELS
        ref = behav[key] in REFUSAL_BEHAVIOR
        t = cells[key[:2]]
        if ref and eng:        t[0] += 1
        elif ref and not eng:  t[1] += 1
        elif not ref and eng:  t[2] += 1
        else:                  t[3] += 1

    rows = []
    for (model, cond), (a, b, c, d) in sorted(cells.items()):
        n_eng, n_not = a + c, b + d
        p_e, p_n = (a / n_eng if n_eng else float("nan")), \
                   (b / n_not if n_not else float("nan"))
        lo_e, hi_e = wilson_ci(a, n_eng)
        lo_n, hi_n = wilson_ci(b, n_not)
        rows.append(dict(model=model, condition=cond, n=a + b + c + d,
                         n_engaged=n_eng, n_not_engaged=n_not,
                         refusal_given_engaged=round(p_e, 3),
                         ci_engaged=f"[{lo_e:.2f},{hi_e:.2f}]",
                         refusal_given_not=round(p_n, 3) if n_not else "NA",
                         ci_not=f"[{lo_n:.2f},{hi_n:.2f}]" if n_not else "NA",
                         odds_ratio=round(odds_ratio(a, b, c, d), 2),
                         fisher_p=fmt_p(fisher_exact_two_sided(a, b, c, d)),
                         a=a, b=b, c=c, d=d))

    with open(os.path.join(out_dir, "conditional_cells.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # pooled per model over romanized conditions (CMH)
    pooled = []
    by_model = defaultdict(list)
    for r in rows:
        if r["condition"] in ROMANIZED_CONDS:
            by_model[r["model"]].append((r["a"], r["b"], r["c"], r["d"]))
    by_model_nh = defaultdict(list)   # non-Hindi strata only (heuristic-safe)
    for r in rows:
        if r["condition"] in ROMANIZED_CONDS and r["condition"] not in HINDI_CONDS:
            by_model_nh[r["model"]].append((r["a"], r["b"], r["c"], r["d"]))
    for model, tabs in sorted(by_model.items()):
        A, B, C, D = map(sum, zip(*tabs))
        nh = by_model_nh.get(model, [])
        pooled.append(dict(model=model, strata=len(tabs),
                           cmh_common_OR=round(cmh_common_or(tabs), 2),
                           cmh_OR_excl_hindi=round(cmh_common_or(nh), 2) if nh else "NA",
                           pooled_ref_given_eng=round(A / (A + C), 3) if A + C else "NA",
                           pooled_ref_given_not=round(B / (B + D), 3) if B + D else "NA",
                           fisher_p_pooled=fmt_p(
                               fisher_exact_two_sided(A, B, C, D))))
    with open(os.path.join(out_dir, "conditional_pooled.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pooled[0].keys()))
        w.writeheader(); w.writerows(pooled)

    # optional logistic regression if statsmodels is present
    try:
        import pandas as pd, statsmodels.formula.api as smf  # noqa
        recs = []
        for key, jlab in judge.items():
            if key[:2][1] in ROMANIZED_CONDS and key in behav:
                recs.append(dict(model=key[0], condition=key[1],
                                 engaged=int(jlab in ENGAGED_LABELS),
                                 refusal=int(behav[key] in REFUSAL_BEHAVIOR)))
        df = pd.DataFrame(recs)
        lines = []
        for model, g in df.groupby("model"):
            try:
                fit = smf.logit("refusal ~ engaged + C(condition)",
                                data=g).fit(disp=0)
                lines.append(f"{model}: beta(engaged)="
                             f"{fit.params['engaged']:.2f} "
                             f"(p={fit.pvalues['engaged']:.3g})")
            except Exception as e:
                lines.append(f"{model}: logit failed ({e})")
        with open(os.path.join(out_dir, "logit_per_model.txt"), "w") as f:
            f.write("\n".join(lines))
    except ImportError:
        print("[info] statsmodels not installed -> skipping logistic "
              "regression (CMH ORs still computed).")

    # LaTeX snippet (romanized cells only)
    tex = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\begin{tabular}{llrrrr}", r"\toprule",
           r"Model & Cond. & $P(\text{ref}\,|\,\text{eng})$ & "
           r"$P(\text{ref}\,|\,\neg\text{eng})$ & OR & $p$ \\",
           r"\midrule"]
    for r in rows:
        if r["condition"] in ROMANIZED_CONDS:
            tex.append(f"{r['model']} & {r['condition'].title()} & "
                       f"{r['refusal_given_engaged']} & "
                       f"{r['refusal_given_not']} & "
                       f"{r['odds_ratio']} & {r['fisher_p']} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Per-item conditional analysis: strict refusal is "
            r"concentrated in the judged-comprehended (engaged) subset. "
            r"OR with Haldane--Anscombe correction; two-sided Fisher exact "
            r"$p$." + (" Strict refusal derived from judge label "
                       "(no separate behavior file found)." if derived else "")
            + r"}", r"\label{tab:conditional}", r"\end{table}"]
    with open(os.path.join(out_dir, "conditional_table.tex"), "w") as f:
        f.write("\n".join(tex))

    # markdown summary
    md = ["# Per-item conditional refusal analysis\n"]
    for r in rows:
        if r["condition"] in ROMANIZED_CONDS:
            md.append(f"- **{r['model']} x {r['condition']}** "
                      f"(n={r['n']}): P(ref|engaged)="
                      f"{r['refusal_given_engaged']} {r['ci_engaged']} vs "
                      f"P(ref|not)={r['refusal_given_not']} {r['ci_not']}, "
                      f"OR={r['odds_ratio']}, Fisher p={r['fisher_p']}")
    md.append("\n## Pooled (CMH over romanized conditions)\n")
    for p in pooled:
        md.append(f"- **{p['model']}**: common OR={p['cmh_common_OR']} "
                  f"across {p['strata']} strata; pooled "
                  f"P(ref|eng)={p['pooled_ref_given_eng']} vs "
                  f"P(ref|not)={p['pooled_ref_given_not']} "
                  f"(Fisher p={p['fisher_p_pooled']})")
    if derived:
        md.append("\n> **Caveat:** strict refusal derived from the judge "
                  "REFUSAL label; wire in the behavior classifier files "
                  "for the paper-grade version.")
    md.append("\n> **ITRANS caveat (do NOT skip):** the heuristic behavior "
              "classifier is blind to refusals written in Devanagari "
              "(verified: 3 heuristic vs 44 judge refusals on one model), so "
              "ITRANS cells computed from heuristic labels are INVALID — use "
              "`cmh_OR_excl_hindi` for the heuristic-based pooled claim and "
              "the multilingual judge labels for any ITRANS number.")
    with open(os.path.join(out_dir, "conditional_summary.md"), "w") as f:
        f.write("\n".join(md))
    print(f"[done] wrote outputs to {out_dir}/")
    return rows, pooled

# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--judge-glob", default="**/judge*.*")
    ap.add_argument("--behavior-glob", default="**/behav*.*")
    ap.add_argument("--out-dir",
                    default="results/phase5/comprehension_validation/conditional_analysis")
    ap.add_argument("--demo", action="store_true",
                    help="run on synthetic data to preview the outputs")
    args = ap.parse_args()

    if args.demo:
        judge, behav = make_demo()
    else:
        judge = load_labels(args.results_dir, args.judge_glob, "judge")
        behav = load_labels(args.results_dir, args.behavior_glob, "behavior")
        if not judge:
            sys.exit("[error] no judge labels found -- adjust --judge-glob "
                     "or FIELD_CANDIDATES to match your schema.")
    analyze(judge, behav, args.out_dir)

if __name__ == "__main__":
    main()
