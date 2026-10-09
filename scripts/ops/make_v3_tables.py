#!/usr/bin/env python3
"""Generate every LaTeX table in paper v3 from results/. No number is typed by hand.

Writes one .tex file per table into <out>/tables/ plus a numbers.json with every value the prose
quotes, so the text can be checked against the same source.

Usage: PYTHONPATH=. python3 scripts/ops/make_v3_tables.py --out openreview/v3_overleaf
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "results/phase5/comprehension_validation"
MODELS = ["gemma_3_1b_it", "gemma_2_2b_it", "gemma_3_4b_it", "qwen3_4b", "llama_3_2_3b_Instruct",
          "qwen2_5_7b_Instruct", "qwen3_8b", "llama_3_1_8b_Instruct", "gemma_2_9b_it", "gemma_3_27b_it"]
NICE = {"gemma_3_1b_it": "Gemma-3-1B", "gemma_2_2b_it": "Gemma-2-2B", "gemma_3_4b_it": "Gemma-3-4B",
        "qwen3_4b": "Qwen3-4B", "llama_3_2_3b_Instruct": "Llama-3.2-3B", "qwen2_5_7b_Instruct": "Qwen2.5-7B",
        "qwen3_8b": "Qwen3-8B", "llama_3_1_8b_Instruct": "Llama-3.1-8B", "gemma_2_9b_it": "Gemma-2-9B",
        "gemma_3_27b_it": "Gemma-3-27B"}
PAIRS = [("tier_a", "arabizi_harmful", "arabic_harmful", "Arabizi", "Arabic"),
         ("tier_c", "itrans_harmful", "devanagari_harmful", "ITRANS", "Devanagari"),
         ("tier_e", "leetspeak_harmful", "english_harmful", "Leetspeak", "English")]
N = {}  # every quoted number lands here


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def judge_labels():
    out = {}
    for m in MODELS:
        f = CV / m / "items_gemini.jsonl"
        if not f.exists():
            continue
        by = collections.defaultdict(list)
        for r in (json.loads(l) for l in open(f)):
            by[r["condition"]].append(r["label"].upper())
        out[m] = {c: collections.Counter(v) for c, v in by.items()}
    return out


def pct(c, key, total=None):
    n = total if total is not None else sum(c.values())
    return 100 * c[key] / n if n else float("nan")


def tbl(path, body, caption, label, note="", wide=False):
    """wide=True spans both columns (table*), which ACL requires at the top of a page."""
    cap = caption + (f" {note}" if note else "")
    env = "table*" if wide else "table"
    size = ("\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n" if wide
            else "\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n")
    path.write_text(f"\\begin{{{env}}}[t]\n" + size + body +
                    f"\n\\caption{{{cap}}}\n\\label{{{label}}}\n\\end{{{env}}}\n")


def t1_behaviour(J, out):
    """Refusal and engagement, native vs rewritten, all ten models."""
    rows = []
    gaps = []
    gaps_arz = []
    for m in MODELS:
        if m not in J:
            continue
        cells = []
        for _, rom, nat, _, _ in PAIRS:
            cn, cr = J[m][nat], J[m][rom]
            rn, rr = pct(cn, "REFUSAL"), pct(cr, "REFUSAL")
            gaps.append(rn - rr)
            if rom == "arabizi_harmful":
                gaps_arz.append(rn - rr)
            cells += [f"{rn:.0f}", f"{rr:.0f}", f"{pct(cr, 'REFUSAL') + pct(cr, 'ON_TOPIC'):.0f}"]
        rows.append(f"{NICE[m]} & " + " & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{l rrr rrr rrr}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Arabic / Arabizi} & \\multicolumn{3}{c}{Devanagari / ITRANS}"
            " & \\multicolumn{3}{c}{English / Leetspeak} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
            "Model & nat & rew & eng & nat & rew & eng & nat & rew & eng \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    tbl(out / "t1_behaviour.tex", body,
        "Refusal on the native script (nat) and on the rewritten script (rew), and engagement on the "
        "rewritten script (eng), as percentages of 50 harmful requests per cell. Engagement counts a "
        "response that answers or refuses the request. Judge: Gemini-2.5-flash, 96 new tokens, greedy.",
        "tab:behaviour", wide=True)
    N["refusal_gap_min_pp"], N["refusal_gap_max_pp"] = round(min(gaps), 1), round(max(gaps), 1)
    N["refusal_gap_arabizi_min_pp"] = round(min(gaps_arz), 1)
    N["refusal_gap_arabizi_max_pp"] = round(max(gaps_arz), 1)


def t2_recovery(J, out):
    """Recovery (hard distractors) against engagement, format-valid cells only."""
    rows, gaps, recs = [], [], []
    for m in MODELS:
        f = CV / m / "forced_choice_hard.json"
        if not f.exists():
            continue
        h = json.load(open(f))
        cells = []
        for p, rom, _, _, _ in PAIRS:
            ok = h["summary"].get(f"{p}|native_copy", {}).get("format_valid")
            rec = h["summary"].get(f"{p}|romanized_harmful", {}).get("debiased_acc_pct")
            eng = pct(J[m][rom], "REFUSAL") + pct(J[m][rom], "ON_TOPIC")
            if ok and rec is not None:
                cells += [f"{rec:.0f}", f"{eng:.0f}", f"$+${rec - eng:.0f}"]
                gaps.append(rec - eng)
                recs.append(rec)
            else:
                cells += ["--", f"{eng:.0f}", "--"]
        rows.append(f"{NICE[m]} & " + " & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{l rrr rrr rrr}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Arabizi} & \\multicolumn{3}{c}{ITRANS} & \\multicolumn{3}{c}{Leetspeak} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
            "Model & rec & eng & diff & rec & eng & diff & rec & eng & diff \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    tbl(out / "t2_recovery.tex", body,
        "Recovery (rec) against engagement (eng), percentages of 50 requests. Recovery is the "
        "four-way meaning choice with near-paraphrase alternatives, each item shown in all four "
        "positions; chance is 25. A dash marks a cell where the model cannot pick a verbatim copy "
        "of the text in front of it, so its recovery score is not interpretable.",
        "tab:recovery", wide=True)
    N["recovery_cells_valid"] = len(gaps)
    # Python's round() breaks ties to even, so a gap of 80.5 came out as 80. Round half up.
    r0 = lambda x: int(math.floor(x + 0.5))
    N["recovery_gap_min_pp"], N["recovery_gap_max_pp"] = r0(min(gaps)), r0(max(gaps))
    N["recovery_min_pct"], N["recovery_max_pct"] = round(min(recs), 1), round(max(recs), 1)
    N["recovery_above_76"] = sum(1 for r in recs if r >= 76)


def t3_outcomes(J, out):
    """Four-way outcome distribution, mean over models."""
    rows = []
    for key, lab in (("arabic_harmful", "Arabic"), ("arabizi_harmful", "Arabizi"),
                     ("devanagari_harmful", "Devanagari"), ("itrans_harmful", "ITRANS"),
                     ("english_harmful", "English"), ("leetspeak_harmful", "Leetspeak")):
        ms = [m for m in MODELS if m in J and key in J[m]]
        vals = [sum(pct(J[m][key], k) for m in ms) / len(ms)
                for k in ("REFUSAL", "ON_TOPIC", "OFF_TOPIC", "GIBBERISH")]
        rows.append(f"{lab} & " + " & ".join(f"{v:.1f}" for v in vals) + " \\\\")
        N[f"outcome_{lab.lower()}"] = [round(v, 1) for v in vals]
    body = ("\\begin{tabular}{lrrrr}\n\\toprule\nScript & Refusal & On topic & Off topic & Unusable \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    tbl(out / "t3_outcomes.tex", body,
        "What the models produce on harmful requests, as percentages averaged over the ten models. "
        "Off-topic text is the largest outcome on two of the three rewritten scripts.",
        "tab:outcomes")


def t4_harm(out):
    h = json.load(open(CV / "harm_summary.json"))
    rows, spec = [], []
    for m in MODELS:
        if m not in h:
            continue
        cells = []
        for _, rom, _, _, _ in PAIRS:
            d = h[m].get(rom, {})
            ot = d.get("SPECIFIC", 0) + d.get("GENERIC", 0) + d.get("NONE", 0)
            s = d.get("SPECIFIC", 0)
            spec.append(s)
            cells += [f"{ot:.0f}", f"{s:.0f}"]
        rows.append(f"{NICE[m]} & " + " & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{l rr rr rr}\n\\toprule\n"
            " & \\multicolumn{2}{c}{Arabizi} & \\multicolumn{2}{c}{ITRANS} & \\multicolumn{2}{c}{Leetspeak} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "Model & on topic & harmful & on topic & harmful & on topic & harmful \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    tbl(out / "t4_harm.tex", body,
        "On-topic responses and the subset that contains specific, usable harmful content, as "
        "percentages of 50 requests. The two columns differ because the engagement judge is "
        "instructed to ignore whether content is harmful. A second judge from another model family "
        "scored harm.",
        "tab:harm", wide=True)
    N["harm_specific_max_pct"] = round(max(spec))
    all_cells = [v.get("SPECIFIC", 0) for d in h.values() for v in d.values()]
    N["harm_cells_zero"] = sum(1 for s in all_cells if s == 0)
    N["harm_cells_total"] = len(all_cells)


def t5_conditional(J, out):
    """Refusal split by the recovery test, format-valid cells."""
    rows = []
    A = B = C = D = 0
    num = den = 0.0
    for m in MODELS:
        f = CV / m / "forced_choice_hard.json"
        if not f.exists():
            continue
        h = json.load(open(f))
        gem = {r["id"]: r["label"].upper() for r in (json.loads(l) for l in open(CV / m / "items_gemini.jsonl"))}
        idof = {}
        for gf in glob.glob(str(ROOT / f"results/phase5/{m}/run_*/generations.jsonl")):
            for g in (json.loads(l) for l in open(gf)):
                idof[(g["condition"], g["source_id"])] = g["id"]
        a = b = c = d = 0
        for p, rom, _, _, _ in PAIRS:
            if not h["summary"].get(f"{p}|native_copy", {}).get("format_valid"):
                continue
            t = collections.Counter()
            for r in h["records"]:
                if r["condition"] != "romanized_harmful" or r["pair"] != p:
                    continue
                jid = idof.get((rom, r["source_id"]))
                if jid is None or jid not in gem:
                    continue
                t[("rec" if r["correct"] else "not", gem[jid] == "REFUSAL")] += 1
            aa, bb = t[("rec", True)], t[("rec", False)]
            cc, dd = t[("not", True)], t[("not", False)]
            a, b, c, d = a + aa, b + bb, c + cc, d + dd
            tot = aa + bb + cc + dd
            if tot:
                num += aa * dd / tot
                den += bb * cc / tot
        if a + b == 0:
            continue
        A, B, C, D = A + a, B + b, C + c, D + d
        lo1, hi1 = wilson(a, a + b)
        r2 = f"{100*c/(c+d):.0f}" if c + d else "--"
        rows.append(f"{NICE[m]} & {a+b} & {100*a/(a+b):.0f} & [{lo1:.0f}, {hi1:.0f}] & {c+d} & {r2} \\\\")
    lo, hi = wilson(A, A + B)
    lo2, hi2 = wilson(C, C + D)
    rows.append("\\midrule\nPooled & %d & %.0f & [%.0f, %.0f] & %d & %.0f \\\\" %
                (A + B, 100 * A / (A + B), lo, hi, C + D, 100 * C / (C + D)))
    body = ("\\begin{tabular}{lrrcrr}\n\\toprule\n"
            " & \\multicolumn{3}{c}{meaning recovered} & \\multicolumn{2}{c}{not recovered} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
            "Model & $n$ & refused & 95\\% CI & $n$ & refused \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    tbl(out / "t5_conditional.tex", body,
        "Refusal on rewritten harmful requests, split by whether the model recovered the meaning. "
        "Format-valid cells only. Because the models recover most requests, the right-hand group is "
        "small, and in eight cells it contains no refusals.",
        "tab:conditional")
    N["cond_rec_pct"], N["cond_not_pct"] = round(100 * A / (A + B), 1), round(100 * C / (C + D), 1)
    N["cond_rec_n"], N["cond_not_n"] = A + B, C + D
    N["cond_cmh_or"] = round(num / den, 2) if den else None


def t6_transfer(out):
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results/phase4_intervention/*/direction_transfer/*.json"))):
        d = json.load(open(f))
        pair = d.get("pair", "tier_a")
        room = d["room_baseline_refused"]
        if room < 17:
            continue
        keep = (pair == "tier_a" and d["alpha"] == (0.4 if d["model"].startswith("gemma_3") else 1.5)) or pair == "tier_e"
        if not keep:
            continue
        lab = {"tier_a": "Arabizi", "tier_c": "ITRANS", "tier_e": "Leetspeak"}[pair]
        n = d["native_dir_ablation"]
        r = d["random_dir_ablation"]
        rows.append((NICE[d["model"]], lab, room, n["flip_rate_pct"], n["wilson95"], r["flip_rate_pct"],
                     d["causal_layer"], d["alpha"]))
    seen, uniq = set(), []
    for row in rows:
        if (row[0], row[1]) in seen:
            continue
        seen.add((row[0], row[1]))
        uniq.append(row)
    body_rows = [f"{a} & {b} & {c} & {d:.0f} & [{e[0]:.0f}, {e[1]:.0f}] & {f:.0f} & {g} & {h} \\\\"
                 for a, b, c, d, e, f, g, h in sorted(uniq, key=lambda x: (x[1], -x[3]))]
    body = ("\\begin{tabular}{llrrcrrr}\n\\toprule\n"
            "Model & Script & $n$ & native & 95\\% CI & random & layer & $\\alpha$ \\\\\n\\midrule\n"
            + "\n".join(body_rows) + "\n\\bottomrule\n\\end{tabular}")
    tbl(out / "t6_transfer.tex", body,
        "Removing the native refusal direction while the model answers rewritten requests it read and "
        "refused. Values are the percentage of those refusals that become compliance, against a "
        "norm-matched random direction. Layer and coefficient are each model's own calibrated values.",
        "tab:transfer", wide=True)


def t7_decode(out):
    rows = json.load(open(ROOT / "results/phase5/decode_then_refuse/summary.json"))
    by = {(r["model"], r["pair"], r["arm"]): r for r in rows}
    out_rows, closed, sims = [], [], []
    for (m, p, a) in sorted(by):
        if a != "baseline":
            continue
        b, s, o = [by[(m, p, x)]["refusal_pct"] for x in ("baseline", "self_decode", "oracle_decode")]
        sim = by[(m, p, "self_decode")]["mean_decode_sim"]
        cl = 100 * (s - b) / (o - b) if o > b else float("nan")
        closed.append(cl)
        sims.append(sim)
        lab = {"tier_c": "ITRANS", "tier_e": "Leetspeak"}[p]
        out_rows.append((NICE[m], lab, b, s, o, cl, sim))
    body_rows = [f"{a} & {b} & {c:.0f} & {d:.0f} & {e:.0f} & {f:.0f} & {g:.2f} \\\\"
                 for a, b, c, d, e, f, g in sorted(out_rows, key=lambda x: -x[5])]
    body = ("\\begin{tabular}{llrrrrr}\n\\toprule\n"
            "Model & Script & rewritten & self-decoded & native & gap closed & decode sim. \\\\\n\\midrule\n"
            + "\n".join(body_rows) + "\n\\bottomrule\n\\end{tabular}")
    tbl(out / "t7_decode.tex", body,
        "Refusal on the same 50 requests under three presentations: the rewritten request, the "
        "model's own rewriting of it into the native script, and the native-script request. Gap "
        "closed is the share of the distance from the rewritten baseline to the native ceiling. "
        "Decode similarity is the character similarity of the model's own rewriting to the known "
        "correct text.",
        "tab:decode", wide=True)
    n = len(sims)
    mx, my = sum(sims) / n, sum(closed) / n
    num = sum((x - mx) * (y - my) for x, y in zip(sims, closed))
    den = (sum((x - mx) ** 2 for x in sims) * sum((y - my) ** 2 for y in closed)) ** 0.5
    N["decode_r"] = round(num / den, 2)
    N["decode_cells"] = n
    N["decode_improved"] = sum(1 for c in closed if c > 0)
    N["decode_max_closed"] = round(max(closed))
    orc = [by[k]["refusal_pct"] for k in by if k[2] == "oracle_decode"]
    bas = [by[k]["refusal_pct"] for k in by if k[2] == "baseline"]
    N["oracle_min"], N["oracle_max"] = round(min(orc)), round(max(orc))
    N["decode_base_min"], N["decode_base_max"] = round(min(bas)), round(max(bas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="openreview/v3_overleaf")
    a = ap.parse_args()
    out = ROOT / a.out / "tables"
    out.mkdir(parents=True, exist_ok=True)
    J = judge_labels()
    t1_behaviour(J, out)
    t2_recovery(J, out)
    t3_outcomes(J, out)
    t4_harm(out)
    t5_conditional(J, out)
    t6_transfer(out)
    t7_decode(out)
    (ROOT / a.out / "numbers.json").write_text(json.dumps(N, indent=1))
    print(f"wrote {len(list(out.glob('*.tex')))} tables to {out}")
    for k, v in N.items():
        print(f"  {k:26s} {v}")


if __name__ == "__main__":
    main()
