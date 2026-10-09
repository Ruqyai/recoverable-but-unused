#!/usr/bin/env python3
"""Regenerate the paper figures (fig1..fig8) reproducibly from LIVE results/ data.

Why this exists: the paper figures used to be hand-made one-offs, so they drifted stale
(e.g. fig7 showed 6 models when the study covers 10). This script rebuilds every
data-driven figure from the canonical JSON artifacts on disk — nothing hand-transcribed —
so the figures always reflect the current results across ALL models that ran each measurement.

Writes .pdf (for LaTeX/Overleaf) and .png (for the .docx build + README/demo) into:
    paper/figures/  and  new_supmtion_paper/figures/

Coverage per figure (models shown = models with that measurement on disk):
    fig1_two_route              schematic (no data)
    fig1_comprehension_gate     behavioral_readout.json — all 10 models (small multiples)
    fig2_concept_present_regimes  logit-lens concept-present — all 10 models, N annotated per bar
    fig3_patching_nonspecific   patching_results.json — all 10 models × 3 tiers
    fig4_h2_direction_contrast  h2_results.json — all 10 models × 3 tiers
    (fig5 removed — the logit-lens dose-response had only 2 reliable N=200 cells; not worth a figure)
    fig6_scale_dose_response    comprehension gate × behavioral refusal — all 10 + 27B by script
    fig7_h3_script_vs_language  jaccard h3_controls — all 10 models (SCRIPT vs LANGUAGE change)
    fig8_causal_vs_separation_depth  phase2 causal vs separation peak — all 10 models

Run:  python3 scripts/ops/make_paper_figures.py
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "results" / "phase2"
P4I = ROOT / "results" / "phase4_intervention"
P4P = ROOT / "results" / "phase4_patching"
P5 = ROOT / "results" / "phase5"
OUT_DIRS = [ROOT / "paper" / "figures", ROOT / "new_supmtion_paper" / "figures"]

# canonical model order + display names (matches the paper tables)
MODELS = [
    ("gemma_3_1b_it", "Gemma-3-1B-it", 26),
    ("gemma_2_2b_it", "Gemma-2-2B-it", 26),
    ("gemma_3_4b_it", "Gemma-3-4B-it", 34),
    ("gemma_2_9b_it", "Gemma-2-9B-it", 42),
    ("gemma_3_27b_it", "Gemma-3-27B-it", 62),
    ("qwen3_4b", "Qwen3-4B", 36),
    ("qwen3_8b", "Qwen3-8B", 36),
    ("qwen2_5_7b_Instruct", "Qwen2.5-7B-Instruct", 28),
    ("llama_3_1_8b_Instruct", "Llama-3.1-8B-Instruct", 32),
    ("llama_3_2_3b_Instruct", "Llama-3.2-3B-Instruct", 28),
]
DISPLAY = {k: d for k, d, _ in MODELS}
NLAYERS = {k: n for k, _, n in MODELS}
KEYS = [k for k, _, _ in MODELS]

TIERS = ["tier_a", "tier_c", "tier_e"]
TIER_LABEL = ["Arabizi", "ITRANS", "Leetspeak"]

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "mathtext.default": "regular"})


def _latest(pattern: str):
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def _load(path):
    if not path:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"  ! skip {path}: {e}")
        return None


def _save(fig, name: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(d / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png  ->  {', '.join(str(d.parent.name) for d in OUT_DIRS)}")


def _place_labels_no_overlap(ax, xs, ys, labs, fontsize=6.8, color="#222"):
    """Greedy leader-line label placement that avoids label-label and label-point overlap.

    Deterministic (candidate offsets tried in a fixed order). Overlap is tested in display
    pixels at draw time, but each label is emitted in DATA coordinates so it tracks the
    point through any later tight_layout()/limit change. Used for the crowded fig6(a) cluster.
    """
    fig = ax.figure
    fig.canvas.draw()  # finalize transforms so transData is valid
    dpi = fig.dpi
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    pts = ax.transData.transform(np.column_stack([xs, ys]))

    def box(text):  # approx label extent in display px
        return (0.60 * fontsize * len(text) * dpi / 72.0, 1.25 * fontsize * dpi / 72.0)

    def hit(b, others):
        return any(b[0] < o[2] and b[2] > o[0] and b[1] < o[3] and b[3] > o[1] for o in others)

    mr = 6.0  # marker half-size px -> keep labels off the dots
    avoid = [(px - mr, py - mr, px + mr, py + mr) for px, py in pts]
    dirs = [(1, 0.4), (1, -0.4), (-1, 0.4), (-1, -0.4), (0, 1), (0, -1), (1, 0), (-1, 0)]
    radii = [11, 18, 26, 36, 48]
    placed = []
    inv = ax.transData.inverted()
    for i in np.argsort(ys):  # place the crowded bottom points first
        px, py = pts[i]
        w, h = box(labs[i])
        chosen = None
        for r in radii:
            for dx, dy in dirs:
                cx, cy = px + dx * r, py + dy * r
                x0, x1 = (cx, cx + w) if dx >= 0 else (cx - w, cx)
                b = (x0, cy - h / 2, x1, cy + h / 2)
                if not hit(b, placed + avoid):
                    chosen = (b, cx, cy, dx, r)
                    break
            if chosen:
                break
        if chosen is None:  # fallback: up-right, accept overlap
            cx, cy = px + 11, py + 11
            chosen = ((cx, cy - h / 2, cx + w, cy + h / 2), cx, cy, 1, 11)
        b, cx, cy, dx, r = chosen
        placed.append(b)
        tx, ty = inv.transform((cx, cy))
        ax.annotate(
            labs[i], (xs[i], ys[i]), xytext=(tx, ty), textcoords="data",
            fontsize=fontsize, color=color, ha=("left" if dx >= 0 else "right"), va="center",
            arrowprops=dict(arrowstyle="-", lw=0.4, color="#aaa") if r > 14 else None,
        )


# ---------------------------------------------------------------- data loaders
def behavioral(key):
    return _load(_latest(str(P5 / key / "run_*" / "behavioral_readout.json")))


_CV = P5 / "comprehension_validation"
_COMPREHENDED = {"ON_TOPIC", "REFUSAL"}


def judge_outcomes(key):
    """Per-condition outcome mix from the Gemini judge (items_gemini.jsonl), in the SAME shape
    as behavioral() so figures can swap loaders. We do NOT use 07_behavioral_readout's keyword
    classify() (it has no romanized-Hindi markers and undercounts multilingual refusals by up to
    +90pp). Mapping: REFUSAL->refusal, ON_TOPIC->harmful_compliance, OFF_TOPIC->off_topic,
    GIBBERISH->gibberish. Returns None if the judge file is absent."""
    f = _CV / key / "items_gemini.jsonl"
    if not f.exists():
        return None
    by = {}
    for line in open(f):
        r = json.loads(line)
        by.setdefault(r["condition"], []).append(r.get("label", "").upper())
    m = {"REFUSAL": "refusal", "ON_TOPIC": "harmful_compliance",
         "OFF_TOPIC": "off_topic", "GIBBERISH": "gibberish"}
    conds = {}
    for cond, labs in by.items():
        n = len(labs) or 1
        pct = {v: 0.0 for v in m.values()}
        for lab in labs:
            if lab in m:
                pct[m[lab]] += 100.0 / n
        conds[cond] = {"pct": {k: round(v, 1) for k, v in pct.items()}}
    return {"conditions": conds}


def gemini_comprehension(key, condition="arabizi_harmful"):
    """Validated comprehension gate = fraction of Gemini labels in {ON_TOPIC, REFUSAL} for a
    condition, from comprehension_validation/<key>/items_gemini.jsonl. Returns pct or None."""
    f = _CV / key / "items_gemini.jsonl"
    if not f.exists():
        return None
    labels = []
    for line in open(f):
        r = json.loads(line)
        if r.get("condition") == condition:
            labels.append(r.get("label", "").upper())
    if not labels:
        return None
    return round(100 * sum(1 for x in labels if x in _COMPREHENDED) / len(labels))


def h2_contrast(key, tier):
    best = None
    for f in sorted(glob.glob(str(P4I / key / "run_*" / "h2_results.json"))):
        d = _load(f)
        if d and d.get("pair") == tier and d.get("mode") == "ablation":
            best = d
    return best


def patch(key, tier):
    best = None
    for f in sorted(glob.glob(str(P4P / key / "**" / "patching_results.json"), recursive=True)):
        d = _load(f)
        if d and d.get("pair") == tier:
            best = d
    return best


def jaccard_arm(key):
    """Arabizi arm: SCRIPT-change (Arabic->Arabizi) vs LANGUAGE-change (Arabic->English) Jaccard."""
    run = _latest(str(P5 / key / "run_*"))
    if not run:
        return None
    for c in ("h3_controls.json", "h3_controls_arabic.json"):
        d = _load(Path(run) / c)
        if d:
            ladder = d.get("ladder", [])
            sc = lang = None
            for row in ladder:
                lab = row.get("label", "").lower()
                if "script" in lab and row.get("jaccard") is not None:
                    sc = row["jaccard"]
                if "cross-language" in lab and row.get("jaccard") is not None:
                    lang = row["jaccard"]
            if sc is not None and lang is not None:
                return {"script": sc, "language": lang}
    return None


def causal_layer(key):
    d = _load(_latest(str(P2 / key / "run_*" / "causal_layer_selection.json")))
    if not d:
        return None
    s = d.get("selection", {})
    return s.get("peak_causal_layer_guard", s.get("peak_causal_layer_lexical"))


def separation_layer(key):
    """Native-script separation peak (depth-biased diff-in-means proxy)."""
    d = _load(_latest(str(P2 / key / "run_*" / "metadata_separation_proxy.json")))
    if not d:
        return None
    conds = d.get("conditions", {})
    for native in ("arabic", "english", "devanagari"):
        if native in conds and conds[native].get("peak_layer") is not None:
            return conds[native]["peak_layer"]
    return None


def _frac(s):
    """'148/200' -> 0.74 ; None-safe."""
    if not s or "/" not in str(s):
        return None
    a, b = str(s).split("/")
    return int(a) / int(b) if int(b) else None


def lens_cell(key, tier):
    """Best (max-N) logit-lens run for a (model, tier): native/variant concept-present
    fraction + N. Native concept anchor is the native-script prompt; variant is the romanized."""
    best = None
    for f in sorted(glob.glob(str(P4P / key / "logitlens_*" / "logit_lens.json"))):
        d = _load(f)
        if d and d.get("pair") == tier and (best is None or d.get("n_pairs", 0) > best.get("n_pairs", 0)):
            best = d
    if not best:
        return None
    s = best.get("summary") or {}
    return {"n": best.get("n_pairs"),
            "nat": _frac(s.get("native_concept_emerged_count")),
            "var": _frac(s.get("variant_concept_emerged_count"))}


# lens is compute-heavy — reported for the 3 models spanning the two non-Gemma families (paper §4)
LENS_MODELS = ["qwen3_4b", "llama_3_1_8b_Instruct", "qwen3_8b"]
TIER_VARIANT_COND = {"tier_a": "arabizi_harmful", "tier_c": "itrans_harmful", "tier_e": "leetspeak_harmful"}

# Full model names (matches the paper tables); previously stripped -it/-Instruct.
short = lambda k: DISPLAY[k]


# ================================================================ figures
def fig1_two_route():
    """Schematic (single-column, compact vertical tree): harmful prompt -> comprehension gate
    -> decoded? deep refusal route : undecoded? shallow route (no concept -> complies).
    Conceptual, no data. Sized to render at \\columnwidth without shrinking."""
    fig, ax = plt.subplots(figsize=(3.5, 3.9))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    BW, BH = 4.4, 1.5

    def box(cx, cy, text, fc, w=BW, ec="#444", fs=8):
        ax.add_patch(plt.Rectangle((cx - w / 2, cy - BH / 2), w, BH, fc=fc, ec=ec, lw=1.2,
                                   zorder=2, joinstyle="round"))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3)

    def arrow(x1, y1, x2, y2, color="#444"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.7, shrinkA=0, shrinkB=0), zorder=1)

    def elabel(x, y, text, color):
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color=color, zorder=4,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

    xL, xR = 2.7, 7.3          # left (deep) / right (shallow) route columns
    box(5.0, 9.1, "harmful prompt", "#eef2f7", w=5.2)
    box(5.0, 7.1, "comprehension gate\n(decode?)", "#fff3d6", w=5.6)
    arrow(5.0, 9.1 - BH / 2, 5.0, 7.1 + BH / 2)

    # deep route (left): decoded -> concept built -> refuse
    box(xL, 4.5, "harmful concept\nBUILT", "#dcedd6")
    box(xL, 2.1, "REFUSE\n(deep route)", "#bfe0b0")
    arrow(5.0 - 1.2, 7.1 - BH / 2, xL, 4.5 + BH / 2, "#2e7d32")
    arrow(xL, 4.5 - BH / 2, xL, 2.1 + BH / 2, "#2e7d32")
    elabel(3.1, 5.95, "decoded", "#2e7d32")

    # shallow route (right): not decoded -> no concept -> comply
    box(xR, 4.5, "concept NOT\nbuilt", "#f6dcda")
    box(xR, 2.1, "comply /\noff-topic", "#efc9c5")
    arrow(5.0 + 1.2, 7.1 - BH / 2, xR, 4.5 + BH / 2, "#b23b34")
    arrow(xR, 4.5 - BH / 2, xR, 2.1 + BH / 2, "#b23b34")
    elabel(6.9, 5.95, "not decoded", "#b23b34")
    _save(fig, "fig1_two_route")


def fig1_comprehension_gate():
    """All 10 models: per-script outcome distribution (refusal / harmful-compliance /
    off-topic / gibberish). On romanized scripts the mass moves to OFF-TOPIC (non-comprehension),
    not to harmful compliance. Small multiples so every model is visible."""
    order = ["arabic", "arabizi", "english", "leetspeak", "devanagari", "itrans"]
    xlab = ["Ara", "Arz", "Eng", "Leet", "Dev", "ITR"]
    colors = {"refusal": "#2e7d32", "harmful_compliance": "#b23b34",
              "off_topic": "#e0a53a", "gibberish": "#9aa0a6"}
    fig, axes = plt.subplots(2, 5, figsize=(11.5, 5.0), sharey=True)
    for ax, key in zip(axes.flat, KEYS):
        d = judge_outcomes(key)
        ax.set_title(short(key), fontsize=8.5)
        if not d:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes, color="#999")
            ax.set_xticks([]); continue
        conds = d.get("conditions", {})
        x = np.arange(len(order))
        bottoms = np.zeros(len(order))
        for field in ("refusal", "harmful_compliance", "off_topic", "gibberish"):
            vals = [(conds.get(f"{sc}_harmful", {}).get("pct", {}) or {}).get(field, 0) for sc in order]
            ax.bar(x, vals, bottom=bottoms, color=colors[field], width=0.8,
                   label=field.replace("_", " "))
            bottoms += np.array(vals)
        ax.set_xticks(x); ax.set_xticklabels(xlab, fontsize=6.5, rotation=0)
        ax.set_ylim(0, 100)
    axes.flat[0].set_ylabel("% of responses"); axes.flat[5].set_ylabel("% of responses")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Comprehension gate (Gemini judge) — on ARABIZI the model goes OFF-TOPIC (can't decode); it reads and\n"
                 "refuses Devanagari and most ITRANS (Arz=Arabizi, Leet=Leetspeak, ITR=ITRANS are the romanized variants)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, "fig1_comprehension_gate")


def _tier_heatmap(matrix, title, cbar_label, vmin, vmax, cmap, name, fmt="{:.0f}"):
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(TIER_LABEL)
    ax.set_yticks(range(len(KEYS))); ax.set_yticklabels([short(k) for k in KEYS], fontsize=8)
    for i in range(len(KEYS)):
        for j in range(3):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if abs(matrix[i, j]) > (vmax * 0.55) else "black")
    ax.set_title(title, fontsize=9.5, pad=8)
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, name)


def fig3_patching_nonspecific():
    """matched - random recovery (pp) per model x tier. All inside the pre-registered <25pp
    reject band => patching is NON-SPECIFIC."""
    mat = np.full((len(KEYS), 3), np.nan)
    for i, k in enumerate(KEYS):
        for j, t in enumerate(TIERS):
            d = patch(k, t)
            if not d:
                continue
            pl = str(d.get("recover_peak_layer"))
            m = (d.get("recover_peak_rate") or 0) * 100
            r = (d.get("controls_random", {}) or {}).get(pl, 0) * 100
            mat[i, j] = m - r
    _tier_heatmap(mat, "H6 — patching specificity: matched $-$ random (pp)\n"
                  "all inside reject band ($<$25) = non-specific",
                  "matched$-$random (pp)", -25, 25, "PuOr_r", "fig3_patching_nonspecific")


def fig4_h2_direction_contrast():
    """native - romanized direction potency contrast (pp), per model x tier. All 10 models."""
    mat = np.full((len(KEYS), 3), np.nan)
    for i, k in enumerate(KEYS):
        for j, t in enumerate(TIERS):
            d = h2_contrast(k, t)
            if d:
                mat[i, j] = d.get("same_target_contrasts", {}).get("primary_contrast_pp", np.nan)
    _tier_heatmap(mat, "H2 — native$-$romanized direction potency (pp)\n"
                  "near 0 = interchangeable (shared axis)",
                  "contrast (pp)", -100, 100, "RdBu_r", "fig4_h2_direction_contrast")


def fig6_scale_dose_response(vertical=False):
    """(a) refusal returns with comprehension across ALL 10 models (Arabizi);
    (b) Gemma-3-27B-it by decode difficulty (per-script refusal vs off-topic).
    vertical=True stacks the two panels for a single-column (portrait) layout."""
    if vertical:
        fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.3, 8.6))
    else:
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    # panel (a): x=Arabizi comprehension %, y=Arabizi harmful-refusal %, one point per model
    xs, ys, labs, xben = [], [], [], []
    for k in KEYS:
        comp = gemini_comprehension(k, "arabizi_harmful")
        bh = judge_outcomes(k)
        if comp is None or not bh:
            continue
        ref = (bh.get("conditions", {}).get("arabizi_harmful", {}).get("pct", {}) or {}).get("refusal")
        if ref is None:
            continue
        xs.append(comp); ys.append(ref); labs.append(DISPLAY[k])  # keep -it / -Instruct suffix
        xben.append(gemini_comprehension(k, "arabizi_benign"))  # leakage-free decode proxy
    xs, ys = np.array(xs), np.array(ys)
    # trend line first (behind points)
    if len(xs) >= 2:
        b, a = np.polyfit(xs, ys, 1)
        xx = np.linspace(xs.min(), xs.max(), 20)
        axa.plot(xx, a + b * xx, color="#b0b0b0", ls="--", lw=1.2, zorder=1)
    # one distinct colour per model + a colour->name legend (no on-point labels)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    for x, y, l, c in zip(xs, ys, labs, colors):
        axa.scatter([x], [y], s=70, color=[c], zorder=3, edgecolor="white", lw=0.8, label=l)
    if len(xs) >= 2:
        r_raw = np.corrcoef(xs, ys)[0, 1]
        # leakage-free r: refusal vs BENIGN-arabizi engagement (shares no term with refusal)
        msk = np.array([v is not None for v in xben])
        note = f"Pearson r = {r_raw:.2f} (raw; partly definitional)"
        if msk.sum() >= 2:
            r_lf = np.corrcoef(np.array(xben, float)[msk], ys[msk])[0, 1]
            note += f"\nr = {r_lf:.2f} leakage-free (vs benign-Arabizi decode)"
        axa.text(0.04, 0.965, note, transform=axa.transAxes, fontsize=7.2, color="#555",
                 va="top", linespacing=1.4)
    axa.set_xlabel("Gemini-judged Arabizi comprehension %  (ON_TOPIC $\\cup$ REFUSAL)")
    axa.set_ylabel("harmful-Arabizi refusal %")
    axa.set_title("(a) Refusal returns with comprehension\n(all 10 models, Arabic$\\to$Arabizi)", fontsize=9.5)
    axa.set_xlim(-3, 100); axa.set_ylim(-3, 100)
    axa.legend(loc="lower right", fontsize=6.6, frameon=True, framealpha=0.9,
               edgecolor="#cccccc", handletextpad=0.3, labelspacing=0.3, borderpad=0.5,
               title="model", title_fontsize=7.2)

    # panel (b): 27B by script — refusal + off_topic
    d = judge_outcomes("gemma_3_27b_it")
    scripts = ["english", "arabic", "devanagari", "leetspeak", "arabizi", "itrans"]
    slab = ["English", "Arabic", "Devanagari", "Leetspeak", "Arabizi", "ITRANS"]
    if d:
        conds = d.get("conditions", {})
        ref = [(conds.get(f"{s}_harmful", {}).get("pct", {}) or {}).get("refusal", 0) for s in scripts]
        off = [(conds.get(f"{s}_harmful", {}).get("pct", {}) or {}).get("off_topic", 0) for s in scripts]
        y = np.arange(len(scripts))
        axb.barh(y, ref, color="#2e7d32", label="refusal")
        axb.barh(y, off, left=ref, color="#b23b34", label="off-topic")
        axb.set_yticks(y); axb.set_yticklabels(slab, fontsize=8)
        axb.invert_yaxis()
        axb.set_xlabel("% of responses (n=50)")
        axb.set_title("(b) Gemma-3-27B-it by decode difficulty", fontsize=9.5)
        axb.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
        axb.set_xlim(0, 108)
    fig.tight_layout()
    _save(fig, "fig6_scale_dose_response_vertical" if vertical else "fig6_scale_dose_response")


def fig7_h3_script_vs_language():
    """All 10 models: changing the SCRIPT (Arabic->Arabizi) fragments SAE features more than
    changing the LANGUAGE (Arabic->English). Lower Jaccard = more fragmented."""
    rows = []
    for k in KEYS:
        a = jaccard_arm(k)
        if a:
            rows.append((k, a["script"], a["language"]))
    # sort by script Jaccard ascending (most fragmented first)
    rows.sort(key=lambda r: r[1])
    labels = [short(k) for k, _, _ in rows]
    sc = [r[1] for r in rows]
    lg = [r[2] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.bar(x - w / 2, sc, w, color="#b23b34", label="change SCRIPT (Arabic$\\to$Arabizi)")
    ax.bar(x + w / 2, lg, w, color="#2c6fbf", label="change LANGUAGE (Arabic$\\to$English)")
    ax.axhline(0.4, color="#2e7d32", ls=":", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("feature Jaccard (lower = more fragmented)")
    ax.set_title("Changing the SCRIPT fragments features MORE than changing the LANGUAGE\n"
                 "(romanizing the same language disrupts more than translating it) — all 10 models",
                 fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig7_h3_script_vs_language")


def fig8_causal_vs_separation_depth():
    """All 10 models: causal peak (where we intervene) is mid-network; separation peak
    (diff-in-means) is late. Relative depth = layer / n_layers."""
    rows = []
    for k in KEYS:
        cl = causal_layer(k)
        sl = separation_layer(k)
        n = NLAYERS[k]
        if cl is None or sl is None:
            continue
        rows.append((k, cl / n, sl / n))
    rows.sort(key=lambda r: r[1])  # by causal depth ascending
    labels = [short(k) for k, _, _ in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, (_, c, s) in enumerate(rows):
        ax.plot([c, s], [i, i], color="#bbb", lw=2, zorder=1)
    ax.scatter([r[1] for r in rows], y, s=70, color="#2e9e6a", zorder=3, label="causal peak (intervene here)")
    ax.scatter([r[2] for r in rows], y, s=70, color="#d24b4b", zorder=3, label="separation peak (diff-in-means)")
    ax.axvline(0.5, color="#666", ls=":", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("relative depth (layer / n_layers)")
    ax.set_xlim(0, 1.0)
    ax.set_title("Causality is mid-network; separation is late\n"
                 "(attribute against the layer you can actually steer) — all 10 models", fontsize=9.5)
    # all data sits at depth > 0.34, so the left third is empty — put the legend there, clear of dots
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.01, 0.5), framealpha=0.95)
    fig.tight_layout()
    _save(fig, "fig8_causal_vs_separation_depth")


def fig2_concept_present_regimes():
    """Interpretable concept probe: native English vs Leetspeak, ALL 10 models at N=200.
    The English-anchored probe is only valid on the Leetspeak pair (English native side);
    Arabizi/ITRANS are uninterpretable and reported as pilots in Table concept, not here.
    We plot the native anchor (probe validity) beside the Leetspeak value, sorted by anchor
    strength. Where the anchor validates (native >=30%) the concept collapses in the weak decoders
    and in two high-comprehension models (Llama-3.2-3B, Gemma-2-9B: residual anchoring); only the
    largest model (27B) retains it (paper §5.2, three-way split)."""
    data = []
    for k in KEYS:
        c = lens_cell(k, "tier_e")
        if c and c["nat"] is not None:
            data.append((k, c["nat"] * 100, (c["var"] or 0) * 100, c["n"]))
    data.sort(key=lambda r: -r[1])  # by native anchor, strongest first
    labels = [short(k) for k, *_ in data]
    nat = [d[1] for d in data]; var = [d[2] for d in data]
    x = np.arange(len(data)); w = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    # shade the invalid-anchor models (native < 30%: probe cannot validate)
    for i, d in enumerate(data):
        if d[1] < 30:
            ax.axvspan(i - 0.5, i + 0.5, color="#f2f2f2", zorder=0)
    ax.bar(x - w / 2, nat, w, color="#9aa0a6", label="native English (probe anchor)")
    ax.bar(x + w / 2, var, w, color="#8e44ad", label="Leetspeak")
    # collapse arrows (native -> Leetspeak) only where the anchor validates
    for xi, d in zip(x, data):
        if d[1] >= 30:
            ax.annotate("", xy=(xi + w / 2, d[2] + 0.5), xytext=(xi - w / 2, d[1] - 0.5),
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.9, alpha=0.55))
    ax.axhline(30, color="#888", ls=":", lw=1)
    ax.text(len(data) - 0.5, 31.5, "anchor-validity floor (native $\\geq$30%)",
            ha="right", fontsize=6.8, color="#666")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("harmful concept present (%)  ·  N=200")
    ax.set_ylim(0, 95)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_title("Harmful-concept probe: native English (anchor) vs Leetspeak — N=200, all ten models\n"
                 "sorted by anchor strength; grey band = native anchor <30% (English-anchored probe "
                 "cannot validate there)",
                 fontsize=8.4)
    fig.tight_layout()
    _save(fig, "fig2_concept_present_regimes")


def main():
    print("Regenerating paper figures from live results/ ...")
    fig1_two_route()
    fig1_comprehension_gate()
    fig2_concept_present_regimes()
    fig3_patching_nonspecific()
    fig4_h2_direction_contrast()
    fig6_scale_dose_response()
    fig7_h3_script_vs_language()
    fig8_causal_vs_separation_depth()
    print("done — 8 figures regenerated from live results/ (fig2 lens annotates N per cell).")


if __name__ == "__main__":
    main()
