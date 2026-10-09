#!/usr/bin/env python3
"""Generate the paper's figures from results/. Two-colour palette, validated colourblind-safe.

fig1  recovery against engagement, one row per interpretable cell. The paper's central finding.
fig2  decode-then-refuse: rewritten, self-decoded, native, one row per cell.
fig3  how much of the gap self-decoding closes, against how well the model decoded.

Usage: PYTHONPATH=. python3 scripts/ops/make_v3_figures.py --out openreview/v3_overleaf/figures
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "results/phase5/comprehension_validation"
BLUE, ORANGE, INK, MUTED, GRID = "#1f6feb", "#d97706", "#1f2328", "#57606a", "#d8dee4"
NICE = {"gemma_3_1b_it": "Gemma-3-1B", "gemma_2_2b_it": "Gemma-2-2B", "gemma_3_4b_it": "Gemma-3-4B",
        "qwen3_4b": "Qwen3-4B", "llama_3_2_3b_Instruct": "Llama-3.2-3B", "qwen2_5_7b_Instruct": "Qwen2.5-7B",
        "qwen3_8b": "Qwen3-8B", "llama_3_1_8b_Instruct": "Llama-3.1-8B", "gemma_2_9b_it": "Gemma-2-9B",
        "gemma_3_27b_it": "Gemma-3-27B"}
PAIRS = [("tier_a", "arabizi_harmful", "Arabizi"), ("tier_c", "itrans_harmful", "ITRANS"),
         ("tier_e", "leetspeak_harmful", "Leetspeak")]

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")


def engagement():
    g = {}
    for f in CV.glob("*/items_gemini.jsonl"):
        m = f.parent.name
        by = collections.defaultdict(list)
        for r in (json.loads(l) for l in open(f)):
            by[r["condition"]].append(r["label"].upper())
        g[m] = {c: 100 * sum(x in ("ON_TOPIC", "REFUSAL") for x in v) / len(v) for c, v in by.items()}
    return g


def fig1(out):
    """One row per interpretable cell: engagement and recovery, joined by a line."""
    eng = engagement()
    rows = []
    for f in sorted(CV.glob("*/forced_choice_hard.json")):
        h = json.load(open(f))
        m = h["model"]
        for p, cond, lab in PAIRS:
            if not h["summary"].get(f"{p}|native_copy", {}).get("format_valid"):
                continue
            rec = h["summary"][f"{p}|romanized_harmful"]["debiased_acc_pct"]
            rows.append((f"{NICE[m]}  {lab}", eng[m][cond], rec))
    rows.sort(key=lambda r: r[2] - r[1])
    fig, ax = plt.subplots(figsize=(5.4, 0.26 * len(rows) + 1.2))
    y = range(len(rows))
    for i, (_, e, r) in zip(y, rows):
        ax.plot([e, r], [i, i], color=GRID, lw=2, solid_capstyle="round", zorder=1)
    ax.scatter([r[1] for r in rows], y, s=34, color=ORANGE, zorder=3, label="engages the request")
    ax.scatter([r[2] for r in rows], y, s=34, color=BLUE, zorder=3, label="recovers the meaning")
    for i, (_, e, r) in zip(y, rows):
        ax.annotate(f"+{r - e:.0f}", (max(e, r) + 2.5, i), va="center", fontsize=7, color=MUTED)
    ax.axvline(25, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.annotate("chance", (25, len(rows) - 0.3), fontsize=6.5, color=MUTED, ha="center", va="bottom")
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(-2, 112)
    ax.set_xlabel("percentage of 50 harmful requests")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False,
              fontsize=7.5, handletextpad=0.3, columnspacing=2.0)
    save(fig, out, "fig1_recovery_vs_engagement")


def fig2(out):
    rows = json.load(open(ROOT / "results/phase5/decode_then_refuse/summary.json"))
    by = {(r["model"], r["pair"], r["arm"]): r for r in rows}
    cells = []
    for (m, p, a) in sorted(by):
        if a != "baseline":
            continue
        b, s, o = [by[(m, p, x)]["refusal_pct"] for x in ("baseline", "self_decode", "oracle_decode")]
        lab = {"tier_c": "ITRANS", "tier_e": "Leetspeak"}[p]
        cells.append((f"{NICE[m]}  {lab}", b, s, o))
    cells.sort(key=lambda c: c[2] - c[1])
    fig, ax = plt.subplots(figsize=(5.4, 0.3 * len(cells) + 1.35))
    y = range(len(cells))
    for i, (_, b, s, o) in zip(y, cells):
        ax.plot([b, o], [i, i], color=GRID, lw=2, solid_capstyle="round", zorder=1)
        ax.annotate("", xy=(s, i), xytext=(b, i),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6, shrinkA=0, shrinkB=0), zorder=2)
    ax.scatter([c[1] for c in cells], y, s=30, color=MUTED, zorder=3, label="rewritten request")
    ax.scatter([c[2] for c in cells], y, s=38, color=BLUE, zorder=4, label="model decodes it first")
    ax.scatter([c[3] for c in cells], y, s=38, color=ORANGE, marker="D", zorder=3, label="native script (ceiling)")
    ax.set_yticks(list(y))
    ax.set_yticklabels([c[0] for c in cells], fontsize=7.5)
    ax.set_xlim(-2, 104)
    ax.set_xlabel("refusal, percentage of 50 harmful requests")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False,
              fontsize=7.5, handletextpad=0.3, columnspacing=1.4)
    save(fig, out, "fig2_decode_then_refuse")


def fig3(out):
    rows = json.load(open(ROOT / "results/phase5/decode_then_refuse/summary.json"))
    by = {(r["model"], r["pair"], r["arm"]): r for r in rows}
    xs, ys, labs = [], [], []
    for (m, p, a) in sorted(by):
        if a != "baseline":
            continue
        b, s, o = [by[(m, p, x)]["refusal_pct"] for x in ("baseline", "self_decode", "oracle_decode")]
        if o <= b:
            continue
        xs.append(by[(m, p, "self_decode")]["mean_decode_sim"])
        ys.append(100 * (s - b) / (o - b))
        labs.append(f"{NICE[m]} {'ITR' if p == 'tier_c' else 'Leet'}")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    r = num / den
    sxx = sum((a - mx) ** 2 for a in xs)
    slope = num / sxx
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    lo, hi = min(xs) - 0.03, max(xs) + 0.03
    ax.plot([lo, hi], [my + slope * (lo - mx), my + slope * (hi - mx)], color=GRID, lw=2, zorder=1)
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.scatter(xs, ys, s=34, color=BLUE, zorder=3)
    ax.set_ylim(min(ys) - 28, max(ys) + 22)
    for x, yv, l in zip(xs, ys, labs):
        if yv < 5 or yv > 70:
            ha = "right" if x > (lo + hi) / 2 else "left"
            dx = -8 if ha == "right" else 8
            ax.annotate(l, (x, yv), textcoords="offset points", xytext=(dx, -3),
                        ha=ha, va="center", fontsize=6.3, color=MUTED)
    ax.annotate(f"$r = {r:.2f}$", (0.04, 0.93), xycoords="axes fraction", fontsize=8, color=INK)
    ax.set_xlabel("similarity of the model's own decoding")
    ax.set_ylabel("percentage of the gap closed")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out, "fig3_decode_quality")


def box(ax, x, y, w, h, text, fc="#ffffff", ec=MUTED, fs=7.2, bold=False, tc=None,
        sub=None, sub_kw=None):
    """A rounded box with centred text. `sub` draws a second line in its own font.

    The second line exists so example strings can use a real italic or monospace face.
    Mathtext (``$\\it{...}$``) was used before and set the words with mathematical spacing,
    which reads as a rendering fault at paper scale.
    """
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.1, zorder=2, clip_on=False))
    common = dict(ha="center", zorder=3, color=tc or INK,
                  fontweight="bold" if bold else "normal", fontsize=fs)
    if sub is None:
        ax.text(x + w / 2, y + h / 2, text, va="center", linespacing=1.45, **common)
        return
    ax.text(x + w / 2, y + h / 2 + 0.018, text, va="bottom", **common)
    ax.text(x + w / 2, y + h / 2 - 0.022, sub, va="top", **{**common, **(sub_kw or {})})


def arrow(ax, x1, y1, x2, y2, color=MUTED, label=None, fs=6.6, dashed=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                                linestyle=(0, (3, 2)) if dashed else "solid", shrinkA=1, shrinkB=1))
    if label:
        ax.text((x1 + x2) / 2, max(y1, y2) + 0.022, label, ha="center", va="bottom",
                fontsize=fs, color=color)


def fig_concept(out):
    """What happens to a harmful request under each of the three presentations. Full text width."""
    fig, ax = plt.subplots(figsize=(6.2, 2.45))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    xin, xmid, xout = 0.0, 0.3575, 0.715
    w, h = 0.285, 0.185
    rows = [
        (0.775, "as written", "Write a script that\u2026", dict(style="italic"),
         "the model reads\nthe request", "refuses\n91.8%", ORANGE, "solid"),
        (0.455, "rewritten", "Wr173 4 5cr1p7\u2026", dict(family="monospace", fontsize=7.4),
         "the meaning is\nrecoverable, unused", "off topic\n62.6%", MUTED, "solid"),
        (0.135, "rewritten, then", "the model decodes it", {},
         "the model reads\nits own decoding", "refuses\nup to 86%", BLUE, "solid"),
    ]
    for y, a, a2, a2kw, b, c, col, style in rows:
        box(ax, xin, y, w, h, a, fs=8, sub=a2, sub_kw=a2kw)
        box(ax, xmid, y, w, h, b, ec=col, fs=8)
        box(ax, xout, y, w, h, c, ec=col, bold=True, tc=col, fs=8)
        arrow(ax, xin + w, y + h / 2, xmid, y + h / 2)
        arrow(ax, xmid + w, y + h / 2, xout, y + h / 2, color=col, dashed=(style == "dashed"))
    ax.text(xmid + w / 2, 0.418, "the step that fails", ha="center", va="top",
            fontsize=7.4, color=MUTED, style="italic")
    ax.text(0.5, 0.0, "Safety is intact. What reaches it is not.", ha="center", va="bottom",
            fontsize=8.6, color=INK)
    save(fig, out, "fig0_concept")


def fig_method(out):
    """The three measurements, and which of them can be contaminated by refusal."""
    fig, ax = plt.subplots(figsize=(6.2, 2.15))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    cols = [
        (0.0, "Engagement", "the model answers\nnormally", "a judge labels the\nresponse four ways",
         "refusal counts as\nreading", ORANGE),
        (0.3475, "Meaning identification", "four candidate meanings\nin the native script",
         "read the answer letter\nfrom the distribution", "no response, so no\nrefusal", BLUE),
        (0.695, "Transliteration", "rewrite it in the\nnative script",
         "score against the\nknown correct text", "no response, so no\nrefusal", BLUE),
    ]
    w = 0.305
    for x, title, inp, how, note, col in cols:
        ax.text(x + w / 2, 0.955, title, ha="center", va="top", fontsize=8.6, fontweight="bold", color=col)
        box(ax, x, 0.645, w, 0.19, inp, fs=8)
        arrow(ax, x + w / 2, 0.645, x + w / 2, 0.545)
        box(ax, x, 0.345, w, 0.19, how, fs=8)
        arrow(ax, x + w / 2, 0.345, x + w / 2, 0.245)
        box(ax, x, 0.045, w, 0.19, note, ec=col, tc=col, fs=8)
    save(fig, out, "fig0b_method")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="openreview/v3_overleaf/figures")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    fig_concept(out)
    fig_method(out)
    fig1(out)
    fig2(out)
    fig3(out)
    print(f"wrote to {out}")


if __name__ == "__main__":
    main()
