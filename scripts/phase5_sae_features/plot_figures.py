#!/usr/bin/env python3
"""Phase 5 figures — behavioral comprehension gate + H3 SAE overlap + judge validation.

Headline: on romanized/distorted scripts the dominant non-refusal outcome is
OFF_TOPIC, not harmful compliance — the "jailbreak" is a comprehension failure, not a
bypassed refusal. Three independent judges (Gemini, Claude, human natives) agree the
gate is real, not a single-judge artifact.

Reads results/phase5/<model>/**/behavioral_readout.json   (4-class outcomes + parity)
      results/phase5/<model>/**/jaccard.json               (H3 SAE feature overlap)
      results/phase5/comprehension_validation/agreement_report.json
Writes results/phase5/figures/.
Run:  python3 scripts/phase5_sae_features/plot_figures.py

Figures
  outcome_distribution.png       4-class outcomes native vs romanized (anchor model)
  comprehension_drop_heatmap.png engagement drop native→romanized, all models
  jaccard_overlap.png            H3 SAE feature overlap per tier (exploratory)
  judge_agreement.png            three-judge binary-gate agreement
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase5" / "figures"
ANCHOR = "gemma_3_4b_it"
MODELS = ["gemma_3_1b_it", "gemma_2_2b_it", "gemma_3_4b_it", "gemma_2_9b_it",
          "gemma_3_27b_it", "qwen3_4b", "qwen3_8b", "qwen2_5_7b_Instruct",
          "llama_3_1_8b_Instruct", "llama_3_2_3b_Instruct"]
short = lambda m: m.replace("_it", "").replace("_Instruct", "").replace("_", " ")
plt.rcParams.update({"font.size": 9, "figure.dpi": 140})
# native, romanized, label
COND_PAIRS = [("arabic_harmful", "arabizi_harmful", "Arabic→Arabizi"),
              ("english_harmful", "leetspeak_harmful", "English→Leetspeak"),
              ("devanagari_harmful", "itrans_harmful", "Devanagari→ITRANS")]
CLASSES = [("refusal", "#3a7d44"), ("harmful_compliance", "#c0392b"),
           ("off_topic", "#e0a83a"), ("gibberish", "#888")]


def _latest(pat):
    fs = sorted(glob.glob(str(ROOT / pat), recursive=True))
    return json.load(open(fs[-1])) if fs else None


def outcome_distribution() -> None:
    bo = _latest(f"results/phase5/{ANCHOR}/**/behavioral_readout.json")
    if not bo:
        print("no anchor behavioral_readout"); return
    conds = bo["conditions"]
    cols, labels = [], []
    for nat, rom, lab in COND_PAIRS:
        for key, tag in [(nat, "native"), (rom, "roman.")]:
            cols.append(key); labels.append(f"{lab.split('→')[0 if tag=='native' else 1]}\n({tag})")
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    x = np.arange(len(cols)); bottom = np.zeros(len(cols))
    for cls, col in CLASSES:
        vals = [conds.get(c, {}).get("pct", {}).get(cls, 0) for c in cols]
        ax.bar(x, vals, 0.7, bottom=bottom, color=col, label=cls.replace("_", " "))
        bottom += np.array(vals)
    for i in range(0, len(cols), 2):
        ax.axvspan(i - 0.5, i + 1.5, color="#f6f6f6", zorder=-1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("% of responses"); ax.set_ylim(0, 100)
    ax.set_title(f"Comprehension gate ({short(ANCHOR)}) — on romanized scripts the model goes "
                 "OFF-TOPIC, it does not comply", fontsize=9)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    p = OUT / "outcome_distribution.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


def comprehension_drop_heatmap() -> None:
    mat = np.full((len(MODELS), 3), np.nan)
    cols = ["Arabizi", "Leetspeak", "ITRANS"]
    keys = ["arabizi_harmful", "leetspeak_harmful", "itrans_harmful"]
    for i, m in enumerate(MODELS):
        bo = _latest(f"results/phase5/{m}/**/behavioral_readout.json")
        if not bo:
            continue
        cp = bo.get("comprehension_parity", {})
        for j, k in enumerate(keys):
            v = cp.get(k, {}).get("comprehension_drop_pp")
            if isinstance(v, (int, float)):
                mat[i, j] = v
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=80, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([short(m) for m in MODELS], fontsize=8)
    for i in range(len(MODELS)):
        for j in range(3):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > 45 else "black")
    ax.set_title("Comprehension gate — engagement drop native$\\rightarrow$romanized (pp)\n"
                 "large = model cannot read the romanized prompt", fontsize=9.5, pad=8)
    fig.colorbar(im, ax=ax, label="drop (pp)", fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = OUT / "comprehension_drop_heatmap.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


def jaccard_overlap() -> None:
    tiers = ["tier_a", "tier_c", "tier_e"]; TL = ["Arabizi", "ITRANS", "Leetspeak"]
    pts = {t: [] for t in tiers}; cis = {t: [] for t in tiers}
    for m in MODELS:
        jf = _latest(f"results/phase5/{m}/**/jaccard.json")
        if not jf:
            continue
        for t in tiers:
            d = jf.get("pairs", {}).get(t)
            if d:
                pts[t].append(d.get("jaccard", np.nan))
                b = d.get("bootstrap_ci", {})
                cis[t].append((b.get("ci_95_low", np.nan), b.get("ci_95_high", np.nan)))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(tiers))
    for j, t in enumerate(tiers):
        ys = np.array(pts[t]); xs = np.full(len(ys), j) + np.linspace(-0.22, 0.22, max(len(ys), 1))
        ax.scatter(xs, ys, s=26, color="#4a6", alpha=0.8, zorder=3)
        if len(ys):
            ax.hlines(np.nanmean(ys), j - 0.3, j + 0.3, color="#245", lw=2, zorder=4)
    ax.axhspan(0, 0.4, color="#dff0df", alpha=0.6)
    ax.axhspan(0.6, 1.0, color="#f0dede", alpha=0.6)
    ax.axhline(0.4, ls="--", c="#3a7", lw=0.9); ax.axhline(0.6, ls="--", c="#c44", lw=0.9)
    ax.text(2.55, 0.2, "SUPPORTED\n(<0.4 shared)", fontsize=6.5, color="#3a7", ha="left", va="center")
    ax.text(2.55, 0.8, "REJECTED\n(≥0.6 shared)", fontsize=6.5, color="#c44", ha="left", va="center")
    ax.set_xticks(x); ax.set_xticklabels(TL); ax.set_xlim(-0.5, 3.4); ax.set_ylim(0, 1)
    ax.set_ylabel("SAE feature Jaccard (top-100)")
    ax.set_title("H3 (exploratory) — SAE feature overlap between native & romanized\n"
                 "per-model points; bar = mean", fontsize=9)
    fig.tight_layout()
    p = OUT / "jaccard_overlap.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


def judge_agreement() -> None:
    rp = ROOT / "results/phase5/comprehension_validation/agreement_report.json"
    if not rp.exists():
        print("no agreement_report.json"); return
    cmp = json.load(open(rp))["comparisons"]
    rows = [("human(ar)↔Claude", "human_arabic_vs_claude"),
            ("human(ar)↔Gemini", "human_arabic_vs_gemini"),
            ("human(en)↔Claude", "human_english_vs_claude"),
            ("human(en)↔Gemini", "human_english_vs_gemini"),
            ("Gemini↔Claude", "gemini_vs_claude")]
    labs, ks = [], []
    for lab, key in rows:
        b = cmp.get(key, {})
        if b.get("kappa_binary") is not None:
            labs.append(lab); ks.append(b["kappa_binary"])
    if not labs:
        print("no filled agreement numbers yet"); return
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    colors = ["#2a7" if "human" in l else "#69c" for l in labs]
    ax.barh(range(len(labs)), ks, color=colors)
    for i, k in enumerate(ks):
        ax.text(k + 0.01, i, f"{k:.2f}", va="center", fontsize=8)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.axvline(0.6, ls="--", c="gray", lw=0.8)
    ax.set_xlabel("Cohen's $\\kappa$ (binary comprehension gate)")
    ax.set_title("Three-judge agreement — the gate is not a single-judge artifact", fontsize=9.5)
    fig.tight_layout()
    p = OUT / "judge_agreement.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    outcome_distribution()
    comprehension_drop_heatmap()
    jaccard_overlap()
    judge_agreement()
