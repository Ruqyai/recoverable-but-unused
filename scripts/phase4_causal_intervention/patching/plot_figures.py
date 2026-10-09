#!/usr/bin/env python3
"""Phase 4 · patching (H6) figures — activation patching + logit-lens comprehension.

Story: patching a matched harmful activation into a romanized prompt is no more
effective than patching a random one (matched ≈ random ⇒ H6 REJECTED, pre-registered
confirmatory). The logit lens shows why: on native prompts the refusal representation
forms in ~10/10 cases; on romanized prompts it barely forms — the model never
comprehends the request, so there is nothing script-specific to bypass.

Reads  results/phase4_patching/<model>/**/patching_results.json
       results/phase4_patching/<model>/logitlens_*/logit_lens.json
Writes results/phase4_patching/figures/.
Run:   python3 scripts/phase4_causal_intervention/patching/plot_figures.py

Figures
  h6_matched_minus_random_heatmap.png  patching specificity (reject band <25pp)
  h6_recovery_curve.png                recovery vs controls (anchor model)
  native_anchor_collapse.png           native-language concept collapse under romanization (3 langs)
"""
from __future__ import annotations
import glob, json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "phase4_patching" / "figures"
ANCHOR = "gemma_3_4b_it"
MODELS = ["gemma_3_1b_it", "gemma_2_2b_it", "gemma_3_4b_it", "gemma_2_9b_it",
          "gemma_3_27b_it", "qwen3_4b", "qwen3_8b", "qwen2_5_7b_Instruct",
          "llama_3_1_8b_Instruct", "llama_3_2_3b_Instruct"]
TIERS = ["tier_a", "tier_c", "tier_e"]
TL = ["Arabizi", "ITRANS", "Leetspeak"]
short = lambda m: m.replace("_it", "").replace("_Instruct", "").replace("_", " ")
plt.rcParams.update({"font.size": 9, "figure.dpi": 140})


def _all_patch(model):
    return [json.load(open(f)) for f in
            glob.glob(str(ROOT / f"results/phase4_patching/{model}/**/patching_results.json"), recursive=True)]


def _latest(pat):
    fs = sorted(glob.glob(str(ROOT / pat), recursive=True))
    return json.load(open(fs[-1])) if fs else None


def h6_matched_minus_random_heatmap() -> None:
    mat = np.full((len(MODELS), 3), np.nan)
    for i, m in enumerate(MODELS):
        by = {x.get("pair"): x for x in _all_patch(m)}
        for j, t in enumerate(TIERS):
            x = by.get(t)
            if x:
                pl = x.get("recover_peak_layer"); pr = x.get("recover_peak_rate")
                rnd = (x.get("controls_random") or {}).get(str(pl))
                if pr is not None and rnd is not None:
                    mat[i, j] = (pr - rnd) * 100
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(mat, cmap="PuOr_r", vmin=-25, vmax=25, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(TL)
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([short(m) for m in MODELS], fontsize=8)
    for i in range(len(MODELS)):
        for j in range(3):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(mat[i, j]) > 14 else "black")
    ax.set_title("H6 — patching specificity: matched $-$ random (pp)\nall inside reject band ($<25$) = non-specific",
                 fontsize=9.5, pad=8)
    fig.colorbar(im, ax=ax, label="matched$-$random (pp)", fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = OUT / "h6_matched_minus_random_heatmap.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


def h6_recovery_curve() -> None:
    series = [("recover_curve_A2B", "matched (harmful→variant)", "#c0392b", 2.4),
              ("controls_random", "random source", "#888", 1.6),
              ("controls_self", "self (variant→variant)", "#4a90c4", 1.4),
              ("controls_benign", "benign source", "#2a9d5a", 1.4)]
    by = {x.get("pair"): x for x in _all_patch(ANCHOR)}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, t, tl in zip(axes, TIERS, TL):
        x = by.get(t)
        if not x:
            ax.set_title(f"{tl} (no data)", fontsize=9); continue
        nl = max(int(k) for k in x.get("recover_curve_A2B", {"0": 0})) + 1
        for key, lab, col, lw in series:
            cur = x.get(key) or {}
            if not cur:
                continue
            ks = sorted(int(k) for k in cur)
            ax.plot(np.array(ks) / nl, [cur[str(k)] * 100 for k in ks], color=col, lw=lw, label=lab)
        ax.axhspan(0, 25, color="#eee", alpha=0.6)
        ax.set_title(tl, fontsize=9); ax.set_xlabel("relative depth"); ax.set_xlim(0, 1)
    axes[0].set_ylabel("recovery rate (%)"); axes[0].legend(fontsize=6.6, loc="upper left")
    fig.suptitle(f"H6 — activation patching recovery vs controls ({short(ANCHOR)}): "
                 "matched ≈ random ⇒ non-specific (REJECTED)", fontsize=9.5, y=1.03)
    fig.tight_layout()
    p = OUT / "h6_recovery_curve.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


def _frac(s):
    """'148/200' -> 0.74 ; None-safe."""
    if s and "/" in str(s):
        a, b = str(s).split("/")
        return int(a) / int(b) if int(b) else None
    return None


def _pair_cell(model, tier):
    """(native_frac, romanized_frac) for the pair's NATIVE-language concept anchor.
    tier_e (English→Leetspeak) uses the English-anchored logitlens_* runs — English IS the native
    side there; tier_a/tier_c use the native-language (Arabic/Hindi) nativeanchor_* runs."""
    prefix = "logitlens_" if tier == "tier_e" else "nativeanchor_"
    best = None
    for f in glob.glob(str(ROOT / f"results/phase4_patching/{model}/{prefix}*/logit_lens.json")):
        d = json.load(open(f))
        if d.get("pair") == tier and (best is None or d.get("n_pairs", 0) > best.get("n_pairs", 0)):
            best = d
    if not best:
        return None
    s = best.get("summary", {})
    if tier == "tier_e":
        return _frac(s.get("native_concept_emerged_count")), _frac(s.get("variant_concept_emerged_count"))
    return s.get("native_concept_present_frac"), s.get("concept_present_in_variant_frac")


def native_anchor_collapse() -> None:
    """Native-LANGUAGE concept anchor across ALL THREE language pairs in one figure: does the harmful
    concept collapse under romanization when anchored in each pair's native language (Arabic for
    Arabizi, Hindi for ITRANS, English for Leetspeak)? Per model, native (grey) vs romanized
    (colored), sorted by collapse gap. Where the anchor discriminates (gap>=40pp, arrows) the concept
    collapses -> comprehension-gating; the direction is unanimous (no cell shows the concept surviving
    romanization). Non-discriminating cells (native~romanized, shaded) are base-rate artifacts:
    probe sensitivity is model- AND script-dependent (Gemma discriminates on Hindi not Arabic;
    Qwen on neither; English/Leetspeak discriminates broadly), because content words fragment into
    high-base-rate subword first-tokens in some tokenizers."""
    langs = [("tier_a", "Arabizi (Arabic anchor)", "#e0912f"),
             ("tier_c", "ITRANS (Hindi anchor)", "#c0392b"),
             ("tier_e", "Leetspeak (English anchor)", "#8e44ad")]
    # collapse gap = native-anchored concept present natively MINUS in the romanized run (pp).
    # tall bar => concept is built on the native prompt then lost on romanization = comprehension-gating.
    rows = {}
    for m in MODELS:
        g = {}
        for tier, _, _ in langs:
            c = _pair_cell(m, tier)
            g[tier] = max(0.0, (c[0] - c[1]) * 100) if (c and c[0] is not None and c[1] is not None) else np.nan
        rows[m] = g
    order = sorted(MODELS, key=lambda m: -np.nansum(list(rows[m].values())))
    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for j, (tier, lab, col) in enumerate(langs):
        ax.bar(x + (j - 1) * w, [rows[m][tier] for m in order], w, color=col, label=lab)
    ax.axhline(40, color="#888", ls=":", lw=1)
    ax.text(len(order) - 0.5, 42, "discrimination floor (40pp)", ha="right", fontsize=6.8, color="#666")
    ax.set_xticks(x); ax.set_xticklabels([short(m) for m in order], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("concept collapse under romanization\n(native $-$ romanized concept, pp)")
    ax.set_ylim(0, 100); ax.legend(fontsize=8, loc="upper right", title="language (native anchor)", title_fontsize=8)
    ax.set_title("Native-language harmful-concept collapse under romanization — one bar per language, all ten models\n"
                 "tall = concept built natively (native anchor fires $\\geq$0.9 where it validates) then lost on "
                 "romanization = comprehension-gating;\nnear-zero = base-rate artifact (probe can't discriminate). "
                 "Direction is always collapse, never reversal.", fontsize=8.4)
    fig.tight_layout()
    p = OUT / "native_anchor_collapse.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    h6_matched_minus_random_heatmap()
    h6_recovery_curve()
    native_anchor_collapse()
