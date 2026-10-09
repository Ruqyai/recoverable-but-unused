#!/usr/bin/env python3
"""Phase 4 · diagnostics figures — pre-intervention sanity audit of the prompts.

Before attributing any comprehension failure to the model, we verify the romanized
prompts are well-formed: (1) romanization fidelity score per distorted script, and
(2) response coherence at baseline. High fidelity + coherent-but-off responses means
the downstream comprehension gate is the model's behavior, not corrupt input data.

Reads  results/phase4_diagnostic/<model>/**/<tier>/summary.json
Writes results/phase4_diagnostic/figures/.
Run:   python3 scripts/phase4_causal_intervention/diagnostics/plot_figures.py

Figures
  romanization_fidelity.png   mean romanization score per distorted script (all models)
  baseline_coherence.png      fraction of coherent responses per distorted script
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "phase4_diagnostic" / "figures"
MODELS = ["gemma_3_1b_it", "gemma_2_2b_it", "gemma_3_4b_it", "gemma_2_9b_it",
          "gemma_3_27b_it", "qwen3_4b", "qwen3_8b", "qwen2_5_7b_Instruct",
          "llama_3_1_8b_Instruct", "llama_3_2_3b_Instruct"]
# distorted script -> (tier, avg_score_key, coherent_key, incoherent_key)
SCRIPTS = [
    ("Arabizi", "tier_a", "average_score", "coherent", "incoherent"),
    ("ITRANS", "tier_c", "itrans_avg_score", "itrans_coherent", "itrans_incoherent"),
    ("Leetspeak", "tier_e", "leetspeak_avg_score", "leetspeak_coherent", "leetspeak_incoherent"),
]
short = lambda m: m.replace("_it", "").replace("_Instruct", "").replace("_", " ")
plt.rcParams.update({"font.size": 9, "figure.dpi": 140})


def _summary(model, tier):
    fs = sorted(glob.glob(str(ROOT / f"results/phase4_diagnostic/{model}/**/{tier}/summary.json"),
                          recursive=True))
    return json.load(open(fs[-1])).get("stats", {}) if fs else None


def _collect():
    fid = np.full((len(MODELS), len(SCRIPTS)), np.nan)
    coh = np.full((len(MODELS), len(SCRIPTS)), np.nan)
    for i, m in enumerate(MODELS):
        for j, (_, tier, ak, ck, ik) in enumerate(SCRIPTS):
            s = _summary(m, tier)
            if not s:
                continue
            if isinstance(s.get(ak), (int, float)):
                fid[i, j] = s[ak]
            c, ic = s.get(ck), s.get(ik)
            if isinstance(c, (int, float)) and isinstance(ic, (int, float)) and (c + ic) > 0:
                coh[i, j] = c / (c + ic)
    return fid, coh


def romanization_fidelity(fid) -> None:
    cols = [s[0] for s in SCRIPTS]
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(fid, cmap="YlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([short(m) for m in MODELS], fontsize=8)
    for i in range(len(MODELS)):
        for j in range(len(cols)):
            if not np.isnan(fid[i, j]):
                ax.text(j, i, f"{fid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if fid[i, j] > 0.85 else "black")
    ax.set_title("Diagnostics — romanization fidelity per distorted script\n"
                 "high & uniform ⇒ prompts are well-formed (failures are the model's)", fontsize=9)
    fig.colorbar(im, ax=ax, label="mean romanization score", fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = OUT / "romanization_fidelity.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


def baseline_coherence(coh) -> None:
    cols = [s[0] for s in SCRIPTS]
    x = np.arange(len(MODELS)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    colors = ["#4a90c4", "#e0a83a", "#c0603a"]
    for j, (c, col) in enumerate(zip(cols, colors)):
        ax.bar(x + (j - 1) * w, coh[:, j] * 100, w, color=col, label=c)
    ax.set_xticks(x); ax.set_xticklabels([short(m) for m in MODELS], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("coherent responses (%)"); ax.set_ylim(0, 105)
    ax.set_title("Diagnostics — baseline response coherence per distorted script", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=3)
    fig.tight_layout()
    p = OUT / "baseline_coherence.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fid, coh = _collect()
    romanization_fidelity(fid)
    baseline_coherence(coh)
