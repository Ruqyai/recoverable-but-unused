#!/usr/bin/env python3
"""Phase 3 figures — geometry of the harmful direction across scripts.

Headline: native and romanized harmful directions START anti-correlated in early
layers, then CONVERGE to a shared axis at mid-network — the geometric signature of
comprehension (once the model "reads" the romanized text, its harmful direction
aligns with the native one).

Reads results/phase3/<model>/**/cosine_similarity.json  (per-layer cross-script cosine)
      results/phase3/<model>/**/silhouette.json          (harmful/benign silhouette)
Writes results/phase3/figures/.
Run:  python3 scripts/phase3_geometric_analysis/plot_figures.py

Figures
  cosine_convergence.png        native vs romanized cosine vs depth (3 tier panels)
  cross_script_cosine_heatmap.png  6x6 mean cross-script cosine (anchor model)
  h5_silhouette.png             harmful/benign silhouette vs Wang 0.24 reference
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3" / "figures"
MODELS = ["gemma_3_1b_it", "gemma_2_2b_it", "gemma_3_4b_it", "gemma_2_9b_it",
          "gemma_3_27b_it", "qwen3_4b", "qwen3_8b", "qwen2_5_7b_Instruct",
          "llama_3_1_8b_Instruct", "llama_3_2_3b_Instruct"]
ANCHOR = "gemma_3_4b_it"
SCRIPTS = ["arabic", "arabizi", "devanagari", "itrans", "english", "leetspeak"]
# native <-> romanized tier pairs
TIER_PAIRS = [("arabic", "arabizi", "Arabic ↔ Arabizi"),
              ("english", "leetspeak", "English ↔ Leetspeak"),
              ("devanagari", "itrans", "Devanagari ↔ ITRANS")]
short = lambda m: m.replace("_it", "").replace("_Instruct", "").replace("_", " ")
plt.rcParams.update({"font.size": 9, "figure.dpi": 140})


def _latest(pat):
    fs = sorted(glob.glob(str(ROOT / pat), recursive=True))
    return json.load(open(fs[-1])) if fs else None


def _pair_curve(cos, a, b):
    pairs = cos.get("pairs", {})
    d = pairs.get(f"{a}__{b}") or pairs.get(f"{b}__{a}")
    if not d:
        return None, None
    ks = sorted(int(k) for k in d)
    return np.array(ks), np.array([d[str(k)] for k in ks])


def cosine_convergence() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, (a, b, title) in zip(axes, TIER_PAIRS):
        allrel, ally = [], []
        for m in MODELS:
            cos = _latest(f"results/phase3/{m}/**/cosine_similarity.json")
            if not cos:
                continue
            ks, ys = _pair_curve(cos, a, b)
            if ks is None:
                continue
            nl = cos.get("n_layers", ks.max() + 1)
            rel = ks / nl
            ax.plot(rel, ys, lw=0.9, color="#9bb", alpha=0.55)
            allrel.append(rel); ally.append(ys)
        # mean curve on a common grid
        if ally:
            grid = np.linspace(0, 1, 40)
            stk = np.vstack([np.interp(grid, r, y) for r, y in zip(allrel, ally)])
            ax.plot(grid, stk.mean(0), lw=2.4, color="#c0392b", label="mean over 10 models")
        ax.axhline(0, c="gray", lw=0.7); ax.set_ylim(-1.05, 1.05); ax.set_xlim(0, 1)
        ax.axhspan(-1.05, 0, color="#f4d3d3", alpha=0.35)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("relative depth")
    axes[0].set_ylabel("cosine(native dir, romanized dir)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Native & romanized harmful directions converge at mid-network "
                 "(anti-correlated early → aligned mid)", fontsize=10, y=1.03)
    fig.tight_layout()
    p = OUT / "cosine_convergence.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


def cross_script_cosine_heatmap() -> None:
    cos = _latest(f"results/phase3/{ANCHOR}/**/cosine_similarity.json")
    if not cos:
        print("no anchor cosine"); return
    summ = cos.get("summary", {})
    n = len(SCRIPTS); mat = np.full((n, n), np.nan)
    for i, a in enumerate(SCRIPTS):
        for j, b in enumerate(SCRIPTS):
            s = summ.get(f"{a}__{b}") or summ.get(f"{b}__{a}")
            if s:
                mat[i, j] = s.get("mean_cos", np.nan)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(SCRIPTS, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(SCRIPTS, fontsize=8)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(mat[i, j]) > 0.55 else "black")
    ax.set_title(f"Mean cross-script cosine ({short(ANCHOR)})\n"
                 "romanized scripts align with their native counterparts", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean cosine over depth")
    fig.tight_layout()
    p = OUT / "cross_script_cosine_heatmap.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


def silhouette() -> None:
    names, ar, az = [], [], []
    for m in MODELS:
        b = (_latest(f"results/phase3/{m}/**/silhouette.json") or {}).get(
            "binary_silhouette_one_vs_rest", {})
        if not b:
            continue
        sv = lambda s: b.get(s, {}).get("silhouette_one_vs_rest")
        names.append(short(m)); ar.append(sv("arabic")); az.append(sv("arabizi"))
    if not names:
        print("no silhouette data"); return
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ax.bar(x - w / 2, ar, w, label="Arabic (native)", color="#69c")
    ax.bar(x + w / 2, az, w, label="Arabizi (romanized)", color="#e88")
    ax.axhline(0.24, ls="--", c="k", lw=0.9, label="Wang 0.24 ref")
    ax.axhline(0, c="gray", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("silhouette"); ax.legend(fontsize=7)
    ax.set_title("H5 — harmful/benign silhouette: Wang 0.24 not reproduced (indeterminate)", fontsize=9)
    fig.tight_layout()
    p = OUT / "h5_silhouette.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cosine_convergence()
    cross_script_cosine_heatmap()
    silhouette()
