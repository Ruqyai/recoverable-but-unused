#!/usr/bin/env python3
"""Phase 2 figures — the causal layer story: causality peaks MID-network,
separation peaks LATE. Attribution must target the layer we can intervene on.

Reads results/phase2/<model>/**/causal_layer_selection.json  (per-layer causal sweep)
      results/phase2/<model>/**/metadata_separation_proxy.json (per-script separation peak)
Writes results/phase2/figures/.
Run:  python3 scripts/phase2_direction_extraction/plot_figures.py

Figures
  causal_flip_curve.png       causal flip-rate vs relative depth, all models overlaid
  causal_vs_separation_depth.png  dumbbell: causal-peak depth vs separation-peak depth
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase2" / "figures"
MODELS = ["gemma_3_1b_it", "gemma_2_2b_it", "gemma_3_4b_it", "gemma_2_9b_it",
          "gemma_3_27b_it", "qwen3_4b", "qwen3_8b", "qwen2_5_7b_Instruct",
          "llama_3_1_8b_Instruct", "llama_3_2_3b_Instruct"]
short = lambda m: m.replace("_it", "").replace("_Instruct", "").replace("_", " ")
plt.rcParams.update({"font.size": 9, "figure.dpi": 140})


def _latest(pat):
    fs = sorted(glob.glob(str(ROOT / pat), recursive=True))
    return json.load(open(fs[-1])) if fs else None


def causal_flip_curve() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(MODELS)))
    peaks = []
    for c, m in zip(cmap, MODELS):
        d = _latest(f"results/phase2/{m}/**/causal_layer_selection.json")
        if not d:
            continue
        nl = d["config"]["n_layers"]
        pl = {int(k): v.get("flip_rate_lexical", np.nan) for k, v in d["per_layer"].items()}
        xs = np.array(sorted(pl)) / nl
        ys = np.array([pl[int(round(x * nl))] for x in xs])
        ax.plot(xs, ys, lw=1.3, color=c, alpha=0.85, label=short(m))
        pk = d["selection"].get("peak_causal_layer_lexical")
        if pk is not None:
            peaks.append(pk / nl)
            ax.scatter([pk / nl], [pl.get(pk, np.nan)], color=c, s=22, zorder=3, edgecolor="k", lw=0.4)
    if peaks:
        ax.axvline(float(np.median(peaks)), ls="--", c="gray", lw=0.9,
                   label=f"median peak depth {np.median(peaks):.2f}")
    ax.set_xlabel("relative depth (layer / n_layers)")
    ax.set_ylabel("causal flip rate (ablation, lexical)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Causal effect peaks mid-network, then decays", fontsize=9.5)
    ax.legend(fontsize=6.2, ncol=2, loc="lower center", framealpha=0.9)
    fig.tight_layout()
    p = OUT / "causal_flip_curve.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


def causal_vs_separation_depth() -> None:
    names, causal, sep = [], [], []
    for m in MODELS:
        d = _latest(f"results/phase2/{m}/**/causal_layer_selection.json")
        s = _latest(f"results/phase2/{m}/**/metadata_separation_proxy.json")
        if not d or not s:
            continue
        nl = d["config"]["n_layers"]
        cp = d["selection"].get("peak_causal_layer_lexical")
        sp = np.mean([c.get("relative_depth_at_peak", np.nan) for c in s["conditions"].values()])
        if cp is not None:
            names.append(short(m)); causal.append(cp / nl); sep.append(sp)
    if not names:
        print("no phase2 depth data"); return
    order = np.argsort(causal)
    names = [names[i] for i in order]; causal = [causal[i] for i in order]; sep = [sep[i] for i in order]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for i in y:
        ax.plot([causal[i], sep[i]], [i, i], color="#bbb", lw=1.8, zorder=1)
    ax.scatter(causal, y, s=48, color="#2a7", zorder=2, label="causal peak (intervene here)")
    ax.scatter(sep, y, s=48, color="#c44", zorder=2, label="separation peak (diff-in-means)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_xlabel("relative depth (layer / n_layers)")
    ax.axvline(0.5, ls=":", c="gray", lw=0.7)
    ax.set_title("Causality is mid-network; separation is late\n"
                 "(attribute against the layer you can actually steer)", fontsize=9.5)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    p = OUT / "causal_vs_separation_depth.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    causal_flip_curve()
    causal_vs_separation_depth()
