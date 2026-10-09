#!/usr/bin/env python3
"""Phase 4 · interventions (H2) figures — forward/reverse causal intervention.

Story: the harmful concept is built from the NATIVE prompt, not the romanized one.
The native-derived direction is potent on the shared target while the romanized-derived
direction is inert, so the "contrast" between them is large positive (romanized carries
no harm-signal to steer) — except on the decodable Leetspeak (the anchoring cells).

Reads  results/phase4_intervention/<model>/**/h2_results.json
Writes results/phase4_intervention/figures/.
Run:   python3 scripts/phase4_causal_intervention/interventions/plot_figures.py

Figures
  h2_contrast_heatmap.png          native-romanized direction potency contrast (pp)
  h2_native_vs_variant_potency.png native direction potent, romanized inert (per tier)
  direction_transfer.png           native vs random ablation on the comprehended-Arabizi subset
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "phase4_intervention" / "figures"
MODELS = ["gemma_3_1b_it", "gemma_2_2b_it", "gemma_3_4b_it", "gemma_2_9b_it",
          "gemma_3_27b_it", "qwen3_4b", "qwen3_8b", "qwen2_5_7b_Instruct",
          "llama_3_1_8b_Instruct", "llama_3_2_3b_Instruct"]
TIERS = ["tier_a", "tier_c", "tier_e"]
TL = ["Arabizi", "ITRANS", "Leetspeak"]
short = lambda m: m.replace("_it", "").replace("_Instruct", "").replace("_", " ")
plt.rcParams.update({"font.size": 9, "figure.dpi": 140})


def _h2(model, tier, mode="ablation"):
    best = None
    for f in glob.glob(str(ROOT / f"results/phase4_intervention/{model}/**/h2_results.json"), recursive=True):
        x = json.load(open(f))
        if x.get("pair") == tier and x.get("mode") == mode:
            best = x
    return best


def h2_contrast_heatmap() -> None:
    mat = np.full((len(MODELS), 3), np.nan)
    for i, m in enumerate(MODELS):
        for j, t in enumerate(TIERS):
            x = _h2(m, t)
            if x:
                mat[i, j] = x.get("same_target_contrasts", {}).get("primary_contrast_pp", np.nan)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-100, vmax=100, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(TL)
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([short(m) for m in MODELS], fontsize=8)
    for i in range(len(MODELS)):
        for j in range(3):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(mat[i, j]) > 55 else "black")
    ax.set_title("H2 — native$-$romanized direction potency (pp)\nnear 0 = interchangeable (shared axis)",
                 fontsize=9.5, pad=8)
    fig.colorbar(im, ax=ax, label="contrast (pp)", fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = OUT / "h2_contrast_heatmap.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


def h2_native_vs_variant_potency() -> None:
    nat = np.full((len(TIERS), len(MODELS)), np.nan)
    var = np.full((len(TIERS), len(MODELS)), np.nan)
    for j, t in enumerate(TIERS):
        for i, m in enumerate(MODELS):
            x = _h2(m, t)
            if x:
                sc = x.get("same_target_contrasts", {})
                nat[j, i] = abs(sc.get("target_native_pp", np.nan))
                var[j, i] = abs(sc.get("target_variant_pp", np.nan))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(TIERS)); w = 0.36
    ax.bar(x - w / 2, np.nanmean(nat, 1), w, yerr=np.nanstd(nat, 1), capsize=3,
           color="#4a90c4", label="native-derived direction")
    ax.bar(x + w / 2, np.nanmean(var, 1), w, yerr=np.nanstd(var, 1), capsize=3,
           color="#e08a4a", label="romanized-derived direction")
    ax.set_xticks(x); ax.set_xticklabels(TL)
    ax.set_ylabel("|flip potency on same target| (pp)")
    ax.set_title("H2 — native-derived direction is potent, romanized-derived is inert\n"
                 "(mean ± sd over 10 models) → romanized prompt carries no harm-signal to steer", fontsize=9)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    p = OUT / "h2_native_vs_variant_potency.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


def _transfer(model):
    """Latest direction-transfer result for a model (or None)."""
    fs = sorted(glob.glob(str(ROOT / f"results/phase4_intervention/{model}/direction_transfer/transfer_*.json")))
    return json.load(open(fs[-1])) if fs else None


def direction_transfer() -> None:
    """The positive half: ablating the NATIVE (Arabic) direction on the comprehended-Arabizi
    subset removes refusal, while a norm-matched RANDOM direction does not. Bars = flip rate;
    room (baseline-refused count) annotated; models with room=0 listed separately."""
    rows, no_room = [], []
    for m in MODELS:
        x = _transfer(m)
        if not x:
            continue
        room = x.get("room_baseline_refused", 0)
        nat = x.get("native_dir_ablation", {}).get("flip_rate_pct")
        rnd = x.get("random_dir_ablation", {}).get("flip_rate_pct")
        alpha = x.get("alpha")
        if room and room > 0 and nat is not None and not np.isnan(nat):
            rows.append((m, room, nat, rnd if rnd is not None else np.nan, alpha))
        else:
            no_room.append(m)
    rows.sort(key=lambda r: r[1])  # by room ascending → highest room at top (barh is bottom-up)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    y = np.arange(len(rows)); h = 0.38
    nat_v = [r[2] for r in rows]
    rnd_v = [r[3] for r in rows]
    ax.barh(y + h / 2, nat_v, h, color="#2e7d32", label="ablate NATIVE (Arabic) direction")
    ax.barh(y - h / 2, rnd_v, h, color="#b0b0b0", label="ablate norm-matched RANDOM (control)")
    for i, r in enumerate(rows):
        ax.text(r[2] + 1.5, i + h / 2, f"{r[2]:.0f}%", va="center", fontsize=7.5, color="#2e7d32")
        rv = 0 if (r[3] is None or np.isnan(r[3])) else r[3]
        ax.text(rv + 1.5, i - h / 2, f"{rv:.0f}%", va="center", fontsize=7.5, color="#666")
    labels = [f"{short(r[0])}\n(room {r[1]}, α {r[4]:g})" for r in rows]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 112); ax.set_xlabel("refusal-removal flip rate (%)")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title("Direction transfer on the comprehended-Arabizi subset\n"
                 "the NATIVE refusal direction mediates refusal once the prompt is read "
                 "(random does not)", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    if no_room:
        ax.text(0.02, -0.30, "room = 0 (comply even when comprehending, nothing to ablate): "
                + ", ".join(short(m) for m in no_room),
                transform=ax.transAxes, fontsize=6.8, color="#777", style="italic")
    fig.tight_layout()
    p = OUT / "direction_transfer.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    h2_contrast_heatmap()
    h2_native_vs_variant_potency()
    direction_transfer()
