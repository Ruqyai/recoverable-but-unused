#!/usr/bin/env python3
"""
03_plot_recover_curves.py — Phase 4b (H6): plot the cross-script recover curves.

Reads the confirmatory `patching_results.json` produced by
`01_activation_patching.py` for one or more models and renders, per model:

  - recover(L) for the MATCHED Arabic->Arabizi patch  (the H6a signal)
  - the RANDOM-pair control curve                      (the specificity floor)
  - the SELF-patch control curve                       (the no-op sanity line)
  - the BENIGN-matched control curve                   (the harm-vs-script
    disambiguator; absent from pre-2026-06-05 runs)

and a combined `matched - random` delta panel across models (the registered
H6a confirmatory metric: confirm parse-bottleneck if delta >= 25pp above the
random-pair control; reject if < 10pp).

NOTE: a near-zero `matched - random` delta REJECTS the parse bottleneck but does
NOT by itself confirm script-anchoring (S). The benign curve + 02_logit_lens.py
are required to disambiguate S from a generic harm / decision-state transfer.

No GPU / model needed — pure plotting from the result JSONs. Output PNGs land
under `plots/phase4_patching/`.

USAGE
  PYTHONPATH=. python3 scripts/phase4_causal_intervention/patching/03_plot_recover_curves.py \
      results/phase4_patching/qwen3_4b/run_20260604_050002/patching_results.json \
      results/phase4_patching/llama_3_1_8b_Instruct/run_20260604_182444/patching_results.json \
      results/phase4_patching/qwen3_8b/run_20260605_145234/patching_results.json
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "plots" / "phase4_patching"

# H6a registered thresholds (matched-minus-random delta, percentage points)
CONFIRM_PP = 25.0
REJECT_PP = 10.0


def _curve(d: dict, key: str) -> tuple[list[int], list[float]]:
    """Return (layers, rates) sorted by integer layer index."""
    raw = d.get(key) or {}
    layers = sorted(int(k) for k in raw)
    return layers, [raw[str(L)] for L in layers]


def plot_model(result_path: Path) -> dict:
    d = json.loads(result_path.read_text())
    model = d["model_key"]
    Lm, matched = _curve(d, "recover_curve_A2B")
    Lr, random_ctrl = _curve(d, "controls_random")
    Ls, self_ctrl = _curve(d, "controls_self")
    Lb, benign_ctrl = _curve(d, "controls_benign")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(Lm, [v * 100 for v in matched], "-o", ms=3, lw=2,
            color="#c0392b", label="matched (Arabic→Arabizi)")
    if Lr:
        ax.plot(Lr, [v * 100 for v in random_ctrl], "--s", ms=3, lw=1.5,
                color="#2980b9", label="random-pair control")
    if Lb:
        ax.plot(Lb, [v * 100 for v in benign_ctrl], "-.^", ms=3, lw=1.5,
                color="#27ae60", label="benign-matched control")
    if Ls:
        ax.plot(Ls, [v * 100 for v in self_ctrl], ":", lw=1.5,
                color="#7f8c8d", label="self-patch control (no-op)")

    pk = d.get("recover_peak_layer")
    if pk is not None:
        ax.axvline(pk, color="#c0392b", alpha=0.25, lw=1)
    ax.set_xlabel("patched layer L")
    ax.set_ylabel("recover rate  (% comply→refuse)")
    ax.set_title(f"H6 activation patching — {model}  ({d['pair']}, N={d['n_pairs']}, room={d.get('n_room')})")
    ax.set_ylim(-2, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"recover_{model}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)

    # delta (matched - random) per shared layer
    rmap = {int(k): v for k, v in (d.get("controls_random") or {}).items()}
    deltas = {L: matched[i] - rmap.get(L, 0.0) for i, L in enumerate(Lm)}
    best_L = max(deltas, key=lambda L: deltas[L]) if deltas else None
    verdict = "REJECT (parse-bottleneck ruled out)"
    if best_L is not None and deltas[best_L] * 100 >= CONFIRM_PP:
        verdict = "CONFIRM parse-bottleneck"
    elif best_L is not None and deltas[best_L] * 100 >= REJECT_PP:
        verdict = "INDETERMINATE"
    bmap = {int(k): v for k, v in (d.get("controls_benign") or {}).items()}
    benign_at_peak = bmap.get(pk) if pk is not None else None
    return {
        "model": model,
        "peak_layer": pk,
        "peak_matched_pp": (d.get("recover_peak_rate") or 0.0) * 100,
        "benign_at_peak_pp": (benign_at_peak * 100) if benign_at_peak is not None else None,
        "benign_present": bool(bmap),
        "max_delta_pp": deltas[best_L] * 100 if best_L is not None else 0.0,
        "max_delta_layer": best_L,
        "verdict": verdict,
        "png": str(out.relative_to(REPO_ROOT)),
        "Lm": Lm, "deltas": [deltas[L] * 100 for L in Lm],
    }


def plot_delta_panel(summaries: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c0392b", "#27ae60", "#8e44ad", "#d35400", "#16a085"]
    for s, c in zip(summaries, colors):
        ax.plot(s["Lm"], s["deltas"], "-o", ms=3, lw=1.8, color=c, label=s["model"])
    ax.axhline(CONFIRM_PP, color="green", ls="--", lw=1, label=f"confirm ≥{CONFIRM_PP:.0f}pp")
    ax.axhline(REJECT_PP, color="orange", ls="--", lw=1, label=f"reject floor {REJECT_PP:.0f}pp")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("patched layer L")
    ax.set_ylabel("matched − random  (pp)")
    ax.set_title("H6a specificity: matched-pair recovery above random-pair control")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    out = OUT_DIR / "delta_matched_minus_random.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("usage: 03_plot_recover_curves.py <patching_results.json> [...]")
        sys.exit(2)
    summaries = [plot_model(p) for p in paths]
    panel = plot_delta_panel(summaries)
    print(f"\nWrote {len(summaries)} per-model plots + delta panel to {OUT_DIR}/\n")
    print(f"{'model':<22}{'peak L':>7}{'matched%':>10}{'benign@pk':>11}{'max Δ(m−r)pp':>14}{'  verdict'}")
    for s in summaries:
        ben = "  n/a" if s["benign_at_peak_pp"] is None else f"{s['benign_at_peak_pp']:>9.1f}%"
        print(f"{s['model']:<22}{str(s['peak_layer']):>7}{s['peak_matched_pp']:>9.1f}%"
              f"{ben}{s['max_delta_pp']:>13.1f} @L{s['max_delta_layer']}   {s['verdict']}")
    if any(not s["benign_present"] for s in summaries):
        print("\n⚠ benign-matched control absent from one or more runs (pre-2026-06-05). "
              "Re-run 01 with --controls (or --benign-topup-from) before any P-vs-S claim.")
    print("⚠ verdict column is H6a (P) ONLY. matched≈random REJECTS P but does NOT confirm S — "
          "needs benign curve + 02_logit_lens.py.")
    print(f"\ndelta panel: {panel.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
