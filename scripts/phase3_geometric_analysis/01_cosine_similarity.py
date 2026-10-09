#!/usr/bin/env python3
"""Phase 3 Stage 1: pairwise cosine similarity between refusal directions.

For each (script_a, script_b) pair, and each transformer layer L, compute
    cos(r̂_a[L], r̂_b[L])
where r̂ is the unit-normalized refusal direction from Phase 2.

Interpretation:
- cos ≈ +1 : the two scripts share a refusal subspace at L (semantic anchoring)
- cos ≈  0 : orthogonal — different mechanisms (orthographic anchoring)
- cos ≈ -1 : opposite — refusal in one script suppresses refusal in the other

Per OSF v4, cosine similarity is a descriptive measure, not a confirmatory test.
The orthographic-anchoring hypothesis predicts LOW cosine between native and
Latin script directions (e.g., cos(r_arabic, r_arabizi) should be small at the
causal layer).

Usage:
    PYTHONPATH=. python3 scripts/phase3_geometric_analysis/01_cosine_similarity.py \\
        --model qwen3_4b
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.models import (
    DEFAULT_MODEL_KEY,
    list_models,
    load_model_config,
)
from scripts.phase3_geometric_analysis.config import (
    SCRIPTS,
    resolve_phase2_source,
    resolve_phase3_run_paths,
    write_latest_pointer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_directions(directions_dir: Path, n_layers: int, hidden_dim: int) -> dict[str, torch.Tensor]:
    """Load one direction tensor per script from a Phase 2 run."""
    out: dict[str, torch.Tensor] = {}
    for script in SCRIPTS:
        path = directions_dir / f"{script}.pt"
        if not path.exists():
            log.warning(f"Direction missing: {path}")
            continue
        t = torch.load(path, map_location="cpu")
        if t.shape != (n_layers, hidden_dim):
            log.error(
                f"Wrong direction shape for {script}: expected ({n_layers},{hidden_dim}), "
                f"got {tuple(t.shape)}. Skipping."
            )
            continue
        out[script] = t
    return out


def pairwise_cosine_per_layer(
    directions: dict[str, torch.Tensor], n_layers: int,
) -> dict[str, dict[int, float]]:
    """Cosine similarity per layer for every (script_a, script_b) pair.

    Returns: {"<a>__<b>": {L: cos_value_at_L, ...}, ...}
    """
    scripts_present = sorted(directions.keys())
    results: dict[str, dict[int, float]] = {}
    for a, b in itertools.combinations_with_replacement(scripts_present, 2):
        ra = directions[a]  # [n_layers, hidden_dim], unit-norm per layer
        rb = directions[b]
        per_layer: dict[int, float] = {}
        for L in range(n_layers):
            cos = float(torch.dot(ra[L], rb[L]))
            per_layer[L] = round(cos, 6)
        results[f"{a}__{b}"] = per_layer
    return results


def summarize(pairs: dict[str, dict[int, float]], n_layers: int) -> dict:
    """Per-pair summary statistics across layers."""
    summary = {}
    for pair_key, per_layer in pairs.items():
        a, b = pair_key.split("__")
        values = [per_layer[L] for L in range(n_layers) if L in per_layer]
        if not values:
            continue
        peak_L = max(per_layer, key=lambda L: per_layer[L])
        min_L = min(per_layer, key=lambda L: per_layer[L])
        summary[pair_key] = {
            "script_a": a,
            "script_b": b,
            "mean_cos": round(sum(values) / len(values), 6),
            "min_cos": round(min(values), 6),
            "max_cos": round(max(values), 6),
            "argmax_layer": int(peak_L),
            "argmin_layer": int(min_L),
            "values_at_layer_5_15_25": {
                str(L): per_layer.get(L) for L in (5, 15, 25)
                if L < n_layers
            },
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Stage 1 — cosine similarity")
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=list_models())
    parser.add_argument(
        "--phase2-run-dir", type=Path, default=None,
        help="Phase 2 run to consume (default: LATEST for this model)",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Phase 3 run to write into (default: LATEST or fresh)",
    )
    parser.add_argument("--new-run", action="store_true")
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate matplotlib heatmaps (optional; requires matplotlib)",
    )
    args = parser.parse_args()

    cfg = load_model_config(args.model)
    phase2 = resolve_phase2_source(cfg.MODEL_KEY, args.phase2_run_dir)
    paths = resolve_phase3_run_paths(cfg.MODEL_KEY, args.run_dir, args.new_run)

    log.info("=" * 70)
    log.info("Phase 3 Stage 1 — Cosine similarity between refusal directions")
    log.info(f"  model        : {cfg.MODEL_ID}  (key={cfg.MODEL_KEY})")
    log.info(f"  layers       : {cfg.N_LAYERS}")
    log.info(f"  phase2 source: {phase2.run_dir}")
    log.info(f"  phase3 out   : {paths.run_dir}")
    log.info("=" * 70)

    directions = load_directions(phase2.directions_dir, cfg.N_LAYERS, cfg.HIDDEN_DIM)
    if not directions:
        log.error("No directions loaded. Did you run Phase 2 Stage 2 for this model?")
        return 1
    log.info(f"Loaded directions for scripts: {sorted(directions.keys())}")

    pairs = pairwise_cosine_per_layer(directions, cfg.N_LAYERS)
    summary = summarize(pairs, cfg.N_LAYERS)

    log.info("\nPairwise summary:")
    for key, s in summary.items():
        log.info(
            f"  {s['script_a']:11} ↔ {s['script_b']:11} | "
            f"mean={s['mean_cos']:+.3f}  min={s['min_cos']:+.3f} (L{s['argmin_layer']})  "
            f"max={s['max_cos']:+.3f} (L{s['argmax_layer']})"
        )

    output = {
        "phase": "phase3_stage1_cosine_similarity",
        "model_key": cfg.MODEL_KEY,
        "model_id": cfg.MODEL_ID,
        "n_layers": cfg.N_LAYERS,
        "hidden_dim": cfg.HIDDEN_DIM,
        "phase2_run_dir": str(phase2.run_dir),
        "phase3_run_dir": str(paths.run_dir),
        "completed_at": datetime.datetime.now().isoformat(),
        "scripts_analyzed": sorted(directions.keys()),
        "pairs": pairs,
        "summary": summary,
        "caveat": (
            "Descriptive measure. Per OSF v4, the orthographic-anchoring "
            "hypothesis is tested causally in Phase 4 (H2), not by cosine "
            "similarity here."
        ),
    }
    paths.cosine_path.write_text(json.dumps(output, indent=2))
    log.info(f"\nWrote {paths.cosine_path}")

    if args.plot:
        try:
            _plot_heatmaps(pairs, summary, cfg.N_LAYERS, paths.plots_dir)
        except ImportError:
            log.warning("matplotlib not available; skipping plots")

    write_latest_pointer(cfg.MODEL_KEY, paths.run_dir)
    log.info(f"LATEST.txt updated → {paths.run_dir.name}")
    return 0


def _plot_heatmaps(pairs: dict, summary: dict, n_layers: int, plots_dir: Path) -> None:
    """One per-layer line plot per pair (only off-diagonal pairs)."""
    import matplotlib.pyplot as plt
    layers = list(range(n_layers))
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, per_layer in pairs.items():
        a, b = key.split("__")
        if a == b:
            continue
        ys = [per_layer.get(L, float("nan")) for L in layers]
        ax.plot(layers, ys, marker="o", markersize=3, label=f"{a} ↔ {b}")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Pairwise cosine similarity between refusal directions")
    ax.axhline(0, color="black", linestyle=":", linewidth=0.5)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = plots_dir / "cosine_per_layer.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"  Plot saved: {out}")


if __name__ == "__main__":
    sys.exit(main())
