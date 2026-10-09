#!/usr/bin/env python3
"""Phase 3 Stage 2: per-layer magnitude trajectory of the (raw) refusal direction.

Phase 2 saves unit-normalized directions (‖r̂_L‖ = 1 by construction). The raw
magnitude ‖r_L‖ = ‖mean(harmful_L) − mean(benign_L)‖ is a descriptive
"signal strength" measure that grows monotonically with residual-stream
accumulation. This script recomputes the raw magnitudes from the saved
activations.

⚠️ Magnitude is depth-biased and NOT a causal-layer indicator (see 2026-05-23
log entry: Qwen2.5-7B's magnitude peaked at L26 but L26 produced 0/5 ablation
flips). Use only descriptively, e.g. in trajectory plots or to compare relative
script "strength" at a single layer.

For OSF v4, the causal layer per model is selected by Phase 4 single-layer
ablation, NOT by this script.

Usage:
    PYTHONPATH=. python3 scripts/phase3_geometric_analysis/02_magnitude_analysis.py \\
        --model qwen3_4b
"""
from __future__ import annotations

import argparse
import datetime
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
from scripts.utils.io import read_jsonl
from scripts.phase3_geometric_analysis.config import (
    BENIGN_FILES,
    HARMFUL_FILES,
    SCRIPTS,
    resolve_phase2_source,
    resolve_phase3_run_paths,
    write_latest_pointer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_activations_by_source_id(
    activation_dir: Path, jsonl_path: Path, n_layers: int, hidden_dim: int,
) -> dict[str, torch.Tensor]:
    """Load activations keyed by source_id (cross-condition pairing key)."""
    if not activation_dir.exists() or not jsonl_path.exists():
        return {}
    id_to_source = {rec.id: rec.source_id for rec in read_jsonl(jsonl_path)}
    out: dict[str, torch.Tensor] = {}
    for path in sorted(activation_dir.glob("*.pt")):
        src = id_to_source.get(path.stem)
        if src is None:
            continue
        t = torch.load(path, map_location="cpu")
        if t.shape != (n_layers, hidden_dim):
            continue
        out.setdefault(src, t)
    return out


def raw_magnitude_per_layer(
    script: str, activations_dir: Path, n_layers: int, hidden_dim: int,
) -> tuple[torch.Tensor | None, dict]:
    """Recompute ‖r_L‖ for the script using only intersection-paired activations.

    Returns (per_layer_magnitudes_tensor, stats_dict). Magnitudes preserve the
    diff-of-means convention from Phase 2 Stage 2 to ensure consistency.
    """
    H = load_activations_by_source_id(
        activations_dir / script / "harmful", HARMFUL_FILES[script], n_layers, hidden_dim,
    )
    B = load_activations_by_source_id(
        activations_dir / script / "benign", BENIGN_FILES[script], n_layers, hidden_dim,
    )
    if not H or not B:
        return None, {"error": "missing_activations"}

    common = sorted(set(H) & set(B))
    if not common:
        return None, {"error": "no_paired_sources"}

    H_stack = torch.stack([H[s] for s in common], dim=0)   # [N, L, D]
    B_stack = torch.stack([B[s] for s in common], dim=0)   # [N, L, D]

    diff = H_stack.mean(dim=0) - B_stack.mean(dim=0)        # [L, D]
    magnitudes = diff.norm(dim=1)                            # [L]

    stats = {
        "n_pairs": len(common),
        "harmful_mean_norm_per_layer": float(H_stack.mean(dim=0).norm(dim=1).mean()),
        "benign_mean_norm_per_layer": float(B_stack.mean(dim=0).norm(dim=1).mean()),
        "diff_mean_magnitude_across_layers": float(magnitudes.mean()),
        "diff_peak_layer": int(magnitudes.argmax().item()),
        "diff_peak_magnitude": float(magnitudes.max().item()),
        "diff_min_layer": int(magnitudes.argmin().item()),
        "diff_min_magnitude": float(magnitudes.min().item()),
    }
    return magnitudes, stats


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Stage 2 — magnitude trajectory")
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=list_models())
    parser.add_argument("--phase2-run-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--new-run", action="store_true")
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate matplotlib trajectory plot (optional)",
    )
    args = parser.parse_args()

    cfg = load_model_config(args.model)
    phase2 = resolve_phase2_source(cfg.MODEL_KEY, args.phase2_run_dir)
    paths = resolve_phase3_run_paths(cfg.MODEL_KEY, args.run_dir, args.new_run)

    log.info("=" * 70)
    log.info("Phase 3 Stage 2 — Raw direction magnitude per layer (DESCRIPTIVE)")
    log.info(f"  model        : {cfg.MODEL_ID}  (key={cfg.MODEL_KEY})")
    log.info(f"  layers       : {cfg.N_LAYERS}")
    log.info(f"  phase2 source: {phase2.run_dir}")
    log.info(f"  phase3 out   : {paths.run_dir}")
    log.info("=" * 70)

    per_script: dict[str, dict] = {}
    for script in SCRIPTS:
        log.info(f"\nAnalyzing script: {script}")
        magnitudes, stats = raw_magnitude_per_layer(
            script, phase2.activations_dir, cfg.N_LAYERS, cfg.HIDDEN_DIM,
        )
        if magnitudes is None:
            log.warning(f"  Skipped: {stats.get('error', 'unknown')}")
            per_script[script] = stats
            continue
        log.info(
            f"  n_pairs={stats['n_pairs']}  peak L{stats['diff_peak_layer']} "
            f"(‖r‖={stats['diff_peak_magnitude']:.3f})  "
            f"mean ‖r‖={stats['diff_mean_magnitude_across_layers']:.3f}"
        )
        per_script[script] = {
            **stats,
            "magnitudes_per_layer": [round(float(m), 4) for m in magnitudes],
            "relative_depth_at_peak": stats["diff_peak_layer"] / max(1, cfg.N_LAYERS - 1),
        }

    # Cross-script normalization: relative magnitude at every layer.
    # Useful for comparing scripts at matched depth (smaller models have fewer layers).
    relative = {}
    for script, data in per_script.items():
        mags = data.get("magnitudes_per_layer")
        if not mags:
            continue
        peak = max(mags)
        relative[script] = [round(m / peak, 4) if peak > 0 else 0.0 for m in mags]

    output = {
        "phase": "phase3_stage2_magnitude_analysis",
        "model_key": cfg.MODEL_KEY,
        "model_id": cfg.MODEL_ID,
        "n_layers": cfg.N_LAYERS,
        "hidden_dim": cfg.HIDDEN_DIM,
        "phase2_run_dir": str(phase2.run_dir),
        "phase3_run_dir": str(paths.run_dir),
        "completed_at": datetime.datetime.now().isoformat(),
        "per_script": per_script,
        "relative_magnitude_per_layer": relative,
        "caveat": (
            "Raw magnitude is depth-biased and grows monotonically with "
            "residual-stream accumulation. NOT a causal layer indicator. "
            "Per OSF v4 the causal layer is selected by Phase 4 single-layer "
            "ablation."
        ),
    }
    paths.magnitude_path.write_text(json.dumps(output, indent=2))
    log.info(f"\nWrote {paths.magnitude_path}")

    if args.plot:
        try:
            _plot_trajectory(per_script, cfg.N_LAYERS, paths.plots_dir)
        except ImportError:
            log.warning("matplotlib not available; skipping plot")

    write_latest_pointer(cfg.MODEL_KEY, paths.run_dir)
    log.info(f"LATEST.txt updated → {paths.run_dir.name}")
    return 0


def _plot_trajectory(per_script: dict, n_layers: int, plots_dir: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = list(range(n_layers))
    for script, data in per_script.items():
        mags = data.get("magnitudes_per_layer")
        if not mags:
            continue
        ax.plot(layers, mags, marker="o", markersize=3, label=script)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("‖r_L‖ (raw direction magnitude)")
    ax.set_title("Refusal direction magnitude per layer (descriptive)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = plots_dir / "magnitude_trajectory.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"  Plot saved: {out}")


if __name__ == "__main__":
    sys.exit(main())
