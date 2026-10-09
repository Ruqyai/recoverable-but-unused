#!/usr/bin/env python3
"""Phase 2 Stage 3: separation-norm peak layer per script (descriptive metadata).

⚠️  Separation norm is a depth-biased proxy. It is NOT the causal layer for
the model. Per OSF v6 § 4, the causal layer is selected by single-layer
ablation in Stage 04 (``04_select_causal_layer.py``), which reads this
script's output to surface the proxy-vs-causal gap directly in its summary.

Renamed 2026-05-29 from ``03_find_peak_layer.py`` to make the narrow scope
explicit. The pre-v4 cross-script ``descriptive_modal_peak_layer`` and
``sensitivity_layers ± 4`` fields were also dropped in the same commit:
v6 § 4 commits to per-model AND per-script causal layers, so averaging
peaks across scripts (the old modal peak) or hard-coding a sensitivity
window are both stale conventions. The remaining output is per-script
separation peak + top-5 + the full per-layer scores.

Consumed by:
  - Stage 04 (``04_select_causal_layer.py``) via ``load_separation_peak``
    — reads ``metadata_separation_proxy.json`` to enrich its causal
    output with the descriptive peak for direct proxy-vs-causal Δ.

Usage:
    PYTHONPATH=. python3 scripts/phase2_direction_extraction/03_find_separation_peak_layer.py \
        --model qwen3_4b
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase2_direction_extraction.config import (
    REPO,
    SCRIPTS,
    resolve_run_paths,
)
from scripts.models import (
    DEFAULT_MODEL_KEY,
    list_models,
    load_model_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_all(directory: Path, n_layers: int, hidden_dim: int) -> list[torch.Tensor]:
    if not directory.exists():
        log.warning(f"Directory not found: {directory}")
        return []
    files = sorted(directory.glob("*.pt"))
    tensors = []
    for path in files:
        try:
            t = torch.load(path, map_location="cpu")
            if t.shape != (n_layers, hidden_dim):
                log.warning(
                    f"Wrong shape in {path.name}: expected ({n_layers}, {hidden_dim}), "
                    f"got {t.shape}. Skipping."
                )
                continue
            tensors.append(t)
        except Exception as e:
            log.warning(f"Failed to load {path.name}: {e}. Skipping.")
            continue
    return tensors


def compute_separation_score(
    direction: torch.Tensor,
    script: str,
    activations_dir: Path,
    n_layers: int,
    hidden_dim: int,
) -> torch.Tensor | None:
    harmful = load_all(activations_dir / script / "harmful", n_layers, hidden_dim)
    benign = load_all(activations_dir / script / "benign", n_layers, hidden_dim)
    if not harmful or not benign:
        log.error(f"Missing activations for {script}")
        return None

    H = torch.stack(harmful, dim=0)
    B = torch.stack(benign, dim=0)

    scores = torch.zeros(n_layers)
    for L in range(n_layers):
        d_L = direction[L]
        scores[L] = (H[:, L, :] @ d_L).mean() - (B[:, L, :] @ d_L).mean()
    return scores


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Find separation-norm peak layer (descriptive, NOT causal)"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_KEY, choices=list_models(),
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Phase 2 run folder. Defaults to LATEST.",
    )
    args = parser.parse_args()

    cfg = load_model_config(args.model)
    run_paths = resolve_run_paths(cfg.MODEL_KEY, run_dir=args.run_dir, new_run=False)

    log.info("=" * 70)
    log.info("Phase 2 Stage 3 — Separation-norm peak (DESCRIPTIVE proxy)")
    log.info(f"  model   : {cfg.MODEL_ID}")
    log.info(f"  run dir : {run_paths.run_dir}")
    log.info("=" * 70)

    if not run_paths.directions_dir.exists():
        log.error(f"Directions directory not found: {run_paths.directions_dir}")
        log.error("Did you run 02_compute_directions.py first?")
        return 1

    metadata = {
        "phase": "phase2_stage3_peak_layer_separation_proxy",
        "model_key": cfg.MODEL_KEY,
        "model_id": cfg.MODEL_ID,
        "n_layers": cfg.N_LAYERS,
        "hidden_dim": cfg.HIDDEN_DIM,
        "run_stamp": run_paths.run_stamp,
        "run_date": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "caveat": (
            "Separation norm is a depth-biased proxy. The OSF v4 primary causal "
            "layer is selected by single-layer ablation in Phase 4, not here."
        ),
        "conditions": {},
    }

    peak_layers = []
    for script in SCRIPTS:
        direction_path = run_paths.directions_dir / f"{script}.pt"
        if not direction_path.exists():
            log.error(f"Direction not found for {script}: {direction_path}")
            continue

        log.info(f"\nAnalyzing script: {script}")
        direction = torch.load(direction_path, map_location="cpu")
        scores = compute_separation_score(
            direction, script, run_paths.activations_dir,
            cfg.N_LAYERS, cfg.HIDDEN_DIM,
        )
        if scores is None:
            continue

        peak_layer = int(scores.argmax().item())
        peak_layers.append(peak_layer)
        top5 = scores.argsort(descending=True)[:5].tolist()
        log.info(f"  Peak layer: {peak_layer}, separation: {scores[peak_layer]:.4f}")
        log.info(f"  Top 5 layers: {top5}")

        harmful_count = len(load_all(
            run_paths.activations_dir / script / "harmful",
            cfg.N_LAYERS, cfg.HIDDEN_DIM,
        ))
        benign_count = len(load_all(
            run_paths.activations_dir / script / "benign",
            cfg.N_LAYERS, cfg.HIDDEN_DIM,
        ))

        # Key is the plain ``peak_layer`` — the file name already encodes the
        # "separation proxy" provenance, so the per-script keys can stay short
        # and Stage 04's ``load_separation_peak`` finds them by this name.
        metadata["conditions"][script] = {
            "n_harmful": harmful_count,
            "n_benign": benign_count,
            "peak_layer": peak_layer,
            "top5_layers": top5,
            "separation_scores": scores.tolist(),
            "relative_depth_at_peak": peak_layer / max(1, cfg.N_LAYERS - 1),
        }

    if not peak_layers:
        log.error("No peak layers computed")
        return 1

    # v6 § 4 commits to per-model AND per-script causal layers. Reporting a
    # cross-script "modal peak" or a hard-coded sensitivity window would
    # imply the layers are comparable across scripts — they are not. Stage 04
    # is the authoritative answer and consumes this file's per-script peaks
    # to surface the proxy-vs-causal Δ directly.

    log.info("")
    log.info("REMINDER: separation peak is a descriptive proxy. Run Stage 04")
    log.info("(04_select_causal_layer.py) for the per-model causal answer.")

    out_path = run_paths.run_dir / "metadata_separation_proxy.json"
    out_path.write_text(json.dumps(metadata, indent=2))
    log.info(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
