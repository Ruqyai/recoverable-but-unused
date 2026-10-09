#!/usr/bin/env python3
"""Phase 2 Stage 2: compute per-layer refusal directions via diff-in-means.

Reads activations from a Phase 2 Stage-1 run folder and writes one
unit-normalized direction tensor per script under `<run>/directions/`.

Usage:
    PYTHONPATH=. python3 scripts/phase2_direction_extraction/02_compute_directions.py \
        --model qwen3_4b                       # uses LATEST run for this model
    PYTHONPATH=. python3 scripts/phase2_direction_extraction/02_compute_directions.py \
        --model qwen3_4b --run-dir results/phase2/qwen3_4b/run_20260526_140530
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

from scripts.phase2_direction_extraction.config import (
    BENIGN_FILES,
    HARMFUL_FILES,
    SCRIPTS,
    resolve_run_paths,
)
from scripts.models import (
    DEFAULT_MODEL_KEY,
    list_models,
    load_model_config,
)
from scripts.utils.io import read_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_activations_by_source_id(
    activation_dir: Path,
    jsonl_path: Path,
    *,
    n_layers: int,
    hidden_dim: int,
) -> dict[str, torch.Tensor]:
    if not activation_dir.exists():
        log.warning(f"Directory not found: {activation_dir}")
        return {}
    if not jsonl_path.exists():
        log.warning(f"JSONL not found, cannot pair by source_id: {jsonl_path}")
        return {}

    id_to_source: dict[str, str] = {}
    for rec in read_jsonl(jsonl_path):
        id_to_source[rec.id] = rec.source_id

    out: dict[str, torch.Tensor] = {}
    n_unmapped = n_bad_shape = n_load_fail = 0
    for path in sorted(activation_dir.glob("*.pt")):
        record_id = path.stem
        source_id = id_to_source.get(record_id)
        if source_id is None:
            n_unmapped += 1
            continue
        try:
            t = torch.load(path, map_location="cpu")
        except Exception as e:
            log.warning(f"Failed to load {path.name}: {e}. Skipping.")
            n_load_fail += 1
            continue
        if t.shape != (n_layers, hidden_dim):
            log.warning(
                f"Wrong shape in {path.name}: expected ({n_layers}, {hidden_dim}), "
                f"got {t.shape}. Skipping."
            )
            n_bad_shape += 1
            continue
        if source_id in out:
            log.warning(
                f"Duplicate source_id {source_id!r} in {activation_dir}; "
                f"keeping first occurrence."
            )
            continue
        out[source_id] = t

    if n_unmapped or n_bad_shape or n_load_fail:
        log.info(
            f"  {activation_dir.name}: unmapped={n_unmapped}, "
            f"bad_shape={n_bad_shape}, load_fail={n_load_fail}"
        )
    return out


def compute_direction(
    script: str,
    activations_dir: Path,
    *,
    n_layers: int,
    hidden_dim: int,
) -> tuple[torch.Tensor | None, dict]:
    log.info(f"\nComputing direction for: {script}")

    harmful_dir = activations_dir / script / "harmful"
    benign_dir = activations_dir / script / "benign"

    harmful_by_src = load_activations_by_source_id(
        harmful_dir, HARMFUL_FILES[script], n_layers=n_layers, hidden_dim=hidden_dim,
    )
    benign_by_src = load_activations_by_source_id(
        benign_dir, BENIGN_FILES[script], n_layers=n_layers, hidden_dim=hidden_dim,
    )

    if not harmful_by_src:
        log.error(f"No harmful activations found for {script}")
        return None, {}
    if not benign_by_src:
        log.error(f"No benign activations found for {script}")
        return None, {}

    common_sources = sorted(set(harmful_by_src) & set(benign_by_src))
    n_drop_h = len(harmful_by_src) - len(common_sources)
    n_drop_b = len(benign_by_src) - len(common_sources)
    if n_drop_h or n_drop_b:
        log.warning(
            f"  Pair filter for {script}: dropped {n_drop_h} unmatched harmful "
            f"and {n_drop_b} unmatched benign (kept {len(common_sources)} pairs)"
        )
    if not common_sources:
        log.error(f"No source_id overlap between harmful and benign for {script}")
        return None, {}

    harmful_acts = [harmful_by_src[s] for s in common_sources]
    benign_acts = [benign_by_src[s] for s in common_sources]

    H = torch.stack(harmful_acts, dim=0)
    B = torch.stack(benign_acts, dim=0)

    h_mean_norm = float(H.mean(dim=0).norm(dim=1).mean())
    b_mean_norm = float(B.mean(dim=0).norm(dim=1).mean())
    log.info(f"  Harmful: {H.shape[0]} prompts, mean norm/layer: {h_mean_norm:.3f}")
    log.info(f"  Benign:  {B.shape[0]} prompts, mean norm/layer: {b_mean_norm:.3f}")

    diff = H.mean(dim=0) - B.mean(dim=0)
    diff_norm_mean = float(diff.norm(dim=1).mean())
    log.info(f"  Diff norm per layer: {diff_norm_mean:.3f}")

    norms = diff.norm(dim=1, keepdim=True).clamp(min=1e-8)
    direction = diff / norms
    log.info(
        f"  After normalization: min={direction.norm(dim=1).min():.6f}, "
        f"max={direction.norm(dim=1).max():.6f} (should be ~1.0)"
    )

    stats = {
        "n_pairs": len(common_sources),
        "n_harmful_loaded": len(harmful_by_src),
        "n_benign_loaded": len(benign_by_src),
        "n_dropped_harmful": n_drop_h,
        "n_dropped_benign": n_drop_b,
        "harmful_mean_norm_per_layer": h_mean_norm,
        "benign_mean_norm_per_layer": b_mean_norm,
        "diff_mean_norm_per_layer": diff_norm_mean,
    }
    return direction, stats


def main():
    parser = argparse.ArgumentParser(description="Compute refusal directions for Phase 2")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_KEY,
        choices=list_models(),
        help=f"Model key (default: {DEFAULT_MODEL_KEY})",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Phase 2 Stage 1 run folder. Defaults to the LATEST run for the model.",
    )
    args = parser.parse_args()

    cfg = load_model_config(args.model)
    run_paths = resolve_run_paths(
        model_key=cfg.MODEL_KEY,
        run_dir=args.run_dir,
        new_run=False,  # this stage always reuses an existing run
    )

    log.info("=" * 70)
    log.info("Phase 2 Stage 2 — Compute Directions (diff-in-means, L2-normalized)")
    log.info(f"  model    : {cfg.MODEL_ID}  (key={cfg.MODEL_KEY})")
    log.info(f"  run dir  : {run_paths.run_dir}")
    log.info("=" * 70)

    run_paths.directions_dir.mkdir(parents=True, exist_ok=True)

    per_script_stats: dict[str, dict] = {}
    results: dict[str, tuple[int, ...]] = {}
    for script in SCRIPTS:
        direction, stats = compute_direction(
            script,
            run_paths.activations_dir,
            n_layers=cfg.N_LAYERS,
            hidden_dim=cfg.HIDDEN_DIM,
        )
        if direction is None:
            log.error(f"Failed to compute direction for {script}")
            continue

        out_path = run_paths.directions_dir / f"{script}.pt"
        torch.save(direction, out_path)
        log.info(f"  Saved to {out_path}")
        results[script] = tuple(direction.shape)
        per_script_stats[script] = stats

    metadata = {
        "phase": "phase2_stage2_compute_directions",
        "model_key": cfg.MODEL_KEY,
        "model_id": cfg.MODEL_ID,
        "n_layers": cfg.N_LAYERS,
        "hidden_dim": cfg.HIDDEN_DIM,
        "run_stamp": run_paths.run_stamp,
        "run_dir": str(run_paths.run_dir),
        "completed_at": datetime.datetime.now().isoformat(),
        "directions": {k: list(v) for k, v in results.items()},
        "per_script_stats": per_script_stats,
    }
    run_paths.metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    log.info(f"\nWrote {run_paths.metadata_path.name}")

    log.info(f"\nCompleted. Saved directions for {len(results)} scripts:")
    for script, shape in results.items():
        log.info(f"  {script}: {shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
