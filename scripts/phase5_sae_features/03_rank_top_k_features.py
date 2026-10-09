#!/usr/bin/env python3
"""Phase 5 Stage 3: rank the top-K SAE features per script (H3).

Reads the per-prompt feature matrices from Stage 02 and, for each script,
ranks features by mean activation across prompts, keeping the top-K (default
100, per the H3 wording). Writes `top100_per_script.json`:

    {
      "k": 100,
      "scripts": {
        "arabic":  {"n_prompts": 200, "top_k": [idx, ...],
                    "scores": {idx: mean_activation, ...}},
        ...
      }
    }

Ranking statistic
-----------------
Mean activation across prompts (`--statistic mean`, default) — SAE/transcoder
features are non-negative (JumpReLU / ReLU-style), so the mean magnitude is a
clean "how active is this feature on this script" score. `--statistic
frequency` ranks by the fraction of prompts on which the feature fires
(activation > --fire-threshold) instead, a sparsity-robust alternative.

Per the deposited OSF preregistration, H3 carries no confirmatory weight.

Usage
-----
  PYTHONPATH=. python3 scripts/phase5_sae_features/03_rank_top_k_features.py --model gemma_2_9b_it
  PYTHONPATH=. python3 scripts/phase5_sae_features/03_rank_top_k_features.py \
      --model gemma_3_1b_it --k 100 --statistic frequency
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

from scripts.models import DEFAULT_MODEL_KEY, list_models, load_model_config
from scripts.phase5_sae_features.config import (
    SCRIPTS,
    resolve_phase5_run_paths,
    write_latest_pointer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def score_features(features: torch.Tensor, statistic: str, fire_threshold: float) -> torch.Tensor:
    """Per-feature score vector [d_sae] from a [N, d_sae] activation matrix."""
    if statistic == "frequency":
        return (features > fire_threshold).float().mean(dim=0)
    # default: mean activation
    return features.mean(dim=0)


def top_k_indices(scores: torch.Tensor, k: int) -> list[int]:
    k = min(k, scores.numel())
    vals, idx = torch.topk(scores, k)
    # keep only features that actually activate (score > 0) to avoid padding the
    # set with dead features when fewer than k features are active
    return [int(i) for i, v in zip(idx.tolist(), vals.tolist(), strict=True) if v > 0]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Phase 5 Stage 3 — top-K SAE features per script (H3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=list_models())
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Phase 5 run with feature_activations/ (default: LATEST).")
    ap.add_argument("--scripts", nargs="+", default=SCRIPTS)
    ap.add_argument("--k", type=int, default=100, help="Top-K features per script.")
    ap.add_argument("--statistic", choices=["mean", "frequency"], default="mean")
    ap.add_argument("--fire-threshold", type=float, default=0.0,
                    help="Activation > threshold counts as 'fired' (frequency mode).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_model_config(args.model)
    paths = resolve_phase5_run_paths(cfg.MODEL_KEY, args.run_dir, new_run=False)

    layer = None
    if paths.sae_meta_path.exists():
        layer = json.loads(paths.sae_meta_path.read_text()).get("layer")

    log.info("=" * 70)
    log.info("Phase 5 Stage 3 — Top-%d features per script (H3)", args.k)
    log.info("  model     : %s  (key=%s)", cfg.MODEL_ID, cfg.MODEL_KEY)
    log.info("  layer     : %s", layer)
    log.info("  statistic : %s", args.statistic)
    log.info("  run dir   : %s", paths.run_dir)
    log.info("=" * 70)

    per_script: dict[str, dict] = {}
    for script in args.scripts:
        if script not in SCRIPTS:
            log.error("Unknown script: %s", script)
            continue
        fpath = paths.feature_acts_dir / f"{script}.pt"
        if not fpath.exists():
            log.warning("  %s missing — run Stage 02 for %s", fpath.name, script)
            continue
        blob = torch.load(fpath, map_location="cpu")
        features = blob["features"]  # [N, d_sae]
        scores = score_features(features, args.statistic, args.fire_threshold)
        top = top_k_indices(scores, args.k)
        per_script[script] = {
            "n_prompts": int(features.shape[0]),
            "d_sae": int(features.shape[1]),
            "n_active_features": int((scores > 0).sum().item()),
            "top_k": top,
            "scores": {str(i): float(scores[i]) for i in top},
        }
        log.info("  %-11s N=%d  active=%d  |top_k|=%d",
                 script, features.shape[0], per_script[script]["n_active_features"], len(top))

    if not per_script:
        log.error("No feature matrices found. Run Stage 02 first.")
        return 1

    output = {
        "phase": "phase5_stage3_top_k_features",
        "model_key": cfg.MODEL_KEY,
        "model_id": cfg.MODEL_ID,
        "layer": layer,
        "k": args.k,
        "statistic": args.statistic,
        "fire_threshold": args.fire_threshold,
        "completed_at": datetime.datetime.now().isoformat(),
        "scripts": per_script,
        "caveat": (
            "top_k may be shorter than k when fewer than k features are active "
            "(dead features are not padded into the set). Per the deposited OSF "
            "preregistration, H3 is exploratory."
        ),
    }
    paths.top_k_path.write_text(json.dumps(output, indent=2))
    log.info("Wrote %s", paths.top_k_path)

    write_latest_pointer(cfg.MODEL_KEY, paths.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
