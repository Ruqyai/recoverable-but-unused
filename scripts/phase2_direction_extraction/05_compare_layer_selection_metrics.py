#!/usr/bin/env python3
"""Phase 2 Stage 5: compare separation-norm vs diff-of-means logit-difference proxies.

Both are depth-biased proxies (see 2026-05-23 log) and NEITHER is the OSF v6
primary-causal-layer test. This script is purely descriptive: it reports where
each proxy peaks per script.

As of 2026-05-29 it ALSO reads the causal-layer-selection output from Stage 04
(``causal_layer_selection.json`` in the same run dir) if present, and emits the
delta between each proxy's peak and the causal layer. The comparison is
optional — if Stage 04 has not yet been run on this model, the script still
produces its descriptive output and records ``causal_comparison: null``.

Usage:
    PYTHONPATH=. python3 scripts/phase2_direction_extraction/05_compare_layer_selection_metrics.py \
        --model qwen3_4b
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase2_direction_extraction.config import SCRIPTS, resolve_run_paths
from scripts.models import (
    DEFAULT_MODEL_KEY,
    list_models,
    load_model_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Stage 04 output reader (optional dependency) ─────────────────────────────

def load_causal_anchor(run_dir: Path) -> Optional[dict]:
    """Read the per-script causal-layer answer produced by Stage 04 if present.

    Returns a dict shaped like::

        {
          "script": "arabic",
          "peak_causal_layer_guard": 14,
          "peak_causal_layer_lexical": 14,
          "max_flip_rate_guard": 0.85,
          "max_flip_rate_lexical": 0.90,
          "source_path": "<absolute path>",
        }

    or ``None`` if Stage 04 has not yet been run on this run dir. Stage 04
    calibrates ONE script per invocation (its ``--script`` argument), so the
    returned dict only carries one script's answer.
    """
    path = run_dir / "causal_layer_selection.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        log.warning(f"Could not parse {path.name}; ignoring causal anchor.")
        return None
    sel = data.get("selection") or {}
    return {
        "script": data.get("config", {}).get("script"),
        "peak_causal_layer_guard": sel.get("peak_causal_layer_guard"),
        "peak_causal_layer_lexical": sel.get("peak_causal_layer_lexical"),
        "max_flip_rate_guard": sel.get("max_flip_rate_guard"),
        "max_flip_rate_lexical": sel.get("max_flip_rate_lexical"),
        "source_path": str(path),
    }


def causal_delta(proxy_peak: int, causal_layer: Optional[int]) -> Optional[int]:
    """Signed delta = proxy_peak − causal_layer. Returns None if no causal anchor."""
    if causal_layer is None:
        return None
    return int(proxy_peak) - int(causal_layer)


# ── Activation loaders / proxy metrics ───────────────────────────────────────

def load_all(directory: Path, n_layers: int, hidden_dim: int) -> list[torch.Tensor]:
    if not directory.exists():
        log.warning(f"Directory not found: {directory}")
        return []
    out = []
    for path in sorted(directory.glob("*.pt")):
        try:
            t = torch.load(path, map_location="cpu")
            if t.shape != (n_layers, hidden_dim):
                log.warning(f"Wrong shape in {path.name}: {t.shape}. Skipping.")
                continue
            out.append(t)
        except Exception as e:
            log.warning(f"Load failure {path.name}: {e}")
    return out


def separation_score(
    direction: torch.Tensor, script: str, activations_dir: Path, n_layers, hidden_dim
) -> torch.Tensor | None:
    H = load_all(activations_dir / script / "harmful", n_layers, hidden_dim)
    B = load_all(activations_dir / script / "benign", n_layers, hidden_dim)
    if not H or not B:
        return None
    H = torch.stack(H, dim=0); B = torch.stack(B, dim=0)
    scores = torch.zeros(n_layers)
    for L in range(n_layers):
        d = direction[L]
        scores[L] = (H[:, L, :] @ d).mean() - (B[:, L, :] @ d).mean()
    return scores


def logit_difference_norm(
    script: str, activations_dir: Path, n_layers, hidden_dim
) -> torch.Tensor | None:
    H = load_all(activations_dir / script / "harmful", n_layers, hidden_dim)
    B = load_all(activations_dir / script / "benign", n_layers, hidden_dim)
    if not H or not B:
        return None
    H = torch.stack(H, dim=0); B = torch.stack(B, dim=0)
    diffs = torch.zeros(n_layers)
    for L in range(n_layers):
        diffs[L] = (H[:, L, :].mean(dim=0) - B[:, L, :].mean(dim=0)).norm(p=2)
    return diffs


def main():
    parser = argparse.ArgumentParser(description="Compare layer-selection proxies (descriptive)")
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=list_models())
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_model_config(args.model)
    run_paths = resolve_run_paths(cfg.MODEL_KEY, args.run_dir, new_run=False)

    log.info("=" * 70)
    log.info("Phase 2 Stage 5 — Compare layer-selection proxies (DESCRIPTIVE)")
    log.info(f"  model   : {cfg.MODEL_ID}")
    log.info(f"  run dir : {run_paths.run_dir}")
    log.info("=" * 70)

    if not run_paths.directions_dir.exists():
        log.error(f"Directions directory not found: {run_paths.directions_dir}")
        return 1

    # Optional causal anchor from Stage 04
    causal_anchor = load_causal_anchor(run_paths.run_dir)
    if causal_anchor is not None:
        log.info(
            f"Stage 04 anchor found: script={causal_anchor['script']!r} "
            f"causal_L_guard={causal_anchor['peak_causal_layer_guard']} "
            f"causal_L_lex={causal_anchor['peak_causal_layer_lexical']} "
            f"max_flip_guard={causal_anchor['max_flip_rate_guard']!r}"
        )
    else:
        log.info("Stage 04 anchor: not found — proxy-vs-causal comparison disabled.")

    results = {}
    for script in SCRIPTS:
        direction_path = run_paths.directions_dir / f"{script}.pt"
        if not direction_path.exists():
            log.error(f"Direction not found: {direction_path}")
            continue

        log.info(f"\n{script.upper()}")
        direction = torch.load(direction_path, map_location="cpu")

        sep = separation_score(
            direction, script, run_paths.activations_dir,
            cfg.N_LAYERS, cfg.HIDDEN_DIM,
        )
        logd = logit_difference_norm(
            script, run_paths.activations_dir,
            cfg.N_LAYERS, cfg.HIDDEN_DIM,
        )
        if sep is None or logd is None:
            continue

        sep_peak = int(sep.argmax().item())
        log_peak = int(logd.argmax().item())
        log.info(f"  Separation-norm peak  : L{sep_peak} (score {sep[sep_peak]:.4f})")
        log.info(f"  Logit-diff norm peak  : L{log_peak} (norm {logd[log_peak]:.4f})")
        log.info(f"  Relative depths       : {sep_peak/max(1,cfg.N_LAYERS-1):.2f} / {log_peak/max(1,cfg.N_LAYERS-1):.2f}")
        log.info(f"  Proxy agreement       : {'YES' if sep_peak == log_peak else f'NO (Δ={abs(sep_peak-log_peak)})'}")

        per_script = {
            "separation_peak": sep_peak,
            "logit_difference_peak": log_peak,
            "separation_scores": sep.tolist(),
            "logit_difference_norms": logd.tolist(),
            "relative_depth_separation": sep_peak / max(1, cfg.N_LAYERS - 1),
            "relative_depth_logit_diff": log_peak / max(1, cfg.N_LAYERS - 1),
        }

        # Causal anchor comparison (only applies to the calibrated script)
        if causal_anchor is not None and causal_anchor["script"] == script:
            cg = causal_anchor["peak_causal_layer_guard"]
            cl = causal_anchor["peak_causal_layer_lexical"]
            per_script["causal_anchor"] = {
                "peak_causal_layer_guard": cg,
                "peak_causal_layer_lexical": cl,
                "max_flip_rate_guard": causal_anchor["max_flip_rate_guard"],
                "max_flip_rate_lexical": causal_anchor["max_flip_rate_lexical"],
                "delta_separation_minus_causal_guard": causal_delta(sep_peak, cg),
                "delta_separation_minus_causal_lexical": causal_delta(sep_peak, cl),
                "delta_logit_diff_minus_causal_guard": causal_delta(log_peak, cg),
                "delta_logit_diff_minus_causal_lexical": causal_delta(log_peak, cl),
                "source_path": causal_anchor["source_path"],
            }
            log.info(
                f"  Causal anchor (Guard) : L{cg}  Δsep={causal_delta(sep_peak, cg)} "
                f"Δlogit={causal_delta(log_peak, cg)}"
            )
            log.info(
                f"  Causal anchor (lex)   : L{cl}  Δsep={causal_delta(sep_peak, cl)} "
                f"Δlogit={causal_delta(log_peak, cl)}"
            )
        else:
            per_script["causal_anchor"] = None
            if causal_anchor is not None:
                log.info(
                    f"  Causal anchor         : (none — Stage 04 calibrated "
                    f"{causal_anchor['script']!r}, not {script!r})"
                )

        results[script] = per_script

    out_path = run_paths.run_dir / "metadata_proxy_comparison.json"
    out_path.write_text(json.dumps({
        "phase": "phase2_stage5_proxy_comparison",
        "model_key": cfg.MODEL_KEY,
        "model_id": cfg.MODEL_ID,
        "n_layers": cfg.N_LAYERS,
        "caveat": (
            "Both proxies are depth-biased and descriptive only. The causal "
            "layer per OSF v6 § 4 is set by Stage 04 (single-layer ablation). "
            "Per-script ``causal_anchor`` shows the Δ between each proxy's peak "
            "and the causal layer when Stage 04 has been run on that script."
        ),
        "causal_comparison_available": causal_anchor is not None,
        "results": results,
    }, indent=2))
    log.info(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
