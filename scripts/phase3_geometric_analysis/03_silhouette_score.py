#!/usr/bin/env python3
"""Phase 3 Stage 3: silhouette analysis on per-prompt activation clouds (H5).

Wang et al. (NeurIPS 2025) reported a silhouette gap of 0.24 (Arabic) vs
0.39 (English) on Qwen2.5-7B. H5 in OSF v4 predicts that the gap NARROWS
when silhouette is computed using Arabizi prompts instead of native Arabic
— specifically Sil_Arabizi > 0.30 (rejection: Sil_Arabizi ∈ [0.19, 0.29]).

We compute silhouette on the activation cloud of harmful prompts at a chosen
layer, treating script identity as cluster label. Two flavors:

    (a) BINARY (default): one script vs all-other-scripts pooled.
        Returns one silhouette score per script.
    (b) MULTI: all scripts as separate clusters, single silhouette score.

For H5, we report flavor (a) for Arabic and Arabizi, compare to Wang's 0.24,
and bootstrap-resample to get a 99% CI.

⚠️ Per OSF v4: H5 is tested on Qwen models (primary Qwen3-4B; replication
Qwen3-8B). On the two Gemma models silhouette is reported as exploratory only,
because Wang's reference value is Qwen2.5-7B-specific.

The chosen layer defaults to the model's relative-depth 0.5 (middle of the
network); override with --layer L. For H5 confirmation in the paper, use the
empirically-selected causal layer from Phase 4.

Usage:
    PYTHONPATH=. python3 scripts/phase3_geometric_analysis/03_silhouette_score.py \\
        --model qwen3_4b
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from sklearn.metrics import silhouette_score, silhouette_samples
except ImportError as e:
    silhouette_score = None
    silhouette_samples = None
    _SKLEARN_IMPORT_ERROR = e
else:
    _SKLEARN_IMPORT_ERROR = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.models import (
    DEFAULT_MODEL_KEY,
    list_models,
    load_model_config,
)
from scripts.utils.io import read_jsonl
from scripts.phase3_geometric_analysis.config import (
    HARMFUL_FILES,
    SCRIPTS,
    resolve_phase2_source,
    resolve_phase3_run_paths,
    write_latest_pointer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


WANG_BASELINE_SIL_ARABIC = 0.24    # Wang et al. (NeurIPS 2025), Qwen2.5-7B
WANG_BASELINE_SIL_ENGLISH = 0.39

# H5 thresholds per OSF v4
H5_CONFIRMATION_THRESHOLD = 0.30   # Sil_Arabizi > 0.30 = NARROWING
H5_REJECTION_RANGE = (0.19, 0.29)  # Sil_Arabizi ∈ [0.19, 0.29] = PERSISTS

# Models on which H5 is confirmatory; on others, results are exploratory.
H5_CONFIRMATORY_MODELS = {"qwen3_4b", "qwen3_8b"}


def load_harmful_activations_at_layer(
    activations_dir: Path,
    script: str,
    layer: int,
    n_layers: int,
    hidden_dim: int,
) -> tuple[np.ndarray, list[str]]:
    """Load all harmful activations for a script at one layer.

    Returns (activations [N, hidden_dim], list_of_source_ids).
    """
    harmful_dir = activations_dir / script / "harmful"
    if not harmful_dir.exists():
        return np.zeros((0, hidden_dim), dtype=np.float32), []
    jsonl = HARMFUL_FILES[script]
    id_to_source = {rec.id: rec.source_id for rec in read_jsonl(jsonl)}

    rows: list[np.ndarray] = []
    sources: list[str] = []
    for path in sorted(harmful_dir.glob("*.pt")):
        src = id_to_source.get(path.stem)
        if src is None:
            continue
        t = torch.load(path, map_location="cpu")
        if t.shape != (n_layers, hidden_dim):
            continue
        rows.append(t[layer].float().numpy())
        sources.append(src)
    if not rows:
        return np.zeros((0, hidden_dim), dtype=np.float32), []
    return np.stack(rows, axis=0), sources


def silhouette_one_vs_rest(
    activations_per_script: dict[str, np.ndarray],
    target_script: str,
    metric: str = "cosine",
) -> float | None:
    """Binary silhouette: target_script vs all other scripts pooled.

    Returns nan if any cluster has <2 samples or only one cluster present.
    """
    if target_script not in activations_per_script:
        return None
    target_X = activations_per_script[target_script]
    rest_arrays = [
        X for s, X in activations_per_script.items()
        if s != target_script and X.shape[0] > 0
    ]
    if target_X.shape[0] < 2 or not rest_arrays:
        return None
    rest_X = np.concatenate(rest_arrays, axis=0)
    if rest_X.shape[0] < 2:
        return None
    X = np.concatenate([target_X, rest_X], axis=0)
    labels = np.concatenate([
        np.zeros(target_X.shape[0], dtype=int),
        np.ones(rest_X.shape[0], dtype=int),
    ])
    return float(silhouette_score(X, labels, metric=metric))


def silhouette_multi(
    activations_per_script: dict[str, np.ndarray],
    metric: str = "cosine",
) -> tuple[float | None, dict[str, float]]:
    """Multi-cluster silhouette: each script is its own cluster."""
    arrays = []
    labels = []
    label_to_script: dict[int, str] = {}
    for idx, (script, X) in enumerate(sorted(activations_per_script.items())):
        if X.shape[0] < 2:
            continue
        label_to_script[idx] = script
        arrays.append(X)
        labels.append(np.full(X.shape[0], idx, dtype=int))
    if len(arrays) < 2:
        return None, {}
    X = np.concatenate(arrays, axis=0)
    y = np.concatenate(labels, axis=0)
    global_sil = float(silhouette_score(X, y, metric=metric))

    samples = silhouette_samples(X, y, metric=metric)
    per_cluster = {}
    for idx, script in label_to_script.items():
        mask = y == idx
        if mask.any():
            per_cluster[script] = float(samples[mask].mean())
    return global_sil, per_cluster


def bootstrap_silhouette_one_vs_rest(
    activations_per_script: dict[str, np.ndarray],
    target_script: str,
    n_bootstrap: int,
    seed: int = 42,
    metric: str = "cosine",
) -> dict:
    """Bootstrap CI for one-vs-rest silhouette of target_script."""
    rng = np.random.default_rng(seed)
    target_X = activations_per_script.get(target_script)
    if target_X is None or target_X.shape[0] < 2:
        return {"error": "insufficient_samples"}

    rest_pool = np.concatenate(
        [X for s, X in activations_per_script.items() if s != target_script and X.shape[0] > 0],
        axis=0,
    )
    if rest_pool.shape[0] < 2:
        return {"error": "no_rest"}

    n_target = target_X.shape[0]
    n_rest = rest_pool.shape[0]
    values: list[float] = []
    for _ in range(n_bootstrap):
        t_idx = rng.integers(0, n_target, n_target)
        r_idx = rng.integers(0, n_rest, n_rest)
        X = np.concatenate([target_X[t_idx], rest_pool[r_idx]], axis=0)
        y = np.concatenate([np.zeros(n_target, dtype=int), np.ones(n_rest, dtype=int)])
        try:
            values.append(float(silhouette_score(X, y, metric=metric)))
        except Exception:
            continue
    if not values:
        return {"error": "all_bootstrap_failed"}

    arr = np.asarray(values)
    return {
        "n_bootstrap_iterations": n_bootstrap,
        "n_successful": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "ci_99_low": float(np.percentile(arr, 0.5)),
        "ci_99_high": float(np.percentile(arr, 99.5)),
        "ci_95_low": float(np.percentile(arr, 2.5)),
        "ci_95_high": float(np.percentile(arr, 97.5)),
    }


def default_layer_for_model(n_layers: int) -> int:
    """Middle of the network (relative depth ~0.5) as a sensible default."""
    return n_layers // 2


def h5_verdict(sil_arabizi: float | None, ci: dict | None) -> str:
    """Apply OSF v4 H5 thresholds and return CONFIRMED / REJECTED / INDETERMINATE."""
    if sil_arabizi is None:
        return "MISSING_DATA"
    if sil_arabizi > H5_CONFIRMATION_THRESHOLD:
        if ci and ci.get("ci_99_low", -np.inf) > H5_CONFIRMATION_THRESHOLD:
            return "CONFIRMED (CI excludes threshold)"
        return "CONFIRMED (point estimate; CI overlaps threshold)"
    lo, hi = H5_REJECTION_RANGE
    if lo <= sil_arabizi <= hi:
        if ci and ci.get("ci_99_high", np.inf) < lo or ci.get("ci_99_low", -np.inf) > hi:
            return "REJECTED (CI excludes rejection range)"
        return "REJECTED (point estimate in rejection range)"
    return "INDETERMINATE (between thresholds)"


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Stage 3 — silhouette analysis (H5)")
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=list_models())
    parser.add_argument("--phase2-run-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--new-run", action="store_true")
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Layer index for silhouette (default: middle of network)",
    )
    parser.add_argument(
        "--metric", default="cosine", choices=["cosine", "euclidean"],
        help="Distance metric (default: cosine — matches Wang)",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=1000,
        help="Bootstrap iterations for 99%% CI (default: 1000; 0 to disable)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Bootstrap seed (default: 42 — matches OSF v4)",
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if silhouette_score is None:
        log.error(
            "scikit-learn is required for Stage 3. Install with:\n"
            "    pip install scikit-learn>=1.5\n"
            "Or reinstall the project extras: pip install -e \".[interp]\"\n"
            f"Original import error: {_SKLEARN_IMPORT_ERROR}"
        )
        return 2

    cfg = load_model_config(args.model)
    phase2 = resolve_phase2_source(cfg.MODEL_KEY, args.phase2_run_dir)
    paths = resolve_phase3_run_paths(cfg.MODEL_KEY, args.run_dir, args.new_run)
    layer = args.layer if args.layer is not None else default_layer_for_model(cfg.N_LAYERS)
    confirmatory = cfg.MODEL_KEY in H5_CONFIRMATORY_MODELS

    log.info("=" * 70)
    log.info("Phase 3 Stage 3 — Silhouette analysis (H5)")
    log.info(f"  model        : {cfg.MODEL_ID}  (key={cfg.MODEL_KEY})")
    log.info(f"  layers       : {cfg.N_LAYERS}")
    log.info(f"  test layer   : L{layer}  (relative depth {layer/max(1,cfg.N_LAYERS-1):.2f})")
    log.info(f"  metric       : {args.metric}")
    log.info(f"  bootstrap N  : {args.bootstrap}")
    log.info(f"  H5 role      : {'CONFIRMATORY' if confirmatory else 'EXPLORATORY (Gemma)'}")
    log.info(f"  phase2 source: {phase2.run_dir}")
    log.info(f"  phase3 out   : {paths.run_dir}")
    log.info("=" * 70)

    # Load harmful activations per script at the chosen layer
    per_script: dict[str, np.ndarray] = {}
    for script in SCRIPTS:
        X, srcs = load_harmful_activations_at_layer(
            phase2.activations_dir, script, layer, cfg.N_LAYERS, cfg.HIDDEN_DIM,
        )
        if X.shape[0] == 0:
            log.warning(f"  No activations loaded for {script}")
            continue
        per_script[script] = X
        log.info(f"  {script:11} → N={X.shape[0]}, dim={X.shape[1]}")

    if len(per_script) < 2:
        log.error("Need at least 2 scripts with activations. Aborting.")
        return 1

    # Per-script binary silhouette (one-vs-rest)
    binary_results = {}
    for script in per_script:
        sil = silhouette_one_vs_rest(per_script, script, metric=args.metric)
        ci = (
            bootstrap_silhouette_one_vs_rest(
                per_script, script, args.bootstrap, args.seed, args.metric,
            )
            if args.bootstrap > 0 else None
        )
        binary_results[script] = {
            "silhouette_one_vs_rest": sil,
            "bootstrap_ci": ci,
        }
        ci_str = f"  (99% CI [{ci['ci_99_low']:+.3f}, {ci['ci_99_high']:+.3f}])" if ci and "ci_99_low" in ci else ""
        log.info(f"  Sil_{script:11} (one-vs-rest) = {sil:+.4f}{ci_str}")

    # Multi-cluster silhouette
    multi_global, multi_per_cluster = silhouette_multi(per_script, metric=args.metric)
    log.info(f"\nMulti-cluster silhouette (all {len(per_script)} scripts):")
    log.info(f"  Global: {multi_global:+.4f}")
    for s, v in multi_per_cluster.items():
        log.info(f"    {s:11}: {v:+.4f}")

    # H5-specific assessment (Arabic & Arabizi)
    sil_arabic = binary_results.get("arabic", {}).get("silhouette_one_vs_rest")
    sil_arabizi = binary_results.get("arabizi", {}).get("silhouette_one_vs_rest")
    arabizi_ci = binary_results.get("arabizi", {}).get("bootstrap_ci")
    h5_status = h5_verdict(sil_arabizi, arabizi_ci)

    log.info("\nH5 — Wang silhouette gap test (Arabic vs Arabizi):")
    log.info(f"  Wang baseline       : Sil_Arabic = {WANG_BASELINE_SIL_ARABIC}")
    log.info(f"  Our Sil_Arabic       : {sil_arabic}")
    log.info(f"  Our Sil_Arabizi      : {sil_arabizi}")
    log.info(f"  Confirmation thresh.: > {H5_CONFIRMATION_THRESHOLD}")
    log.info(f"  Rejection range     : {H5_REJECTION_RANGE}")
    log.info(f"  Verdict             : {h5_status}")
    if not confirmatory:
        log.warning(
            "  ⚠️  This model is not in the H5 confirmatory set (Qwen3-4B/8B). "
            "Treating result as EXPLORATORY per OSF v4."
        )

    output = {
        "phase": "phase3_stage3_silhouette",
        "model_key": cfg.MODEL_KEY,
        "model_id": cfg.MODEL_ID,
        "n_layers": cfg.N_LAYERS,
        "hidden_dim": cfg.HIDDEN_DIM,
        "test_layer": layer,
        "relative_depth": layer / max(1, cfg.N_LAYERS - 1),
        "metric": args.metric,
        "bootstrap_iterations": args.bootstrap,
        "bootstrap_seed": args.seed,
        "phase2_run_dir": str(phase2.run_dir),
        "phase3_run_dir": str(paths.run_dir),
        "completed_at": datetime.datetime.now().isoformat(),
        "scripts_loaded": {s: int(X.shape[0]) for s, X in per_script.items()},
        "binary_silhouette_one_vs_rest": binary_results,
        "multi_cluster_silhouette": {
            "global": multi_global,
            "per_cluster": multi_per_cluster,
        },
        "h5_test": {
            "is_confirmatory_model": confirmatory,
            "wang_baseline_sil_arabic": WANG_BASELINE_SIL_ARABIC,
            "wang_baseline_sil_english": WANG_BASELINE_SIL_ENGLISH,
            "our_sil_arabic": sil_arabic,
            "our_sil_arabizi": sil_arabizi,
            "confirmation_threshold": H5_CONFIRMATION_THRESHOLD,
            "rejection_range": H5_REJECTION_RANGE,
            "verdict": h5_status,
            "arabizi_bootstrap_ci": arabizi_ci,
        },
        "caveat": (
            "Layer choice defaults to relative depth 0.5. For OSF v4 H5 "
            "confirmation, re-run with --layer set to the empirically-identified "
            "causal layer from Phase 4 on this model. Gemma models are exploratory; "
            "Wang's 0.24 baseline is Qwen2.5-7B-specific."
        ),
    }
    paths.silhouette_path.write_text(json.dumps(output, indent=2))
    log.info(f"\nWrote {paths.silhouette_path}")

    if args.plot:
        try:
            _plot_silhouettes(binary_results, multi_per_cluster, layer, paths.plots_dir)
        except ImportError:
            log.warning("matplotlib not available; skipping plot")

    write_latest_pointer(cfg.MODEL_KEY, paths.run_dir)
    log.info(f"LATEST.txt updated → {paths.run_dir.name}")
    return 0


def _plot_silhouettes(
    binary: dict, multi_per_cluster: dict, layer: int, plots_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    scripts = sorted(binary.keys())
    ovr_values = [binary[s]["silhouette_one_vs_rest"] for s in scripts]
    multi_values = [multi_per_cluster.get(s, 0.0) for s in scripts]
    x = np.arange(len(scripts))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, ovr_values, width, label="One-vs-rest (binary)")
    ax.bar(x + width/2, multi_values, width, label="Per-cluster (multi)")
    ax.axhline(WANG_BASELINE_SIL_ARABIC, color="red", linestyle="--",
               linewidth=1, label=f"Wang Arabic baseline ({WANG_BASELINE_SIL_ARABIC})")
    ax.axhline(H5_CONFIRMATION_THRESHOLD, color="green", linestyle=":",
               linewidth=1, label=f"H5 confirmation > {H5_CONFIRMATION_THRESHOLD}")
    ax.set_xticks(x)
    ax.set_xticklabels(scripts)
    ax.set_ylabel(f"Silhouette score at L{layer}")
    ax.set_title("Silhouette scores per script (H5 analysis)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = plots_dir / "silhouette_per_script.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"  Plot saved: {out}")


if __name__ == "__main__":
    sys.exit(main())
