#!/usr/bin/env python3
"""Phase 5 Stage 4: top-K-feature Jaccard between scripts + H3 verdict.

For each H3 script pair (Tier A: arabic↔arabizi headline; Tier C:
devanagari↔itrans; Tier E: english↔leetspeak), computes the Jaccard similarity
of the two top-K feature sets and a bootstrap 99% CI (resample prompts → re-rank
→ re-Jaccard). Applies the pre-registered H3 thresholds:

    Jaccard < 0.4  → SUPPORTED (orthographic anchoring; features mostly disjoint)
    Jaccard >= 0.6 → REJECTED  (features largely shared)
    in between     → INDETERMINATE

Reads `top100_per_script.json` (Stage 03) for the point estimate and the
per-prompt matrices in `feature_activations/` for the bootstrap. Writes
`jaccard.json`.

Per the deposited OSF preregistration, H3 is EXPLORATORY (no
confirmatory weight); the Gemma models carry the official-SAE variant, the
others (incl. the Qwen3 transcoders) are exploratory-extra and flagged.

Usage
-----
  PYTHONPATH=. python3 scripts/phase5_sae_features/04_jaccard_similarity.py --model gemma_2_9b_it
  PYTHONPATH=. python3 scripts/phase5_sae_features/04_jaccard_similarity.py \
      --model gemma_3_1b_it --k 100 --bootstrap 1000
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.models import DEFAULT_MODEL_KEY, list_models, load_model_config
from scripts.phase5_sae_features.config import (
    H3_CONFIRMATORY_MODELS,
    H3_SCRIPT_PAIRS,
    h3_verdict,
    resolve_phase5_run_paths,
    write_latest_pointer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def score_features(features: torch.Tensor, statistic: str, fire_threshold: float) -> torch.Tensor:
    if statistic == "frequency":
        return (features > fire_threshold).float().mean(dim=0)
    return features.mean(dim=0)


def top_k_set(features: torch.Tensor, k: int, statistic: str, fire_threshold: float) -> set[int]:
    scores = score_features(features, statistic, fire_threshold)
    k = min(k, scores.numel())
    vals, idx = torch.topk(scores, k)
    return {int(i) for i, v in zip(idx.tolist(), vals.tolist(), strict=True) if v > 0}


def bootstrap_jaccard(
    feats_a: torch.Tensor,
    feats_b: torch.Tensor,
    k: int,
    statistic: str,
    fire_threshold: float,
    n_boot: int,
    seed: int,
) -> dict:
    """Bootstrap CI for the top-K Jaccard by resampling prompts per script."""
    rng = np.random.default_rng(seed)
    na, nb = feats_a.shape[0], feats_b.shape[0]
    if na < 2 or nb < 2:
        return {"error": "insufficient_prompts"}
    vals: list[float] = []
    for _ in range(n_boot):
        ia = rng.integers(0, na, na)
        ib = rng.integers(0, nb, nb)
        sa = top_k_set(feats_a[ia], k, statistic, fire_threshold)
        sb = top_k_set(feats_b[ib], k, statistic, fire_threshold)
        j = jaccard(sa, sb)
        if not np.isnan(j):
            vals.append(j)
    if not vals:
        return {"error": "all_bootstrap_failed"}
    arr = np.asarray(vals)
    return {
        "n_bootstrap": n_boot,
        "n_successful": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "ci_99_low": float(np.percentile(arr, 0.5)),
        "ci_99_high": float(np.percentile(arr, 99.5)),
        "ci_95_low": float(np.percentile(arr, 2.5)),
        "ci_95_high": float(np.percentile(arr, 97.5)),
    }


def load_features(feature_acts_dir: Path, script: str) -> torch.Tensor | None:
    p = feature_acts_dir / f"{script}.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu")["features"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Phase 5 Stage 4 — top-K Jaccard + H3 verdict.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=list_models())
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Phase 5 run with top100_per_script.json (default: LATEST).")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--statistic", choices=["mean", "frequency"], default="mean")
    ap.add_argument("--fire-threshold", type=float, default=0.0)
    ap.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iters (0 to disable).")
    ap.add_argument("--seed", type=int, default=42, help="Bootstrap seed (matches the project).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_model_config(args.model)
    paths = resolve_phase5_run_paths(cfg.MODEL_KEY, args.run_dir, new_run=False)

    if not paths.top_k_path.exists():
        log.error("Missing %s — run Stage 03 first.", paths.top_k_path)
        return 1
    top_k_data = json.loads(paths.top_k_path.read_text())
    per_script_top = top_k_data.get("scripts", {})
    layer = top_k_data.get("layer")
    is_gemma_official = cfg.MODEL_KEY in H3_CONFIRMATORY_MODELS

    meta = json.loads(paths.sae_meta_path.read_text()) if paths.sae_meta_path.exists() else {}
    is_transcoder = bool(meta.get("is_transcoder"))
    snap = int(meta.get("layer_snap_distance", 0) or 0)

    log.info("=" * 70)
    log.info("Phase 5 Stage 4 — Top-%d Jaccard + H3 verdict", args.k)
    log.info("  model     : %s  (key=%s)", cfg.MODEL_ID, cfg.MODEL_KEY)
    log.info("  layer     : %s  (snap %s)", layer, snap)
    log.info("  H3 role   : %s%s",
             "official Gemma SAE" if is_gemma_official else "exploratory-extra",
             "  [TRANSCODER analogue]" if is_transcoder else "")
    log.info("  bootstrap : %s", args.bootstrap)
    log.info("=" * 70)

    results: dict[str, dict] = {}
    for tier, (s_a, s_b) in H3_SCRIPT_PAIRS.items():
        ta = per_script_top.get(s_a)
        tb = per_script_top.get(s_b)
        if ta is None or tb is None:
            log.info("  %-7s (%s↔%s): skipped (missing top-K for one side)", tier, s_a, s_b)
            continue
        set_a = set(ta["top_k"])
        set_b = set(tb["top_k"])
        point = jaccard(set_a, set_b)

        ci = None
        if args.bootstrap > 0:
            fa = load_features(paths.feature_acts_dir, s_a)
            fb = load_features(paths.feature_acts_dir, s_b)
            if fa is not None and fb is not None:
                ci = bootstrap_jaccard(fa, fb, args.k, args.statistic,
                                       args.fire_threshold, args.bootstrap, args.seed)
            else:
                log.warning("  %s: feature matrices missing; skipping bootstrap CI", tier)

        verdict = h3_verdict(point, ci)
        results[tier] = {
            "pair": [s_a, s_b],
            "k": args.k,
            "intersection": len(set_a & set_b),
            "union": len(set_a | set_b),
            "jaccard": point,
            "bootstrap_ci": ci,
            "verdict": verdict,
            "is_headline": tier == "tier_a",
        }
        ci_str = (f"  99% CI [{ci['ci_99_low']:.3f}, {ci['ci_99_high']:.3f}]"
                  if ci and "ci_99_low" in ci else "")
        log.info("  %-7s (%s↔%s): J=%.3f  |∩|=%d |∪|=%d  → %s%s",
                 tier, s_a, s_b, point, len(set_a & set_b), len(set_a | set_b), verdict, ci_str)

    if not results:
        log.error("No H3 pairs computable — were both scripts of a pair extracted + ranked?")
        return 1

    output = {
        "phase": "phase5_stage4_jaccard",
        "model_key": cfg.MODEL_KEY,
        "model_id": cfg.MODEL_ID,
        "layer": layer,
        "layer_snap_distance": snap,
        "k": args.k,
        "statistic": args.statistic,
        "bootstrap_iterations": args.bootstrap,
        "bootstrap_seed": args.seed,
        "is_official_gemma_sae": is_gemma_official,
        "is_transcoder_analogue": is_transcoder,
        "thresholds": {"supported_below": 0.4, "rejected_at_or_above": 0.6},
        "completed_at": datetime.datetime.now().isoformat(),
        "pairs": results,
        "caveat": (
            "H3 is EXPLORATORY per the deposited OSF preregistration — no "
            "confirmatory weight. Headline pair is tier_a (Arabic↔Arabizi). "
            "Transcoder-analogue results (Qwen3) and snapped-layer results "
            "(Gemma IT SAEs at non-causal layers) are not directly comparable to "
            "the official Gemma residual-SAE Jaccard at the exact causal layer."
        ),
    }
    paths.jaccard_path.write_text(json.dumps(output, indent=2))
    log.info("Wrote %s", paths.jaccard_path)

    # Headline banner
    headline = results.get("tier_a")
    if headline:
        log.info("\n" + "=" * 70)
        log.info("H3 HEADLINE (Tier A, Arabic↔Arabizi): J=%.3f → %s",
                 headline["jaccard"], headline["verdict"])
        if not is_gemma_official:
            log.info("  ⚠️  exploratory-extra model — not the registered H3 Gemma pair")
        if is_transcoder:
            log.info("  ⚠️  transcoder analogue — not a residual-SAE Jaccard")
        if snap:
            log.info("  ⚠️  SAE layer snapped %d from the causal layer", snap)
        log.info("=" * 70)

    write_latest_pointer(cfg.MODEL_KEY, paths.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
