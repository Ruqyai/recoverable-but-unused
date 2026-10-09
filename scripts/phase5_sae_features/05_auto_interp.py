#!/usr/bin/env python3
"""Phase 5 Stage 5 (optional): pull auto-interp labels for top-K features.

Best-effort enrichment: for each top-K feature (Stage 03) it queries the public
Neuronpedia API for that feature's auto-interp label, so the H3 overlap/disjoint
sets can be read qualitatively (e.g. "is the shared feature an 'Arabic script'
detector or a generic 'harmful-request' detector?"). This stage is OPTIONAL and
purely descriptive — it never changes the H3 verdict.

It degrades gracefully: no network / unknown Neuronpedia model id / HTTP errors
→ the feature is recorded with `label: null` and the reason, and the stage still
exits 0. Pass `--neuronpedia-model` to set the Neuronpedia model slug (the
default map covers the models that are actually hosted there).

Per the deposited OSF preregistration, H3 is exploratory; labels here
are interpretive aids only.

Usage
-----
  PYTHONPATH=. python3 scripts/phase5_sae_features/05_auto_interp.py --model gemma_2_9b_it
  PYTHONPATH=. python3 scripts/phase5_sae_features/05_auto_interp.py \
      --model gemma_2_9b_it --neuronpedia-model gemma-2-9b-it \
      --neuronpedia-sae 20-gemmascope-res-16k --max-features 50
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.models import DEFAULT_MODEL_KEY, list_models, load_model_config
from scripts.phase5_sae_features.config import (
    resolve_phase5_run_paths,
    write_latest_pointer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NEURONPEDIA_API = "https://www.neuronpedia.org/api/feature"

# Best-known Neuronpedia model slugs (only models actually hosted there).
NEURONPEDIA_MODEL_SLUG = {
    "gemma_2_9b_it": "gemma-2-9b-it",
    "gemma_3_1b_it": "gemma-3-1b-it",
    "qwen2_5_7b_Instruct": "qwen2.5-7b-it",
    "qwen3_4b": "qwen3-4b",
}


def fetch_label(model_slug: str, sae_slug: str, feature_idx: int, timeout: float) -> dict:
    """Query Neuronpedia for one feature's label. Returns {label, error}."""
    url = f"{NEURONPEDIA_API}/{model_slug}/{sae_slug}/{feature_idx}"
    req = Request(url, headers={"User-Agent": "comprehension-gated-refusal/phase5"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except HTTPError as e:
        return {"label": None, "error": f"http_{e.code}"}
    except (URLError, TimeoutError) as e:
        return {"label": None, "error": f"net_{type(e).__name__}"}
    except Exception as e:  # malformed JSON, etc.
        return {"label": None, "error": f"parse_{type(e).__name__}"}

    # Neuronpedia returns a list of explanations under "explanations"; take the
    # first description if present.
    label = None
    expls = data.get("explanations") if isinstance(data, dict) else None
    if isinstance(expls, list) and expls:
        label = expls[0].get("description")
    return {"label": label, "error": None if label else "no_explanation"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Phase 5 Stage 5 — optional Neuronpedia auto-interp labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=list_models())
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Phase 5 run with top100_per_script.json (default: LATEST).")
    ap.add_argument("--neuronpedia-model", default=None,
                    help="Neuronpedia model slug (default: per-model map; required if unmapped).")
    ap.add_argument("--neuronpedia-sae", default=None,
                    help="Neuronpedia SAE slug, e.g. '20-gemmascope-res-16k'. "
                         "Required — Neuronpedia addresses features by SAE slug, not HF release.")
    ap.add_argument("--max-features", type=int, default=100,
                    help="Cap features queried per script (politeness / rate limits).")
    ap.add_argument("--sleep", type=float, default=0.2, help="Seconds between requests.")
    ap.add_argument("--timeout", type=float, default=15.0, help="Per-request timeout.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_model_config(args.model)
    paths = resolve_phase5_run_paths(cfg.MODEL_KEY, args.run_dir, new_run=False)

    if not paths.top_k_path.exists():
        log.error("Missing %s — run Stage 03 first.", paths.top_k_path)
        return 1
    top_k_data = json.loads(paths.top_k_path.read_text())
    per_script = top_k_data.get("scripts", {})

    model_slug = args.neuronpedia_model or NEURONPEDIA_MODEL_SLUG.get(cfg.MODEL_KEY)
    if not model_slug:
        log.error("No Neuronpedia model slug for %s — pass --neuronpedia-model. "
                  "(This stage is optional; skip it if the model is not on Neuronpedia.)",
                  cfg.MODEL_KEY)
        return 1
    if not args.neuronpedia_sae:
        log.error("Pass --neuronpedia-sae (e.g. '20-gemmascope-res-16k'). Neuronpedia "
                  "addresses features by SAE slug, which differs from the sae-lens id.")
        return 1

    log.info("=" * 70)
    log.info("Phase 5 Stage 5 — Neuronpedia auto-interp (optional)")
    log.info("  model        : %s → neuronpedia '%s'", cfg.MODEL_KEY, model_slug)
    log.info("  sae slug     : %s", args.neuronpedia_sae)
    log.info("  max features : %d/script", args.max_features)
    log.info("=" * 70)

    # De-dup features across scripts so each is fetched once.
    all_feats: set[int] = set()
    for s in per_script.values():
        all_feats.update(int(i) for i in s.get("top_k", [])[: args.max_features])
    log.info("Unique features to label: %d", len(all_feats))

    labels: dict[str, dict] = {}
    n_ok = 0
    for j, fidx in enumerate(sorted(all_feats)):
        res = fetch_label(model_slug, args.neuronpedia_sae, fidx, args.timeout)
        labels[str(fidx)] = res
        if res["label"]:
            n_ok += 1
        if (j + 1) % 25 == 0:
            log.info("  %d/%d fetched (%d labelled)", j + 1, len(all_feats), n_ok)
        time.sleep(args.sleep)

    output = {
        "phase": "phase5_stage5_auto_interp",
        "model_key": cfg.MODEL_KEY,
        "neuronpedia_model": model_slug,
        "neuronpedia_sae": args.neuronpedia_sae,
        "layer": top_k_data.get("layer"),
        "n_features": len(all_feats),
        "n_labelled": n_ok,
        "completed_at": datetime.datetime.now().isoformat(),
        "feature_labels": labels,
        "per_script_top_k": {s: v.get("top_k", []) for s, v in per_script.items()},
        "caveat": (
            "Descriptive only — labels do not affect the H3 verdict. Missing "
            "labels (label=null) are expected when a feature has no Neuronpedia "
            "explanation or the model/SAE is not hosted. Per the deposited OSF "
            "preregistration, H3 is exploratory."
        ),
    }
    paths.feature_labels_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    log.info("Wrote %s  (%d/%d labelled)", paths.feature_labels_path, n_ok, len(all_feats))

    write_latest_pointer(cfg.MODEL_KEY, paths.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
