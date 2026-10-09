#!/usr/bin/env python3
"""Phase 2 Stage 6: registered logit-difference proxy via lm_head projection.

Δlogit(L) = lm_head(r̂_L)[refusal_token] − mean(lm_head(r̂_L)[compliance_tokens])

This is the "logit-difference" proxy mentioned in the OSF preregistration as
a DESCRIPTIVE depth metric. Per OSF v6 it is NOT used to select the primary
causal layer — Stage 04 (single-layer ablation) does that. This script reports
per-layer Δlogit so the discrepancy with the causal layer can be quantified per
model.

As of 2026-05-29 it ALSO reads ``causal_layer_selection.json`` (Stage 04
output) from the same run dir, if present, and emits per-script deltas between
the Δlogit peak and the causal layer. The comparison is optional — if Stage
04 has not been run on this model, the script still produces its descriptive
output and records ``causal_anchor: null``.

Usage:
    PYTHONPATH=. python3 scripts/phase2_direction_extraction/06_compute_logit_difference.py \
        --model qwen3_4b
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.device import DeviceConfig, detect_device, load_hf_model, log_device_info
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

    Returns the same shape as the Stage 5 helper. None if absent.
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
    """Signed delta = proxy_peak − causal_layer. None if no anchor."""
    if causal_layer is None:
        return None
    return int(proxy_peak) - int(causal_layer)


# ── Model + token setup ──────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str, dev: DeviceConfig, model_class=None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    model = load_hf_model(
        model_id, dev, hf_token=os.environ.get("HF_TOKEN"), model_class=model_class,
    )
    model.eval()
    return model, tokenizer


def get_token_ids(tokenizer) -> dict:
    refusal_token_id = tokenizer.encode("I", add_special_tokens=False)[0]
    candidates = [" (", "Here", "Sure", "Yes"]
    compliance_ids = []
    for c in candidates:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if ids:
            compliance_ids.append(ids[0])
    return {
        "refusal_token_id": refusal_token_id,
        "refusal_token": tokenizer.decode([refusal_token_id]),
        "compliance_token_ids": compliance_ids,
        "compliance_tokens": [tokenizer.decode([t]) for t in compliance_ids],
    }


def compute_logit_difference(
    model, direction_tensor: torch.Tensor, token_ids: dict, dev: DeviceConfig, n_layers: int,
) -> torch.Tensor:
    lm_head_weight = model.lm_head.weight.to(dev.device).float()  # [vocab, hidden]
    direction_tensor = direction_tensor.to(dev.device).float()
    refusal_id = token_ids["refusal_token_id"]
    compliance_ids = token_ids["compliance_token_ids"]

    delta = torch.zeros(n_layers, device=dev.device)
    for L in range(n_layers):
        r_L = direction_tensor[L]
        logits = lm_head_weight @ r_L  # [vocab]
        refusal_logit = logits[refusal_id].item()
        compliance_logits = [logits[i].item() for i in compliance_ids]
        compliance_mean = sum(compliance_logits) / len(compliance_logits)
        delta[L] = refusal_logit - compliance_mean
    return delta.cpu()


def main():
    parser = argparse.ArgumentParser(description="Logit-difference proxy via lm_head projection")
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=list_models())
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--prefer-device",
        type=str,
        default=None,
        choices=["cuda", "xla", "mps", "cpu"],
        help="Force a specific device. Default: auto-detect.",
    )
    args = parser.parse_args()

    cfg = load_model_config(args.model)
    run_paths = resolve_run_paths(cfg.MODEL_KEY, args.run_dir, new_run=False)

    log.info("=" * 70)
    log.info("Phase 2 Stage 6 — Logit-difference proxy (DESCRIPTIVE)")
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
            f"causal_L_lex={causal_anchor['peak_causal_layer_lexical']}"
        )
    else:
        log.info("Stage 04 anchor: not found — Δlogit-vs-causal comparison disabled.")

    dev = detect_device(prefer=args.prefer_device)
    log_device_info(dev)
    model, tokenizer = load_model_and_tokenizer(
        cfg.MODEL_ID, dev, model_class=getattr(cfg, "MODEL_CLASS", None),
    )
    token_ids = get_token_ids(tokenizer)
    log.info(f"Refusal token:   {token_ids['refusal_token']!r} (id {token_ids['refusal_token_id']})")
    log.info(f"Compliance ids:  {token_ids['compliance_tokens']}")

    results = {}
    for script in SCRIPTS:
        direction_path = run_paths.directions_dir / f"{script}.pt"
        if not direction_path.exists():
            log.error(f"Direction not found: {direction_path}")
            continue

        log.info(f"\n{script.upper()}")
        direction = torch.load(direction_path, map_location="cpu")
        delta = compute_logit_difference(model, direction, token_ids, dev, cfg.N_LAYERS)

        peak = int(delta.argmax().item())
        top5 = delta.argsort(descending=True)[:5].tolist()
        log.info(f"  Peak layer  : L{peak} (Δlogit {delta[peak]:.4f})")
        log.info(f"  Top 5       : {top5}")
        log.info(f"  Rel. depth  : {peak/max(1, cfg.N_LAYERS-1):.2f}")

        per_script = {
            "logit_difference_peak": peak,
            "logit_difference_per_layer": delta.tolist(),
            "top5_layers": top5,
            "relative_depth": peak / max(1, cfg.N_LAYERS - 1),
        }

        # Causal anchor comparison (only on the calibrated script)
        if causal_anchor is not None and causal_anchor["script"] == script:
            cg = causal_anchor["peak_causal_layer_guard"]
            cl = causal_anchor["peak_causal_layer_lexical"]
            per_script["causal_anchor"] = {
                "peak_causal_layer_guard": cg,
                "peak_causal_layer_lexical": cl,
                "max_flip_rate_guard": causal_anchor["max_flip_rate_guard"],
                "max_flip_rate_lexical": causal_anchor["max_flip_rate_lexical"],
                "delta_logit_diff_minus_causal_guard": causal_delta(peak, cg),
                "delta_logit_diff_minus_causal_lexical": causal_delta(peak, cl),
                "source_path": causal_anchor["source_path"],
            }
            log.info(
                f"  Causal anchor (Guard) : L{cg}  Δlogit_peak−causal={causal_delta(peak, cg)}"
            )
            log.info(
                f"  Causal anchor (lex)   : L{cl}  Δlogit_peak−causal={causal_delta(peak, cl)}"
            )
        else:
            per_script["causal_anchor"] = None
            if causal_anchor is not None:
                log.info(
                    f"  Causal anchor         : (none — Stage 04 calibrated "
                    f"{causal_anchor['script']!r}, not {script!r})"
                )

        results[script] = per_script

    out_path = run_paths.run_dir / "metadata_logit_difference.json"
    out_path.write_text(json.dumps({
        "phase": "phase2_stage6_logit_difference_proxy",
        "model_key": cfg.MODEL_KEY,
        "model_id": cfg.MODEL_ID,
        "n_layers": cfg.N_LAYERS,
        "device": dev.to_dict(),
        "token_ids": token_ids,
        "causal_comparison_available": causal_anchor is not None,
        "results": results,
        "caveat": (
            "Descriptive proxy. The causal layer per OSF v6 § 4 is set by "
            "Stage 04 (single-layer ablation). Per-script ``causal_anchor`` "
            "carries the Δ between this proxy's peak and the causal layer "
            "when Stage 04 has been run on that script."
        ),
    }, indent=2))
    log.info(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
