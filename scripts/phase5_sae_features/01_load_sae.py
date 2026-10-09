#!/usr/bin/env python3
"""Phase 5 Stage 1: load the chosen SAE / transcoder and record its metadata (H3).

Resolves the per-model causal layer (Phase 2 Stage 04), looks up a VERIFIED
`sae-lens` (release, sae_id) for that model at that layer, loads the SAE /
transcoder (caching weights into the HF cache), and writes `sae_meta.json`.
Stage 02 reads that file and re-loads the (now cached) module.

Verified source catalog (`SAE_CATALOG`)
---------------------------------------
Every entry below was confirmed present in the installed sae-lens pretrained
directory (`get_pretrained_saes_directory()`, sae-lens 6.x). Crucially, the
official Gemma *instruction-tuned* residual SAEs are released only at a FEW
layers (1B: 7/13/17/22; 4B: 9/17/22/29; 9B: 9/20/31), so the requested causal
layer is SNAPPED to the nearest available SAE layer (the snap distance is
recorded). The Gemma-2-2B (PT) SAE and the Qwen3 transcoders cover every layer.

  model_key             release                            layers          kind
  gemma_3_1b_it         gemma-scope-2-1b-it-res            7,13,17,22      SAE
  gemma_2_9b_it         gemma-scope-9b-it-res-canonical    9,20,31         SAE
  gemma_2_2b_it         gemma-scope-2b-pt-res-canonical    0..25 (all)     SAE (PT)
  gemma_3_4b_it         gemma-scope-2-4b-it-res            9,17,22,29      SAE *
  qwen2_5_7b_Instruct   qwen2.5-7b-instruct-andyrdt        3,7,11,15,19,23,27   SAE
  llama_3_1_8b_Instruct llama-3.1-8b-instruct-andyrdt      3,7,11,15,19,23,27   SAE
  qwen3_4b              mwhanna-qwen3-4b-transcoders       0..35 (all)     transcoder
  qwen3_8b              mwhanna-qwen3-8b-transcoders       0..35 (all)     transcoder
  (llama_3_2_3b_Instruct: NO public SAE / transcoder — not loadable)

  * gemma_3_4b_it is multimodal; the SAE loads, but Stage 02 cannot extract its
    residuals on the text-only path until it branches on LAYERS_PATH.

Transcoders vs SAEs (Qwen3): the mwhanna transcoders encode the MLP input, not
`resid_post`. Stage 02 feeds the layer residual to `.encode()` either way, so a
Qwen3 H3 result is an *MLP-transcoder-feature analogue* of the residual-SAE H3 —
exploratory, methodology-mismatched, NOT directly comparable to the Gemma
residual-SAE Jaccard. The `is_transcoder` / `input_caveat` fields in
`sae_meta.json` flag this.

Per the deposited OSF preregistration, H3 carries no confirmatory weight.

Dependencies
------------
Phase 5 depends ONLY on Phase 1 data + a causal-layer number + an SAE. It does
NOT use Phase 2 directions and is fully independent of Phase 4 (causal
intervention). The causal layer comes from Phase 2 Stage 04 by default, but
`--layer L` makes Phase 5 STANDALONE — no Phase 2 run is read or created.

Usage
-----
  # Default: read the causal layer from Phase 2 Stage 04.
  PYTHONPATH=. python3 scripts/phase5_sae_features/01_load_sae.py --model gemma_2_9b_it
  PYTHONPATH=. python3 scripts/phase5_sae_features/01_load_sae.py --model qwen3_4b
  # Standalone (no Phase 2 needed at all):
  PYTHONPATH=. python3 scripts/phase5_sae_features/01_load_sae.py --model gemma_3_1b_it --layer 13
  # Override anything:
  PYTHONPATH=. python3 scripts/phase5_sae_features/01_load_sae.py \
      --model gemma_3_1b_it --layer 13 --width 16k \
      --sae-release gemma-scope-2-1b-it-res --sae-id "layer_13_width_16k_l0_big"
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from sae_lens import SAE

    try:
        from sae_lens import Transcoder
    except Exception:  # older sae-lens without a public Transcoder export
        Transcoder = None
except ImportError as e:  # pragma: no cover - only without [interp]
    SAE = None
    Transcoder = None
    _SAE_LENS_IMPORT_ERROR = e
else:
    _SAE_LENS_IMPORT_ERROR = None

from scripts.models import DEFAULT_MODEL_KEY, list_models, load_model_config
from scripts.phase5_sae_features.config import (
    H3_CONFIRMATORY_MODELS,
    read_causal_layer,
    resolve_phase2_source,
    resolve_phase5_run_paths,
    write_latest_pointer,
)
from scripts.phase5_sae_features.sae_loading import (
    load_hf_disk_module,
    load_qwen_scope_module,
    load_registry_module,
)
from scripts.utils.device import detect_device, log_device_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Verified sae-lens source catalog ─────────────────────────────────────────
# id_pattern uses {L} = layer, {W} = width token. `kind`: "sae" | "transcoder".
# `layers`: the layers actually released (requested layer is snapped to nearest).
SAE_CATALOG: dict[str, dict] = {
    "gemma_3_1b_it": {
        "release": "gemma-scope-2-1b-it-res",
        "id_pattern": "layer_{L}_width_{W}_l0_big",
        "layers": [7, 13, 17, 22],
        "kind": "sae",
        "default_width": "16k",
    },
    "gemma_2_9b_it": {
        "release": "gemma-scope-9b-it-res-canonical",
        "id_pattern": "layer_{L}/width_{W}/canonical",
        "layers": [9, 20, 31],
        "kind": "sae",
        "default_width": "16k",
    },
    "gemma_2_2b_it": {
        "release": "gemma-scope-2b-pt-res-canonical",
        "id_pattern": "layer_{L}/width_{W}/canonical",
        "layers": list(range(26)),
        "kind": "sae",
        "default_width": "16k",
    },
    "gemma_3_4b_it": {
        "release": "gemma-scope-2-4b-it-res",
        "id_pattern": "layer_{L}_width_{W}_l0_big",
        "layers": [9, 17, 22, 29],
        "kind": "sae",
        "default_width": "16k",
    },
    "qwen2_5_7b_Instruct": {
        "release": "qwen2.5-7b-instruct-andyrdt",
        "id_pattern": "resid_post_layer_{L}_trainer_1",
        "layers": [3, 7, 11, 15, 19, 23, 27],
        "kind": "sae",
        "default_width": None,
    },
    "llama_3_1_8b_Instruct": {
        "release": "llama-3.1-8b-instruct-andyrdt",
        "id_pattern": "resid_post_layer_{L}_trainer_1",
        "layers": [3, 7, 11, 15, 19, 23, 27],
        "kind": "sae",
        "default_width": None,
    },
    "qwen3_4b": {
        "release": "mwhanna-qwen3-4b-transcoders",
        "id_pattern": "layer_{L}",
        "layers": list(range(36)),
        "kind": "transcoder",
        "default_width": None,
    },
    "qwen3_8b": {
        "release": "mwhanna-qwen3-8b-transcoders",
        "id_pattern": "layer_{L}",
        "layers": list(range(36)),
        "kind": "transcoder",
        "default_width": None,
    },
    # llama_3_2_3b: NOT in the sae-lens registry, but `Geaming/Llama-3.2-3B-Instruct_SAEs`
    # ships residual SAEs in sae-lens training-output format (cfg.json +
    # sae_weights.safetensors), loadable via SAE.load_from_disk. Verified
    # cfg: architecture=jumprelu, d_in=3072, d_sae=49152, hook=blocks.{L}.hook_resid_post.
    # Variants in the repo: BT(F)/BT(P)/FAST × {8X,16X}_2048 × {jumprelu,standard};
    # default below = BT(F) 16X jumprelu. Override the subdir with --sae-id.
    "llama_3_2_3b_Instruct": {
        "source": "hf_disk",
        "hf_repo": "Geaming/Llama-3.2-3B-Instruct_SAEs",
        "id_pattern": "BT(F)/blocks_{L}_hook_resid_post_16X_2048_jumprelu",
        "layers": [4, 12, 20],
        "kind": "sae",
        "default_width": None,
    },
    # ── Large-model extension (RERUN_PLAN §7; exploratory) ────────────────────
    # gemma-3-27b-it: official Gemma Scope 2 residual SAEs, sae-lens registry.
    # resid_post layers verified against the HF repo 2026-06-21: [16, 31, 40, 53],
    # widths 16k/65k/262k/1m, l0 tags big/medium/small. Same id_pattern as the 4b.
    "gemma_3_27b_it": {
        "release": "gemma-scope-2-27b-it-res",
        "id_pattern": "layer_{L}_width_{W}_l0_big",
        "layers": [16, 31, 40, 53],
        "kind": "sae",
        "default_width": "16k",
    },
    # qwen3.5-27b: Qwen-Scope residual TopK SAEs, ALL 64 layers, width 80K (d_sae
    # 81920), k=50. NOT sae-lens — `.pt` dicts loaded via load_qwen_scope_module
    # (source="qwen_scope_pt"). The L0_100 variant repo is the denser alternative.
    "qwen3_5_27b": {
        "source": "qwen_scope_pt",
        "hf_repo": "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50",
        "id_pattern": "layer{L}.sae.pt",
        "layers": list(range(64)),
        "kind": "sae",
        "default_width": "80k",
        "k": 50,
    },
}


def nearest_layer(requested: int, available: list[int]) -> tuple[int, int]:
    """Snap `requested` to the nearest available SAE layer.

    Returns (chosen_layer, snap_distance). snap_distance == 0 means an exact hit.
    """
    chosen = min(available, key=lambda L: (abs(L - requested), L))
    return chosen, abs(chosen - requested)


def build_sae_id(entry: dict, layer: int, width: str | None) -> str:
    pat = entry["id_pattern"]
    w = width or entry.get("default_width") or "16k"
    return pat.format(L=layer, W=w)


def _sae_attr(obj_root, *names, default=None):
    """Read the first present attribute from the module, its `.cfg`, or
    `.cfg.metadata` (sae-lens moved fields across versions)."""
    cfg = getattr(obj_root, "cfg", None)
    meta = getattr(cfg, "metadata", None)
    for obj in (obj_root, cfg, meta):
        if obj is None:
            continue
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
    return default


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Phase 5 Stage 1 — load SAE/transcoder + write sae_meta.json (H3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=list_models())
    ap.add_argument("--phase2-run-dir", type=Path, default=None,
                    help="Phase 2 run to read the causal layer from (default: LATEST). "
                         "Ignored when --layer is given (Phase 5 then runs standalone).")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Existing Phase 5 run to write into (default: LATEST/new).")
    ap.add_argument("--new-run", action="store_true", help="Force a fresh Phase 5 run folder.")
    ap.add_argument("--layer", type=int, default=None,
                    help="Causal layer. If given, Phase 5 runs STANDALONE (Phase 2 not "
                         "read or created). Default: read from Phase 2 Stage 04.")
    ap.add_argument("--width", default=None,
                    help="SAE width token (default: catalog default, usually 16k).")
    ap.add_argument("--sae-release", default=None,
                    help="Explicit sae-lens release (overrides catalog).")
    ap.add_argument("--sae-id", default=None,
                    help="Explicit sae-lens sae_id (registry) OR in-repo subdir (hf_disk).")
    ap.add_argument("--source", choices=["registry", "hf_disk"], default=None,
                    help="Override the load source. With 'hf_disk', --sae-release is the HF repo "
                         "and --sae-id is the in-repo subdir (sae-lens disk format). Use for "
                         "ad-hoc SAEs not in the catalog (e.g. syzhang gemma-2-2b-it).")
    ap.add_argument("--prefer-device", choices=["cuda", "xla", "mps", "cpu"], default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # sae-lens is required for the registry / hf_disk sources, but NOT for the
    # qwen_scope_pt source (pure-torch .pt loader). Enforce at load time once the
    # source is resolved, so Qwen-Scope H3 can run without sae-lens installed.

    cfg = load_model_config(args.model)
    entry = SAE_CATALOG.get(cfg.MODEL_KEY)

    if entry is None and not (args.sae_release and args.sae_id):
        log.error(
            "No SAE / transcoder is catalogued for %s, and no --sae-release/--sae-id "
            "override was given. %s has no public residual SAE or transcoder in the "
            "sae-lens registry, so H3 is not runnable for it.",
            cfg.MODEL_KEY, cfg.MODEL_ID,
        )
        return 1

    if getattr(cfg, "IS_MULTIMODAL", False):
        log.info(
            "%s is MULTIMODAL (%s). Stage 02 extracts residuals from the language "
            "submodule via LAYERS_PATH=%s (supported); verify the SAE repo matches "
            "that hook point before trusting H3 numbers.",
            cfg.MODEL_KEY, getattr(cfg, "MODEL_CLASS", "?"),
            getattr(cfg, "LAYERS_PATH", "?"),
        )

    paths = resolve_phase5_run_paths(cfg.MODEL_KEY, args.run_dir, args.new_run)
    dev = detect_device(prefer=args.prefer_device)
    log_device_info(dev)

    # Resolve causal layer. With --layer, Phase 5 is fully standalone and never
    # touches Phase 2 (no run dir resolved, none created). Otherwise read the
    # layer from Phase 2 Stage 04's causal_layer_selection.json.
    phase2_run_dir = None
    if args.layer is not None:
        requested_layer = args.layer
    else:
        phase2 = resolve_phase2_source(cfg.MODEL_KEY, args.phase2_run_dir)
        phase2_run_dir = phase2.run_dir
        requested_layer = read_causal_layer(phase2_run_dir)
        if requested_layer is None:
            log.error(
                "No causal layer in %s. Run Phase 2 Stage 04 there, or pass "
                "--layer L to run Phase 5 standalone (no Phase 2 needed).",
                phase2_run_dir,
            )
            return 1

    kind = entry["kind"] if entry else "sae"

    # Snap to the nearest available SAE layer (catalogued models only)
    snap_distance = 0
    layer = requested_layer
    if entry is not None and not args.sae_id:
        layer, snap_distance = nearest_layer(requested_layer, entry["layers"])
        if snap_distance > 0:
            log.warning(
                "Causal layer L%s is not in %s's released SAE layers %s — "
                "snapping to nearest L%s (distance %s). Report the layer "
                "mismatch in any H3 write-up.",
                requested_layer, cfg.MODEL_KEY, entry["layers"], layer, snap_distance,
            )

    width = args.width or (entry.get("default_width") if entry else None)
    # --source overrides the catalog (e.g. force hf_disk for an ad-hoc IT SAE).
    source = args.source or (entry.get("source", "registry") if entry else "registry")

    # Resolve the load address. For registry sources this is a sae-lens
    # (release, sae_id); for hf_disk sources `release` is the HF repo and
    # `sae_id` is the in-repo subdir (sae-lens training-output folder).
    if source in ("hf_disk", "qwen_scope_pt"):
        release = args.sae_release or (entry.get("hf_repo") if entry else None)
        sae_id = args.sae_id or (build_sae_id(entry, layer, width) if entry else None)  # subdir / file
    else:
        release = args.sae_release or (entry["release"] if entry else None)
        sae_id = args.sae_id or (build_sae_id(entry, layer, width) if entry else None)
    if not release or not sae_id:
        log.error("Could not resolve (release, sae_id). Pass --sae-release and --sae-id.")
        return 1

    is_gemma_official = cfg.MODEL_KEY in H3_CONFIRMATORY_MODELS
    log.info("=" * 70)
    log.info("Phase 5 Stage 1 — Load SAE/transcoder (H3)")
    log.info("  model         : %s  (key=%s)", cfg.MODEL_ID, cfg.MODEL_KEY)
    log.info("  causal layer  : requested L%s → using L%s (snap %s)",
             requested_layer, layer, snap_distance)
    rel_label = f"{release} (HF repo)" if source == "hf_disk" else release
    id_label = f"{sae_id} (subdir)" if source == "hf_disk" else sae_id
    log.info("  kind          : %s", kind)
    log.info("  source        : %s", source)
    log.info("  width         : %s", width)
    log.info("  sae release   : %s", rel_label)
    log.info("  sae id        : %s", id_label)
    log.info("  H3 role       : %s", "Gemma official SAE (headline H3)"
             if is_gemma_official else "exploratory-extra")
    log.info("  phase2 source : %s", phase2_run_dir or "standalone (--layer; Phase 2 not used)")
    log.info("  phase5 out    : %s", paths.run_dir)
    log.info("=" * 70)

    if source in ("registry", "hf_disk") and SAE is None:
        log.error(
            "sae-lens is required for the %s source. Install with:\n"
            "    pip install -e \".[interp]\"\n"
            "Original import error: %s", source, _SAE_LENS_IMPORT_ERROR,
        )
        return 2

    log.info("Loading SAE (downloads + caches on first run; transcoders are large)...")
    try:
        if source == "qwen_scope_pt":
            # Qwen-Scope TopK SAE `.pt` dict — pure torch, no sae-lens needed.
            k = int(entry.get("k", 50)) if entry else 50
            sae = load_qwen_scope_module(release, layer, k, dev.device, file_template=sae_id)
        elif source == "hf_disk":
            sae = load_hf_disk_module(SAE, release, sae_id, dev.device)
        else:
            sae = load_registry_module(SAE, Transcoder, release, sae_id, dev.device, kind)
    except Exception as e:
        log.error("Load failed for (release=%r, sae_id=%r): %s", release, sae_id, e)
        log.error("Verify the address (sae-lens registry or HF repo subdir), then pass "
                  "--sae-release / --sae-id explicitly.")
        return 1

    d_sae = int(_sae_attr(sae, "d_sae", default=-1))
    d_in = int(_sae_attr(sae, "d_in", default=-1))
    hook_name = _sae_attr(sae, "hook_name", "hook", default=getattr(cfg, "SAE_HOOK_DEFAULT", ""))

    if d_in not in (-1, cfg.HIDDEN_DIM):
        log.warning("SAE d_in=%s != model HIDDEN_DIM=%s — wrong SAE for this model?",
                    d_in, cfg.HIDDEN_DIM)

    input_caveat = (
        "Transcoder: encoder was trained on the MLP input, not resid_post. Stage 02 "
        "feeds the layer residual to .encode(), so the Qwen3 H3 is an MLP-transcoder "
        "analogue of the residual-SAE H3 — exploratory and not directly comparable to "
        "the Gemma residual-SAE Jaccard."
        if kind == "transcoder" else
        "Residual SAE on resid_post — Stage 02 feeds hidden_states[layer+1], an exact match."
    )

    meta = {
        "phase": "phase5_stage1_load_sae",
        "model_key": cfg.MODEL_KEY,
        "model_id": cfg.MODEL_ID,
        "sae_repo": getattr(cfg, "SAE_REPO", ""),
        "source": source,
        "sae_release": release,
        "sae_id": sae_id,
        "kind": kind,
        "is_transcoder": kind == "transcoder",
        "requested_layer": requested_layer,
        "layer": layer,
        "layer_snap_distance": snap_distance,
        "available_sae_layers": entry["layers"] if entry else None,
        "width": width,
        "hook_name": str(hook_name),
        "d_sae": d_sae,
        "d_in": d_in,
        "hidden_dim": cfg.HIDDEN_DIM,
        "n_layers": cfg.N_LAYERS,
        "h3_official_gemma_sae": is_gemma_official,
        "causal_layer_source": "--layer" if args.layer is not None else "phase2_stage04",
        "phase2_run_dir": str(phase2_run_dir) if phase2_run_dir else None,
        "device": dev.to_dict(),
        "completed_at": datetime.datetime.now().isoformat(),
        "input_caveat": input_caveat,
        "caveat": (
            "Per the deposited OSF preregistration, H3 is exploratory "
            "(no confirmatory weight). d_in must equal hidden_dim. If "
            "layer_snap_distance > 0 the SAE is not exactly at the causal layer."
        ),
    }
    paths.sae_meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Wrote %s", paths.sae_meta_path)
    log.info("  d_sae=%s  d_in=%s  hook=%s  kind=%s", d_sae, d_in, hook_name, kind)

    write_latest_pointer(cfg.MODEL_KEY, paths.run_dir)
    log.info("LATEST.txt → %s", paths.run_dir.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
