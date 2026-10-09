#!/usr/bin/env python3
"""Phase 5 Stage 2: extract per-prompt SAE feature activations at the causal layer.

Reads `sae_meta.json` (Stage 01), loads the model + SAE/transcoder, runs every
Phase 1 prompt through the model, captures the residual stream at the SAE's
layer (the model's `hidden_states[layer+1]` = resid_post of that layer) at the
measurement position, encodes it through the SAE, and saves the per-prompt
feature matrix per script:

    feature_activations/<script>.pt   →  {"features": [N, d_sae] float32,
                                          "ids": [...], "source_ids": [...]}

Harmful prompts are extracted by default (H3 compares harmful Arabic vs Arabizi,
etc.); add --include-benign to also dump the Tier B matched benigns.

Consistency with Phase 2: residuals are taken exactly as in
`01_extract_activations.py` — `base_model(output_hidden_states=True)`, layer
index `layer+1`, last non-pad position — so the H3 features sit on the same
representation the refusal direction was extracted from.

Per the deposited OSF preregistration, H3 carries no confirmatory weight.

Usage
-----
  PYTHONPATH=. python3 scripts/phase5_sae_features/02_extract_feature_activations.py \
      --model gemma_2_9b_it
  PYTHONPATH=. python3 scripts/phase5_sae_features/02_extract_feature_activations.py \
      --model gemma_3_1b_it --scripts arabic arabizi --include-benign
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from sae_lens import SAE

    try:
        from sae_lens import Transcoder
    except Exception:
        Transcoder = None
except ImportError as e:  # pragma: no cover
    SAE = None
    Transcoder = None
    _SAE_LENS_IMPORT_ERROR = e
else:
    _SAE_LENS_IMPORT_ERROR = None

from scripts.models import DEFAULT_MODEL_KEY, list_models, load_model_config
from scripts.phase5_sae_features.config import (
    BENIGN_FILES,
    HARMFUL_FILES,
    SCRIPTS,
    resolve_phase5_run_paths,
    write_latest_pointer,
)
from scripts.phase5_sae_features.sae_loading import load_hf_disk_module, load_registry_module
from scripts.utils.device import (
    DeviceConfig,
    detect_device,
    get_text_base_model,
    load_hf_model,
    log_device_info,
    resolve_dtype,
)
from scripts.utils.io import read_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def format_prompt(tokenizer, prompt: str, fallback_template: str) -> str:
    """Prefer tokenizer.apply_chat_template; fall back to the manual template.

    Mirrors Phase 2 Stage 01 so the residual matches the extracted direction.
    """
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception as e:
        log.debug("apply_chat_template failed (%r); using manual template", e)
    return fallback_template.format(prompt=prompt)


def encode_features(sae, resid: torch.Tensor) -> torch.Tensor:
    """Encode a [B, d_in] residual batch into [B, d_sae] feature activations.

    sae-lens SAEs and transcoders both expose `.encode`. Cast to the module's
    dtype/device first (SAEs load float32 by default).
    """
    p = next(sae.parameters())
    x = resid.to(device=p.device, dtype=p.dtype)
    with torch.no_grad():
        feats = sae.encode(x)
    if isinstance(feats, tuple):  # some encoders return (acts, ...)
        feats = feats[0]
    return feats.detach().float().cpu()


def resolve_decoder_layers(model, base_model, cfg):
    """Return the decoder-layer ModuleList.

    Standard text models: `base_model.layers` (base_model = model.model). A
    multimodal model can override the dotted path via cfg.LAYERS_PATH **relative
    to the full model** (e.g. 'model.language_model.layers' for Gemma-3 VLMs).
    """
    layers_path = getattr(cfg, "LAYERS_PATH", None)
    if layers_path:
        obj = model
        for attr in layers_path.split("."):
            obj = getattr(obj, attr)
        return obj
    layers = getattr(base_model, "layers", None)
    if layers is not None:
        return layers
    return model.model.layers


def extract_for_file(
    model,
    base_model,
    tokenizer,
    sae,
    data_path: Path,
    out_path: Path,
    dev: DeviceConfig,
    *,
    layer: int,
    n_layers: int,
    chat_template: str,
    batch_size: int,
    mlp_module=None,
) -> int:
    """Extract SAE/transcoder features for every record in one JSONL → one .pt.

    SAE (mlp_module=None): input = `resid_post` at `layer` = hidden_states[layer+1].
    Transcoder (mlp_module given): input = the MLP INPUT at `layer`, captured via a
    forward_pre_hook on the layer's MLP module — because mwhanna transcoders encode
    `blocks.{L}.mlp.hook_in`, NOT resid_post. Feeding resid_post to them would be
    out-of-distribution; this hook gives the transcoder the activation it expects.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info("  %s already done — skipping", out_path.name)
        return 0

    records = list(read_jsonl(data_path))
    rows: list[torch.Tensor] = []
    ids: list[str] = []
    source_ids: list[str] = []

    for start in tqdm(range(0, len(records), batch_size),
                      desc=f"  {data_path.name}", unit="batch"):
        batch = records[start : start + batch_size]
        texts = [format_prompt(tokenizer, r.prompt, chat_template) for r in batch]
        inputs = tokenizer(texts, padding=True, return_tensors="pt",
                           truncation=True, max_length=2048)
        inputs = {k: v.to(dev.device) for k, v in inputs.items()}

        if mlp_module is not None:
            # Transcoder: capture the MLP input (mlp.hook_in) via a pre-hook.
            captured: dict[str, torch.Tensor] = {}

            def _pre_hook(_m, args, _store=captured):
                _store["x"] = args[0].detach()

            handle = mlp_module.register_forward_pre_hook(_pre_hook)
            try:
                with torch.no_grad():
                    base_model(**inputs, use_cache=False)
            finally:
                handle.remove()
            act = captured["x"]  # [B, seq, d_in] = MLP input at `layer`
        else:
            with torch.no_grad():
                outputs = base_model(**inputs, output_hidden_states=True, use_cache=False)
            assert len(outputs.hidden_states) == n_layers + 1, (
                f"expected {n_layers + 1} hidden states, got {len(outputs.hidden_states)}"
            )
            # resid_post of `layer` == hidden_states[layer+1]
            act = outputs.hidden_states[layer + 1]

        attn = inputs["attention_mask"]
        resids = []
        for i in range(len(batch)):
            last = attn[i].sum() - 1
            resids.append(act[i, last, :].float())
        resid_batch = torch.stack(resids, dim=0)  # [B, d_in]

        feats = encode_features(sae, resid_batch)  # [B, d_sae]
        for i, rec in enumerate(batch):
            rows.append(feats[i])
            ids.append(rec.id)
            source_ids.append(rec.source_id)

        dev.empty_cache()

    if not rows:
        log.warning("  No records in %s", data_path.name)
        return 0

    features = torch.stack(rows, dim=0)  # [N, d_sae]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "ids": ids, "source_ids": source_ids}, out_path)
    log.info("  Wrote %s  [%d, %d]", out_path.name, features.shape[0], features.shape[1])
    return features.shape[0]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Phase 5 Stage 2 — extract SAE feature activations (H3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=list_models())
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Phase 5 run with sae_meta.json (default: LATEST).")
    ap.add_argument("--scripts", nargs="+", default=SCRIPTS,
                    help="Scripts to extract (default: all six).")
    ap.add_argument("--include-benign", action="store_true",
                    help="Also extract Tier B matched benigns per script.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override per-model BATCH_SIZE (skips device multiplier).")
    ap.add_argument("--prefer-device", choices=["cuda", "xla", "mps", "cpu"], default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if SAE is None:
        log.error("sae-lens required. pip install -e \".[interp]\"  (%s)", _SAE_LENS_IMPORT_ERROR)
        return 2

    cfg = load_model_config(args.model)
    paths = resolve_phase5_run_paths(cfg.MODEL_KEY, args.run_dir, new_run=False)

    if not paths.sae_meta_path.exists():
        log.error("Missing %s — run Stage 01 (01_load_sae.py) first.", paths.sae_meta_path)
        return 1
    meta = json.loads(paths.sae_meta_path.read_text())
    layer = int(meta["layer"])
    release, sae_id, kind = meta["sae_release"], meta["sae_id"], meta.get("kind", "sae")
    source = meta.get("source", "registry")

    is_multimodal = bool(getattr(cfg, "IS_MULTIMODAL", False))

    dev = detect_device(prefer=args.prefer_device)
    log_device_info(dev)
    batch_size = (
        args.batch_size if args.batch_size is not None
        else dev.apply_batch_size(cfg.BATCH_SIZE)
    )

    log.info("=" * 70)
    log.info("Phase 5 Stage 2 — Extract SAE features (H3)")
    log.info("  model        : %s  (key=%s)", cfg.MODEL_ID, cfg.MODEL_KEY)
    log.info("  layer        : L%s  (resid_post = hidden_states[%s])", layer, layer + 1)
    log.info("  sae          : %s :: %s  (kind=%s)", release, sae_id, kind)
    log.info("  scripts      : %s  benign=%s", args.scripts, args.include_benign)
    log.info("  batch size   : %s", batch_size)
    log.info("  run dir      : %s", paths.run_dir)
    log.info("=" * 70)

    # Load model (base submodule skips lm_head — same optimisation as Phase 2).
    # `AutoModelForCausalLM` loads Gemma-3 VLMs as Gemma3ForConditionalGeneration; for
    # text-only residual extraction we run the LANGUAGE submodule
    # (model.model.language_model = Gemma3TextModel), which accepts input_ids +
    # output_hidden_states and returns n_text_layers+1 hidden states. Verified for
    # google/gemma-3-4b-it (35 states, dim 2560).
    pref_dtype = resolve_dtype(getattr(cfg, "PREFERRED_DTYPE", None))
    if pref_dtype is not None:
        log.info("  PREFERRED_DTYPE override: %s", pref_dtype)
    log.info("Loading model %s ...", cfg.MODEL_ID)
    model = load_hf_model(cfg.MODEL_ID, dev, hf_token=os.environ.get("HF_TOKEN"), dtype=pref_dtype,
                          model_class=getattr(cfg, "MODEL_CLASS", None))
    model.eval()
    base_model = get_text_base_model(
        model, is_multimodal=is_multimodal,
        submodule_path=getattr(cfg, "TEXT_SUBMODULE_PATH", None),
    )
    if is_multimodal:
        log.info("Multimodal model — extracting from language submodule (%s)",
                 type(base_model).__name__)
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SAE (cached by Stage 01); same source the meta recorded.
    log.info("Loading SAE/transcoder %s :: %s  (source=%s) ...", release, sae_id, source)
    try:
        if source == "hf_disk":
            sae = load_hf_disk_module(SAE, release, sae_id, dev.device)
        else:
            sae = load_registry_module(SAE, Transcoder, release, sae_id, dev.device, kind)
    except Exception as e:
        log.error("SAE load failed: %s", e)
        return 1

    # Transcoders encode the MLP input (mlp.hook_in), not resid_post — resolve the
    # layer's MLP module so extract_for_file can capture it via a forward_pre_hook.
    mlp_module = None
    if kind == "transcoder":
        try:
            mlp_module = resolve_decoder_layers(model, base_model, cfg)[layer].mlp
        except Exception as e:
            log.error("Could not resolve decoder layer %d's MLP for the transcoder "
                      "input hook: %s", layer, e)
            return 1
        log.info("Transcoder input hook: decoder_layers[%d].mlp (captures mlp.hook_in)", layer)

    total = 0
    for script in args.scripts:
        if script not in SCRIPTS:
            log.error("Unknown script: %s", script)
            continue
        log.info("Script: %s", script)
        total += extract_for_file(
            model, base_model, tokenizer, sae,
            HARMFUL_FILES[script], paths.feature_acts_dir / f"{script}.pt", dev,
            layer=layer, n_layers=cfg.N_LAYERS, chat_template=cfg.CHAT_TEMPLATE,
            batch_size=batch_size, mlp_module=mlp_module,
        )
        if args.include_benign:
            total += extract_for_file(
                model, base_model, tokenizer, sae,
                BENIGN_FILES[script], paths.feature_acts_dir / f"{script}_benign.pt", dev,
                layer=layer, n_layers=cfg.N_LAYERS, chat_template=cfg.CHAT_TEMPLATE,
                batch_size=batch_size, mlp_module=mlp_module,
            )

    # Stamp extraction metadata next to the features
    (paths.feature_acts_dir / "extraction_meta.json").write_text(json.dumps({
        "phase": "phase5_stage2_extract_features",
        "model_key": cfg.MODEL_KEY,
        "layer": layer,
        "sae_release": release,
        "sae_id": sae_id,
        "kind": kind,
        "scripts": args.scripts,
        "include_benign": args.include_benign,
        "batch_size": batch_size,
        "device": dev.to_dict(),
        "completed_at": datetime.datetime.now().isoformat(),
    }, indent=2))

    write_latest_pointer(cfg.MODEL_KEY, paths.run_dir)
    log.info("Total feature vectors written: %d", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
