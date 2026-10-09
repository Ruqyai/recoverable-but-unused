#!/usr/bin/env python3
"""Phase 5 Stage 6: H3 validity controls — the calibration ladder.

A low Tier-A Jaccard (Arabic↔Arabizi harmful) is only meaningful if (a) the
measure actually rises when meaning is shared, and (b) the low overlap is
specific to harm rather than a generic surface-form effect. This stage computes
the calibration ladder from the H3-controls brainstorm (2026-06-10). Each pair
holds MEANING constant and isolates ONE variable:

  Tier D   Arabic↔Arabic-paraphrase   — isolates WORDING (same lang+script).
                                         **Validity check:** if meaning drives
                                         features, this must be HIGH.
  Tier A   Arabic↔Arabizi (harmful)   — isolates SCRIPT (same language). The
                                         headline test; LOW = orthographic anchoring.
  X-lang   Arabic↔English (harmful)   — isolates LANGUAGE (same meaning, matched
                                         AdvBench source_ids). A foreign-language
                                         floor: is Arabizi (script-only) as low as
                                         a full language change?
  Tier B   Arabic-benign↔Arabizi-benign — script change on BENIGN content. If HIGH
                                         while Tier-A-harmful is LOW → the
                                         fragmentation is HARM-SPECIFIC.

Interpretation we hope for: **Tier D (high) ≫ Tier A (low).** If additionally
Tier A ≈ the X-lang floor, romanizing one's own language makes the model treat it
**like a foreign language**. If Tier D is NOT high, the measure does not track
meaning and H3 is uninterpretable (cut it).

Reuses the Phase 5 extraction + ranking + Jaccard machinery (incl. the bf16 /
multimodal / transcoder paths). Reads the SAE address from the run's
`sae_meta.json`. Per the deposited OSF preregistration, H3 is exploratory.

Usage
-----
  PYTHONPATH=. python3 scripts/phase5_sae_features/06_h3_controls.py --model gemma_2_9b_it
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import torch
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
    REPO,
    h3_verdict,
    resolve_phase5_run_paths,
    write_latest_pointer,
)
from scripts.phase5_sae_features.sae_loading import load_hf_disk_module, load_registry_module
from scripts.utils.device import (
    detect_device,
    get_text_base_model,
    load_hf_model,
    log_device_info,
    resolve_dtype,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_P5 = Path(__file__).resolve().parent


def _load_stage(stem: str):
    spec = importlib.util.spec_from_file_location(f"phase5_{stem}", _P5 / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Per-arm condition sets (each holds MEANING constant, moves one variable) ──
# Generic condition keys make the ladder/banner arm-agnostic:
#   native_harmful · script_harmful · xlang_harmful · native_paraphrase ·
#   native_benign · script_benign  (xlang_harmful absent on the Hindi arm).
# The Arabic & English harmful arms share AdvBench source_ids (clean X-lang to each
# other). The Hindi arm is Nemotron-sourced → no matched cross-language control.
ARMS: dict[str, dict] = {
    "arabic": {
        "label": "Arabic ↔ Arabizi (chatspeak romanization)",
        "conditions": {
            "native_harmful":    REPO / "data/tier_a/tier_a_arabic.jsonl",
            "script_harmful":    REPO / "data/tier_a/tier_a_arabizi.jsonl",
            "xlang_harmful":     REPO / "data/tier_e/tier_e_english.jsonl",
            "native_paraphrase": REPO / "data/tier_d/tier_d_arabic_paraphrased.jsonl",
            "native_benign":     REPO / "data/tier_b/benign_arabic.jsonl",
            "script_benign":     REPO / "data/tier_b/benign_arabizi.jsonl",
        },
        "xlang_label": "English",
    },
    # Tier E: orthography within ENGLISH (high-resource) → targets the rarity confound.
    "english": {
        "label": "English ↔ Leetspeak (within-language orthography)",
        "conditions": {
            "native_harmful":    REPO / "data/tier_e/tier_e_english.jsonl",
            "script_harmful":    REPO / "data/tier_e/tier_e_leetspeak.jsonl",
            "xlang_harmful":     REPO / "data/tier_a/tier_a_arabic.jsonl",
            "native_paraphrase": REPO / "data/tier_d/tier_d_english_paraphrased.jsonl",
            "native_benign":     REPO / "data/tier_b/benign_english.jsonl",
            "script_benign":     REPO / "data/tier_b/benign_leetspeak.jsonl",
        },
        "xlang_label": "Arabic",
    },
    # Tier C: STANDARD deterministic transliteration (ITRANS) → targets the RomanLens
    # tension. Nemotron-sourced ⇒ no matched X-lang (partial ladder).
    "hindi": {
        "label": "Hindi Devanagari ↔ ITRANS (standard transliteration)",
        "conditions": {
            "native_harmful":    REPO / "data/tier_c/tier_c_devanagari.jsonl",
            "script_harmful":    REPO / "data/tier_c/tier_c_itrans.jsonl",
            "native_paraphrase": REPO / "data/tier_d/tier_d_devanagari_paraphrased.jsonl",
            "native_benign":     REPO / "data/tier_b/benign_devanagari.jsonl",
            "script_benign":     REPO / "data/tier_b/benign_itrans.jsonl",
        },
        "xlang_label": None,  # no matched cross-language source
    },
}


def build_ladder(arm: str) -> list[tuple]:
    """(label, cond_a, cond_b, expectation) pairs for an arm, in decreasing-J order."""
    xl = ARMS[arm].get("xlang_label")
    ladder = [
        ("tier_d (meaning⇒feature VALIDITY; words only)", "native_harmful", "native_paraphrase",
         "HIGH expected if the measure tracks meaning (same lang+script, paraphrase)"),
        ("headline (SCRIPT change, same language)", "native_harmful", "script_harmful",
         "LOW expected (orthographic anchoring)"),
    ]
    if xl and "xlang_harmful" in ARMS[arm]["conditions"]:
        ladder.append(
            (f"cross-language (LANGUAGE change → {xl}, same meaning)", "native_harmful",
             "xlang_harmful", f"isolates LANGUAGE (meaning held); foreign-language floor ({xl})"))
    ladder.append(
        ("tier_b (harm-specificity; benign script change)", "native_benign", "script_benign",
         "if HIGH while headline is LOW → fragmentation is harm-specific"))
    return ladder


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Phase 5 Stage 6 — H3 validity controls (calibration ladder).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=list_models())
    ap.add_argument("--arm", default="arabic", choices=list(ARMS.keys()),
                    help="Language pair / orthography regime: arabic (↔Arabizi), "
                         "english (↔Leetspeak; targets rarity confound), hindi (Devanagari↔ITRANS; "
                         "targets RomanLens tension — no X-lang).")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Phase 5 run with sae_meta.json (default: LATEST).")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--statistic", choices=["mean", "frequency"], default="mean")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--prefer-device", choices=["cuda", "xla", "mps", "cpu"], default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if SAE is None:
        log.error("sae-lens required. pip install -e \".[interp]\"  (%s)", _SAE_LENS_IMPORT_ERROR)
        return 2

    s02 = _load_stage("02_extract_feature_activations")
    s04 = _load_stage("04_jaccard_similarity")

    cfg = load_model_config(args.model)
    arm = args.arm
    conditions = ARMS[arm]["conditions"]
    ladder_spec = build_ladder(arm)
    paths = resolve_phase5_run_paths(cfg.MODEL_KEY, args.run_dir, new_run=False)
    if not paths.sae_meta_path.exists():
        log.error("Missing %s — run Stage 01 first.", paths.sae_meta_path)
        return 1
    meta = json.loads(paths.sae_meta_path.read_text())
    layer = int(meta["layer"])
    release, sae_id, kind = meta["sae_release"], meta["sae_id"], meta.get("kind", "sae")
    source = meta.get("source", "registry")
    is_multimodal = bool(getattr(cfg, "IS_MULTIMODAL", False))

    dev = detect_device(prefer=args.prefer_device)
    log_device_info(dev)
    batch_size = args.batch_size or dev.apply_batch_size(cfg.BATCH_SIZE)
    pref_dtype = resolve_dtype(getattr(cfg, "PREFERRED_DTYPE", None))

    log.info("=" * 70)
    log.info("Phase 5 Stage 6 — H3 validity controls (calibration ladder)")
    log.info("  arm       : %s  — %s", arm, ARMS[arm]["label"])
    log.info("  model     : %s  (key=%s)", cfg.MODEL_ID, cfg.MODEL_KEY)
    log.info("  layer/SAE : L%s :: %s :: %s (kind=%s, source=%s)",
             layer, release, sae_id, kind, source)
    log.info("  run dir   : %s", paths.run_dir)
    log.info("=" * 70)

    # ── Load model + SAE (bf16 / multimodal / source aware) ───────────────────
    log.info("Loading model %s ...", cfg.MODEL_ID)
    model = load_hf_model(cfg.MODEL_ID, dev, hf_token=os.environ.get("HF_TOKEN"), dtype=pref_dtype,
                          model_class=getattr(cfg, "MODEL_CLASS", None))
    model.eval()
    base_model = get_text_base_model(
        model, is_multimodal=is_multimodal,
        submodule_path=getattr(cfg, "TEXT_SUBMODULE_PATH", None),
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading SAE %s :: %s (source=%s) ...", release, sae_id, source)
    try:
        if source == "hf_disk":
            sae = load_hf_disk_module(SAE, release, sae_id, dev.device)
        else:
            sae = load_registry_module(SAE, Transcoder, release, sae_id, dev.device, kind)
    except Exception as e:
        log.error("SAE load failed: %s", e)
        return 1

    mlp_module = None
    if kind == "transcoder":
        mlp_module = s02.resolve_decoder_layers(model, base_model, cfg)[layer].mlp
        log.info("Transcoder input hook: decoder_layers[%d].mlp", layer)

    # ── Extract features for every control condition (arm-prefixed to avoid collisions) ──
    ctrl_dir = paths.feature_acts_dir / "controls"
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    feats: dict[str, torch.Tensor] = {}
    for name, src in conditions.items():
        if not src.exists():
            log.error("Missing control data: %s", src)
            return 1
        out_path = ctrl_dir / f"{arm}_{name}.pt"
        log.info("Condition: %s  ←  %s", name, src.name)
        s02.extract_for_file(
            model, base_model, tokenizer, sae, src, out_path, dev,
            layer=layer, n_layers=cfg.N_LAYERS, chat_template=cfg.CHAT_TEMPLATE,
            batch_size=batch_size, mlp_module=mlp_module,
        )
        feats[name] = torch.load(out_path, map_location="cpu")["features"]

    # ── Compute the ladder ────────────────────────────────────────────────────
    results = []
    for label, a, b, expect in ladder_spec:
        fa, fb = feats[a], feats[b]
        sa = s04.top_k_set(fa, args.k, args.statistic, 0.0)
        sb = s04.top_k_set(fb, args.k, args.statistic, 0.0)
        j = s04.jaccard(sa, sb)
        ci = (s04.bootstrap_jaccard(fa, fb, args.k, args.statistic, 0.0, args.bootstrap, args.seed)
              if args.bootstrap > 0 else None)
        results.append({"label": label, "pair": [a, b], "jaccard": j,
                        "intersection": len(sa & sb), "union": len(sa | sb),
                        "bootstrap_ci": ci, "expectation": expect,
                        "verdict": h3_verdict(j, ci)})
        ci_s = (f"  99%CI[{ci['ci_99_low']:.3f},{ci['ci_99_high']:.3f}]"
                if ci and "ci_99_low" in ci else "")
        log.info("  %-46s J=%.3f%s   (%s)", label, j, ci_s, expect)

    out = {
        "phase": "phase5_stage6_h3_controls",
        "arm": arm, "arm_label": ARMS[arm]["label"],
        "model_key": cfg.MODEL_KEY, "model_id": cfg.MODEL_ID, "layer": layer,
        "sae_release": release, "sae_id": sae_id, "kind": kind,
        "k": args.k, "statistic": args.statistic, "bootstrap": args.bootstrap, "seed": args.seed,
        "completed_at": datetime.datetime.now().isoformat(),
        "ladder": results,
        "interpretation": (
            "VALID measure ⇒ tier_d HIGH (meaning⇒features) AND headline (script) LOW. "
            "If tier_b benign is also LOW, the fragmentation is generic (surface), not "
            "harm-specific. If tier_d is LOW, the measure does not track meaning → H3 "
            "uninterpretable. Per OSF prereg, H3 is exploratory."
        ),
    }
    ctrl_path = paths.run_dir / f"h3_controls_{arm}.json"
    ctrl_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info("\nWrote %s", ctrl_path)

    # ── Verdict banner (generic keys) ─────────────────────────────────────────
    by = {r["pair"][0] + "|" + r["pair"][1]: r["jaccard"] for r in results}
    td = by.get("native_harmful|native_paraphrase")   # wording
    ta = by.get("native_harmful|script_harmful")      # script change, same language
    cx = by.get("native_harmful|xlang_harmful")        # language change (None on hindi arm)
    tb = by.get("native_benign|script_benign")        # benign script change

    def _f(x):
        return f"{x:.3f}" if x is not None else "  n/a"

    log.info("\n" + "=" * 70)
    log.info("H3 CALIBRATION LADDER — %s · %s", cfg.MODEL_KEY, ARMS[arm]["label"])
    log.info("  Tier D  wording (same lang+script)    : J=%s  %s", _f(td),
             "✓ tracks meaning" if td and td >= 0.5 else "✗ does NOT track meaning")
    log.info("  Headline SCRIPT change, same language : J=%s", _f(ta))
    if cx is not None:
        log.info("  X-lang  LANGUAGE change (%s)         : J=%s", ARMS[arm]["xlang_label"], _f(cx))
    log.info("  Tier B  benign script change          : J=%s", _f(tb))
    if td is not None and ta is not None:
        if td < 0.4:
            log.info("  → Tier D LOW: measure does NOT track meaning → H3 UNINTERPRETABLE.")
        elif td >= 0.5 and ta < 0.4:
            note = ""
            if cx is not None:
                note = ("  Script change disrupts MORE than the language change → orthography "
                        "is the most disruptive variable." if cx > ta + 0.03 else
                        "  Script ≈ language change in disruption.")
            log.info("  → Tier D ≫ headline: low cross-script overlap is MEANINGFUL.%s", note)
            if tb is not None:
                log.info("  → harm-specificity: Tier B %s headline (%s).",
                         "≈" if abs(tb - ta) < 0.06 else (">" if tb > ta else "<"),
                         "generic/orthographic" if abs(tb - ta) < 0.06 else "differs")
        else:
            log.info("  → Ambiguous: Tier D not clearly high; interpret with caution.")
    log.info("=" * 70)

    write_latest_pointer(cfg.MODEL_KEY, paths.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
