#!/usr/bin/env python3
"""Direction transfer on the comprehended subset (reviewer ask #1 -> the main-track lever).

The conditional analysis (Table 2) proves refusal *correlates* with comprehension.
It does NOT prove the *same native refusal direction* mediates that refusal on
comprehended-romanized items. This script closes that gap directly:

  On the COMPREHENDED Arabizi subset (items Gemini judged ON_TOPIC/REFUSAL, where the
  model demonstrably read the request and mostly refused), ablate the NATIVE (Arabic)
  refusal direction and measure whether refusal is removed (refuse -> coherent comply).

  Prediction under "intact-but-starved": ablating the native direction flips refusal on
  comprehended-Arabizi at rates comparable to ablating it on native Arabic (i.e. the SAME
  direction mediates), and far above a norm-matched RANDOM direction (specificity).

This is the one experiment that converts the paper's asserted positive half ("safety is
intact, merely starved") into a demonstrated one. It reuses the EXACT ablation mechanics
of 01_h2_forward_reverse.py (imported, not re-implemented) so the result is method-identical
to the paper's H2. Needs a GPU; ~200 generations x 3 arms x model.

Run (when a GPU is available):
  PYTHONPATH=. python3 scripts/phase4_causal_intervention/interventions/\
02_direction_transfer_comprehended.py --model gemma_3_4b_it --n 50 --guard

Outputs: results/phase4_intervention/<model>/direction_transfer/transfer_<stamp>.json
"""
from __future__ import annotations
import argparse, importlib.util, json, glob, math
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))
from scripts.models import DEFAULT_MODEL_KEY, list_models, load_model_config  # noqa: E402
from scripts.phase2_direction_extraction.config import resolve_run_paths  # noqa: E402
from scripts.utils.device import detect_device, load_hf_model, log_device_info, resolve_dtype  # noqa: E402

# ---- import the exact H2 mechanics from the numeric-named sibling module ----
_H2 = ROOT / "scripts/phase4_causal_intervention/interventions/01_h2_forward_reverse.py"
_spec = importlib.util.spec_from_file_location("_h2mod", _H2)
h2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h2)  # exposes make_ablation_hook, format_prompt, generate,
#   get_layer_module, classify_response, baseline_pass, run_cell,
#   load_causal_layer, GUARD_MODEL_ID


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


# Each pair: which romanized condition to test, and which NATIVE direction mediates it.
# Arabizi is native-Arabic; Leetspeak is native-English; ITRANS is native-Devanagari.
# Running tier_e/tier_c gives WELL-POWERED cross-family cells where Arabizi is too
# poorly comprehended to leave any room (reviewer ask: cross-family causal evidence).
PAIR_SPEC = {
    "tier_a": {"condition": "arabizi_harmful", "native_dir": "arabic.pt", "subset": "comprehended_arabizi"},
    "tier_e": {"condition": "leetspeak_harmful", "native_dir": "english.pt", "subset": "comprehended_leetspeak"},
    "tier_c": {"condition": "itrans_harmful", "native_dir": "devanagari.pt", "subset": "comprehended_itrans"},
}


def comprehended_ids(model_key, condition):
    """Item ids Gemini judged ON_TOPIC/REFUSAL on the romanized condition (comprehended subset)."""
    fs = glob.glob(str(ROOT / f"results/phase5/comprehension_validation/{model_key}/items_gemini.jsonl"))
    ids = set()
    if fs:
        for line in open(fs[0]):
            r = json.loads(line)
            if r["condition"] == condition and r["label"].upper() in ("ON_TOPIC", "REFUSAL"):
                ids.add(r["id"])
    return ids


def load_romanized_prompts(model_key, condition, ids, n):
    """(source_id, user_text) for the comprehended romanized harmful items."""
    fs = sorted(glob.glob(str(ROOT / f"results/phase5/{model_key}/**/generations.jsonl"), recursive=True))
    out = []
    if fs:
        for line in open(fs[-1]):
            r = json.loads(line)
            if r["condition"] == condition and r["id"] in ids:
                out.append((r["id"], r["prompt"]))
    return out[:n] if n else out


def flip_rate(model, tok, cfg, device, layer, alpha, dir_layer, prompts, base, guard, mnt):
    """Ablation flip (refuse -> coherent comply) using the SAME run_cell as H2."""
    raw, gated, _ = h2.run_cell(model, tok, cfg, device, layer, alpha, "ablation",
                                dir_layer, prompts, base, guard, mnt)
    room = sum(1 for b in base if b["refused"])          # baseline-refused = room to flip
    flips = sum(1 for g, b in zip(gated, base) if g and b["refused"])
    return flips, room


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(list_models()), default=DEFAULT_MODEL_KEY)
    ap.add_argument("--pair", choices=sorted(PAIR_SPEC), default="tier_a",
                    help="tier_a=Arabizi/Arabic, tier_e=Leetspeak/English, tier_c=ITRANS/Devanagari")
    ap.add_argument("--alpha", type=float, default=None, help="override calibrated alpha")
    ap.add_argument("--causal-layer", type=int, default=None, help="override phase-2 causal layer")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--guard", action="store_true", help="use the guard classifier for refusal")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_model_config(args.model)
    dev = detect_device()
    log_device_info(dev)
    device = dev.device

    # causal layer + calibrated alpha (mirror phase-2 selection / H2 defaults)
    paths = resolve_run_paths(args.model)
    # Default to the layer and alpha of this model's own H2 ablation run, so the transfer test is
    # read at the same setting as the necessity result it complements. The old fallback (alpha 0.4 for
    # every model without cfg.ALPHA, layer from the lexical Phase-2 peak) silently mismatched H2 for the
    # non-Gemma-3 models, which H2 ran at alpha 1.5 (fixed 2026-10-08).
    h2_layer, h2_alpha, h2_src = None, None, None
    for f in sorted(glob.glob(str(ROOT / f"results/phase4_intervention/{args.model}/run_*/h2_results.json")), reverse=True):
        d = json.load(open(f))
        if d.get("mode") == "ablation" and d.get("pair") == args.pair:
            h2_layer, h2_alpha, h2_src = d.get("layer"), d.get("alpha"), f
            break
    if h2_layer is None:  # same model, any pair: H2 uses one layer/alpha per model
        for f in sorted(glob.glob(str(ROOT / f"results/phase4_intervention/{args.model}/run_*/h2_results.json")), reverse=True):
            d = json.load(open(f))
            if d.get("mode") == "ablation":
                h2_layer, h2_alpha, h2_src = d.get("layer"), d.get("alpha"), f
                break
    layer = args.causal_layer if args.causal_layer is not None else (h2_layer if h2_layer is not None else h2.load_causal_layer(paths.run_dir))
    if layer is None:
        raise SystemExit("no causal layer found; pass --causal-layer N")
    alpha = args.alpha if args.alpha is not None else (h2_alpha if h2_alpha is not None else getattr(cfg, "ALPHA", None))
    if alpha is None:
        raise SystemExit("no calibrated alpha (no H2 ablation run and no cfg.ALPHA); pass --alpha")
    print(f"[transfer] layer={layer} alpha={alpha} (source: {'CLI' if args.alpha is not None else (h2_src or 'cfg')})", flush=True)

    # load model with the same dtype/class safeguards as H2 (Gemma-3 needs bfloat16)
    model = load_hf_model(cfg.MODEL_ID, dev, dtype=resolve_dtype(getattr(cfg, "PREFERRED_DTYPE", None)),
                          model_class=getattr(cfg, "MODEL_CLASS", None))
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg.MODEL_ID)

    spec = PAIR_SPEC[args.pair]
    # native direction (Arabic/English/Devanagari) at the causal layer + a norm-matched random control
    nat = torch.load(paths.directions_dir / spec["native_dir"], map_location="cpu")[layer].to(device).float()
    nat = nat / nat.norm()
    g = torch.Generator().manual_seed(args.seed)
    rand = torch.randn(nat.shape, generator=g).to(device).float()
    rand = rand / rand.norm()

    # comprehended romanized subset for this pair
    ids = comprehended_ids(args.model, spec["condition"])
    prompts = load_romanized_prompts(args.model, spec["condition"], ids, args.n)
    if not prompts:
        raise SystemExit(f"no comprehended {spec['condition']} items found for this model")

    guard = None
    if args.guard:
        gm = load_hf_model(h2.GUARD_MODEL_ID, dev)
        gm.eval()
        guard = (gm, AutoTokenizer.from_pretrained(h2.GUARD_MODEL_ID))
    mnt = args.max_new_tokens
    base = h2.baseline_pass(model, tok, cfg, device, prompts, guard, mnt)

    # arm 1: ablate NATIVE direction on comprehended Arabizi (the diagnostic)
    f_nat, room = flip_rate(model, tok, cfg, device, layer, alpha, nat, prompts, base, guard, mnt)
    # arm 2: ablate a RANDOM norm-matched direction (specificity control)
    f_rand, _ = flip_rate(model, tok, cfg, device, layer, alpha, rand, prompts, base, guard, mnt)

    def rate(k):
        return (100 * k / room) if room else float("nan")
    lo_n, hi_n = wilson(f_nat, room)
    lo_r, hi_r = wilson(f_rand, room)
    out = {
        "phase": "direction_transfer_on_comprehended_subset",
        "model": args.model, "pair": args.pair, "subset": spec["subset"],
        "romanized_condition": spec["condition"], "native_direction": spec["native_dir"],
        "causal_layer": layer, "alpha": alpha,
        "n_comprehended": len(prompts), "room_baseline_refused": room,
        "native_dir_ablation": {"flips": f_nat, "flip_rate_pct": rate(f_nat),
                                "wilson95": [100 * lo_n, 100 * hi_n]},
        "random_dir_ablation": {"flips": f_rand, "flip_rate_pct": rate(f_rand),
                                "wilson95": [100 * lo_r, 100 * hi_r]},
        "specificity_gap_pp": rate(f_nat) - rate(f_rand),
        "layer_alpha_source": "CLI" if args.alpha is not None else (h2_src or "cfg"),
        "interpretation": ("descriptive only: a high native flip rate with a low random flip rate means the "
                           "native direction mediates refusal on these items; read both with their Wilson "
                           "intervals and the room."),
        "completed_at": datetime.now().isoformat(),
    }
    outdir = ROOT / f"results/phase4_intervention/{args.model}/direction_transfer"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = outdir / f"transfer_{args.pair}_{stamp}.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n[done] {p}")
    print(f"native-dir flip {rate(f_nat):.0f}% vs random-dir flip {rate(f_rand):.0f}% "
          f"(room={room}); gap {out['specificity_gap_pp']:.0f}pp")


if __name__ == "__main__":
    main()
