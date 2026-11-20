#!/usr/bin/env python3
"""Extract clean bfloat16 last-token native directions for Gemma-2-9B-it and refresh Tier-E transfer.

Fixes the June 28 MPS extraction artefact where Gemma's default left-padding caused
`attention_mask[i].sum() - 1` to index mid-prompt/<pad> positions on short English prompts
in float16. Extracts all 200 paired prompts per native script (english, arabic, devanagari)
in bfloat16 at the true final token (-1 under left-padding) and refreshes Gemma-2-9B-it's
comprehended-Leetspeak direction transfer (Table 3).
"""
from __future__ import annotations

import gc
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.models import load_model_config  # noqa: E402
from scripts.phase2_direction_extraction.config import BENIGN_FILES, HARMFUL_FILES  # noqa: E402
from scripts.utils.device import detect_device, get_text_base_model, load_hf_model  # noqa: E402

_H2 = ROOT / "scripts/phase4_causal_intervention/interventions/01_h2_forward_reverse.py"
_spec = importlib.util.spec_from_file_location("_h2mod", _H2)
h2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h2)

_DT = ROOT / "scripts/phase4_causal_intervention/interventions/02_direction_transfer_comprehended.py"
_spec2 = importlib.util.spec_from_file_location("_dtmod", _DT)
dt = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(dt)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - rad), 100 * (c + rad)]


def extract_script_direction(base_model, tok, cfg, device: str, script: str, batch_size: int = 4) -> torch.Tensor:
    harm_map = {r["source_id"]: r["prompt"] for r in (json.loads(l) for l in open(HARMFUL_FILES[script]) if l.strip())}
    ben_map = {r["source_id"]: r["prompt"] for r in (json.loads(l) for l in open(BENIGN_FILES[script]) if l.strip())}
    sids = sorted(set(harm_map) & set(ben_map))
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    def run_side(prompts: list[str]) -> torch.Tensor:
        rows = []
        for i in range(0, len(prompts), batch_size):
            chunk = [h2.format_prompt(tok, p, cfg) for p in prompts[i:i + batch_size]]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                out = base_model(**enc, output_hidden_states=True, use_cache=False)
            # Stack [batch, n_layers, hidden_dim] at final real token (-1 under left padding)
            layer_stack = torch.stack([out.hidden_states[L][:, -1, :].float().cpu()
                                       for L in range(1, cfg.N_LAYERS + 1)], dim=1)
            rows.append(layer_stack)
        return torch.cat(rows, dim=0)

    H = run_side([harm_map[s] for s in sids])
    B = run_side([ben_map[s] for s in sids])
    diff = H.mean(dim=0) - B.mean(dim=0)
    norms = diff.norm(dim=1, keepdim=True).clamp(min=1e-8)
    print(f"  extracted {script:11s}: {len(sids)} pairs, mean diff norm={diff.norm(dim=1).mean():.2f}", flush=True)
    return diff / norms


def generate_batch(model, tok, cfg, device: str, prompts: list[str], max_new_tokens: int, batch_size: int = 4) -> list[str]:
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    formatted = [h2.format_prompt(tok, p, cfg) for p in prompts]
    outs = []
    for i in range(0, len(formatted), batch_size):
        chunk = formatted[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for row in out:
            outs.append(tok.decode(row[plen:], skip_special_tokens=True).strip())
    return outs


def main() -> None:
    model_key = "gemma_2_9b_it"
    layer, alpha = 19, 1.5
    cfg = load_model_config(model_key)
    dev = detect_device()
    clean_dir = ROOT / f"results/phase2/{model_key}/directions_bf16_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {cfg.MODEL_ID} in bfloat16 on {dev.device}...", flush=True)
    model = load_hf_model(cfg.MODEL_ID, dev, dtype=torch.bfloat16)
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg.MODEL_ID)
    base_model = get_text_base_model(model)

    for script in ("english", "arabic", "devanagari"):
        out_pt = clean_dir / f"{script}.pt"
        if out_pt.exists():
            print(f"  [cached] {out_pt}", flush=True)
            continue
        vec = extract_script_direction(base_model, tok, cfg, dev.device, script, batch_size=4)
        torch.save(vec, out_pt)
        dev.empty_cache()

    # Refresh comprehended-Leetspeak direction transfer (Table 3)
    ids = dt.comprehended_ids(model_key, "leetspeak_harmful")
    pairs = dt.load_romanized_prompts(model_key, "leetspeak_harmful", ids, 50)
    prompts = [p for _, p in pairs]
    print(f"Running comprehended Leetspeak transfer on {len(prompts)} items (L={layer}, alpha={alpha})...", flush=True)

    lmod = h2.get_layer_module(model, layer, cfg)
    en_vec = torch.load(clean_dir / "english.pt", map_location="cpu")[layer].to(dev.device).float()
    en_vec = en_vec / en_vec.norm()
    g = torch.Generator().manual_seed(42)
    rand_vec = torch.randn(en_vec.shape, generator=g).to(dev.device).float()
    rand_vec = rand_vec / rand_vec.norm()

    base_resps = generate_batch(model, tok, cfg, dev.device, prompts, 64, batch_size=4)
    hk = lmod.register_forward_hook(h2.make_ablation_hook(en_vec, alpha))
    try:
        nat_resps = generate_batch(model, tok, cfg, dev.device, prompts, 64, batch_size=4)
    finally:
        hk.remove()
    hk = lmod.register_forward_hook(h2.make_ablation_hook(rand_vec, alpha))
    try:
        rnd_resps = generate_batch(model, tok, cfg, dev.device, prompts, 64, batch_size=4)
    finally:
        hk.remove()

    base_cls = [h2.classify_response(p, r, None) for p, r in zip(prompts, base_resps)]
    nat_cls = [h2.classify_response(p, r, None) for p, r in zip(prompts, nat_resps)]
    rnd_cls = [h2.classify_response(p, r, None) for p, r in zip(prompts, rnd_resps)]

    room = sum(1 for b in base_cls if b["refused"])
    fn = sum(1 for b, r in zip(base_cls, nat_cls) if b["refused"] and (not r["refused"]) and r["coherent"])
    fr = sum(1 for b, r in zip(base_cls, rnd_cls) if b["refused"] and (not r["refused"]) and r["coherent"])
    rn = 100 * fn / room if room else float("nan")
    rr = 100 * fr / room if room else float("nan")

    out_json = ROOT / f"results/phase4_intervention/{model_key}/direction_transfer/transfer_tier_e_20261119_013805.json"
    payload = {
        "phase": "direction_transfer_on_comprehended_subset",
        "model": model_key,
        "pair": "tier_e",
        "subset": "comprehended_leetspeak",
        "romanized_condition": "leetspeak_harmful",
        "native_direction": "directions_bf16_clean/english.pt",
        "dtype": "bfloat16",
        "causal_layer": layer,
        "alpha": alpha,
        "n_comprehended": len(prompts),
        "room_baseline_refused": room,
        "native_dir_ablation": {"flips": fn, "flip_rate_pct": rn, "wilson95": wilson(fn, room)},
        "random_dir_ablation": {"flips": fr, "flip_rate_pct": rr, "wilson95": wilson(fr, room)},
        "specificity_gap_pp": rn - rr,
        "layer_alpha_source": str(ROOT / f"results/phase4_intervention/{model_key}/run_20260702_072549/h2_results.json"),
        "completed_at": datetime.now().isoformat(),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[saved] {out_json}: native={fn}/{room} ({rn:.1f}%), random={fr}/{room} ({rr:.1f}%)", flush=True)

    del model, base_model
    gc.collect()
    dev.empty_cache()


if __name__ == "__main__":
    main()
