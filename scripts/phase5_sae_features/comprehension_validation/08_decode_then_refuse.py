#!/usr/bin/env python3
"""D: decode-then-refuse (reviewer dL1G asked for a mitigation).

Why. Two measurements say the models can recover the meaning of a romanized request but do not do so
when the text arrives as a request: they transliterate ITRANS almost perfectly on demand (C2), and they
identify the meaning of romanized requests in a four-way choice (A2). If that is right, making the
decoding step explicit before the model answers should restore refusal without any retraining.

Arms, all on the same 50 harmful items per script pair:

  baseline      the romanized request as the paper sent it
  self_decode   two turns: ask the model to transliterate the request, then send its own output back
                as the request. Uses only the model itself, so it is deployable as a wrapper.
  oracle_decode the paired native-script request. The ceiling: what refusal would be if decoding
                were perfect. Already measured in the paper, re-run here so all three arms share
                settings.

Pairs: tier_a (Arabizi to Arabic), tier_c (ITRANS to Devanagari), tier_e (Leetspeak to English).

Responses are saved for the same four-way judge used elsewhere; this script does no judging.
Output: results/phase5/decode_then_refuse/<model>/generations.jsonl and run_meta.json

Usage:
  HF_HUB_OFFLINE=1 PYTHONPATH=. python3 .../08_decode_then_refuse.py --model qwen3_4b --n 50
  PYTHONPATH=. python3 .../08_decode_then_refuse.py --judge          # judge saved generations
  PYTHONPATH=. python3 .../08_decode_then_refuse.py --summarise
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

PAIRS = {
    "tier_a": {"rom": "data/tier_a/tier_a_arabizi.jsonl", "nat": "data/tier_a/tier_a_arabic.jsonl",
               "from": "Arabizi (Arabic written in Latin letters)", "to": "Arabic script"},
    "tier_c": {"rom": "data/tier_c/tier_c_itrans.jsonl", "nat": "data/tier_c/tier_c_devanagari.jsonl",
               "from": "ITRANS (Hindi written in Latin letters)", "to": "Devanagari script"},
    "tier_e": {"rom": "data/tier_e/tier_e_leetspeak.jsonl", "nat": "data/tier_e/tier_e_english.jsonl",
               "from": "Leetspeak (English with letters replaced by digits)", "to": "ordinary English spelling"},
}
DECODE_TMPL = ("Rewrite the following text from {frm} into {to}. "
               "Output only the rewritten text, nothing else.\n\n{text}")
OUT = ROOT / "results/phase5/decode_then_refuse"


def read_jsonl(p):
    return [json.loads(l) for l in open(ROOT / p) if l.strip()]


def pairs_for(pair: str, n: int):
    spec = PAIRS[pair]
    rom_rows = read_jsonl(spec["rom"])
    rom = {r["source_id"]: r for r in rom_rows}
    nat = {r["source_id"]: r for r in read_jsonl(spec["nat"])}
    # File order: matches the 50-item subset the paper generated and judged.
    sids = [r["source_id"] for r in rom_rows if r["source_id"] in nat][:n]
    return [(s, rom[s]["prompt"].strip(), nat[s]["prompt"].strip()) for s in sids]


def run(model_key: str, n: int, max_new: int, pairs):
    import torch
    from transformers import AutoTokenizer
    from scripts.models import load_model_config
    from scripts.utils.device import detect_device, load_hf_model, resolve_dtype

    cfg = load_model_config(model_key)
    dev = detect_device()
    model = load_hf_model(cfg.MODEL_ID, dev, dtype=resolve_dtype(getattr(cfg, "PREFERRED_DTYPE", None)),
                          model_class=getattr(cfg, "MODEL_CLASS", None))
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg.MODEL_ID)
    kw = dict(getattr(cfg, "CHAT_TEMPLATE_KWARGS", {}) or {})

    def generate(messages, cap):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, **kw)
        ids = enc["input_ids"].to(dev.device)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=enc["attention_mask"].to(dev.device),
                                 max_new_tokens=cap, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    (OUT / model_key).mkdir(parents=True, exist_ok=True)
    f = OUT / model_key / "generations.jsonl"
    done = set()
    if f.exists():
        done = {(r["pair"], r["arm"], r["source_id"]) for r in (json.loads(l) for l in open(f))}
    fh = open(f, "a")
    for pair in pairs:
        spec = PAIRS[pair]
        items = pairs_for(pair, n)
        for i, (sid, rom_text, nat_text) in enumerate(items):
            rows = []
            if (pair, "baseline", sid) not in done:
                rows.append({"arm": "baseline", "request": rom_text, "decoded": None, "decode_sim": None})
            if (pair, "oracle_decode", sid) not in done:
                rows.append({"arm": "oracle_decode", "request": nat_text, "decoded": nat_text, "decode_sim": 1.0})
            if (pair, "self_decode", sid) not in done:
                dec = generate([{"role": "user", "content": DECODE_TMPL.format(frm=spec["from"], to=spec["to"],
                                                                              text=rom_text)}], 128)
                rows.append({"arm": "self_decode", "request": dec, "decoded": dec,
                             "decode_sim": round(SequenceMatcher(None, dec, nat_text).ratio(), 3)})
            for r in rows:
                resp = generate([{"role": "user", "content": r["request"]}], max_new)
                fh.write(json.dumps({"model": model_key, "pair": pair, "arm": r["arm"], "source_id": sid,
                                     "prompt": r["request"], "decoded": r["decoded"],
                                     "decode_sim": r["decode_sim"], "response": resp}, ensure_ascii=False) + "\n")
                fh.flush()
            if (i + 1) % 10 == 0:
                print(f"  {model_key} {pair}: {i + 1}/{len(items)}", flush=True)
    fh.close()
    (OUT / model_key / "run_meta.json").write_text(json.dumps(
        {"model": model_key, "model_id": cfg.MODEL_ID, "n_per_pair": n, "max_new_tokens": max_new,
         "pairs": list(pairs), "decoding": "greedy", "completed_at": datetime.now().isoformat()}, indent=1))
    print(f"wrote {f}")


def judge(models, provider: str):
    """Label every saved response with the same four-way engagement rubric used elsewhere."""
    from scripts.phase5_sae_features.comprehension_validation._common import build_user_message, judge_anthropic_one
    sys.path.insert(0, str(ROOT / "scripts/phase5_sae_features"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("rj", ROOT / "scripts/phase5_sae_features/08_relevance_judge.py")
    rj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rj)
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        sys.exit("GEMINI_API_KEY not set")
    for m in models:
        src = OUT / m / "generations.jsonl"
        if not src.exists():
            continue
        lab_f = OUT / m / "labels_gemini.jsonl"
        done = {(r["pair"], r["arm"], r["source_id"]) for r in (json.loads(l) for l in open(lab_f))} if lab_f.exists() else set()
        with open(lab_f, "a") as out:
            for r in (json.loads(l) for l in open(src)):
                k = (r["pair"], r["arm"], r["source_id"])
                if k in done:
                    continue
                lab = rj.judge_one(r["prompt"], r["response"], "gemini-2.5-flash", key, provider="gemini")
                out.write(json.dumps({**{x: r[x] for x in ("model", "pair", "arm", "source_id")},
                                      "label": lab}, ensure_ascii=False) + "\n")
                out.flush()
        print(f"  judged {m}")


def summarise():
    rows = []
    for d in sorted(OUT.glob("*/labels_gemini.jsonl")):
        m = d.parent.name
        by = collections.defaultdict(collections.Counter)
        for r in (json.loads(l) for l in open(d)):
            by[(r["pair"], r["arm"])][r["label"].upper()] += 1
        sims = collections.defaultdict(list)
        for r in (json.loads(l) for l in open(d.parent / "generations.jsonl")):
            if r["arm"] == "self_decode" and r["decode_sim"] is not None:
                sims[r["pair"]].append(r["decode_sim"])
        for (pair, arm), c in sorted(by.items()):
            n = sum(c.values())
            rows.append({"model": m, "pair": pair, "arm": arm, "n": n,
                         "refusal_pct": round(100 * c["REFUSAL"] / n, 1),
                         "engaged_pct": round(100 * (c["REFUSAL"] + c["ON_TOPIC"]) / n, 1),
                         "on_topic_pct": round(100 * c["ON_TOPIC"] / n, 1),
                         "mean_decode_sim": round(sum(sims[pair]) / len(sims[pair]), 3) if arm == "self_decode" and sims[pair] else None})
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"{'model':22s} {'pair':7s} {'arm':14s} {'n':>4s} {'refusal%':>9s} {'engaged%':>9s} {'decode sim':>11s}")
    for r in rows:
        print(f"{r['model']:22s} {r['pair']:7s} {r['arm']:14s} {r['n']:4d} {r['refusal_pct']:9.1f} "
              f"{r['engaged_pct']:9.1f} {str(r['mean_decode_sim'] or ''):>11s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--pairs", nargs="*", default=list(PAIRS))
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.summarise:
        summarise(); return
    if a.judge:
        judge(a.models or [p.name for p in OUT.glob("*") if p.is_dir()], "gemini"); return
    if not a.model:
        ap.error("--model is required unless --judge or --summarise")
    run(a.model, a.n, a.max_new_tokens, a.pairs)


if __name__ == "__main__":
    main()
