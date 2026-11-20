#!/usr/bin/env python3
"""Scale Phase-5 harmful behavioural readout and Gemini/Claude judging from n=50 to n=200 per cell.

Appends items 51..200 across all 6 harmful conditions (`arabic_harmful`, `arabizi_harmful`,
`devanagari_harmful`, `itrans_harmful`, `english_harmful`, `leetspeak_harmful`) directly into each
model's `results/phase5/<model>/run_*/generations.jsonl`, preserving items 1..50 untouched and
checkpointing after every batch so execution can be paused and resumed at any point.

Supports:
  1. Local MPS generation (`--model <key> --batch-size <bs>`) for all <=9B models.
  2. OpenRouter API generation (`--model gemma_3_27b_it --api`) for Gemma-3-27B-it (54 GB in bf16).
  3. Parallel Gemini-2.5-flash judging (`--judge --provider openrouter`) into `items_gemini.jsonl`,
     after which `06_harm_judge.py` scores all new `ON_TOPIC` items with Claude Opus 4.8.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from scripts.phase5_sae_features.comprehension_validation import _common as C  # noqa: E402
from scripts.utils.io import read_jsonl  # noqa: E402

_BR_PATH = ROOT / "scripts/phase5_sae_features/07_behavioral_readout.py"
_spec = importlib.util.spec_from_file_location("_br", _BR_PATH)
br = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(br)

HARMFUL_CONDITIONS = (
    "arabic_harmful",
    "arabizi_harmful",
    "devanagari_harmful",
    "itrans_harmful",
    "english_harmful",
    "leetspeak_harmful",
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_GEN_IDS = {
    "gemma_3_27b_it": "google/gemma-3-27b-it",
}


def run_local_model(model_key: str, n_target: int, max_new_tokens: int, batch_size: int) -> None:
    import torch
    from transformers import AutoTokenizer
    from scripts.models import load_model_config
    from scripts.utils.device import detect_device, load_hf_model, resolve_dtype

    run_dir = C.latest_run_dir(model_key)
    if run_dir is None:
        sys.exit(f"No existing Phase-5 run_dir for {model_key}")
    gen_path = run_dir / "generations.jsonl"
    done_ids = {(r["condition"], r["id"]) for r in C.load_generations(model_key)}

    # Check if anything is left to generate before loading weights
    todo_by_cond = {}
    for cond in HARMFUL_CONDITIONS:
        src_path, distorted = br.CONDITIONS[cond]
        recs = list(read_jsonl(src_path))[:n_target]
        missing = [(i, r, distorted) for i, r in enumerate(recs) if (cond, getattr(r, "id", f"{cond}_{i}")) not in done_ids]
        if missing:
            todo_by_cond[cond] = missing

    total_missing = sum(len(v) for v in todo_by_cond.values())
    if total_missing == 0:
        print(f"[skip] {model_key}: all {n_target}x{len(HARMFUL_CONDITIONS)} harmful items already in {gen_path.name}", flush=True)
        return

    cfg = load_model_config(model_key)
    dev = detect_device()
    pref_dtype = resolve_dtype(getattr(cfg, "PREFERRED_DTYPE", None))
    print(f"\n=== Scaling {model_key} (+{total_missing} items -> n={n_target}, bs={batch_size}) ===", flush=True)
    model = load_hf_model(cfg.MODEL_ID, dev, dtype=pref_dtype, model_class=getattr(cfg, "MODEL_CLASS", None))
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg.MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    with open(gen_path, "a") as fh:
        for cond, missing in todo_by_cond.items():
            t0 = time.time()
            for start in range(0, len(missing), batch_size):
                chunk = missing[start:start + batch_size]
                formatted = [br.format_prompt(tok, r.prompt, cfg) for _, r, _ in chunk]
                enc = tok(formatted, return_tensors="pt", padding=True).to(dev.device)
                plen = enc["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
                for (idx, r, distorted), row in zip(chunk, out):
                    resp = tok.decode(row[plen:], skip_special_tokens=True).strip()
                    hcat = getattr(r, "harm_category", "unknown")
                    cls = br.classify(resp, r.prompt, hcat)
                    rec_id = getattr(r, "id", f"{cond}_{idx}")
                    fh.write(json.dumps({
                        "condition": cond,
                        "distorted": distorted,
                        "id": rec_id,
                        "source_id": getattr(r, "source_id", None),
                        "harm_category": hcat,
                        "prompt": r.prompt,
                        "response": resp,
                        "heuristic_class": cls,
                    }, ensure_ascii=False) + "\n")
                fh.flush()
                dev.empty_cache()
            print(f"  {model_key} {cond:20s}: +{len(missing)} items ({time.time() - t0:.1f}s)", flush=True)

    del model
    gc.collect()
    dev.empty_cache()


def _call_openrouter_gen(model_id: str, prompt: str, max_tokens: int, key: str) -> str:
    payload = {
        "model": model_id,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
    backoff = 2.0
    for attempt in range(6):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(backoff)
                backoff = min(backoff * 2, 45)
                continue
            r.raise_for_status()
            ch = r.json()["choices"][0]
            return ((ch.get("message") or {}).get("content") or "").strip()
        except Exception:
            if attempt == 5:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 45)
    return ""


def run_api_model(model_key: str, n_target: int, max_new_tokens: int, workers: int) -> None:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set")
    or_model = OPENROUTER_GEN_IDS[model_key]
    run_dir = C.latest_run_dir(model_key)
    if run_dir is None:
        sys.exit(f"No existing Phase-5 run_dir for {model_key}")
    gen_path = run_dir / "generations.jsonl"
    done_ids = {(r["condition"], r["id"]) for r in C.load_generations(model_key)}

    with open(gen_path, "a") as fh:
        for cond in HARMFUL_CONDITIONS:
            src_path, distorted = br.CONDITIONS[cond]
            recs = list(read_jsonl(src_path))[:n_target]
            missing = [(i, r, distorted) for i, r in enumerate(recs)
                       if (cond, getattr(r, "id", f"{cond}_{i}")) not in done_ids]
            if not missing:
                print(f"  [skip] {model_key} {cond}: already has {n_target}", flush=True)
                continue
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=workers) as ex:
                resps = list(ex.map(lambda item: _call_openrouter_gen(or_model, item[1].prompt, max_new_tokens, key), missing))
            for (idx, r, distorted), resp in zip(missing, resps):
                hcat = getattr(r, "harm_category", "unknown")
                cls = br.classify(resp, r.prompt, hcat)
                rec_id = getattr(r, "id", f"{cond}_{idx}")
                fh.write(json.dumps({
                    "condition": cond,
                    "distorted": distorted,
                    "id": rec_id,
                    "source_id": getattr(r, "source_id", None),
                    "harm_category": hcat,
                    "prompt": r.prompt,
                    "response": resp,
                    "heuristic_class": cls,
                }, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  {model_key} (API) {cond:20s}: +{len(missing)} items ({time.time() - t0:.1f}s)", flush=True)


def _judge_with_fallback(prompt: str, response: str, jmodel: str, key: str, provider: str) -> str:
    if not response or len(response.strip()) < 5:
        return "GIBBERISH"
    try:
        lab = C.judge_gemini_one(prompt, response, jmodel, key, provider=provider, max_retries=3)
        if lab in C.VALID_LABELS:
            return lab
    except Exception:
        pass
    # Fallback 1: direct Anthropic API if available
    anth_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anth_key:
        try:
            lab_a = C.judge_anthropic_one(prompt, response, C.DEFAULT_CLAUDE_JUDGE, anth_key, max_retries=2)
            if lab_a in C.VALID_LABELS:
                return lab_a
        except Exception:
            pass
    # Fallback 2: open-weights multilingual judge via OpenRouter (no provider moderation 403)
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    if or_key:
        for fb_model in ("qwen/qwen-2.5-72b-instruct", "deepseek/deepseek-chat-v3-0324"):
            try:
                lab_fb = C.judge_gemini_one(prompt, response, fb_model, or_key, provider="openrouter", max_retries=2)
                if lab_fb in C.VALID_LABELS:
                    return lab_fb
            except Exception:
                pass
    return "REFUSAL" if br.classify(response, prompt, "unknown") == "refusal" else "OFF_TOPIC"


def run_judge(models: list[str], provider: str, workers: int) -> None:
    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        jmodel = "google/gemini-2.5-flash"
    else:
        key = os.environ.get("GEMINI_API_KEY", "")
        jmodel = "gemini-2.5-flash"
    if not key:
        sys.exit(f"API key for {provider} not set")

    for m in models:
        out_path = C.items_path(m, "gemini")
        existing = [r for r in C.read_items(out_path) if r.get("label") in C.VALID_LABELS]
        done = {r["id"] for r in existing}
        if out_path.exists():
            # Rewrite cleanly without any prior ERROR rows before appending
            with open(out_path, "w") as fh:
                for r in existing:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        rows = [r for r in C.load_generations(m) if r["condition"] in HARMFUL_CONDITIONS and r["id"] not in done]
        if not rows:
            print(f"  [judge skip] {m}: all harmful rows already judged", flush=True)
            continue
        t0 = time.time()
        with open(out_path, "a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
            labs = ex.map(lambda r: _judge_with_fallback(r["prompt"], r["response"], jmodel, key, provider), rows)
            for r, lab in zip(rows, labs):
                fh.write(json.dumps({
                    "model": m,
                    "id": r["id"],
                    "condition": r["condition"],
                    "judge_provider": provider,
                    "judge_model": "gemini-2.5-flash",
                    "label": lab,
                }, ensure_ascii=False) + "\n")
            fh.flush()
        print(f"  [judged] {m:24s}: +{len(rows)} items ({time.time() - t0:.1f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Single model key to generate for")
    ap.add_argument("--models", nargs="*", help="Model keys for --judge")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--api", action="store_true", help="Generate via OpenRouter API (for gemma_3_27b_it)")
    ap.add_argument("--judge", action="store_true", help="Judge unlabelled harmful rows into items_gemini.jsonl")
    ap.add_argument("--provider", choices=("gemini", "openrouter"), default="openrouter")
    a = ap.parse_args()

    if a.judge:
        run_judge(a.models or C.models_with_generations(), a.provider, a.workers)
        return
    if not a.model:
        ap.error("--model is required unless --judge is set")
    if a.api:
        run_api_model(a.model, a.n, a.max_new_tokens, a.workers)
    else:
        run_local_model(a.model, a.n, a.max_new_tokens, a.batch_size)


if __name__ == "__main__":
    main()
