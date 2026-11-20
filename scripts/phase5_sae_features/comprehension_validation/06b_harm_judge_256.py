#!/usr/bin/env python3
"""Judge whether 256-token ON_TOPIC responses in decode_then_refuse contain specific harmful content.

Reuses the exact rubric, system prompt, and Claude Opus 4.8 judge of 06_harm_judge.py on the
256-token generations in results/phase5/decode_then_refuse/ across all three arms (baseline,
self_decode, oracle_decode).

Outputs:
  results/phase5/decode_then_refuse/<model>/labels_harm_claude.jsonl
  results/phase5/decode_then_refuse/harm_256_summary.json
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

_HARM_MOD_PATH = ROOT / "scripts/phase5_sae_features/comprehension_validation/06_harm_judge.py"
_spec = importlib.util.spec_from_file_location("_harm_judge", _HARM_MOD_PATH)
hj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hj)

DTR = ROOT / "results/phase5/decode_then_refuse"
MODELS = ["gemma_3_4b_it", "qwen3_4b", "llama_3_2_3b_Instruct", "qwen3_8b", "gemma_2_9b_it"]


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default=hj.DEFAULT_CLAUDE_JUDGE)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--provider", choices=("anthropic", "openrouter"), default="openrouter")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY" if a.provider == "openrouter" else "ANTHROPIC_API_KEY", "")
    call = hj.judge_openrouter if a.provider == "openrouter" else hj.judge

    todo = []
    for m in MODELS:
        mdir = DTR / m
        gem = {(r["pair"], r["arm"], r["source_id"]): r["label"].upper()
               for r in read_jsonl(mdir / "labels_gemini.jsonl")}
        done = {(r["pair"], r["arm"], r["source_id"])
                for r in read_jsonl(mdir / "labels_harm_claude.jsonl") if r.get("label") != "ERROR"}
        for g in read_jsonl(mdir / "generations.jsonl"):
            k = (g["pair"], g["arm"], g["source_id"])
            if gem.get(k) == "ON_TOPIC" and k not in done:
                todo.append((m, g))

    print(f"256-token ON_TOPIC items to judge: {len(todo)}")
    if a.dry_run:
        return
    if todo and not key:
        sys.exit(f"API key for provider {a.provider} not set")

    def work(item):
        m, g = item
        lab, raw = call(g["prompt"], g["response"], a.judge_model, key)
        return m, {
            "model": m,
            "pair": g["pair"],
            "arm": g["arm"],
            "source_id": g["source_id"],
            "judge_provider": a.provider,
            "judge_model": a.judge_model,
            "label": lab,
            "raw": raw,
        }

    if todo:
        n_done = 0
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for m, rec in ex.map(work, todo):
                with open(DTR / m / "labels_harm_claude.jsonl", "a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_done += 1
                if n_done % 20 == 0 or n_done == len(todo):
                    print(f"  judged {n_done}/{len(todo)} (last {m} {rec['pair']} {rec['arm']}: {rec['label']})",
                          flush=True)

    # Summarize per (model, pair, arm)
    summary = {}
    for m in MODELS:
        mdir = DTR / m
        gem = {(r["pair"], r["arm"], r["source_id"]): r["label"].upper()
               for r in read_jsonl(mdir / "labels_gemini.jsonl")}
        harm = {(r["pair"], r["arm"], r["source_id"]): r["label"]
                for r in read_jsonl(mdir / "labels_harm_claude.jsonl") if r.get("label") != "ERROR"}
        summary[m] = {}
        for pair in ("tier_a", "tier_c", "tier_e"):
            for arm in ("baseline", "self_decode", "oracle_decode"):
                keys = [k for k in gem if k[0] == pair and k[1] == arm]
                if not keys:
                    continue
                n = len(keys)
                c = collections.Counter()
                on_topic_count = 0
                for k in keys:
                    lab = gem[k]
                    if lab == "ON_TOPIC":
                        on_topic_count += 1
                        c[harm.get(k, "ON_TOPIC_UNJUDGED")] += 1
                    else:
                        c[lab] += 1
                summary[m][f"{pair}|{arm}"] = {
                    "n": n,
                    "on_topic_pct": round(100 * on_topic_count / n, 1),
                    "specific_pct": round(100 * c.get("SPECIFIC", 0) / n, 1),
                    "generic_pct": round(100 * c.get("GENERIC", 0) / n, 1),
                    "none_pct": round(100 * c.get("NONE", 0) / n, 1),
                    "refusal_pct": round(100 * c.get("REFUSAL", 0) / n, 1),
                    "off_topic_pct": round(100 * c.get("OFF_TOPIC", 0) / n, 1),
                    "gibberish_pct": round(100 * c.get("GIBBERISH", 0) / n, 1),
                    "counts": dict(c),
                }
    out_file = DTR / "harm_256_summary.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
