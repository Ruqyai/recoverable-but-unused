#!/usr/bin/env python3
"""Stage 01 — per-item comprehension judge (Gemini OR Claude), one label per row.

Phase 5 Stage 08 wrote only per-condition COUNTS; an agreement (kappa) analysis
needs per-ITEM labels so the two LLM judges and the human annotators can be
merged row-by-row. This stage re-runs the *identical* Stage 08 classification
task (same system prompt, same 4 labels, same user template — imported, not
copied) over a model's ``generations.jsonl`` and writes:

    results/phase5/comprehension_validation/<model>/items_<judge>.jsonl
        {model, id, condition, judge_provider, judge_model, label}

Two providers:
  --provider gemini     direct Google AI Studio (free tier, gemini-2.5-flash).
                        Needs GEMINI_API_KEY in the ENVIRONMENT (not auto-loaded).
  --provider anthropic  Claude (default claude-opus-4-8) — the SECOND judge.
                        Needs ANTHROPIC_API_KEY in the ENVIRONMENT.

Reproducing the existing Stage 08 numbers: run --provider gemini and the
per-condition comprehension% recomputed here will match relevance_judged.json.

Usage
-----
  set -a; source .env; set +a   # export GEMINI_API_KEY / ANTHROPIC_API_KEY
  # Gemini, all models with generations:
  PYTHONPATH=. python3 scripts/phase5_sae_features/comprehension_validation/01_judge_per_item.py --provider gemini
  # Claude (second judge), one model:
  PYTHONPATH=. python3 scripts/phase5_sae_features/comprehension_validation/01_judge_per_item.py \
      --provider anthropic --models gemma_2_9b_it
  # Smoke test without spending much (2 rows per condition):
  PYTHONPATH=. python3 scripts/phase5_sae_features/comprehension_validation/01_judge_per_item.py \
      --provider gemini --models gemma_3_1b_it --limit 2

Resumable: rows already present in items_<judge>.jsonl for a model are skipped,
so an interrupted run continues where it stopped (use --overwrite to redo).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.phase5_sae_features.comprehension_validation import _common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_GEMINI_JUDGE = "gemini-2.5-flash"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Per-item comprehension judge (Gemini or Claude).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--provider", choices=["gemini", "anthropic"], required=True)
    ap.add_argument(
        "--models", nargs="*", default=None,
        help="Model keys (default: all with a Phase 5 generations.jsonl).",
    )
    ap.add_argument(
        "--judge-model", default=None,
        help="Override judge model id (default: gemini-2.5-flash / claude-opus-4-8).",
    )
    ap.add_argument(
        "--conditions", nargs="*", default=None,
        help="Restrict to these conditions (default: all 6).",
    )
    ap.add_argument("--limit", type=int, default=None, help="Cap rows per condition (smoke test).")
    ap.add_argument("--sleep", type=float, default=0.5, help="Seconds between calls (RPM throttle).")
    ap.add_argument("--overwrite", action="store_true", help="Re-judge rows already labeled.")
    return ap.parse_args()


def judge_tag(provider: str, judge_model: str) -> str:
    """File tag: 'gemini' or 'claude' (so items_gemini.jsonl / items_claude.jsonl)."""
    if provider == "gemini":
        return "gemini"
    return "claude"


def main() -> int:
    args = parse_args()
    provider = args.provider
    judge_model = args.judge_model or (
        DEFAULT_GEMINI_JUDGE if provider == "gemini" else C.DEFAULT_CLAUDE_JUDGE
    )
    key_name = "GEMINI_API_KEY" if provider == "gemini" else "ANTHROPIC_API_KEY"
    api_key = os.environ.get(key_name)
    if not api_key:
        log.error("%s missing from environment. `set -a; source .env; set +a` first.", key_name)
        return 2

    models = args.models or C.models_with_generations()
    if not models:
        log.error("No models with a Phase 5 generations.jsonl found under %s", C.PHASE5_DIR)
        return 1
    tag = judge_tag(provider, judge_model)
    log.info("Judge=%s (%s via %s) over models: %s", tag, judge_model, provider, ", ".join(models))

    for model_key in models:
        try:
            rows = C.load_generations(model_key)
        except FileNotFoundError as e:
            log.warning("skip %s: %s", model_key, e)
            continue
        if args.conditions:
            rows = [r for r in rows if r["condition"] in set(args.conditions)]
        if args.limit:
            seen: dict[str, int] = {}
            capped = []
            for r in rows:
                c = r["condition"]
                if seen.get(c, 0) >= args.limit:
                    continue
                seen[c] = seen.get(c, 0) + 1
                capped.append(r)
            rows = capped

        out_path = C.items_path(model_key, tag)
        done: set[str] = set()
        if out_path.exists() and not args.overwrite:
            done = {r["id"] for r in C.read_items(out_path)}
        mode = "w" if args.overwrite else "a"

        judged = 0
        with out_path.open(mode) as fh:
            for i, r in enumerate(rows):
                if r["id"] in done:
                    continue
                if provider == "gemini":
                    label = C.judge_gemini_one(
                        r["prompt"], r["response"], judge_model, api_key, provider="gemini"
                    )
                else:
                    label = C.judge_anthropic_one(
                        r["prompt"], r["response"], judge_model, api_key
                    )
                fh.write(json.dumps({
                    "model": model_key,
                    "id": r["id"],
                    "condition": r["condition"],
                    "judge_provider": provider,
                    "judge_model": judge_model,
                    "label": label,
                }, ensure_ascii=False) + "\n")
                fh.flush()
                judged += 1
                if args.sleep:
                    time.sleep(args.sleep)
                if judged % 25 == 0:
                    log.info("  %s: judged %d (last=%s)", model_key, judged, label)
        log.info("%s → %s  (+%d rows, %d already present)", model_key, out_path, judged, len(done))

    return 0


if __name__ == "__main__":
    sys.exit(main())
