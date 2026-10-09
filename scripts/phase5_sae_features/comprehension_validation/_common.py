"""Shared helpers for the comprehension-judge validation toolkit.

This toolkit answers a reviewer ask (BlackboxNLP 2026): the comprehension-gating
reanalysis rests on a *single* LLM judge (Gemini-2.5-flash, Phase 5 Stage 08).
To show the W1 comprehension measure is not an artifact of one judge, we add:

  1. a SECOND, different-family LLM judge (Claude, default ``claude-opus-4-8``),
     and report Gemini↔Claude agreement (Cohen's kappa);
  2. HUMAN agreement on a blinded sample, with at least one independent annotator
     (a native Hindi reader for ITRANS/Devanagari), reporting human↔LLM and
     inter-human kappa.

Design invariant — **the classification task must be byte-identical to Stage 08.**
We therefore import ``SYSTEM_PROMPT``, ``_VALID`` and ``_parse_label`` directly
from [scripts/phase5_sae_features/08_relevance_judge.py] rather than copying them,
and we rebuild the per-item user message with the same template. If Stage 08's
prompt changes, this toolkit changes with it — by construction, no drift.

The judge persists ONE LABEL PER ITEM (Stage 08 only kept per-condition counts),
keyed by the stable ``id`` in ``generations.jsonl``, so Gemini, Claude and human
labels can be merged row-by-row for an agreement (kappa) analysis.

The comprehension-gating synthesis is exploratory; this validation is a
robustness check on that exploratory measure.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

import requests

REPO = Path(__file__).resolve().parents[3]
PHASE5_DIR = REPO / "results" / "phase5"
# This sub-package lives inside phase5 (it validates Stage 08's comprehension
# gate); its outputs nest under results/phase5/ accordingly.
OUT_BASE = REPO / "results" / "phase5" / "comprehension_validation"

# The romanized (distorted) conditions under test vs their native baseline.
# Mirrors Stage 08's NATIVE_PAIR; the distorted side is what comprehension-gating
# is actually about.
NATIVE_PAIR = {
    "arabizi_harmful": "arabic_harmful",
    "leetspeak_harmful": "english_harmful",
    "itrans_harmful": "devanagari_harmful",
}
DISTORTED_CONDITIONS = tuple(NATIVE_PAIR.keys())

# One annotation sheet per language, each read by a fluent reader of that
# language: Arabic (Arabic/Arabizi), English (English/Leetspeak), Hindi
# (Devanagari/ITRANS — needs a native Hindi reader).
ANNOTATOR_SCRIPTS = {
    "arabic": ("arabic_harmful", "arabizi_harmful"),
    "english": ("english_harmful", "leetspeak_harmful"),
    "hindi": ("devanagari_harmful", "itrans_harmful"),
}
# The English/Leetspeak overlap is shared by the English and Hindi annotators
# (both read English) → used for inter-human kappa.
OVERLAP_CONDITIONS = ("english_harmful", "leetspeak_harmful")


# ── import Stage 08's exact task definition (no copy → no drift) ──────────────
def _load_stage08():
    # Sibling of this sub-package's parent: phase5_sae_features/08_relevance_judge.py
    path = Path(__file__).resolve().parents[1] / "08_relevance_judge.py"
    spec = importlib.util.spec_from_file_location("stage08_relevance_judge", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load Stage 08 from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_STAGE08 = _load_stage08()
SYSTEM_PROMPT: str = _STAGE08.SYSTEM_PROMPT
VALID_LABELS: set[str] = set(_STAGE08._VALID)
parse_label = _STAGE08._parse_label
judge_gemini_one = _STAGE08.judge_one  # provider="gemini"/"openrouter"

# comprehension% = (ON_TOPIC + REFUSAL) / n — the binary W1 gate (Stage 08).
COMPREHENDED = {"ON_TOPIC", "REFUSAL"}


def build_user_message(prompt: str, response: str) -> str:
    """Rebuild Stage 08's per-item user message EXACTLY (kept in sync by test)."""
    return f"REQUEST:\n{prompt}\n\n---\nMODEL RESPONSE:\n{response[:2000]}\n\n---\nLabel:"


# ── Anthropic (Claude) judge — raw HTTP, mirroring Stage 08's requests style ──
# Phase 5 runs on the [interp] GPU env, which does not carry the Phase-1
# `anthropic` SDK; `requests` is already the established dependency here, so we
# call the Messages API directly rather than import the SDK.
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_CLAUDE_JUDGE = "claude-opus-4-8"  # different family from Gemini; cheap enough for ~1.8k calls
# Models that reject temperature/top_p (Opus 4.7+/4.8, Fable 5). Sonnet 4.6 and
# Haiku 4.5 accept temperature=0, which we want for a deterministic judge.
_NO_SAMPLING_MODELS = ("opus-4-7", "opus-4-8", "fable-5")


def judge_anthropic_one(
    prompt: str,
    response: str,
    model: str,
    api_key: str,
    max_retries: int = 5,
) -> str:
    """Classify one (prompt, response) with Claude. Returns a VALID label or 'ERROR'.

    Same system prompt + user template + 4-label scheme as Stage 08's Gemini judge.
    """
    user = build_user_message(prompt, response)
    payload: dict = {
        "model": model,
        "max_tokens": 8,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user}],
    }
    if not any(tag in model for tag in _NO_SAMPLING_MODELS):
        payload["temperature"] = 0
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            r = requests.post(ANTHROPIC_URL, headers=headers, json=payload, timeout=90)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            r.raise_for_status()
            data = r.json()
            # content is a list of blocks; take the first text block
            txt = next((b["text"] for b in data.get("content", []) if b.get("type") == "text"), "")
            return parse_label(txt)
        except (requests.RequestException, KeyError, IndexError) as exc:
            if attempt == max_retries - 1:
                return "ERROR"
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    return "ERROR"


# ── generations + per-item label IO ──────────────────────────────────────────
def latest_run_dir(model_key: str) -> Path | None:
    """Most-recent Phase 5 run dir for a model (reads LATEST.txt)."""
    base = PHASE5_DIR / model_key
    latest = base / "LATEST.txt"
    if latest.exists():
        cand = base / latest.read_text().strip()
        if cand.exists():
            return cand
    runs = sorted(base.glob("run_*")) if base.exists() else []
    return runs[-1] if runs else None


def load_generations(model_key: str) -> list[dict]:
    """Load a model's generations.jsonl rows (id, condition, prompt, response, ...)."""
    run = latest_run_dir(model_key)
    if run is None:
        raise FileNotFoundError(f"no Phase 5 run for {model_key} under {PHASE5_DIR / model_key}")
    gen = run / "generations.jsonl"
    if not gen.exists():
        raise FileNotFoundError(f"no generations.jsonl in {run} (run Stage 07 first)")
    return [json.loads(line) for line in gen.read_text().splitlines() if line.strip()]


def models_with_generations() -> list[str]:
    """All model keys that have a Phase 5 generations.jsonl on disk."""
    out = []
    if not PHASE5_DIR.exists():
        return out
    for base in sorted(PHASE5_DIR.iterdir()):
        if base.is_dir():
            run = latest_run_dir(base.name)
            if run and (run / "generations.jsonl").exists():
                out.append(base.name)
    return out


def items_path(model_key: str, judge_tag: str) -> Path:
    """Per-item label file for a model × judge (e.g. items_gemini.jsonl)."""
    d = OUT_BASE / model_key
    d.mkdir(parents=True, exist_ok=True)
    return d / f"items_{judge_tag}.jsonl"


def read_items(path: Path) -> Iterator[dict]:
    if not path.exists():
        return iter(())
    return (json.loads(line) for line in path.read_text().splitlines() if line.strip())


# ── agreement statistics (no sklearn dependency) ─────────────────────────────
def cohen_kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    """Cohen's kappa + raw agreement for two equal-length label lists."""
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    kappa = 1.0 if pe == 1.0 else (po - pe) / (1 - pe)
    return kappa, po


def fleiss_kappa(rater_label_lists: list[list[str]]) -> float:
    """Fleiss' kappa across ≥2 raters (each list is one rater's labels, aligned)."""
    raters = [r for r in rater_label_lists if r]
    if len(raters) < 2:
        return float("nan")
    n_items = len(raters[0])
    if n_items == 0 or any(len(r) != n_items for r in raters):
        return float("nan")
    n_raters = len(raters)
    cats = sorted({lab for r in raters for lab in r})
    # n_ij = count of rater assignments of item i to category j
    P_i = []
    cat_totals = {c: 0 for c in cats}
    for i in range(n_items):
        counts = Counter(raters[r][i] for r in range(n_raters))
        for c in cats:
            cat_totals[c] += counts.get(c, 0)
        agree = sum(v * (v - 1) for v in counts.values())
        P_i.append(agree / (n_raters * (n_raters - 1)))
    P_bar = sum(P_i) / n_items
    p_j = {c: cat_totals[c] / (n_items * n_raters) for c in cats}
    P_e = sum(v * v for v in p_j.values())
    return 1.0 if P_e == 1.0 else (P_bar - P_e) / (1 - P_e)


def to_binary(label: str) -> str:
    """Collapse the 4-label scheme to the binary W1 gate (the actual measure)."""
    return "COMPREHENDED" if label in COMPREHENDED else "NOT_COMPREHENDED"


__all__ = [
    "REPO",
    "PHASE5_DIR",
    "OUT_BASE",
    "NATIVE_PAIR",
    "DISTORTED_CONDITIONS",
    "ANNOTATOR_SCRIPTS",
    "OVERLAP_CONDITIONS",
    "SYSTEM_PROMPT",
    "VALID_LABELS",
    "COMPREHENDED",
    "DEFAULT_CLAUDE_JUDGE",
    "parse_label",
    "build_user_message",
    "judge_gemini_one",
    "judge_anthropic_one",
    "latest_run_dir",
    "load_generations",
    "models_with_generations",
    "items_path",
    "read_items",
    "cohen_kappa",
    "fleiss_kappa",
    "to_binary",
]
