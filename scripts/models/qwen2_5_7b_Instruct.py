"""Model config: Qwen/Qwen2.5-7B-Instruct (legacy / exploratory).

NOT part of the OSF v4 confirmatory set. This was the pre-2026-05-22 primary
model — rejected after the diagnostic showed 15/15 Arabizi gibberish (see
CLAUDE.md "REJECTED" list; rejected 2026-05-22). Kept here
because:

  - It has public SAEs (andyrdt/saes-qwen2.5-7b-instruct, 7 layers, 131k
    features) which Qwen3-4B and Qwen3-8B lack, useful for ad-hoc H3 probing.
  - Wang et al. (NeurIPS 2025) ran their Arabic-silhouette analysis on this
    exact model; comparing v4 results back to Qwen2.5-7B at the same
    hidden_dim (3584) lets us anchor cross-paper claims.
  - The archived activations at
    results/phase2/archive_pre_refactor_20260525_qwen2_5_7b_Instruct/ were
    extracted from this model; reproducing them requires this registry entry.

Adding this model to the v4 *confirmatory* set requires an OSF amendment.
Until then, treat all results from this config as exploratory only.
"""
from __future__ import annotations

MODEL_KEY = "qwen2_5_7b_Instruct"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Architecture
N_LAYERS = 28
HIDDEN_DIM = 3584  # matches Gemma-2-9b-it (3584) — NOT Qwen3-4B (which is 2560) — for cross-model direction comparison

# Extraction
BATCH_SIZE = 4  # 7B model; matches Qwen3-8B's batch size
MEASUREMENT_POSITION = -1  # trailing newline of "<|im_start|>assistant\n"

CHAT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# Qwen2.5 has no thinking mode (that's Qwen3 only) — no special kwargs needed.
CHAT_TEMPLATE_KWARGS: dict[str, object] = {}

# System prompts used by Phase-4 diagnostics under --use-system-prompt.
# See qwen3_4b.py for rationale; defaults from models/__init__.py are used
# when None. NOT used during Phase-2 activation extraction.
SYSTEM_PROMPTS: dict[str, str] | None = None

# Public SAEs — Arditi's Qwen2.5-7B-Instruct SAEs, Apache-2.0.
# Seven checkpoints at layers 3, 7, 11, 15, 19, 23, 27 (every 4 layers).
# 131,072 features per layer; auto-interp labels on Neuronpedia release
# "misaligned-persona". Hook: blocks.{N}.hook_resid_post.
SAE_REPO = "andyrdt/saes-qwen2.5-7b-instruct"
SAE_HOOK_DEFAULT = "resid_post"
SAE_WIDTH_DEFAULT = "131k"
SAE_AVAILABLE_LAYERS = (3, 7, 11, 15, 19, 23, 27)

NOTES = (
    "LEGACY exploratory model (NOT in OSF v4 confirmatory set). 7.61B params, "
    "28 layers, 3584 hidden dim, Apache-2.0. Original Phase-2 primary "
    "(pre-2026-05-22) — rejected for v4 because the baseline diagnostic showed "
    "15/15 Arabizi gibberish. Retained for cross-paper anchoring against Wang "
    "et al. NeurIPS 2025 (same model) and for SAE-instrumented probing where "
    "the Qwen3 models lack public SAEs. Any v4-relevant result on this model "
    "requires an OSF amendment before it can move from exploratory to "
    "confirmatory."
)
