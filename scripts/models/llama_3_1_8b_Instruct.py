"""Model config: meta-llama/Llama-3.1-8B-Instruct (legacy / exploratory).

NOT part of the OSF v4 confirmatory set. The 2026-05-25 diagnostic flagged
Llama-3.1-8B for cross-family generalisation concerns:
  - Devanagari coherence was the weakest of the candidates evaluated
    (2/15 in the original CLAUDE.md note; the 2026-05-28 multilingual
    rescore put DV refusals at 3/15 and IT refusals at 0/15, suggesting
    Tier C is partially usable but the parsing failure rate is high).
  - Arabizi coherence after rescore is 14/15 — comparable to the OSF v4
    confirmatory set, so Llama is fine for H1/H2 *exploratory* runs on
    Tier A.

Kept registered because:
  - Cross-architecture probe (Llama vs Qwen vs Gemma) is a natural ablation
    once the OSF v4 confirmatory set is collected; useful for the discussion
    section of the paper, not the headline result.
  - The translit_first / parsing-bottleneck classification done on the
    2026-05-28 replication check used Llama as a sibling baseline; this
    config makes those runs reproducible.
  - Public SAEs do exist for Llama-3.1-8B-Instruct (Goodfire's release at
    google/gemma-scope-... wait no, those are Gemma. Llama SAEs available
    via `Goodfire/Llama-3-8B-instruct-sparse-autoencoder` and EleutherAI
    releases — none of which are pinned for our project; leave SAE_REPO
    empty until a specific release is committed to.

Promoting to the v4 confirmatory set requires an OSF amendment. All
results from this config are exploratory until then.
"""
from __future__ import annotations

MODEL_KEY = "llama_3_1_8b_Instruct"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Architecture
N_LAYERS = 32
HIDDEN_DIM = 4096  # matches Qwen3-8B at 4096; useful for cross-family direction comparison

# Extraction
BATCH_SIZE = 4  # 8B model; matches Qwen3-8B's batch size
MEASUREMENT_POSITION = -1  # trailing newline of "<|start_header_id|>assistant<|end_header_id|>\n\n"

# Llama-3 chat template (manual fallback; scripts prefer tokenizer.apply_chat_template).
# The double-newline after the assistant header is the canonical position for the
# refusal-direction probe (analogous to Qwen's "\n" after "<|im_start|>assistant\n").
CHAT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{prompt}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

# No thinking mode (Llama has no equivalent of Qwen3's <think> blocks).
CHAT_TEMPLATE_KWARGS: dict[str, object] = {}

# System prompts used by Phase-4 diagnostics under --use-system-prompt.
# See qwen3_4b.py for rationale; defaults from models/__init__.py are used
# when None. NOT used during Phase-2 activation extraction.
SYSTEM_PROMPTS: dict[str, str] | None = None

# Public SAEs — none pinned for this project. Several third-party releases
# exist (Goodfire, EleutherAI, etc.) but none are committed to the v4
# methodology. Empty to make this explicit.
SAE_REPO = ""
SAE_HOOK_DEFAULT = ""
SAE_WIDTH_DEFAULT = ""

NOTES = (
    "LEGACY exploratory model (NOT in OSF v4 confirmatory set). 8B params, "
    "32 layers, 4096 hidden dim, Llama 3.1 Community License. Original "
    "2026-05-25 diagnostic flagged weak Devanagari coherence; the 2026-05-28 "
    "multilingual rescore showed Arabizi 14/15 coherent + 1 refusal and "
    "Tier C 3 DV refusals + 0 IT refusals — usable for Arabizi exploratory "
    "runs, marginal for ITRANS. Useful as a cross-architecture sibling to "
    "the Qwen and Gemma families when the OSF confirmatory set is complete. "
    "Promotion to v4 confirmatory requires an OSF amendment."
)
