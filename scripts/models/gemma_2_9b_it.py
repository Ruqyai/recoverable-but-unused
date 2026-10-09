"""Model config: google/gemma-2-9b-it (Phase-2 cross-family large + H3 SAE source)."""
from __future__ import annotations

MODEL_KEY = "gemma_2_9b_it"
MODEL_ID = "google/gemma-2-9b-it"

# Architecture
N_LAYERS = 42
HIDDEN_DIM = 3584

# Extraction
BATCH_SIZE = 2  # 9B model; conservative batch for M2 32GB
MEASUREMENT_POSITION = -1

CHAT_TEMPLATE = (
    "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
)

# Extra kwargs passed into tokenizer.apply_chat_template(...).
# Gemma has no thinking mode; left empty for symmetry with the Qwen configs.
CHAT_TEMPLATE_KWARGS: dict[str, object] = {}

# System prompts used by Phase-4 diagnostics under --use-system-prompt.
# See qwen3_4b.py for rationale; defaults from models/__init__.py are used
# when None. NOT used during Phase-2 activation extraction.
SYSTEM_PROMPTS: dict[str, str] | None = None

# Public SAEs — original Gemma Scope release.
# Residual stream is the primary hook for refusal analysis; mlp / attn variants
# are separate repos and are used as H3 sensitivity checks.
SAE_REPO = "google/gemma-scope-9b-it-res"
SAE_HOOK_DEFAULT = "resid_post"
SAE_WIDTH_DEFAULT = "16k"  # primary width released for the 9B IT res variant
SAE_AVAILABLE_HOOKS = ("resid_post", "mlp_out", "attn_out")
SAE_REPO_MLP = "google/gemma-scope-9b-it-mlp"
SAE_REPO_ATT = "google/gemma-scope-9b-it-att"

NOTES = (
    "Cross-family large replication per OSF v4 AND secondary H3 SAE source. "
    "9B parameters, 42 layers, 3584 hidden dim, Gemma Terms of Use. "
    "Hidden dim 3584 matches Qwen2.5-7B (NOT Qwen3-4B, which is 2560), enabling "
    "direct cross-family direction comparison with the Qwen2.5-7B exploratory "
    "model. Gemma Scope SAEs split across three repos: "
    "google/gemma-scope-9b-it-{res,mlp,att}."
)
