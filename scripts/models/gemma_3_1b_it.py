"""Model config: google/gemma-3-1b-it (Phase-2 cross-family small + H3 SAE source)."""
from __future__ import annotations

MODEL_KEY = "gemma_3_1b_it"
MODEL_ID = "google/gemma-3-1b-it"

# Architecture
N_LAYERS = 26
HIDDEN_DIM = 1152

# Extraction
BATCH_SIZE = 16  # 1B model fits comfortably; bigger batch for throughput
MEASUREMENT_POSITION = -1

# Gemma chat template; scripts should still prefer tokenizer.apply_chat_template.
# The literal '{prompt}' placeholder is replaced by str.format() in the manual fallback.
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

# Public SAEs — Gemma Scope V3 covers all 26 layers, widths 16k/64k/256k/1m,
# at hook points resid_post / mlp_out / attn_out.
SAE_REPO = "google/gemma-scope-2-1b-it"
SAE_HOOK_DEFAULT = "resid_post"
SAE_WIDTH_DEFAULT = "64k"
SAE_AVAILABLE_WIDTHS = ("16k", "64k", "256k", "1m")
SAE_AVAILABLE_HOOKS = ("resid_post", "mlp_out", "attn_out")
# Cross-Layer Transcoder repo (compatible with circuit-tracer in the parent project)
CLT_REPO = "mwhanna/gemma-scope-2-1b-it"

NOTES = (
    "Cross-family small replication per OSF v4 AND primary H3 SAE source. "
    "1B parameters, 26 layers, 1152 hidden dim, Gemma Terms of Use. "
    "Gemma Scope V3 SAEs at google/gemma-scope-2-1b-it cover all layers with "
    "four widths (16k/64k/256k/1m) and three hook points (resid_post/mlp_out/attn_out). "
    "CLT at mwhanna/gemma-scope-2-1b-it is compatible with circuit-tracer."
)
