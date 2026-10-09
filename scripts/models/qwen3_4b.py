"""Model config: Qwen/Qwen3-4B (Phase-2 PRIMARY model per OSF v4)."""
from __future__ import annotations

MODEL_KEY = "qwen3_4b"
MODEL_ID = "Qwen/Qwen3-4B"

# Architecture (verified against HF config.json: num_hidden_layers=36, hidden_size=2560)
N_LAYERS = 36
HIDDEN_DIM = 2560

# Extraction
BATCH_SIZE = 8
MEASUREMENT_POSITION = -1  # last token of "<|im_start|>assistant\n" template

# Chat template (manual fallback; scripts prefer tokenizer.apply_chat_template)
CHAT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# Extra kwargs passed into tokenizer.apply_chat_template(...).
# Qwen3 enables <think>...</think> reasoning blocks by default; with finite
# max_new_tokens the model exhausts the budget inside the thinking trace and
# never reaches its actual answer. Disabling it makes the model answer directly.
# See the 2026-05-25 (PM, 15:26) [BUGFIX].
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# System prompts used by Phase-4 diagnostics under --use-system-prompt.
# Inherits from DEFAULT_SYSTEM_PROMPTS in models/__init__.py when None.
# Override here (e.g. {"arabizi": "..."}) to customise for this model.
# NOT used during Phase-2 activation extraction — those runs must stay
# template-only to keep the extracted directions comparable to the
# unprimed-baseline literature (Wang 2025, Arditi 2024).
SYSTEM_PROMPTS: dict[str, str] | None = None

# Public SAEs — none available for Qwen3-4B at v4 deposit time.
# H3 (SAE Jaccard) is therefore not testable on this model; it is tested on
# the two Gemma models. See OSF v4 limitation #5.
SAE_REPO = ""
SAE_HOOK_DEFAULT = ""
SAE_WIDTH_DEFAULT = ""

NOTES = (
    "Primary analysis model per OSF v4. 4B parameters, 36 transformer layers, "
    "2560 hidden dim, Apache-2.0. NOTE (2026-05-30 [BUGFIX]): hidden dim is 2560, "
    "NOT 3584 — it does NOT match Qwen2.5-7B (3584). The earlier claim that the "
    "shared 3584 dim enabled an H5 silhouette comparison was based on a wrong "
    "architecture constant; silhouette is computed within each model's own space "
    "and does not require matching hidden dims, so H5 is unaffected, but any "
    "cross-model reasoning that assumed dimensional parity must be re-checked. "
    "No public SAEs available, so H3 is delegated to the two Gemma models."
)
