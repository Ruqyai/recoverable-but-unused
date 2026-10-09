"""Model config: google/gemma-3-4b-it (OPTIONAL exploratory — OSF v8, MULTIMODAL).

NOT part of the v6 six-model grid. Added to the OSF v8 optional-exploratory set
(the deposited pre-registration): "run only after deposit; do NOT enter
any confirmation rule". Role: a **within-generation scale pair** with
gemma_3_1b_it — same Gemma-3 generation, 1B vs 4B — so the 1B↔4B contrast
isolates size from generation (the complement to gemma_2_2b_it's cross-size,
same-Gemma-2 pairing).

⚠️ MULTIMODAL. google/gemma-3-4b-it loads as `Gemma3ForConditionalGeneration`,
NOT `AutoModelForCausalLM`, and the decoder layers live under
`model.model.language_model.layers`, NOT `model.model.layers`.

SUPPORTED as of 2026-06-21: the text-only Phase 2/4/5 path now branches on the
hints below. `device.load_hf_model(..., model_class=cfg.MODEL_CLASS)` loads the
correct VLM class, and every hidden-state / hook site resolves the text decoder
via `get_text_base_model(..., submodule_path=cfg.TEXT_SUBMODULE_PATH)` →
`...language_model.layers`. The three hints (IS_MULTIMODAL / MODEL_CLASS /
LAYERS_PATH + TEXT_SUBMODULE_PATH) drive that branching; text-only configs leave
them unset and keep the original `model.model.layers` path unchanged.

H3 eligibility: SAEs should be available via Gemma Scope 2 (covers the Gemma-3
family); the repo names below NEED VERIFICATION against HF before any H3 run.

Adding this model to a confirmatory rule requires an OSF amendment.
"""
from __future__ import annotations

MODEL_KEY = "gemma_3_4b_it"
MODEL_ID = "google/gemma-3-4b-it"

# Architecture (verified via AutoConfig 2026-06-09: Gemma3ForConditionalGeneration,
# text_config.num_hidden_layers=34, text_config.hidden_size=2560).
N_LAYERS = 34
HIDDEN_DIM = 2560

# Extraction
BATCH_SIZE = 8  # 4B language submodule; mirrors qwen3_4b
MEASUREMENT_POSITION = -1

# Gemma chat template; scripts should still prefer tokenizer.apply_chat_template.
CHAT_TEMPLATE = (
    "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
)

# Gemma has no thinking mode; left empty for symmetry with the Qwen configs.
CHAT_TEMPLATE_KWARGS: dict[str, object] = {}

# System prompts used by Phase-4 diagnostics under --use-system-prompt.
# See qwen3_4b.py for rationale; defaults from models/__init__.py are used
# when None. NOT used during Phase-2 activation extraction.
SYSTEM_PROMPTS: dict[str, str] | None = None

# --- Multimodal hints (additive; absent on the text-only configs). ---
# Read downstream with getattr(cfg, "IS_MULTIMODAL", False) etc., so adding
# these does not affect any existing single-modal model.
IS_MULTIMODAL = True
MODEL_CLASS = "Gemma3ForConditionalGeneration"   # AutoModelForCausalLM maps to this
# Text decoder layers, relative to the FULL model object (not via base_model).
LAYERS_PATH = "model.language_model.layers"
# Extract from the language submodule (text path of the VLM):
TEXT_SUBMODULE_PATH = "model.language_model"      # full_model.model.language_model = Gemma3TextModel

# REQUIRED: Gemma-3-4B's residual norms (≈40k at mid layers) OVERFLOW float16 on MPS →
# NaN at deeper layers. bfloat16 (float32 exponent range) is mandatory. Verified 2026-06-10.
PREFERRED_DTYPE = "bfloat16"

# Public SAEs — official Gemma Scope 2 for Gemma-3-4B-it: https://huggingface.co/google/gemma-scope-2-4b-it
# (verified 2026-06-10). sae-lens release `gemma-scope-2-4b-it-res`, layers 9/17/22/29.
SAE_REPO = "google/gemma-scope-2-4b-it"
SAE_HOOK_DEFAULT = "resid_post"
SAE_WIDTH_DEFAULT = "16k"
SAE_AVAILABLE_HOOKS = ("resid_post", "mlp_out", "attn_out")
# Cross-Layer Transcoder repo (compatible with circuit-tracer); verify before use.
CLT_REPO = "mwhanna/gemma-scope-2-4b-it"

NOTES = (
    "OPTIONAL exploratory model (OSF v8 optional set, post-deposit; NO "
    "confirmatory weight, enters no confirmation rule). 4B params, 34 layers, "
    "2560 hidden dim, Gemma Terms of Use. Within-generation scale pair with "
    "gemma_3_1b_it (Gemma-3, 1B vs 4B) to isolate size from generation. "
    "MULTIMODAL: loads as Gemma3ForConditionalGeneration with decoder layers at "
    "model.model.language_model.layers — see IS_MULTIMODAL/MODEL_CLASS/"
    "LAYERS_PATH. SUPPORTED (2026-06-21): Phase 2/4/5 branch on these hints and "
    "run this model on the text path. SAE/CLT repo names "
    "(google/gemma-scope-2-4b-it, mwhanna/gemma-scope-2-4b-it) NEED VERIFICATION "
    "against HF before an H3 run. Promotion to a confirmatory rule requires an "
    "OSF amendment."
)
