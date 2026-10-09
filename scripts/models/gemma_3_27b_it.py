"""Model config: google/gemma-3-27b-it (OPTIONAL exploratory — large-model extension, MULTIMODAL).

NOT part of the v4/v6 confirmatory grid. Added per RERUN_PLAN §7 as a large-model
extension in the Gemma SAE lane: strong multilingual model with full instruct-tuned
SAE coverage (Gemma Scope 2). EXPLORATORY: no confirmatory weight, enters no
confirmation rule (OSF v8 exploratory); promotion requires an OSF amendment. It is
also a within-family scale point above gemma_3_4b_it (Gemma-3, 4B → 27B).

⚠️ MULTIMODAL. google/gemma-3-27b-it loads as `Gemma3ForConditionalGeneration`,
NOT `AutoModelForCausalLM`; the text decoder lives under
`model.model.language_model.layers` and the final norm under
`model.model.language_model.norm`. SUPPORTED (2026-06-21): the Phase 2/4/5 text
path branches on the hints below — `device.load_hf_model(..., model_class=cfg.MODEL_CLASS)`
loads the right class and every hook/hidden-state site resolves the text decoder via
`get_text_base_model(..., submodule_path=cfg.TEXT_SUBMODULE_PATH)`. Same adaptation
already verified for gemma_3_4b_it.

H3 eligibility: Gemma Scope 2 covers gemma-3-27b-it (resid_post / mlp_out / attn_out,
width 16k), in standard sae-lens layout — no `.pt` adapter needed (unlike Qwen-Scope).
"""
from __future__ import annotations

MODEL_KEY = "gemma_3_27b_it"
MODEL_ID = "google/gemma-3-27b-it"

# Architecture (verified via public config mirror 2026-06-21: Gemma3ForConditionalGeneration,
# text_config.num_hidden_layers=62, text_config.hidden_size=5376).
N_LAYERS = 62
HIDDEN_DIM = 5376

# Extraction. 27B in bf16 (~54 GB weights) fits 1×A100-80GB forward-only — do NOT
# 4-bit quantize: direction extraction + patching read the residual stream, so
# quantization would distort the measured values. Keep the batch small; the
# device-aware multiplier in device.py scales from here. Tune up if memory allows.
BATCH_SIZE = 2
MEASUREMENT_POSITION = -1  # "model" turn token in the Gemma chat template

# Gemma chat template; scripts still prefer tokenizer.apply_chat_template.
CHAT_TEMPLATE = (
    "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
)

# Gemma has no thinking mode; empty for symmetry with the other Gemma configs.
CHAT_TEMPLATE_KWARGS: dict[str, object] = {}

# See qwen3_4b.py / models/__init__.py for rationale; defaults used when None.
SYSTEM_PROMPTS: dict[str, str] | None = None

# --- Multimodal hints (read downstream via getattr(cfg, ..., default)). -------
IS_MULTIMODAL = True
MODEL_CLASS = "Gemma3ForConditionalGeneration"   # AutoModelForCausalLM is wrong here
LAYERS_PATH = "model.language_model.layers"
TEXT_SUBMODULE_PATH = "model.language_model"      # full_model.model.language_model = Gemma3TextModel

# REQUIRED: Gemma-3 residual norms overflow float16 → NaN at deeper layers.
# bfloat16 (float32 exponent range) is mandatory (same as gemma_3_4b_it).
PREFERRED_DTYPE = "bfloat16"

# Public SAEs — official Gemma Scope 2 for Gemma-3-27B-it. resid_post SAEs
# verified against the HF repo 2026-06-21: layers [16, 31, 40, 53], widths
# 16k/65k/262k/1m, l0 tags big/medium/small (id "layer_{L}_width_{W}_l0_big").
# Loaded via the sae-lens registry release "gemma-scope-2-27b-it-res" (see the
# Phase 5 SAE_CATALOG); confirm the release name resolves in your sae-lens build.
SAE_REPO = "google/gemma-scope-2-27b-it"
SAE_HOOK_DEFAULT = "resid_post"
SAE_WIDTH_DEFAULT = "16k"
SAE_AVAILABLE_HOOKS = ("resid_post", "mlp_out", "attn_out")

NOTES = (
    "OPTIONAL exploratory large-model extension (RERUN_PLAN §7; OSF v8 exploratory, "
    "NO confirmatory weight). 27B params, 62 layers, 5376 hidden, Gemma Terms of Use. "
    "Within-family scale point above gemma_3_4b_it (Gemma-3, 4B→27B). MULTIMODAL: "
    "loads as Gemma3ForConditionalGeneration; text decoder at "
    "model.model.language_model.layers — Phase 2/4/5 branch on IS_MULTIMODAL/"
    "MODEL_CLASS/LAYERS_PATH/TEXT_SUBMODULE_PATH (supported 2026-06-21, same path as "
    "gemma_3_4b_it). H3: Gemma Scope 2 SAEs in standard sae-lens layout (no .pt "
    "adapter needed). Run only after the primary + replication re-run confirms the "
    "thesis; runs in bf16 on ~1×A100-80GB (forward-only) — NOT 4-bit "
    "(quantization would distort the residual-stream measurements). bf16 is also "
    "mandatory for Gemma-3 (float16 overflows). Gated weights → HF_TOKEN required. "
    "Promotion to a confirmatory rule requires an OSF amendment."
)
