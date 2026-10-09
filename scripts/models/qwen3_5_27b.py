"""Model config: Qwen/Qwen3.5-27B (OPTIONAL exploratory — large-model extension, MULTIMODAL).

NOT part of the v4/v6 confirmatory grid. Added per RERUN_PLAN §7 as the proposed
large-model extension: a strongly-multilingual model in our PRIMARY family (Qwen)
to probe comprehension at scale — separating "can't decode the romanized surface"
from "anchored despite comprehension". EXPLORATORY: no confirmatory weight, enters
no confirmation rule (OSF v8 exploratory); promotion requires an OSF amendment.

⚠️ MULTIMODAL. Qwen/Qwen3.5-27B loads as `Qwen3_5ForConditionalGeneration`
(text + vision), NOT `AutoModelForCausalLM`. The text decoder lives under
`model.model.language_model.layers` and the final norm under
`model.model.language_model.norm` — the SAME layout as Gemma-3 VLMs. The Phase
2/4/5 text path branches on the hints below (verified 2026-06-21):
`device.load_hf_model(..., model_class=cfg.MODEL_CLASS)` loads the right class and
every hook/hidden-state site resolves the text decoder via
`get_text_base_model(..., submodule_path=cfg.TEXT_SUBMODULE_PATH)`.

H3 eligibility: Qwen-Scope residual SAEs cover ALL 64 layers (width 80K). They
ship as `.pt` state dicts (`layer{n}.sae.pt`), NOT sae-lens format, so Phase 5
needs a small loader adapter before H3 can run on this model. See SAE_* below.
"""
from __future__ import annotations

MODEL_KEY = "qwen3_5_27b"
MODEL_ID = "Qwen/Qwen3.5-27B"

# Architecture (verified via AutoConfig 2026-06-21: Qwen3_5ForConditionalGeneration,
# text_config.num_hidden_layers=64, text_config.hidden_size=5120).
N_LAYERS = 64
HIDDEN_DIM = 5120

# Extraction. 27B in bf16 (~54 GB weights) fits 1×A100-80GB forward-only — do NOT
# 4-bit quantize: direction extraction + patching read the residual stream, so
# quantization would distort the measured values. Keep the batch small; the
# device-aware multiplier in device.py scales from here. Tune up if memory allows.
BATCH_SIZE = 2
MEASUREMENT_POSITION = -1  # last token of "<|im_start|>assistant\n" template

# Chat template (manual fallback; scripts prefer tokenizer.apply_chat_template)
CHAT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# Qwen3 family enables <think>...</think> by default; with finite max_new_tokens
# the model can exhaust the budget inside the thinking trace. Disable for direct
# answers (mirrors qwen3_4b). VERIFY the Qwen3.5 template accepts this kwarg.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# See qwen3_4b.py / models/__init__.py for rationale; defaults used when None.
SYSTEM_PROMPTS: dict[str, str] | None = None

# --- Multimodal hints (read downstream via getattr(cfg, ..., default)). -------
IS_MULTIMODAL = True
MODEL_CLASS = "Qwen3_5ForConditionalGeneration"  # AutoModelForCausalLM is wrong here
# Text decoder layers / submodule, relative to the FULL model object. Verified via
# named_modules on a meta-device build 2026-06-21: layers at model.language_model.layers
# (n=64), final norm at model.language_model.norm, lm_head at top level.
LAYERS_PATH = "model.language_model.layers"
TEXT_SUBMODULE_PATH = "model.language_model"     # full_model.model.language_model = Qwen3_5TextModel

# Qwen weights are bf16; safe default everywhere (and required on MPS like Gemma-3).
PREFERRED_DTYPE = "bfloat16"

# Public SAEs — Qwen-Scope residual SAEs, ALL 64 layers, width 80K, instruct-trained.
# ⚠️ FORMAT: these are `.pt` state dicts (one `layer{n}.sae.pt` per layer), NOT
# sae-lens repos. Phase 5 (01_load_sae / sae_loading.py) needs a `.pt` loader
# adapter before H3 runs. Two sparsity variants are published:
#   - L0_50  (default here)            : Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50
#   - L0_100 (denser, alternative)     : Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50"
SAE_REPO_L0_100 = "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100"
SAE_HOOK_DEFAULT = "resid_post"
SAE_WIDTH_DEFAULT = "80k"
SAE_FORMAT = "qwen_scope_pt"          # marker for the Phase-5 loader branch (not sae-lens)
SAE_FILE_TEMPLATE = "layer{layer}.sae.pt"

NOTES = (
    "OPTIONAL exploratory large-model extension (RERUN_PLAN §7; OSF v8 exploratory, "
    "NO confirmatory weight). 27B params, 64 layers, 5120 hidden. PRIMARY family "
    "(Qwen) → best continuity with the qwen3_4b primary. MULTIMODAL: loads as "
    "Qwen3_5ForConditionalGeneration; text decoder at model.model.language_model.layers "
    "(same layout as Gemma-3 VLMs) — Phase 2/4/5 branch on IS_MULTIMODAL/MODEL_CLASS/"
    "LAYERS_PATH/TEXT_SUBMODULE_PATH (supported 2026-06-21). H3: Qwen-Scope residual "
    "SAEs cover all 64 layers (W80K, L0_50/L0_100) but ship as .pt dicts → needs a "
    "Phase-5 loader adapter (SAE_FORMAT='qwen_scope_pt'). Run only after the primary + "
    "replication re-run confirms the thesis; runs in bf16 on ~1×A100-80GB "
    "(forward-only) — NOT 4-bit (quantization would distort the residual-stream "
    "measurements). "
    "Promotion to a confirmatory rule requires an OSF amendment."
)
