"""Model config: google/gemma-2-2b-it (OPTIONAL exploratory — OSF v8 optional set).

NOT part of the v6 six-model grid. Added to the OSF v8 optional-exploratory set
(the deposited pre-registration): "run only after deposit; do NOT enter
any confirmation rule". Role: a **generation-controlled scale pair** with
gemma-2-9b-it — same Gemma-2 generation, different size (2B vs 9B) — so the
2B↔9B contrast isolates size from generation within one family.

H3 eligibility: SAEs are available via Gemma Scope's 2B residual release
(google/gemma-scope-2b-pt-res), but those are trained on the **base** model
(gemma-2-2b), not the IT variant. Applying PT-trained SAEs to IT activations is
the documented community path (Gemma Scope reports the basis transfers), but it
carries the same base-vs-IT caveat called out for the Qwen3-8B-Base SAEs in the
Phase 5 README — report any H3 result here as exploratory and flag the mismatch.

Adding this model to a confirmatory rule requires an OSF amendment.
"""
from __future__ import annotations

MODEL_KEY = "gemma_2_2b_it"
MODEL_ID = "google/gemma-2-2b-it"

# Architecture (verified via AutoConfig 2026-06-09: Gemma2ForCausalLM,
# num_hidden_layers=26, hidden_size=2304).
N_LAYERS = 26
HIDDEN_DIM = 2304

# Extraction
BATCH_SIZE = 8  # 2B model; between gemma-3-1b (16) and gemma-2-9b (2)
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

# Public SAEs — Gemma Scope 2B residual (PT-trained; see module docstring caveat).
# Covers all 26 layers at width 16k (canonical). Sibling mlp/att repos exist.
SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_HOOK_DEFAULT = "resid_post"
SAE_WIDTH_DEFAULT = "16k"
SAE_AVAILABLE_HOOKS = ("resid_post", "mlp_out", "attn_out")
SAE_REPO_MLP = "google/gemma-scope-2b-pt-mlp"
SAE_REPO_ATT = "google/gemma-scope-2b-pt-att"

NOTES = (
    "OPTIONAL exploratory model (OSF v8 optional set, post-deposit; NO "
    "confirmatory weight, enters no confirmation rule). 2B params, 26 layers, "
    "2304 hidden dim, Gemma Terms of Use. Generation-controlled scale pair with "
    "gemma-2-9b-it (same Gemma-2 generation, 2B vs 9B) to isolate size from "
    "generation. H3 via google/gemma-scope-2b-pt-res — but those SAEs are "
    "PT-trained (base gemma-2-2b), so any IT H3 result carries a base-vs-IT "
    "caveat (parallel to the Qwen3-8B-Base SAE caveat in the Phase 5 README). "
    "Promotion to a confirmatory rule requires an OSF amendment."
)
