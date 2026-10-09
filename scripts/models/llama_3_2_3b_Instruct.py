"""Model config: meta-llama/Llama-3.2-3B-Instruct (OPTIONAL exploratory — OSF v8).

NOT part of the v6 six-model grid. Added to the OSF v8 optional-exploratory set
(the deposited pre-registration): "run only after deposit; do NOT enter
any confirmation rule". Role: a **scale series** with llama_3_1_8b_Instruct —
same Llama family, 3B vs 8B — to probe how the refusal asymmetry scales within
the Llama architecture (cross-architecture sibling to the Qwen and Gemma pairs).

⚠️ The HF repo is GATED (403 without an approved Llama license on HF_TOKEN), so
the architecture constants below come from the published Llama-3.2-3B spec, not
a local AutoConfig read. Re-verify N_LAYERS / HIDDEN_DIM on first successful
load once access is granted.

H3 eligibility: none — no public SAE covers Llama-3.2-3B-Instruct (andyrdt's
release is Llama-3.1-8B only). SAE fields left empty; H3 is not testable here.

Adding this model to a confirmatory rule requires an OSF amendment.
"""
from __future__ import annotations

MODEL_KEY = "llama_3_2_3b_Instruct"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# Architecture (published Llama-3.2-3B spec; repo gated — verify on first load).
N_LAYERS = 28
HIDDEN_DIM = 3072

# Extraction
BATCH_SIZE = 8  # 3B model; between gemma-3-1b (16) and the 7-8B configs (4)
MEASUREMENT_POSITION = -1  # trailing newline of the assistant header (see below)

# Llama-3 chat template (manual fallback; scripts prefer tokenizer.apply_chat_template).
# Identical structure to llama_3_1_8b_Instruct — the double-newline after the
# assistant header is the canonical refusal-direction probe position.
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

# Public SAEs — none for Llama-3.2-3B-Instruct. andyrdt's BatchTopK release
# covers Llama-3.1-8B only; nothing public at the 3B size. Empty = not H3-eligible.
SAE_REPO = ""
SAE_HOOK_DEFAULT = ""
SAE_WIDTH_DEFAULT = ""

NOTES = (
    "OPTIONAL exploratory model (OSF v8 optional set, post-deposit; NO "
    "confirmatory weight, enters no confirmation rule). 3B params, 28 layers, "
    "3072 hidden dim, Llama 3.2 Community License. Scale series with "
    "llama_3_1_8b_Instruct (3B vs 8B). Repo is GATED — constants from the "
    "published spec; re-verify on first successful load. No public SAE at 3B, "
    "so H3 is not testable here. Promotion to a confirmatory rule requires an "
    "OSF amendment."
)
