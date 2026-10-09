"""Model config: Qwen/Qwen3-8B (Phase-2 within-Qwen size-scaling replication)."""
from __future__ import annotations

MODEL_KEY = "qwen3_8b"
MODEL_ID = "Qwen/Qwen3-8B"

# Architecture
N_LAYERS = 36
HIDDEN_DIM = 4096

# Extraction
BATCH_SIZE = 4  # smaller batch; 8B model uses ~2× memory of 4B
MEASUREMENT_POSITION = -1

CHAT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# Extra kwargs passed into tokenizer.apply_chat_template(...).
# Disable Qwen3 thinking mode — see qwen3_4b.py for rationale and log entry.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# System prompts used by Phase-4 diagnostics under --use-system-prompt.
# See qwen3_4b.py for rationale; defaults from models/__init__.py are used
# when None. NOT used during Phase-2 activation extraction.
SYSTEM_PROMPTS: dict[str, str] | None = None

# Public SAEs — none available for Qwen3-8B at v4 deposit time.
SAE_REPO = ""
SAE_HOOK_DEFAULT = ""
SAE_WIDTH_DEFAULT = ""

NOTES = (
    "Size-scaling replication within the Qwen family per OSF v4. 8B parameters, "
    "36 layers, 4096 hidden dim, Apache-2.0. Used alongside Qwen3-4B to isolate "
    "size-of-parameter-count effects from family-level effects. Tied with Qwen3-4B "
    "on the 2026-05-25 diagnostic (44/45 total coherent)."
)
