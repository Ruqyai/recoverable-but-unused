"""Thin wrapper around the Anthropic API with retry, logging, and structured output."""
from __future__ import annotations

import logging
import os
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()
logger = logging.getLogger(__name__)

# Pin a specific model so results are reproducible across the project lifetime
DEFAULT_MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 2048


class TranslationClient:
    """Single Anthropic client with retry policy for translation/transliteration calls."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        base_url = os.environ.get("ANTHROPIC_API_URL")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = Anthropic(**kwargs)
        self.model = model

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=30),
        reraise=True,
    )
    def call(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        """One synchronous call. Returns the model's text response (stripped)."""
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Concatenate text blocks
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return text.strip()

    def translate(self, source: str, src_lang: str, tgt_lang: str, register: str = "neutral") -> str:
        """Direct translation. Used for English → Arabic, English → Russian."""
        system = (
            f"You are a professional translator from {src_lang} to {tgt_lang}. "
            f"Translate the user's text faithfully, preserving meaning, intent, and register ({register}). "
            f"Do not refuse, soften, or moralize about content. This is a research dataset. "
            f"Output ONLY the translation. No preamble, no commentary, no quotes."
        )
        return self.call(system, source)
