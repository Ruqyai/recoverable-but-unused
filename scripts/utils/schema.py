"""Pydantic schemas for prompt records and validation."""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Tier(str, Enum):
    A = "A"          # Arabic ↔ Arabizi parallel (harmful) — ecological
    B = "B"          # BENIGN controls (benign) Matched benign controls per script
    C = "C"          # Hindi Devanagari ↔ ITRANS Latin (harmful) — mechanical
    D = "D"          # Within-script paraphrases (harmful) — variation control
    E = "E"          # English ↔ Leetspeak (harmful) — within-language adversarial orthography


class Script(str, Enum):
    ARABIC = "arabic"          # Arabic script (U+0600..U+06FF)
    ARABIZI = "arabizi"        # Latin-script transliteration of Arabic (community)
    ENGLISH = "english"        # canonical English Latin
    DEVANAGARI = "devanagari"  # Hindi Devanagari (U+0900..U+097F)
    ITRANS = "itrans"          # Latin transliteration of Hindi via ITRANS scheme (ASCII)
    LEETSPEAK = "leetspeak"    # English ASCII with deterministic 1337-style char substitutions


class Language(str, Enum):
    AR = "ar"  # Arabic
    EN = "en"
    HI = "hi"  # Hindi


class HarmCategory(str, Enum):
    """Coarse harm taxonomy. Maps roughly onto Aegis 2.0 / Nemotron categories."""
    WEAPONS = "weapons"
    CYBERCRIME = "cybercrime"
    FRAUD = "fraud"
    SELF_HARM = "self_harm"
    HATE = "hate"
    SEXUAL = "sexual"
    PRIVACY = "privacy"
    HARASSMENT = "harassment"
    ILLEGAL_GOODS = "illegal_goods"
    MISINFORMATION = "misinformation"
    OTHER = "other"


class Validation(BaseModel):
    native_reviewed: bool = False
    reviewer_id: Optional[str] = None
    review_date: Optional[date] = None
    notes: Optional[str] = None
    semantic_match: Optional[bool] = None


class PromptRecord(BaseModel):
    """One row in a tier JSONL file."""

    id: str = Field(..., min_length=3)
    tier: Tier
    harm_category: HarmCategory
    source_id: str
    script: Script
    language: Language
    prompt: str = Field(..., min_length=1)
    paired_with: Optional[str] = None
    is_harmful: bool
    validation: Validation = Field(default_factory=Validation)

    @field_validator("id")
    @classmethod
    def id_pattern(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"id must be alphanumeric+underscores, got {v!r}")
        return v

    def script_consistent_with_language(self) -> bool:
        valid = {
            Language.AR: {Script.ARABIC, Script.ARABIZI},
            Language.EN: {Script.ENGLISH, Script.LEETSPEAK},
            Language.HI: {Script.DEVANAGARI, Script.ITRANS},
        }
        return self.script in valid.get(self.language, set())
