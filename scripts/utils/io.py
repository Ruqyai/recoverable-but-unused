"""Shared I/O utilities for JSONL data files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import jsonlines

from .schema import PromptRecord


def write_jsonl(records: Iterable[PromptRecord], path: Path) -> int:
    """Write records to JSONL, return count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with jsonlines.open(path, mode="w") as writer:
        for rec in records:
            writer.write(rec.model_dump(mode="json"))
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[PromptRecord]:
    """Stream records from JSONL with validation."""
    with jsonlines.open(path, mode="r") as reader:
        for obj in reader:
            yield PromptRecord(**obj)


def make_id(tier: str, idx: int, script: str) -> str:
    """Canonical ID format for new records.

    Harmful tiers (A, C, D, E): ``tier_<tier>_<idx>_<script>``  e.g. ``tier_a_001_arabic``.
    Benign records (pass ``tier="BENIGN"``): ``benign_<script>_<idx>``  e.g. ``benign_english_042``.

    Legacy on-disk note (frozen, do NOT regenerate): the pre-2026-05-28 benign
    files (``benign_arabic.jsonl``, ``benign_arabizi.jsonl``,
    ``benign_devanagari.jsonl``, ``benign_itrans.jsonl``) use a third pattern
    ``tier_b_<script>_<idx>`` (e.g. ``tier_b_arabic_001``) produced by an
    earlier version of this function. Their ``paired_with`` references point
    to ``tier_a_<idx>_<script>`` (cross-tier benign↔harmful counterpart). The
    newer benign files (``benign_english.jsonl``, ``benign_leetspeak.jsonl``,
    written by scripts 12/13 on 2026-05-28) use the ``benign_<script>_<idx>``
    pattern produced here. Both patterns coexist on disk and both pass schema
    validation (``id`` is a string). Per the 2026-05-29 ``[DECISION]`` log
    entry, the data is frozen as-is; consumers must tolerate both patterns.
    """
    if tier.upper() == "BENIGN":
        return f"benign_{script}_{idx:03d}"
    return f"tier_{tier.lower()}_{idx:03d}_{script}"
