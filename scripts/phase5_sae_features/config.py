"""Phase 5 path + run config (SAE-feature overlap, H3).

Phase 5 reads a Phase 2 run folder (directions + the empirically-selected
causal layer from Stage 04) and a Gemma-Scope SAE, then computes the H3
top-100-feature Jaccard between scripts. Outputs land in a timestamped Phase 5
run folder per model:

    results/phase5/
    ├── README.md
    ├── <model_key>/
    │   ├── LATEST.txt                      ← pointer to most-recent run
    │   └── run_YYYYMMDD_HHMMSS/
    │       ├── run_metadata.json           ← model, phase2 source, device, args
    │       ├── sae_meta.json               ← Stage 01 (release / sae_id / d_sae / layer)
    │       ├── feature_activations/<script>.pt   ← Stage 02 (per-prompt SAE features)
    │       ├── top100_per_script.json      ← Stage 03 (top-K feature indices)
    │       ├── jaccard.json                ← Stage 04 (pairwise Jaccard + H3 verdict)
    │       └── feature_labels.json         ← Stage 05 (optional Neuronpedia labels)
    └── ...

Each stage accepts:
    --model <key>            : any registered model key
    --phase2-run-dir <path>  : Phase 2 run to consume (default: model's LATEST)
    --run-dir <path>         : existing Phase 5 run to resume (default: LATEST)
    --new-run                : force a fresh timestamped Phase 5 folder

H3 (OSF prereg § 3.3): top-100 SAE features at the per-model
causal layer show Jaccard < 0.4 between Arabic vs Arabizi. REJECTION: Jaccard
>= 0.6. Testable on the two Gemma models (Gemma Scope SAEs); exploratory
elsewhere. Per the deposited preregistration, H3 is designated EXPLORATORY
(foreknowledge-clean but no confirmatory weight) — every Stage 04 output
carries that caveat.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path

from scripts.phase2_direction_extraction.config import (  # noqa: F401  (re-exported)
    BENIGN_FILES,
    HARMFUL_FILES,
    SCRIPTS,
)
from scripts.phase2_direction_extraction.config import (
    resolve_run_paths as resolve_phase2_run_paths,
)

REPO = Path(__file__).resolve().parents[2]
BASE_RESULTS_DIR = REPO / "results" / "phase5"


# ── H3 thresholds + scope (OSF prereg § 3.3) ──────────────────
H3_CONFIRMATION_THRESHOLD = 0.4   # Jaccard < 0.4 → orthographic-anchoring supported
H3_REJECTION_THRESHOLD = 0.6      # Jaccard >= 0.6 → features largely shared (reject)

# Models on which H3 is (exploratorily) testable — the two Gemma models carry
# Gemma Scope SAEs. The optional-set Gemma models (2b / 3-4b) have community/
# unverified SAEs and are exploratory-extra; everything else lacks a usable SAE.
H3_CONFIRMATORY_MODELS = {"gemma_3_1b_it", "gemma_2_9b_it"}

# Ordered script pairs tested for feature overlap, by tier. The first (Tier A,
# Arabic vs Arabizi) is the headline H3 pair; the others are within-design
# extensions (Tier C deterministic transliteration, Tier E within-language).
H3_SCRIPT_PAIRS = {
    "tier_a": ("arabic", "arabizi"),
    "tier_c": ("devanagari", "itrans"),
    "tier_e": ("english", "leetspeak"),
}


@dataclass(frozen=True)
class Phase5RunPaths:
    """Resolved output paths for a single Phase 5 run on one model."""

    model_key:           str
    run_stamp:           str
    run_dir:             Path
    sae_meta_path:       Path     # run_dir / "sae_meta.json"
    feature_acts_dir:    Path     # run_dir / "feature_activations"
    top_k_path:          Path     # run_dir / "top100_per_script.json"
    jaccard_path:        Path     # run_dir / "jaccard.json"
    feature_labels_path: Path     # run_dir / "feature_labels.json"
    run_metadata_path:   Path     # run_dir / "run_metadata.json"


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def model_base_dir(model_key: str) -> Path:
    return BASE_RESULTS_DIR / model_key


def resolve_phase5_run_paths(
    model_key: str,
    run_dir: Path | None = None,
    new_run: bool = False,
) -> Phase5RunPaths:
    """Resolve a Phase 5 output run folder.

    Precedence: explicit --run-dir > --new-run > resume LATEST > fresh run.
    Mirrors phase2/phase3 `resolve_*_run_paths`.
    """
    base = model_base_dir(model_key)
    base.mkdir(parents=True, exist_ok=True)

    if run_dir is not None:
        run_dir = Path(run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        stamp = run_dir.name.replace("run_", "")
    else:
        latest_file = base / "LATEST.txt"
        if not new_run and latest_file.exists():
            latest_name = latest_file.read_text().strip()
            candidate = base / latest_name
            if candidate.exists():
                run_dir = candidate
                stamp = latest_name.replace("run_", "")
            else:
                stamp = _timestamp()
                run_dir = base / f"run_{stamp}"
        else:
            stamp = _timestamp()
            run_dir = base / f"run_{stamp}"

    run_dir.mkdir(parents=True, exist_ok=True)
    feature_acts_dir = run_dir / "feature_activations"
    feature_acts_dir.mkdir(parents=True, exist_ok=True)

    return Phase5RunPaths(
        model_key=model_key,
        run_stamp=stamp,
        run_dir=run_dir,
        sae_meta_path=run_dir / "sae_meta.json",
        feature_acts_dir=feature_acts_dir,
        top_k_path=run_dir / "top100_per_script.json",
        jaccard_path=run_dir / "jaccard.json",
        feature_labels_path=run_dir / "feature_labels.json",
        run_metadata_path=run_dir / "run_metadata.json",
    )


def write_latest_pointer(model_key: str, run_dir: Path) -> None:
    """Update <model_key>/LATEST.txt to point at run_dir."""
    (model_base_dir(model_key) / "LATEST.txt").write_text(run_dir.name + "\n")


def resolve_phase2_source(model_key: str, phase2_run_dir: Path | None = None):
    """Resolve which Phase 2 run to consume (defaults to the model's LATEST).

    Returns the Phase 2 `RunPaths` dataclass, so callers read `activations_dir`,
    `directions_dir`, `run_dir` from a consistent source.
    """
    return resolve_phase2_run_paths(
        model_key=model_key,
        run_dir=phase2_run_dir,
        new_run=False,
    )


def read_causal_layer(phase2_run_dir: Path) -> int | None:
    """Read the per-model causal layer chosen by Phase 2 Stage 04.

    Prefers the Llama-Guard-classified peak (`peak_causal_layer_guard`); falls
    back to the lexical peak if Guard was skipped. Returns None if Stage 04 has
    not been run on this Phase 2 run (callers must then require an explicit
    --layer override).
    """
    path = Path(phase2_run_dir) / "causal_layer_selection.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    sel = data.get("selection", {})
    for key in ("peak_causal_layer_guard", "peak_causal_layer_lexical"):
        if sel.get(key) is not None:
            return int(sel[key])
    return None


def h3_verdict(jaccard: float, ci: dict | None = None) -> str:
    """Apply the OSF prereg H3 thresholds (§ 3.3).

    < 0.4 → SUPPORTED (orthographic anchoring); >= 0.6 → REJECTED (shared
    features); in-between → INDETERMINATE. When a bootstrap CI is supplied the
    verdict notes whether the CI clears the threshold.
    """
    if jaccard < H3_CONFIRMATION_THRESHOLD:
        if ci and ci.get("ci_99_high") is not None and ci["ci_99_high"] < H3_CONFIRMATION_THRESHOLD:
            return "SUPPORTED (CI excludes threshold)"
        return "SUPPORTED (point estimate; CI overlaps threshold)"
    if jaccard >= H3_REJECTION_THRESHOLD:
        if ci and ci.get("ci_99_low") is not None and ci["ci_99_low"] >= H3_REJECTION_THRESHOLD:
            return "REJECTED (CI excludes threshold)"
        return "REJECTED (point estimate; CI overlaps threshold)"
    return "INDETERMINATE (0.4 <= Jaccard < 0.6)"


__all__ = [
    "REPO",
    "BASE_RESULTS_DIR",
    "SCRIPTS",
    "HARMFUL_FILES",
    "BENIGN_FILES",
    "H3_CONFIRMATION_THRESHOLD",
    "H3_REJECTION_THRESHOLD",
    "H3_CONFIRMATORY_MODELS",
    "H3_SCRIPT_PAIRS",
    "Phase5RunPaths",
    "model_base_dir",
    "resolve_phase5_run_paths",
    "write_latest_pointer",
    "resolve_phase2_source",
    "read_causal_layer",
    "h3_verdict",
]
