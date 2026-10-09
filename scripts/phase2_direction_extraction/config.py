"""Phase 2 path + run config.

Per-model architecture constants live in `scripts/models/`.
This file resolves a model key + optional run timestamp into a concrete directory
layout under `results/phase2/`:

    results/phase2/
    ├── README.md
    ├── archive_pre_refactor_<date>/        ← pre-2026-05-25 single-model layout
    ├── <model_key>/
    │   ├── LATEST.txt                      ← pointer to most-recent run
    │   └── run_YYYYMMDD_HHMMSS/
    │       ├── activations/<script>/<harmful|benign>/<record_id>.pt
    │       ├── directions/<script>.pt
    │       ├── metadata.json
    │       └── run_metadata.json
    └── ...

Each invocation of `01_extract_activations.py` or `02_compute_directions.py`
accepts `--model <key>` and either `--run-dir <path>` (resume an existing run)
or `--new-run` (force a fresh timestamped folder).
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

# --- shared data paths (independent of model) ---------------------------------

REPO = Path(__file__).resolve().parents[2]

HARMFUL_FILES = {
    "arabic":     REPO / "data/tier_a/tier_a_arabic.jsonl",
    "arabizi":    REPO / "data/tier_a/tier_a_arabizi.jsonl",
    "devanagari": REPO / "data/tier_c/tier_c_devanagari.jsonl",
    "itrans":     REPO / "data/tier_c/tier_c_itrans.jsonl",
    # Tier E (English ↔ Leetspeak) — first-class confirmatory tier as of OSF v6
    "english":    REPO / "data/tier_e/tier_e_english.jsonl",
    "leetspeak":  REPO / "data/tier_e/tier_e_leetspeak.jsonl",
}

BENIGN_FILES = {
    "arabic":     REPO / "data/tier_b/benign_arabic.jsonl",
    "arabizi":    REPO / "data/tier_b/benign_arabizi.jsonl",
    "devanagari": REPO / "data/tier_b/benign_devanagari.jsonl",
    "itrans":     REPO / "data/tier_b/benign_itrans.jsonl",
    "english":    REPO / "data/tier_b/benign_english.jsonl",
    "leetspeak":  REPO / "data/tier_b/benign_leetspeak.jsonl",
}

# Six v6 scripts. Tier E (english/leetspeak) added 2026-05-30 — the Phase 2
# config previously wired only the original four, leaving Tier E unextractable.
SCRIPTS = ["arabic", "arabizi", "devanagari", "itrans", "english", "leetspeak"]

BASE_RESULTS_DIR = REPO / "results/phase2"


# --- per-run layout -----------------------------------------------------------

@dataclass(frozen=True)
class RunPaths:
    """Resolved output paths for a single Phase 2 run on one model."""

    model_key:        str
    run_stamp:        str       # "YYYYMMDD_HHMMSS"
    run_dir:          Path      # results/phase2/<model_key>/run_<stamp>/
    activations_dir:  Path      # run_dir / "activations"
    directions_dir:   Path      # run_dir / "directions"
    metadata_path:    Path      # run_dir / "metadata.json"
    run_metadata_path: Path     # run_dir / "run_metadata.json"


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def model_base_dir(model_key: str) -> Path:
    return BASE_RESULTS_DIR / model_key


def resolve_run_paths(
    model_key: str,
    run_dir: Path | None = None,
    new_run: bool = False,
) -> RunPaths:
    """Resolve the output paths for a Phase 2 invocation.

    Precedence:
        1. If `run_dir` is given, use it verbatim (resume mode).
        2. Else if `new_run=True` or no prior run exists, create a fresh
           `run_<stamp>/` folder under `<model_key>/`.
        3. Else reuse the most-recent `run_*/` pointed to by `LATEST.txt`.
    """
    base = model_base_dir(model_key)
    base.mkdir(parents=True, exist_ok=True)

    if run_dir is not None:
        run_dir = Path(run_dir).resolve()
        if not run_dir.exists():
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
                # Stale pointer; start fresh.
                stamp = _timestamp()
                run_dir = base / f"run_{stamp}"
        else:
            stamp = _timestamp()
            run_dir = base / f"run_{stamp}"

    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        model_key=model_key,
        run_stamp=stamp,
        run_dir=run_dir,
        activations_dir=run_dir / "activations",
        directions_dir=run_dir / "directions",
        metadata_path=run_dir / "metadata.json",
        run_metadata_path=run_dir / "run_metadata.json",
    )


def write_latest_pointer(model_key: str, run_dir: Path) -> None:
    """Update <model_key>/LATEST.txt to point at run_dir."""
    pointer = model_base_dir(model_key) / "LATEST.txt"
    pointer.write_text(run_dir.name + "\n")
