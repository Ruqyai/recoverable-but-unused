"""Phase 3 path + run config.

Phase 3 reads activations and directions from a Phase 2 run folder and writes
its analyses into a timestamped Phase 3 run folder per model:

    results/phase3/
    ├── README.md
    ├── <model_key>/
    │   ├── LATEST.txt                      ← pointer to most-recent run
    │   └── run_YYYYMMDD_HHMMSS/
    │       ├── run_metadata.json           ← model, phase2 source, device, args
    │       ├── cosine_similarity.json      ← Stage 1 output
    │       ├── magnitude_analysis.json     ← Stage 2 output
    │       ├── silhouette.json             ← Stage 3 output (H5)
    │       └── plots/                      ← optional matplotlib figures
    │           ├── cosine_heatmap_<script>_vs_<script>.png
    │           ├── magnitude_trajectory.png
    │           └── silhouette_per_layer.png
    └── ...

Each script accepts:
    --model <key>            : one of qwen3_4b / qwen3_8b / gemma_3_1b_it / gemma_2_9b_it
    --phase2-run-dir <path>  : Phase 2 run to consume (default: model's LATEST)
    --run-dir <path>         : existing Phase 3 run to resume (default: LATEST)
    --new-run                : force a fresh timestamped Phase 3 folder
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

from scripts.phase2_direction_extraction.config import (
    BASE_RESULTS_DIR as PHASE2_BASE_DIR,
    HARMFUL_FILES,
    BENIGN_FILES,
    SCRIPTS,
    model_base_dir as phase2_model_base_dir,
    resolve_run_paths as resolve_phase2_run_paths,
)

REPO = Path(__file__).resolve().parents[2]
BASE_RESULTS_DIR = REPO / "results" / "phase3"


@dataclass(frozen=True)
class Phase3RunPaths:
    """Resolved output paths for a single Phase 3 run on one model."""
    model_key:           str
    run_stamp:           str
    run_dir:             Path
    cosine_path:         Path     # run_dir / "cosine_similarity.json"
    magnitude_path:      Path     # run_dir / "magnitude_analysis.json"
    silhouette_path:     Path     # run_dir / "silhouette.json"
    run_metadata_path:   Path     # run_dir / "run_metadata.json"
    plots_dir:           Path     # run_dir / "plots"


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def model_base_dir(model_key: str) -> Path:
    return BASE_RESULTS_DIR / model_key


def resolve_phase3_run_paths(
    model_key: str,
    run_dir: Path | None = None,
    new_run: bool = False,
) -> Phase3RunPaths:
    """Resolve a Phase 3 output run folder.

    Precedence: explicit --run-dir > --new-run > resume LATEST > fresh run.
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
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    return Phase3RunPaths(
        model_key=model_key,
        run_stamp=stamp,
        run_dir=run_dir,
        cosine_path=run_dir / "cosine_similarity.json",
        magnitude_path=run_dir / "magnitude_analysis.json",
        silhouette_path=run_dir / "silhouette.json",
        run_metadata_path=run_dir / "run_metadata.json",
        plots_dir=plots_dir,
    )


def write_latest_pointer(model_key: str, run_dir: Path) -> None:
    """Update <model_key>/LATEST.txt to point at run_dir."""
    (model_base_dir(model_key) / "LATEST.txt").write_text(run_dir.name + "\n")


def resolve_phase2_source(model_key: str, phase2_run_dir: Path | None = None):
    """Resolve which Phase 2 run to consume.

    Returns the Phase 2 `RunPaths` dataclass (from phase2 config), so callers
    can read `activations_dir`, `directions_dir`, `metadata_path` from a
    consistent source. Defaults to the model's Phase 2 LATEST.
    """
    return resolve_phase2_run_paths(
        model_key=model_key,
        run_dir=phase2_run_dir,
        new_run=False,
    )


__all__ = [
    "REPO",
    "BASE_RESULTS_DIR",
    "Phase3RunPaths",
    "SCRIPTS",
    "HARMFUL_FILES",
    "BENIGN_FILES",
    "model_base_dir",
    "resolve_phase3_run_paths",
    "resolve_phase2_source",
    "write_latest_pointer",
]
