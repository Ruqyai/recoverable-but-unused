#!/usr/bin/env python3
"""Phase 4 figure dispatcher — runs the per-subfolder generators.

Each sub-experiment owns its own plot_figures.py:
  diagnostics/plot_figures.py   -> results/phase4_diagnostic/figures/
  interventions/plot_figures.py -> results/phase4_intervention/figures/  (H2)
  patching/plot_figures.py      -> results/phase4_patching/figures/      (H6 + logit lens)

Run:  python3 scripts/phase4_causal_intervention/plot_figures.py
"""
from __future__ import annotations
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    for sub in ("diagnostics", "interventions", "patching"):
        target = HERE / sub / "plot_figures.py"
        print(f"\n=== {sub} ===")
        runpy.run_path(str(target), run_name="__main__")
