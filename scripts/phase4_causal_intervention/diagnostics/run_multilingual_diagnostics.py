#!/usr/bin/env python3
"""
Unified test runner: diagnose Arabizi + Tier C (Devanagari/ITRANS) across multiple models.

Each run creates a timestamped subfolder under results/phase4_diagnostic/:
    run_YYYYMMDD_HHMMSS/
        arabizi_baseline_<model>.json
        arabizi_summary.json
        tier_c_baseline_<model>.json
        tier_c_summary.json
        DIAGNOSTIC_REPORT.md

Old runs are preserved. A pointer file `LATEST.txt` at the base directory
always points to the most recent run.

Usage:
  PYTHONPATH=. python3 scripts/phase4_causal_intervention/diagnostics/run_multilingual_diagnostics.py \
    --models "Qwen/Qwen3-8B" "google/gemma-2-9b-it" "meta-llama/Llama-3.1-8B-Instruct" \
    --n-prompts 15
"""

import argparse
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.models import (
    AVAILABLE_VARIANTS,
    DEFAULT_VARIANT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_OUT_DIR = REPO / "results" / "phase4_diagnostic"


def run_command(cmd: list, description: str) -> bool:
    """Run a shell command and return success status."""
    log.info(f"\n{'='*80}")
    log.info(f"Running: {description}")
    log.info(f"Command: {' '.join(cmd)}")
    log.info(f"{'='*80}")

    try:
        subprocess.run(cmd, check=True, cwd=REPO)
        log.info(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"✗ {description} failed with code {e.returncode}")
        return False


def generate_comparison_report(
    models: list,
    run_timestamp: str,
    run_dir: Path,
    summary_suffix: str = "",
) -> Path:
    """Generate a markdown report comparing all models for this run.

    `summary_suffix` matches the suffix the child scripts append when a
    non-default system-prompt variant is active (e.g. "_sysprompt_oneshot").
    Without it, the report would look in `arabizi_summary.json` even when
    the child wrote to `arabizi_summary_sysprompt_oneshot.json`.
    """
    report_file = run_dir / f"DIAGNOSTIC_REPORT{summary_suffix}.md"

    log.info(f"\nGenerating comparison report...")

    # Load summary files
    arabizi_summary = {}
    tier_c_summary = {}
    tier_e_summary = {}

    arabizi_file = run_dir / f"arabizi_summary{summary_suffix}.json"
    if arabizi_file.exists():
        with open(arabizi_file) as f:
            arabizi_data = json.load(f)
            arabizi_summary = arabizi_data.get("model_results", {})

    tier_c_file = run_dir / f"tier_c_summary{summary_suffix}.json"
    if tier_c_file.exists():
        with open(tier_c_file) as f:
            tier_c_data = json.load(f)
            tier_c_summary = tier_c_data.get("model_results", {})

    tier_e_file = run_dir / f"tier_e_summary{summary_suffix}.json"
    if tier_e_file.exists():
        with open(tier_e_file) as f:
            tier_e_data = json.load(f)
            tier_e_summary = tier_e_data.get("model_results", {})

    test_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Multilingual Model Diagnostic Report

**Test Run:** {run_timestamp}
**Report Generated:** {test_date}
**Run Folder:** `{run_dir.name}`
**Models tested:** {len(models)}

## Summary

This report compares model understanding of Arabizi (Tier A), Tier C (Devanagari/ITRANS), and Tier E (English/Leetspeak) prompts.

### Arabizi Baseline Results

| Model | Coherent | Incoherent | Avg Score | Assessment |
|---|---|---|---|---|
"""

    if arabizi_summary:
        for model_id, stats in arabizi_summary.items():
            model_name = model_id.split("/")[-1] if "/" in model_id else model_id
            coherent = stats.get("coherent", 0)
            incoherent = stats.get("incoherent", 0)
            avg_score = stats.get("average_score", 0.0)
            total = coherent + incoherent if (coherent + incoherent) > 0 else 15

            if coherent >= 10:
                assessment = "✅ Good (likely usable)"
            elif coherent >= 5:
                assessment = "⚠️ Mixed (needs inspection)"
            else:
                assessment = "❌ Poor (unintelligible)"

            report += f"| {model_name:30} | {coherent:2}/{total} | {incoherent:2}/{total} | {avg_score:.2f} | {assessment} |\n"

    report += """
### Tier C Baseline Results (Devanagari vs ITRANS)

| Model | Devanagari | ITRANS | Avg Dv | Avg IT | ROM Issues | Assessment |
|---|---|---|---|---|---|---|
"""

    if tier_c_summary:
        for model_id, stats in tier_c_summary.items():
            model_name = model_id.split("/")[-1] if "/" in model_id else model_id
            dv_coherent = stats.get("devanagari_coherent", 0)
            it_coherent = stats.get("itrans_coherent", 0)
            dv_avg = stats.get("devanagari_avg_score", 0.0)
            it_avg = stats.get("itrans_avg_score", 0.0)
            rom_issues = stats.get("romanization_issues", 0)
            total = 15

            if it_coherent < 5 and dv_coherent >= 10:
                assessment = "✅ ITRANS-specific failure (romanization issue)"
            elif it_coherent < 5 and dv_coherent < 5:
                assessment = "❌ General Hindi failure"
            elif it_coherent >= 10 and dv_coherent >= 10:
                assessment = "✅ Both scripts understood"
            else:
                assessment = "⚠️ Mixed"

            report += f"| {model_name:30} | {dv_coherent:2}/{total} | {it_coherent:2}/{total} | {dv_avg:.2f} | {it_avg:.2f} | {rom_issues:2} | {assessment} |\n"

    report += """
### Tier E Baseline Results (English vs Leetspeak)

| Model | English | Leetspeak | Avg EN | Avg LEET | LEET Issues | Assessment |
|---|---|---|---|---|---|---|
"""

    if tier_e_summary:
        for model_id, stats in tier_e_summary.items():
            model_name = model_id.split("/")[-1] if "/" in model_id else model_id
            en_coherent = stats.get("english_coherent", 0)
            leet_coherent = stats.get("leetspeak_coherent", 0)
            en_avg = stats.get("english_avg_score", 0.0)
            leet_avg = stats.get("leetspeak_avg_score", 0.0)
            leet_issues = stats.get("leetspeak_issues", 0)
            total = 15

            if leet_coherent < 5 and en_coherent >= 10:
                assessment = "✅ LEETSPEAK-specific failure (within-language orthographic anchoring)"
            elif leet_coherent < 5 and en_coherent < 5:
                assessment = "❌ General English failure (unexpected)"
            elif leet_coherent >= 10 and en_coherent >= 10:
                assessment = "⚠️ Both intelligible (no Leetspeak anchoring)"
            else:
                assessment = "⚠️ Mixed"

            report += f"| {model_name:30} | {en_coherent:2}/{total} | {leet_coherent:2}/{total} | {en_avg:.2f} | {leet_avg:.2f} | {leet_issues:2} | {assessment} |\n"

    report += """
## Interpretation Guide

### Arabizi Results
- **✅ 10+/15**: Model understands Arabizi reasonably well
- **⚠️ 5-9/15**: Mixed; check manually for patterns
- **❌ <5/15**: Model produces gibberish on Arabizi (unintelligible)

### Tier C Results
- **Devanagari good, ITRANS bad**: Evidence for romanization-specific failure
- **Both bad**: General Hindi understanding issue
- **Both good**: Model handles both scripts equally (best case)

### Tier E Results
- **English good, Leetspeak bad**: Within-language orthographic anchoring (cleanest evidence for surface-level refusal mechanism)
- **Both intelligible**: Model robust to within-language perturbation
- **Asymmetric refusal rates with both intelligible**: H2-style intent-anchored vs script-anchored distinction

## Next Steps

1. Review baseline responses in this run folder
2. Run Phase 2 direction extraction on the selected model(s)
"""

    with open(report_file, "w") as f:
        f.write(report)

    log.info(f"✓ Report saved to {report_file}")
    return report_file


def update_latest_pointer(run_dir: Path) -> None:
    """Write a pointer file at the base directory pointing to the latest run."""
    latest_file = BASE_OUT_DIR / "LATEST.txt"
    latest_file.write_text(f"{run_dir.name}\n")
    log.info(f"✓ Latest pointer updated: {latest_file} → {run_dir.name}")


def main():
    parser = argparse.ArgumentParser(description="Run unified diagnostics across models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Qwen/Qwen3-8B",
            "google/gemma-2-9b-it",
            "meta-llama/Llama-3.1-8B-Instruct",
        ],
        help="Model IDs to test",
    )
    parser.add_argument("--n-prompts", type=int, default=15, help="Number of prompts to test")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory. If not provided, creates run_YYYYMMDD_HHMMSS/ subfolder.",
    )
    parser.add_argument(
        "--use-system-prompt",
        action="store_true",
        help=(
            "Forward --use-system-prompt to both child diagnostics. When on, "
            "Arabizi gets the arabizi pedagogy prompt and ITRANS gets the "
            "itrans pedagogy prompt; Devanagari (canonical) stays un-primed. "
            "Outputs land in suffixed filenames so they never overwrite the "
            "un-primed baseline runs used by downstream aggregation."
        ),
    )
    parser.add_argument(
        "--system-prompt-variant",
        choices=AVAILABLE_VARIANTS,
        default=DEFAULT_VARIANT,
        help=(
            "Which system-prompt variant to use when --use-system-prompt "
            f"is set. Available: {AVAILABLE_VARIANTS} (default: "
            f"{DEFAULT_VARIANT!r}). Forwarded verbatim to both child scripts."
        ),
    )
    parser.add_argument(
        "--input-file-arabizi",
        type=str,
        default=None,
        help=(
            "Override the Arabizi input JSONL forwarded to 00_diagnose_arabizi_baseline.py. "
            "Default: data/tier_a/tier_a_arabizi.jsonl. Use "
            "data/tier_b/benign_arabizi.jsonl as a paired benign control."
        ),
    )
    parser.add_argument(
        "--input-file-devanagari",
        type=str,
        default=None,
        help=(
            "Override the Devanagari input JSONL forwarded to 01_diagnose_tier_c_baseline.py. "
            "Default: data/tier_c/tier_c_devanagari.jsonl. Pair with --input-file-itrans."
        ),
    )
    parser.add_argument(
        "--input-file-itrans",
        type=str,
        default=None,
        help=(
            "Override the ITRANS input JSONL forwarded to 01_diagnose_tier_c_baseline.py. "
            "Default: data/tier_c/tier_c_itrans.jsonl. Pair with --input-file-devanagari."
        ),
    )
    parser.add_argument(
        "--input-file-english",
        type=str,
        default=None,
        help=(
            "Override the English input JSONL forwarded to 02_diagnose_tier_e_baseline.py. "
            "Default: data/tier_e/tier_e_english.jsonl. Pair with --input-file-leetspeak."
        ),
    )
    parser.add_argument(
        "--input-file-leetspeak",
        type=str,
        default=None,
        help=(
            "Override the Leetspeak input JSONL forwarded to 02_diagnose_tier_e_baseline.py. "
            "Default: data/tier_e/tier_e_leetspeak.jsonl. Pair with --input-file-english."
        ),
    )
    parser.add_argument(
        "--skip-tier-e",
        action="store_true",
        help=(
            "Skip the Tier E (English ↔ Leetspeak) diagnostic. Useful when "
            "Tier E corpus is not built or when only Arabic + Hindi runs "
            "are needed. The Arabizi and Tier C diagnostics always run."
        ),
    )
    parser.add_argument(
        "--prompt-range",
        type=str,
        default=None,
        help=(
            "Test a specific 1-indexed inclusive prompt range instead of the "
            "default first 15. Forwarded verbatim to BOTH child scripts so "
            "the same slice is used for Arabizi and Tier C runs. Format: "
            "'START-END' (e.g. '100-115' for 16 prompts)."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Override the model config's `enable_thinking` chat-template kwarg "
            "(Qwen3 only — ignored by non-thinking models). Forwarded verbatim "
            "to both child scripts. 'on' should be paired with "
            "--max-new-tokens ≥1024 or the <think> block will truncate."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Override MAX_NEW_TOKENS for response generation. Forwarded "
            "verbatim to both child scripts. Recommended ≥1024 when "
            "--enable-thinking on."
        ),
    )
    args = parser.parse_args()

    # Create timestamp for this test run
    run_timestamp = datetime.datetime.now().isoformat()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve run output directory
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = BASE_OUT_DIR / f"run_{stamp}"

    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("="*80)
    log.info("MULTILINGUAL MODEL DIAGNOSTICS")
    log.info("="*80)
    log.info(f"Test run started at: {run_timestamp}")
    log.info(f"Run folder: {run_dir}")
    log.info(f"Models: {args.models}")
    log.info(f"Prompts per test: {args.n_prompts}")
    if args.use_system_prompt:
        log.info(
            f"System-prompt injection ENABLED, variant={args.system_prompt_variant!r}"
        )

    # Save run metadata
    metadata = {
        "run_timestamp": run_timestamp,
        "run_stamp": stamp,
        "models": args.models,
        "n_prompts": args.n_prompts,
        "use_system_prompt": args.use_system_prompt,
        "system_prompt_variant": args.system_prompt_variant if args.use_system_prompt else None,
        "input_file_arabizi": args.input_file_arabizi,
        "input_file_devanagari": args.input_file_devanagari,
        "input_file_itrans": args.input_file_itrans,
        "input_file_english": args.input_file_english,
        "input_file_leetspeak": args.input_file_leetspeak,
        "skip_tier_e": args.skip_tier_e,
        "prompt_range": args.prompt_range,
        "enable_thinking": args.enable_thinking,
        "max_new_tokens": args.max_new_tokens,
        "run_dir": str(run_dir),
    }
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Child-script flags shared by both diagnostics. The sysprompt suffix is
    # mirrored here so we can find the resulting summary files; the children
    # produce e.g. arabizi_summary_sysprompt_oneshot.json when the flag is on.
    sysprompt_args: list[str] = []
    if args.use_system_prompt:
        sysprompt_args = [
            "--use-system-prompt",
            "--system-prompt-variant", args.system_prompt_variant,
        ]
    summary_suffix = (
        f"_sysprompt_{args.system_prompt_variant}" if args.use_system_prompt else ""
    )
    if args.prompt_range:
        summary_suffix += f"_p{args.prompt_range.replace('-', '-')}"

    if (args.input_file_arabizi or args.input_file_devanagari or args.input_file_itrans or
        args.input_file_english or args.input_file_leetspeak):
        log.warning(
            "⚠️  --input-file-* override is set. The comparison report will "
            "look for default-prefix summary files (arabizi_summary*, "
            "tier_c_summary*, tier_e_summary*) but child scripts will write "
            "to alternate prefixes (e.g. benign_arabizi_summary*, "
            "benign_tier_c_summary*, benign_tier_e_summary*). "
            "The DIAGNOSTIC_REPORT.md may be empty; inspect per-model JSONs directly."
        )

    # --prompt-range forwarded verbatim to both children.
    prompt_range_args: list[str] = (
        ["--prompt-range", args.prompt_range] if args.prompt_range else []
    )

    # --enable-thinking and --max-new-tokens forwarded verbatim to both children.
    thinking_args: list[str] = []
    if args.enable_thinking != "auto":
        thinking_args += ["--enable-thinking", args.enable_thinking]
    if args.max_new_tokens is not None:
        thinking_args += ["--max-new-tokens", str(args.max_new_tokens)]

    # Tag the summary_suffix so the comparison-report helper finds the right
    # aggregate-summary filenames (children's filenames carry the same tags).
    if args.enable_thinking == "on":
        summary_suffix += "_thinkON"
    elif args.enable_thinking == "off":
        summary_suffix += "_thinkOFF"
    if args.max_new_tokens is not None:
        summary_suffix += f"_max{args.max_new_tokens}"

    # Per-script input-file overrides. Each child takes its own flag(s):
    #   00_  takes --input-file (singular, Arabizi)
    #   01_  takes --input-file-devanagari + --input-file-itrans (paired)
    #   02_ takes --input-file-english + --input-file-leetspeak (paired)
    arabizi_input_args: list[str] = (
        ["--input-file", args.input_file_arabizi] if args.input_file_arabizi else []
    )
    tier_c_input_args: list[str] = []
    if args.input_file_devanagari:
        tier_c_input_args += ["--input-file-devanagari", args.input_file_devanagari]
    if args.input_file_itrans:
        tier_c_input_args += ["--input-file-itrans", args.input_file_itrans]
    tier_e_input_args: list[str] = []
    if args.input_file_english:
        tier_e_input_args += ["--input-file-english", args.input_file_english]
    if args.input_file_leetspeak:
        tier_e_input_args += ["--input-file-leetspeak", args.input_file_leetspeak]

    # Run Arabizi diagnostics
    arabizi_cmd = [
        sys.executable,
        str(REPO / "scripts" / "phase4_causal_intervention" / "diagnostics" / "00_diagnose_arabizi_baseline.py"),
        "--output-dir", str(run_dir),
        *sysprompt_args,
        *arabizi_input_args,
        *prompt_range_args,
        *thinking_args,
        "--models",
        *args.models,
    ]
    run_command(arabizi_cmd, "Arabizi baseline diagnostics")

    # Run Tier C diagnostics
    tier_c_cmd = [
        sys.executable,
        str(REPO / "scripts" / "phase4_causal_intervention" / "diagnostics" / "01_diagnose_tier_c_baseline.py"),
        "--output-dir", str(run_dir),
        *sysprompt_args,
        *tier_c_input_args,
        *prompt_range_args,
        *thinking_args,
        "--models",
        *args.models,
    ]
    run_command(tier_c_cmd, "Tier C baseline diagnostics")

    # Run Tier E diagnostics (unless skipped)
    if not args.skip_tier_e:
        tier_e_cmd = [
            sys.executable,
            str(REPO / "scripts" / "phase4_causal_intervention" / "diagnostics"
                / "02_diagnose_tier_e_baseline.py"),
            "--output-dir", str(run_dir),
            *sysprompt_args,
            *tier_e_input_args,
            *prompt_range_args,
            *thinking_args,
            "--models",
            *args.models,
        ]
        run_command(tier_e_cmd, "Tier E (English ↔ Leetspeak) baseline diagnostics")
    else:
        log.info("Skipping Tier E diagnostics (--skip-tier-e)")

    # Generate report
    log.info("\n" + "="*80)
    log.info("GENERATING COMPARISON REPORT")
    log.info("="*80)
    report_file = generate_comparison_report(
        args.models, run_timestamp, run_dir, summary_suffix=summary_suffix,
    )

    # Update latest pointer
    update_latest_pointer(run_dir)

    # Build the additive per-model reviewer view (non-breaking; the flat run_dir
    # above stays the aggregator's source of truth). See reorg_per_model.py.
    try:
        from reorg_per_model import reorganize_run
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from reorg_per_model import reorganize_run
    per_model_dirs = reorganize_run(run_dir, metadata)
    for d in per_model_dirs:
        log.info(f"✓ Per-model view: {d.relative_to(REPO)}")

    log.info("\n" + "="*80)
    log.info("DIAGNOSTICS COMPLETE")
    log.info("="*80)
    log.info(f"\nRun folder: {run_dir}")
    log.info(f"Report: {report_file}")
    log.info("\nFiles in run folder:")
    log.info("  - run_metadata.json")
    log.info("  - arabizi_baseline_*.json")
    log.info("  - arabizi_summary.json")
    log.info("  - tier_c_baseline_*.json")
    log.info("  - tier_c_summary.json")
    if not args.skip_tier_e:
        log.info("  - tier_e_baseline_*.json")
        log.info("  - tier_e_summary.json")
    log.info("  - DIAGNOSTIC_REPORT.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
