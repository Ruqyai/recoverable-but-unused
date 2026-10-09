#!/usr/bin/env python3
"""Per-model reviewer-facing view of a diagnostic run (added 2026-06-11).

The diagnostic runner writes a flat, timestamp-first `run_<stamp>/` folder whose
filenames embed the model + config suffix. That folder stays untouched (it is
what downstream aggregation reads from). This module
builds an ADDITIVE per-model view that mirrors the rest of the repo's layout
(`<phase>/<model_key>/run_<stamp>/`), so a reviewer finds the latest run for a
model the same way as Phase 2/3/5/H2:

    results/phase4_diagnostic/<model_key>/
        LATEST.txt                         ← latest run_<stamp> for this model
        run_<stamp>/
            run_config.json                ← FULL metadata (think on/off, variant,
                                              n, date, input files, think_status)
            arabizi/   (Tier A)            ← per-tier subfolder
                baseline_<model_key>.json
                summary.json               ← this model's slice + run metadata
            tier_c/    (Tier C)
            tier_e/    (Tier E)

Copies, never moves — the flat folder remains the aggregator's source of truth,
so this is non-breaking. Old flat runs that predate this module simply have no
per-model view (regenerate with `--reorg-only` if ever needed).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BASE_OUT_DIR = REPO / "results" / "phase4_diagnostic"

# flat-file prefix → {data-tier folder name, lang/script token for the filename}.
# Folder is the DATA TIER (tier_a/tier_c/tier_e), not the script, for consistency
# with the rest of the repo's tier vocabulary.
TIERS = {
    "arabizi": {"folder": "tier_a", "langs": "arabizi"},            # Tier A: Arabic ↔ Arabizi
    "tier_c":  {"folder": "tier_c", "langs": "devanagari_itrans"},  # Tier C: Devanagari ↔ ITRANS
    "tier_e":  {"folder": "tier_e", "langs": "english_leetspeak"},  # Tier E: English ↔ Leetspeak
}


def _resolve_think(metadata: dict, run_dir: Path, slug: str) -> str:
    """Resolve the EFFECTIVE think state → 'ON' or 'OFF' (never 'AUTO').
    --enable-thinking on/off is authoritative; 'auto' is decided from the
    actual generations' think_status (a real <think> block ⇒ ON)."""
    et = (metadata.get("enable_thinking") or "auto").lower()
    if et == "on":
        return "ON"
    if et == "off":
        return "OFF"
    # auto → inspect this model's baselines
    for b in run_dir.glob(f"*_baseline*_{slug}.json"):
        if ".rescored" in b.name:
            continue
        ts = _think_status_summary(b)
        if ts.get("closed", 0) or ts.get("truncated", 0):
            return "ON"
        if ts.get("no_think", 0):
            return "OFF"
    return "OFF"


DEFAULT_N_PROMPTS = 15


def _arg_signature(metadata: dict, think_label: str) -> str:
    """Compact arg signature in each filename — ONLY non-default settings appear.
    Defaults (thinkOFF, no system prompt, n=15, no max/range override) are
    omitted entirely. Returns '' when everything is default, so the filename
    carries no suffix. Examples: '' (all default), 'thinkON',
    'sysprompt_translit_first_thinkON_max8192_p100-115'."""
    parts = []
    if metadata.get("use_system_prompt"):
        parts.append(f"sysprompt_{metadata.get('system_prompt_variant')}")
    if think_label == "ON":                       # thinkOFF is default → hidden
        parts.append("thinkON")
    if metadata.get("max_new_tokens"):
        parts.append(f"max{metadata['max_new_tokens']}")
    if metadata.get("prompt_range"):
        parts.append(f"p{metadata['prompt_range']}")
    n = metadata.get("n_prompts")
    if n not in (None, DEFAULT_N_PROMPTS):
        parts.append(f"n{n}")
    return "_".join(parts)


def _model_key_map() -> dict[str, str]:
    """model_id → registry model_key (fallback: slug of the id)."""
    try:
        from scripts.models import _REGISTRY, load_model_config
        out = {}
        for k in _REGISTRY:
            out[load_model_config(k).MODEL_ID] = load_model_config(k).MODEL_KEY
        return out
    except Exception:
        return {}


def _slug(model_id: str) -> str:
    """Match the child scripts' filename token: last path part, '.'→'-'."""
    return model_id.split("/")[-1].replace(".", "-")


def _think_status_summary(baseline_path: Path) -> dict[str, int]:
    """Count think_status values across a baseline file's responses."""
    counts: dict[str, int] = {}
    try:
        data = json.loads(baseline_path.read_text())
    except Exception:
        return counts

    def scan(obj):
        if isinstance(obj, dict):
            ts = obj.get("think_status")
            if isinstance(ts, str):
                counts[ts] = counts.get(ts, 0) + 1
            for v in obj.values():
                scan(v)
        elif isinstance(obj, list):
            for v in obj:
                scan(v)

    scan(data)
    return counts


def reorganize_run(run_dir: Path, metadata: dict) -> list[Path]:
    """Build the per-model view for one flat run_dir. Returns the per-model run dirs created."""
    run_dir = Path(run_dir)
    stamp = metadata.get("run_stamp") or run_dir.name.replace("run_", "")
    key_map = _model_key_map()
    created: list[Path] = []

    for model_id in metadata.get("models", []):
        model_key = key_map.get(model_id) or _slug(model_id).replace("-", "_")
        slug = _slug(model_id)
        # per-model arg signature, with think resolved to ON/OFF from this model's data
        think_label = _resolve_think(metadata, run_dir, slug)
        arg_sig = _arg_signature(metadata, think_label)
        per_model_run = BASE_OUT_DIR / model_key / f"run_{stamp}"
        if per_model_run.exists():
            shutil.rmtree(per_model_run)  # idempotent re-reorg

        think_summary_all: dict[str, int] = {}
        tiers_present: list[str] = []

        for prefix, spec in TIERS.items():
            folder, langs = spec["folder"], spec["langs"]
            # flat baseline files for this model+tier (any config suffix), excluding rescored dupes
            baselines = sorted(
                p for p in run_dir.glob(f"{prefix}_baseline*_{slug}.json")
                if ".rescored" not in p.name
            )
            if not baselines:
                continue
            tiers_present.append(folder)
            tier_dir = per_model_run / folder
            tier_dir.mkdir(parents=True, exist_ok=True)

            # copy baseline(s) → <model>_<tier>_<langs>[_<argmeta>].json
            # (argmeta omitted entirely when all args are default)
            suffix = f"_{arg_sig}" if arg_sig else ""
            fname = f"{model_key}_{folder}_{langs}{suffix}.json"
            for b in baselines:
                shutil.copy2(b, tier_dir / fname)
                for k, v in _think_status_summary(b).items():
                    think_summary_all[k] = think_summary_all.get(k, 0) + v

            # this model's slice of the tier summary
            summaries = sorted(run_dir.glob(f"{prefix}_summary*.json"))
            slice_stats = None
            for s in summaries:
                if ".rescored" in s.name:
                    continue
                try:
                    sd = json.loads(s.read_text())
                except Exception:
                    continue
                mr = sd.get("model_results", {})
                if model_id in mr:
                    slice_stats = mr[model_id]
                    break
            (tier_dir / "summary.json").write_text(json.dumps({
                "tier": folder,
                "langs": langs,
                "model_id": model_id,
                "model_key": model_key,
                "run_stamp": stamp,
                "test_date": metadata.get("run_timestamp"),
                "stats": slice_stats,
            }, indent=2, ensure_ascii=False))

        if not tiers_present:
            continue

        # full per-model run_config
        run_config = {
            "model_id": model_id,
            "model_key": model_key,
            "run_stamp": stamp,
            "test_date": metadata.get("run_timestamp"),
            "tiers": tiers_present,
            "n_prompts": metadata.get("n_prompts"),
            "use_system_prompt": metadata.get("use_system_prompt"),
            "system_prompt_variant": metadata.get("system_prompt_variant"),
            "enable_thinking": metadata.get("enable_thinking"),
            "enable_thinking_effective": think_label,  # resolved ON/OFF (used in filenames)
            "max_new_tokens": metadata.get("max_new_tokens"),
            "prompt_range": metadata.get("prompt_range"),
            "input_files": {
                "arabizi": metadata.get("input_file_arabizi"),
                "devanagari": metadata.get("input_file_devanagari"),
                "itrans": metadata.get("input_file_itrans"),
                "english": metadata.get("input_file_english"),
                "leetspeak": metadata.get("input_file_leetspeak"),
            },
            "think_status_summary": think_summary_all,
            "flat_run_dir": str(run_dir),
        }
        (per_model_run / "run_config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False)
        )
        (BASE_OUT_DIR / model_key / "LATEST.txt").write_text(f"run_{stamp}\n")
        created.append(per_model_run)

    return created


def _known_slugs() -> dict[str, tuple[str, str]]:
    """filename slug → (model_key, model_id), e.g. 'Qwen3-4B' → ('qwen3_4b', 'Qwen/Qwen3-4B')."""
    out: dict[str, tuple[str, str]] = {}
    try:
        from scripts.models import _REGISTRY, load_model_config
        for k in _REGISTRY:
            c = load_model_config(k)
            out[_slug(c.MODEL_ID)] = (c.MODEL_KEY, c.MODEL_ID)
    except Exception:
        pass
    return out


def _normalize_tag(tags: str) -> str:
    """Apply the new naming rule to a legacy flat-filename tag: drop default
    think tokens (thinkOFF/thinkAUTO) so only non-default settings remain."""
    parts = [p for p in tags.split("_") if p and p not in ("thinkOFF", "thinkAUTO")]
    return "_".join(parts)


def _stamp_sort_key(run_name: str) -> str:
    """Sort key from the leading YYYYMMDD_HHMMSS of a run folder name."""
    rest = run_name[len("run_"):] if run_name.startswith("run_") else run_name
    bits = rest.split("_")
    return "_".join(bits[:2]) if len(bits) >= 2 else rest


def _backfill_from_filenames(run_dir: Path) -> list[str]:
    """Per-model view for a flat run WITHOUT usable run_metadata: parse
    `{prefix}_baseline_{tags}_{slug}.json` filenames directly. Returns the
    model_keys touched. Copies (archive stays the raw source)."""
    slugs = _known_slugs()
    stamp = run_dir.name[len("run_"):] if run_dir.name.startswith("run_") else run_dir.name
    meta = {}
    mp = run_dir / "run_metadata.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            meta = {}
    touched: set[str] = set()

    for prefix, spec in TIERS.items():
        folder, langs = spec["folder"], spec["langs"]
        for f in run_dir.glob(f"{prefix}_baseline*.json"):
            if any(x in f.name for x in (".rescored", ".harm_audited", ".partial")):
                continue
            stem = f.name[:-5]                       # strip .json
            rest = stem[len(prefix) + len("_baseline"):].lstrip("_")  # "{tags}_{slug}"
            match = next(
                ((s, mk, mid) for s, (mk, mid) in slugs.items()
                 if rest == s or rest.endswith("_" + s)),
                None,
            )
            if not match:
                continue                              # aggregate file (e.g. *_15) — skip
            slug, model_key, model_id = match
            tags = _normalize_tag(rest[: -len(slug)].rstrip("_") if rest != slug else "")
            tier_dir = BASE_OUT_DIR / model_key / f"run_{stamp}" / folder
            tier_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_{tags}" if tags else ""
            shutil.copy2(f, tier_dir / f"{model_key}_{folder}_{langs}{suffix}.json")
            touched.add(model_key)

    # minimal run_config per touched model (archive backfill: filename-derived)
    for model_key in touched:
        slug_for = next((s for s, (mk, _mid) in slugs.items() if mk == model_key), None)
        mid = slugs.get(slug_for, (model_key, model_key))[1] if slug_for else model_key
        (BASE_OUT_DIR / model_key / f"run_{stamp}" / "run_config.json").write_text(
            json.dumps({
                "model_id": mid,
                "model_key": model_key,
                "run_stamp": stamp,
                "source": "archive_backfill_from_filenames",
                "flat_run_dir": str(run_dir),
                "run_metadata": meta or None,
            }, indent=2, ensure_ascii=False)
        )
    return sorted(touched)


def _fix_latest_pointers() -> None:
    """Point each model's LATEST.txt at its newest run_* (by leading timestamp)."""
    for model_dir in BASE_OUT_DIR.iterdir():
        if not model_dir.is_dir() or model_dir.name in ("archive",):
            continue
        runs = [p.name for p in model_dir.glob("run_*") if p.is_dir()]
        if runs:
            newest = max(runs, key=_stamp_sort_key)
            (model_dir / "LATEST.txt").write_text(f"{newest}\n")


def reorg_archive(archive_dir: Path | None = None) -> dict:
    """Backfill the per-model view for EVERY flat run under archive/ (same style
    as live runs). Metadata-bearing runs use the rich path; the rest are parsed
    from filenames. Copies only — archive stays the raw aggregator source."""
    archive_dir = Path(archive_dir) if archive_dir else (BASE_OUT_DIR / "archive")
    runs = sorted(
        (p for p in archive_dir.glob("run_*") if p.is_dir()),
        key=lambda p: _stamp_sort_key(p.name),
    )
    summary = {"runs": 0, "rich": 0, "filename": 0, "models": set()}
    for run_dir in runs:
        summary["runs"] += 1
        mp = run_dir / "run_metadata.json"
        meta = {}
        if mp.exists():
            try:
                meta = json.loads(mp.read_text())
            except Exception:
                meta = {}
        if meta.get("models"):
            meta.setdefault("run_stamp", run_dir.name[len("run_"):])
            created = reorganize_run(run_dir, meta)
            summary["rich"] += 1
            summary["models"].update(p.parent.name for p in created)
        else:
            touched = _backfill_from_filenames(run_dir)
            summary["filename"] += 1
            summary["models"].update(touched)
    _fix_latest_pointers()
    summary["models"] = sorted(summary["models"])
    return summary


def reorg_from_flat(flat_run_dir: Path) -> list[Path]:
    """Re-build the per-model view from an existing flat run_dir (reads its
    run_metadata.json). No model generation — cheap, idempotent."""
    flat_run_dir = Path(flat_run_dir)
    meta_path = flat_run_dir / "run_metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if not metadata.get("models"):
        # infer models from baseline filenames if metadata is absent
        slugs = {p.name.split("_baseline")[0] for p in flat_run_dir.glob("*_baseline*.json")}
        raise SystemExit(f"run_metadata.json missing/incomplete in {flat_run_dir}; "
                         f"cannot infer model ids (found prefixes: {slugs})")
    return reorganize_run(flat_run_dir, metadata)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args == ["--archive"]:
        s = reorg_archive()
        print(f"✓ archive backfill: {s['runs']} runs "
              f"({s['rich']} via metadata, {s['filename']} via filenames)")
        print(f"  models: {', '.join(s['models'])}")
    elif len(args) == 1 and not args[0].startswith("-"):
        out = reorg_from_flat(Path(args[0]))
        for d in out:
            print(f"✓ {d}")
    else:
        print("usage: reorg_per_model.py <flat_run_dir>   # rebuild one flat run")
        print("       reorg_per_model.py --archive         # backfill all archive/run_* "
              "into per-model views")
        raise SystemExit(2)
