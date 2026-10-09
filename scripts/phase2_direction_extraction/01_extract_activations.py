#!/usr/bin/env python3
"""Phase 2 Stage 1: extract residual-stream activations for the six v6 scripts.

Per-model architecture constants come from the registry in
`scripts/models/`. Outputs are written to
`results/phase2/<model_key>/run_YYYYMMDD_HHMMSS/activations/...`.

Usage:
    PYTHONPATH=. python3 scripts/phase2_direction_extraction/01_extract_activations.py \
        --model qwen3_4b
    PYTHONPATH=. python3 scripts/phase2_direction_extraction/01_extract_activations.py \
        --model gemma_3_1b_it --scripts arabic arabizi
    PYTHONPATH=. python3 scripts/phase2_direction_extraction/01_extract_activations.py \
        --model qwen3_4b --run-dir results/phase2/qwen3_4b/run_20260526_140530  # resume
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.models import (
    DEFAULT_MODEL_KEY,
    list_models,
    load_model_config,
)
from scripts.phase2_direction_extraction.config import (
    BENIGN_FILES,
    HARMFUL_FILES,
    SCRIPTS,
    resolve_run_paths,
    write_latest_pointer,
)
from scripts.utils.device import (
    DeviceConfig,
    detect_device,
    get_text_base_model,
    load_hf_model,
    log_device_info,
    resolve_dtype,
)
from scripts.utils.io import read_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_model_and_tokenizer(model_id: str, dev: DeviceConfig, dtype=None, model_class=None):
    model = load_hf_model(
        model_id, dev, hf_token=os.environ.get("HF_TOKEN"), dtype=dtype,
        model_class=model_class,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def format_prompt(tokenizer, prompt: str, fallback_template: str) -> str:
    """Prefer tokenizer.apply_chat_template; fall back to manual template."""
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception as e:
        log.debug(f"apply_chat_template failed ({e!r}); using manual template")
    return fallback_template.format(prompt=prompt)


def is_done(out_path: Path) -> bool:
    return out_path.exists() and out_path.stat().st_size > 0


def process_file(
    model,
    tokenizer,
    data_path: Path,
    out_dir: Path,
    dev: DeviceConfig,
    *,
    batch_size: int,
    n_layers: int,
    chat_template: str,
    is_multimodal: bool = False,
    text_submodule_path: str | None = None,
) -> int:
    records = list(read_jsonl(data_path))
    to_process = [r for r in records if not is_done(out_dir / f"{r.id}.pt")]

    if not to_process:
        log.info(f"All records in {data_path.name} already processed")
        return 0

    log.info(f"Processing {len(to_process)}/{len(records)} records from {data_path.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 only ever reads hidden_states, never logits. Resolve the base
    # transformer once so the forward below skips the lm_head projection over the
    # full vocabulary (~151k tokens for Qwen3) — that projection otherwise
    # allocates a multi-GB logits tensor and OOMs MPS on longer inputs (e.g.
    # Devanagari). hidden_states from the base model are identical to those the
    # full *ForCausalLM returns, so previously-extracted activations stay valid.
    # Multimodal VLMs (Gemma-3): extract from the language submodule instead.
    base_model = get_text_base_model(
        model, is_multimodal=is_multimodal, submodule_path=text_submodule_path
    )

    processed = 0
    for batch_idx in tqdm(
        range(0, len(to_process), batch_size),
        desc=f"Batches: {data_path.name}",
        unit="batch",
    ):
        batch = to_process[batch_idx : batch_idx + batch_size]
        texts = [format_prompt(tokenizer, r.prompt, chat_template) for r in batch]

        inputs = tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(dev.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = base_model(**inputs, output_hidden_states=True, use_cache=False)

        assert len(outputs.hidden_states) == n_layers + 1, (
            f"Expected {n_layers + 1} hidden states (embedding + {n_layers} layers), "
            f"got {len(outputs.hidden_states)}"
        )

        attention_mask = inputs["attention_mask"]
        for i, record in enumerate(batch):
            last_real_pos = attention_mask[i].sum() - 1

            acts = []
            for layer_idx in range(1, n_layers + 1):
                hidden = outputs.hidden_states[layer_idx]
                h = hidden[i, last_real_pos, :].float().cpu()
                acts.append(h)

            act_tensor = torch.stack(acts, dim=0)

            out_path = out_dir / f"{record.id}.pt"
            torch.save(act_tensor, out_path)

            processed += 1
            if processed % 20 == 0:
                log.info(f"Processed {processed} records from {data_path.name}")

        dev.empty_cache()

    return processed


def write_run_metadata(
    cfg,
    run_paths,
    dev: DeviceConfig,
    batch_size: int,
    args_dict: dict,
) -> None:
    payload = {
        "phase": "phase2_stage1_extract_activations",
        "model_key": cfg.MODEL_KEY,
        "model_id": cfg.MODEL_ID,
        "n_layers": cfg.N_LAYERS,
        "hidden_dim": cfg.HIDDEN_DIM,
        "model_default_batch_size": cfg.BATCH_SIZE,
        "effective_batch_size": batch_size,
        "measurement_position": cfg.MEASUREMENT_POSITION,
        "device": dev.to_dict(),
        "run_stamp": run_paths.run_stamp,
        "run_dir": str(run_paths.run_dir),
        "started_at": datetime.datetime.now().isoformat(),
        "args": args_dict,
        "notes": cfg.NOTES,
    }
    run_paths.run_metadata_path.write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Extract activations for Phase 2")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_KEY,
        choices=list_models(),
        help=f"Model key (default: {DEFAULT_MODEL_KEY}). One of: {list_models()}",
    )
    parser.add_argument(
        "--scripts",
        type=str,
        nargs="+",
        default=SCRIPTS,
        help="Scripts to process (default: all six v6 scripts)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing run_YYYYMMDD_HHMMSS/ to resume. Mutually exclusive with --new-run.",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Force a fresh timestamped run folder (default: reuse LATEST if exists).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-model BATCH_SIZE (skips device-aware multiplier).",
    )
    parser.add_argument(
        "--prefer-device",
        type=str,
        default=None,
        choices=["cuda", "xla", "mps", "cpu"],
        help="Force a specific device. Default: auto-detect (CUDA > XLA > MPS > CPU).",
    )
    args = parser.parse_args()

    if args.run_dir and args.new_run:
        parser.error("--run-dir and --new-run are mutually exclusive")

    cfg = load_model_config(args.model)
    run_paths = resolve_run_paths(
        model_key=cfg.MODEL_KEY,
        run_dir=args.run_dir,
        new_run=args.new_run,
    )

    dev = detect_device(prefer=args.prefer_device)
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = dev.apply_batch_size(cfg.BATCH_SIZE)

    log.info("=" * 70)
    log.info("Phase 2 Stage 1 — Extract Activations")
    log.info(f"  model         : {cfg.MODEL_ID}  (key={cfg.MODEL_KEY})")
    log.info(f"  layers        : {cfg.N_LAYERS}")
    log.info(f"  hidden dim    : {cfg.HIDDEN_DIM}")
    log.info(f"  model default : batch_size={cfg.BATCH_SIZE}")
    log.info(f"  effective     : batch_size={batch_size}")
    log.info(f"  scripts       : {args.scripts}")
    log.info(f"  run dir       : {run_paths.run_dir}")
    log.info("=" * 70)
    log_device_info(dev)

    write_run_metadata(
        cfg=cfg,
        run_paths=run_paths,
        dev=dev,
        batch_size=batch_size,
        args_dict={
            "scripts": args.scripts,
            "batch_size": batch_size,
            "resume": bool(args.run_dir),
            "new_run": bool(args.new_run),
            "prefer_device": args.prefer_device,
        },
    )

    pref_dtype = resolve_dtype(getattr(cfg, "PREFERRED_DTYPE", None))
    is_multimodal = bool(getattr(cfg, "IS_MULTIMODAL", False))
    text_submodule_path = getattr(cfg, "TEXT_SUBMODULE_PATH", None)
    if pref_dtype is not None:
        log.info(f"  PREFERRED_DTYPE override: {pref_dtype}")
    if is_multimodal:
        log.info(f"  MULTIMODAL: extracting from language submodule ({text_submodule_path})")
    model, tokenizer = load_model_and_tokenizer(
        cfg.MODEL_ID, dev, dtype=pref_dtype,
        model_class=getattr(cfg, "MODEL_CLASS", None),
    )

    total_processed = 0
    for script in args.scripts:
        if script not in SCRIPTS:
            log.error(f"Unknown script: {script}")
            continue

        harmful_path = HARMFUL_FILES[script]
        benign_path = BENIGN_FILES[script]
        out_dir_harmful = run_paths.activations_dir / script / "harmful"
        out_dir_benign = run_paths.activations_dir / script / "benign"

        log.info(f"\n{'=' * 60}")
        log.info(f"Processing script: {script}")
        log.info(f"{'=' * 60}")

        n_harmful = process_file(
            model, tokenizer, harmful_path, out_dir_harmful, dev,
            batch_size=batch_size, n_layers=cfg.N_LAYERS,
            chat_template=cfg.CHAT_TEMPLATE,
            is_multimodal=is_multimodal, text_submodule_path=text_submodule_path,
        )
        n_benign = process_file(
            model, tokenizer, benign_path, out_dir_benign, dev,
            batch_size=batch_size, n_layers=cfg.N_LAYERS,
            chat_template=cfg.CHAT_TEMPLATE,
            is_multimodal=is_multimodal, text_submodule_path=text_submodule_path,
        )

        log.info(
            f"Completed {script}: {n_harmful} harmful, {n_benign} benign records"
        )
        total_processed += n_harmful + n_benign

    write_latest_pointer(cfg.MODEL_KEY, run_paths.run_dir)

    log.info(f"\nTotal processed: {total_processed} records")
    log.info(f"Run dir: {run_paths.run_dir}")
    log.info(f"LATEST.txt updated to: {run_paths.run_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
