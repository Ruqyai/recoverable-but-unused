#!/usr/bin/env python3
"""
02_logit_lens.py — Phase 4b (H6 companion): per-layer logit lens (RomanLens-style).

Companion to 01_activation_patching.py and the GATE on its interpretation. While
patching is *causal* (which activations matter), the logit lens is *descriptive*
(what is represented). The registered H6c decision rule cannot be applied without
it: BOTH the parse-bottleneck (P) and script-anchoring (S) verdicts require
evidence here that the harmful concept IS internally represented in the romanized
(Arabizi) run.

  - P needs: strong mid-layer patch recovery, matched>>random, AND concept present.
  - S needs: late/absent patch recovery DESPITE concept present.
  - If the concept is NOT represented at all, recovery being non-specific points to
    a parse failure / decision-state transfer, NOT script-anchoring.

It applies the model's final norm + unembedding (lm_head) to the **intermediate**
residual at LENS_POS (-1) for every layer, on the Arabizi (and, for contrast, the
Arabic) prompt, and reads the per-layer next-token distribution.

THREE PROBES (the P-vs-S triangulation, with 01)
  1. Latent romanization / shared semantics: do native-script or English-meaning
     tokens of the prompt's content appear in the mid-layer lens of the *Arabizi*
     run? (RomanLens's claim — the model internally decodes the romanization.)
  2. Harmful-concept emergence: at which layer do the prompt's English content
     tokens (from the AdvBench seed) first enter the lens top-k of the *Arabizi*
     run? present -> meaning IS formed (S-compatible); absent -> parse failure (P).
  3. Refusal-token emergence: at which layer do refusal tokens start to dominate,
     and does that layer differ between Arabic (early/strong) and Arabizi (late/weak)?

Descriptive, exploratory. Read trends across depth, not single-layer absolutes
(early layers are off-distribution for the unembedding — RomanLens caveat).

USAGE
  PYTHONPATH=. python3 scripts/phase4_causal_intervention/patching/02_logit_lens.py \
      --model qwen3_4b --pair tier_a --n-pairs 200 --prefer-device cuda
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.models import DEFAULT_MODEL_KEY, list_models, load_model_config
from scripts.utils.device import (
    detect_device,
    get_text_base_model,
    load_hf_model,
    log_device_info,
)
from scripts.utils.io import read_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
LENS_POS = -1

PAIRS = {
    "tier_a": ("arabic", "arabizi", REPO / "data/tier_a/tier_a_arabic.jsonl", REPO / "data/tier_a/tier_a_arabizi.jsonl"),
    # tier_c is now AdvBench-derived Hindi (advbench_* ids, the _advbench suffix was
    # dropped and the content merged into these plain filenames). It HAS an English
    # concept anchor in advbench_sample_en.jsonl, so the lens runs (200/200 coverage).
    "tier_c": ("devanagari", "itrans", REPO / "data/tier_c/tier_c_devanagari.jsonl", REPO / "data/tier_c/tier_c_itrans.jsonl"),
    "tier_e": ("english", "leetspeak", REPO / "data/tier_e/tier_e_english.jsonl", REPO / "data/tier_e/tier_e_leetspeak.jsonl"),
}

# English-meaning anchor for the harmful-concept probe (shared source_id with all tiers).
ENGLISH_SEED = REPO / "data/raw/advbench_sample_en.jsonl"

# short refusal-token cues for the "refusal emergence" probe (EN/AR)
REFUSAL_CUES = {"cannot", "sorry", "unable", "لا", "آسف", "عذر"}  # refusal-specific, whole-token

STOPWORDS = {
    "the", "a", "an", "to", "of", "for", "in", "on", "and", "or", "that", "this",
    "with", "how", "can", "you", "your", "i", "it", "is", "are", "be", "as", "at",
    "by", "from", "write", "create", "develop", "make", "give", "provide", "explain",
}


def format_prompt(tok, user_text, cfg):
    kw = dict(getattr(cfg, "CHAT_TEMPLATE_KWARGS", {}) or {})
    return tok.apply_chat_template([{"role": "user", "content": user_text}],
                                   tokenize=False, add_generation_prompt=True, **kw)


# Decoder-layer container, resolved once in main() (VLM-aware). Text models →
# model.model; multimodal VLMs (Gemma-3) → the language submodule. None until set.
_LAYER_BASE = None


def get_final_norm(model):
    """Final RMSNorm before lm_head — model.model.norm on Qwen/Gemma/Llama, or
    the language-submodule norm on multimodal VLMs (resolved via _LAYER_BASE)."""
    base = _LAYER_BASE if _LAYER_BASE is not None else getattr(model, "model", model)
    for attr in ("norm", "final_layernorm"):
        m = getattr(base, attr, None)
        if m is not None:
            return m
    return torch.nn.Identity()


def concept_token_ids(tok, english_text: str, max_words: int = 6) -> set[int]:
    """First-token ids of the salient English content words (the harmful concept).

    Tokenized with a leading space (natural BPE form). Used to detect whether the
    harmful meaning surfaces in the romanized run's logit lens."""
    if not english_text:
        return set()
    words = [w for w in re.findall(r"[a-zA-Z]+", english_text.lower())
             if len(w) >= 4 and w not in STOPWORDS]
    ids: set[int] = set()
    for w in words[:max_words]:
        for form in (" " + w, w):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return ids


def collect_layer_residuals(model, tok, prompt_text, device, n_layers):
    """One forward; stash residual[LENS_POS] at every layer."""
    enc = tok(prompt_text, return_tensors="pt").to(device)
    res: dict[int, torch.Tensor] = {}
    handles = []

    def mk(L):
        def hook(_m, _i, output):
            hidden = output[0] if isinstance(output, tuple) else output
            res[L] = hidden[:, LENS_POS, :].detach().float()
        return hook

    base = _LAYER_BASE if _LAYER_BASE is not None else getattr(model, "model", model)
    for L in range(n_layers):
        handles.append(base.layers[L].register_forward_hook(mk(L)))
    try:
        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()
    return res


def logit_lens(model, tok, prompt_text, device, n_layers, top_k, concept_top_k,
               target_ids: set[int], W, norm):
    """Return (lens_tokens, refusal_layer, concept_layer).

    refusal_layer = first layer a refusal cue enters top-k.
    concept_layer = first layer a harmful-concept (English content) token enters
                    the top-`concept_top_k` (None if never)."""
    res = collect_layer_residuals(model, tok, prompt_text, device, n_layers)
    out = {}
    refusal_layer = None
    concept_layer = None
    k = max(top_k, concept_top_k)
    target = torch.tensor(sorted(target_ids), device=W.device) if target_ids else None
    with torch.no_grad():
        for L in range(n_layers):
            h = norm(res[L].to(W.device)) if not isinstance(norm, torch.nn.Identity) else res[L].to(W.device)
            logits = (h @ W.T).squeeze(0)  # [vocab]
            top = torch.topk(logits, k).indices
            top_list = top.tolist()
            toks = [tok.decode([t]).strip() for t in top_list[:top_k]]
            out[L] = toks
            if refusal_layer is None and any(w.strip().lower() in REFUSAL_CUES for w in toks):
                refusal_layer = L
            if concept_layer is None and target is not None:
                top_concept = set(top_list[:concept_top_k])
                if any(t in top_concept for t in target_ids):
                    concept_layer = L
    return out, refusal_layer, concept_layer


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(list_models()), default=DEFAULT_MODEL_KEY)
    ap.add_argument("--pair", choices=sorted(PAIRS), default="tier_a")
    ap.add_argument("--n-pairs", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=10, help="Tokens stored per layer for inspection.")
    ap.add_argument("--concept-top-k", type=int, default=50,
                    help="Top-k window for harmful-concept emergence detection (probe 2).")
    ap.add_argument("--prefer-device", choices=["cuda", "xla", "mps", "cpu"], default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    log.info("=" * 80); log.info("PHASE 4b — Logit Lens (per-layer unembedding)"); log.info("=" * 80)
    dev = detect_device(prefer=args.prefer_device); log_device_info(dev)
    cfg = load_model_config(args.model)
    native, variant, nat_file, var_file = PAIRS[args.pair]

    nat = {r.source_id: r.prompt for r in read_jsonl(nat_file)}
    var = {r.source_id: r.prompt for r in read_jsonl(var_file)}
    # Raw AdvBench seed uses "prompt_en" (not "prompt") and lacks PromptRecord fields — read directly.
    eng = {}
    if ENGLISH_SEED.exists():
        import json as _json
        for line in ENGLISH_SEED.read_text().splitlines():
            if line.strip():
                obj = _json.loads(line)
                if "source_id" in obj and "prompt_en" in obj:
                    eng[obj["source_id"]] = obj["prompt_en"]
    common = sorted(sid for sid in nat if sid in var)[: args.n_pairs]
    log.info(f"Model {cfg.MODEL_KEY} | pair={args.pair} | {len(common)} pairs | "
             f"top_k={args.top_k} concept_top_k={args.concept_top_k} | english_anchor={len(eng)>0}")
    anchored = [sid for sid in common if eng.get(sid)]
    cov = len(anchored) / max(1, len(common))
    log.info(f"Concept-anchor coverage (common ∩ english seed): {len(anchored)}/{len(common)} ({cov:.1%})")
    if not anchored:
        log.error("NO pair has an English concept anchor — concept_present_in_variant_frac would be a "
                  "meaningless 0.0 and the H6c P-vs-S gate cannot be applied. Cause: the english seed's "
                  "source_id scheme does not match the english seed's advbench_* ids. "
                  "Aborting."); return 1
    if cov < 0.9:
        log.warning(f"Only {cov:.1%} of pairs have a concept anchor; the frac-over-N metric is diluted "
                    "by anchor-less pairs. Use the *_anchored field added below.")

    model = load_hf_model(
        cfg.MODEL_ID, dev, model_class=getattr(cfg, "MODEL_CLASS", None),
    ); model.eval()
    # Resolve the decoder-layer container once (VLM-aware) for hooks + final norm.
    global _LAYER_BASE
    _LAYER_BASE = get_text_base_model(
        model,
        is_multimodal=bool(getattr(cfg, "IS_MULTIMODAL", False)),
        submodule_path=getattr(cfg, "TEXT_SUBMODULE_PATH", None),
    )
    tok = AutoTokenizer.from_pretrained(cfg.MODEL_ID)
    W = model.lm_head.weight.float()
    norm = get_final_norm(model)

    records = []
    ref_nat, ref_var, con_nat, con_var = [], [], [], []
    for sid in common:
        target = concept_token_ids(tok, eng.get(sid, ""))
        nat_lens, nat_ref, nat_con = logit_lens(
            model, tok, format_prompt(tok, nat[sid], cfg), dev.device,
            cfg.N_LAYERS, args.top_k, args.concept_top_k, target, W, norm)
        var_lens, var_ref, var_con = logit_lens(
            model, tok, format_prompt(tok, var[sid], cfg), dev.device,
            cfg.N_LAYERS, args.top_k, args.concept_top_k, target, W, norm)
        if nat_ref is not None: ref_nat.append(nat_ref)
        if var_ref is not None: ref_var.append(var_ref)
        if nat_con is not None: con_nat.append(nat_con)
        if var_con is not None: con_var.append(var_con)
        records.append({
            "source_id": sid,
            "native_refusal_emerges_layer": nat_ref, "variant_refusal_emerges_layer": var_ref,
            "native_concept_emerges_layer": nat_con, "variant_concept_emerges_layer": var_con,
            "n_concept_targets": len(target),
            "native_lens": nat_lens, "variant_lens": var_lens})

    def mean(xs):
        return sum(xs) / len(xs) if xs else None
    N = len(common)
    summary = {
        "native_refusal_emerges_mean_layer": mean(ref_nat),
        "variant_refusal_emerges_mean_layer": mean(ref_var),
        "native_refusal_emerged_count": f"{len(ref_nat)}/{N}",
        "variant_refusal_emerged_count": f"{len(ref_var)}/{N}",
        "native_concept_emerges_mean_layer": mean(con_nat),
        "variant_concept_emerges_mean_layer": mean(con_var),
        "native_concept_emerged_count": f"{len(con_nat)}/{N}",
        "variant_concept_emerged_count": f"{len(con_var)}/{N}",
        "concept_present_in_variant_frac": (len(con_var) / N) if N else None,
        "n_concept_anchored": len(anchored),
        "concept_present_in_variant_frac_anchored": (len(con_var) / len(anchored)) if anchored else None,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / "phase4_patching" / cfg.MODEL_KEY / f"logitlens_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logit_lens.json").write_text(json.dumps(
        {"phase": "phase4b_logit_lens", "model_key": cfg.MODEL_KEY, "pair": args.pair,
         "n_pairs": N, "top_k": args.top_k, "concept_top_k": args.concept_top_k,
         "lens_position": LENS_POS, "summary": summary, "records": records,
         "completed_at": datetime.now().isoformat()},
        ensure_ascii=False, indent=2))

    log.info("=" * 80); log.info("LOGIT-LENS SUMMARY")
    log.info(f"  HARMFUL CONCEPT present in VARIANT run: {summary['variant_concept_emerged_count']} "
             f"(mean emergence layer {summary['variant_concept_emerges_mean_layer']})")
    log.info(f"    (native contrast: {summary['native_concept_emerged_count']}, "
             f"layer {summary['native_concept_emerges_mean_layer']})")
    log.info(f"  refusal tokens emerge — native: layer {summary['native_refusal_emerges_mean_layer']} "
             f"({summary['native_refusal_emerged_count']})")
    log.info(f"  refusal tokens emerge — variant: layer {summary['variant_refusal_emerges_mean_layer']} "
             f"({summary['variant_refusal_emerged_count']})")
    log.info("  H6c GATE: S is only declarable if the harmful concept IS present in the variant run.")
    log.info(f"  Wrote: {out_dir / 'logit_lens.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
