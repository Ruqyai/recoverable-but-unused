# Phase 2 — Refusal-direction extraction + causal-layer selection

The **foundation** phase: for every model it extracts a per-script **refusal
direction** (diff-in-means of harmful − benign residuals) at every layer, and
picks the model's single **causal layer**. Everything downstream — H2
(intervention), Phase 3 (geometry/H5), H6 (patching) — reads these directions
and this layer.

Complete for **all 10 models** (6 JSON + direction/activation tensors each).

---

## Layout (per model)

```
phase2/<model>/run_YYYYMMDD_HHMMSS/
├── directions/<script>.pt              # 6 refusal directions (per-layer tensors)
│     arabic, arabizi, devanagari, itrans, english, leetspeak
├── activations/<script>/…              # cached harmful/benign residuals (per prompt)
├── metadata.json                       # Stage 2: directions + per_script_stats
├── causal_layer_selection.json  ⭐     # Stage 4: the per-model causal layer
├── metadata_separation_proxy.json      # descriptive separation-norm proxy
├── metadata_logit_difference.json      # logit-difference proxy
├── metadata_proxy_comparison.json      # proxy-vs-proxy comparison
└── run_metadata.json                   # run config / provenance
```

Scripts = the 3 native/romanized pairs (Arabic/Arabizi, Devanagari/ITRANS,
English/Leetspeak); N=200 matched harmful+benign per script.

### The direction files
`directions/<script>.pt` is the diff-in-means refusal direction **for that
script, at every layer** (shape ≈ `[n_layers, hidden_dim]`). H2/H6 load the row
at the causal layer; Phase 3 uses all layers for cosine/geometry.

### `causal_layer_selection.json` ⭐ (the load-bearing output)
Selects the layer at which the intervention is run. **Definition (OSF v6 §4):**
the causal layer is the **peak-flip-rate layer from single-layer ablation**, with
a centrality tiebreaker at relative depth 0.5 — read `selection.peak_causal_layer_guard`
(or `…_lexical` if the guard was skipped).

> ⚠️ **Do not** use the separation-norm peak (`metadata_separation_proxy.json`)
> to pick the intervention layer — it is a *descriptive* proxy only. Separation
> magnitude peaks late in the network (RMSNorm-inflated) where intervention does
> **not** actually flip behaviour. The three `metadata_*proxy*` files exist to
> show the proxy≠causal gap, not to drive layer choice.

---

## Per-model causal layer (feeds H2 & H6)

| model | n_layers | causal layer | rel. depth |
|-------|:---:|:---:|:---:|
| gemma_3_1b_it | 26 | 9 | 0.35 |
| gemma_2_2b_it | 26 | 12 | 0.46 |
| gemma_3_4b_it | 34 | 16 | 0.47 |
| gemma_2_9b_it | 42 | 20 | 0.48 |
| gemma_3_27b_it | 62 | 32 | 0.52 |
| qwen3_4b | 36 | 18 | 0.50 |
| qwen3_8b | 36 | 15 | 0.42 |
| qwen2_5_7b_Instruct | 28 | 14 | 0.50 |
| llama_3_1_8b_Instruct | 32 | 11 | 0.34 |
| llama_3_2_3b_Instruct | 28 | 14 | 0.50 |

Layers cluster near mid-depth (≈0.34–0.52) — **not** the late peak-separation
layers. These indices are **not** cross-comparable across models (different
depths/architectures); each model must use its own direction at its own layer.

---

## Notes on the `.pt` tensors
- `directions/*.pt` are small (per-layer vectors) and are the reusable artifact.
- `activations/*` are the raw per-prompt residual caches (large; regenerable).
  They exist so directions/geometry can be recomputed without re-running the
  model. On a fresh checkout these may be pulled from the HF dataset rather than
  shipped in git.

## Provenance
- Extraction: `scripts/phase2_direction_extraction/` (Stage 1 activations → Stage 2 directions → Stage 3 proxies → Stage 4 causal-layer selection). Seed 42; `add_special_tokens` verified not to move the direction.
- Downstream consumers: H2 (`results/phase4_intervention/`), geometry (`results/phase3/`), H6 (`results/phase4_patching/`).
