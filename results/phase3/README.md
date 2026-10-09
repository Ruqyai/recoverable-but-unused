# Phase 3 — Geometry of the refusal direction (H5)

Complete for **all 10 models** (3 files each). This phase characterises the
*geometry* of the per-script refusal direction: is it the **same** direction
across scripts, how **strongly** does it separate harmful from benign, and does
the harmful/benign **cluster separation** (Wang's silhouette) reflect
orthographic interference? It feeds **H5**.

Directions come from Phase 2 (diff-in-means, per script). Analysis layer =
per-model causal layer (`test_layer`); bootstrap = 1000, seed = 42.

---

## Files (per model)

```
phase3/<model>/[run_*/]
├── cosine_similarity.json    # Stage 1 — per-layer cosine between script directions
├── magnitude_analysis.json   # Stage 2 — harmful vs benign residual norms per script
└── silhouette.json           # Stage 3 — harmful/benign cluster silhouette (Wang 0.24 test)
```

Six scripts analysed: `arabic, arabizi, devanagari, itrans, english, leetspeak`
(N=200 harmful+benign pairs per script).

### 1. `cosine_similarity.json`
`pairs["<A>__<B>"]` = cosine between script-A and script-B refusal directions at
**each layer** (`{"0": …, "1": …}`). Self-pairs (`arabic__arabic`) ≈ 1.0
(sanity). Cross-script pairs (`arabic__arabizi`) test whether native and
romanized refusal share one axis (high cosine) or sit in different subspaces (low).

### 2. `magnitude_analysis.json`
`per_script[<script>]`: `harmful_mean_norm_per_layer`, `benign_mean_norm_per_layer`,
`diff_mean_magnitude_across_layers`, and the `diff_peak_layer` / `diff_peak_magnitude`
(where harmful−benign separation is largest). Bigger diff ⇒ stronger, more
linearly-separable refusal signal at that script.

### 3. `silhouette.json`
`binary_silhouette_one_vs_rest[<script>]`: silhouette of the harmful-vs-benign
split along that script's direction, with a 1000-iteration `bootstrap_ci`
(99%/95%). Also `multi_cluster_silhouette`. This is the **Wang (2024) reference
test**: H5 asks whether the Arabizi silhouette differs from the Arabic 0.24 the
prior work reports.

---

## Headline — silhouette (H5) at the causal layer

Harmful/benign silhouette (one-vs-rest). Wang reference = **0.24**.

| model | test_layer | arabic | arabizi |
|-------|:---:|:---:|:---:|
| gemma_3_1b_it | 13 | −0.06 | 0.14 |
| gemma_2_2b_it | 13 | 0.01 | 0.02 |
| gemma_3_4b_it | 17 | 0.05 | 0.25 |
| gemma_2_9b_it | 21 | −0.05 | 0.10 |
| gemma_3_27b_it | 31 | −0.13 | 0.10 |
| qwen3_4b | 18 | 0.18 | 0.06 |
| qwen3_8b | 18 | 0.14 | 0.05 |
| qwen2_5_7b_Instruct | 14 | 0.16 | 0.01 |
| llama_3_1_8b_Instruct | 16 | 0.07 | 0.10 |
| llama_3_2_3b_Instruct | 14 | 0.13 | 0.12 |

**H5 verdict: indeterminate.** Wang's 0.24 is **not reproduced** — silhouettes are
low and inconsistent (max ≈ 0.25 on a single cell, most well below 0.24, several
negative). The harmful/benign split is not cleanly separable along a single
per-script direction at the causal layer, so the orthographic-interference claim
cannot be adjudicated from geometry alone. (Silhouette is also a magnitude-sensitive
metric; see the late-layer RMSNorm caveat in the project README.)

## Cross-script direction cosine (native ↔ romanized) at the causal layer

| model | cos(arabic, arabizi) | cos(devanagari, itrans) | cos(english, leetspeak) |
|-------|:---:|:---:|:---:|
| gemma_3_1b_it | −0.47 | 0.97 | 0.80 |
| gemma_2_2b_it | 0.65 | 0.25 | 0.29 |
| gemma_3_4b_it | 0.58 | 0.95 | −0.71 |
| gemma_2_9b_it | 0.29 | 0.38 | 0.60 |
| gemma_3_27b_it | 0.80 | 0.37 | 0.70 |
| qwen3_4b | 0.27 | 0.52 | 0.84 |
| qwen3_8b | 0.44 | 0.56 | 0.67 |
| qwen2_5_7b_Instruct | 0.40 | 0.59 | 0.74 |
| llama_3_1_8b_Instruct | 0.36 | 0.38 | 0.47 |
| llama_3_2_3b_Instruct | 0.18 | 0.38 | 0.48 |

The native↔romanized refusal direction is **not consistently aligned** — cosine
ranges from strongly positive to negative across models and scripts. This says
the geometry alone is ambiguous about script-anchoring; the causal question is
settled by intervention (H2) and patching (H6), not by cosine.

---

## Provenance
- Phase 2 directions: `results/phase2/<model>/run_*` (diff-in-means, per script, per layer).
- Stages 1–3: `scripts/phase3_geometry/…`; bootstrap 1000, seed 42; analysis at the per-model causal layer (`test_layer`).
- H5 is a reference-reproduction check; it is **indeterminate**, not a positive result — cite it as a non-replication of Wang (2024), not as evidence for the two-route model.
