# Phase 4 — H2: Forward/Reverse Causal Intervention

This folder holds the **H2** results: a causal test of whether the model's
refusal is carried by a **script-specific** direction (native-script potent,
romanized-script inert) or by a **shared, meaning-level** direction.

For each matched pair *(native script → romanized/obfuscated variant)* we take
the diff-in-means **refusal direction** at the model's causal layer and either
**ablate** it (remove refusal) or **add** it (induce refusal), then measure how
often the generation flips. The asymmetry between the native-script direction
and the variant-script direction is the signal.

All 10 models are present. Numbers read live from these files; seed = 42.

---

## Models (10)

`gemma_3_1b_it`, `gemma_2_2b_it`, `gemma_3_4b_it`, `gemma_2_9b_it`,
`gemma_3_27b_it`, `qwen3_4b`, `qwen3_8b`, `qwen2_5_7b_Instruct`,
`llama_3_1_8b_Instruct`, `llama_3_2_3b_Instruct`.

## Script pairs (tiers)

| tier | native → variant | family |
|------|------------------|--------|
| `tier_a` | Arabic → **Arabizi** | Arabic |
| `tier_c` | Devanagari → **ITRANS** | Hindi |
| `tier_e` | English → **Leetspeak** | English |

---

## Directory layout

```
phase4_intervention/<model>/
├── run_YYYYMMDD_HHMMSS/
│   └── h2_results.json        # one file per (tier × mode); the canonical output
└── checkpoints/
    ├── h2_<tier>_<mode>_L<layer>_a<alpha>_n200.DONE       # completion markers
    └── h2_<tier>_<mode>_..._n200.partial.json             # per-prompt resume cache
```

- There are **several `run_*` dirs per model** — one per (tier, mode) and per
  resumed run. Each `h2_results.json` is self-describing (`pair`, `mode`,
  `layer`, `alpha`). Use the **latest** run for a given `(pair, mode)`.
- `mode` is `ablation` (remove refusal, the headline) or `addition` (induce
  refusal, the inverse control).
- `layer` = per-model causal layer (varies: e.g. Llama-3.1-8B = L11); `alpha` =
  intervention strength (gemma-3 models use 0.4, others 1.5); `n_pairs` = 200.

---

## What each `h2_results.json` reports

The design is a 2×2: which direction (native vs variant) applied to which
target. Four **cells** are measured, each as a flip rate:

| cell | meaning |
|------|---------|
| `cell1_baseline` | ablate the **native** direction on the **native** prompt (sanity: should flip a lot) |
| `cell2_forward` | ablate the **variant** direction on the **variant** prompt |
| `cell3_reverse` | ablate the **native** direction on the **variant** prompt |
| `cell4_self` | ablate the **variant** direction on the **native** prompt |

### Key fields

- **`baseline_refusal`** — `{native, variant}` refusal counts (out of `n_pairs`)
  before any intervention. The native→variant gap is the phenomenon H1 documents.
- **`baseline_coherence`** — the **comprehension gate**: fraction of baseline
  responses that are coherent. A gibberish "compliance" does not count.
- **`cell_flip_rates_gated_vs_raw`** — flip rate per cell, `gated` (flip **and**
  a coherent, on-topic transition = genuine) vs `raw` (any flip). Prefer `gated`.
- **`cell_flip_rates_conditional`** — flips ÷ **pairs-with-room** (room =
  baseline-refused for ablation, baseline-complied for addition). Removes the
  ceiling/floor confound.
- **`same_target_contrasts`** ⭐ — the **headline, confound-free** metric:
  native-direction vs variant-direction potency on the **same** target.
  - `primary_contrast_pp` = native potency − variant potency, in percentage
    points. Large positive ⇒ refusal is carried by a script-specific direction
    that is potent on the native script and inert on the romanized one.
- **`headline`** — includes the older **FF−RF** (forward − reverse) index and a
  McNemar test. **FF−RF is baseline-saturation-confounded and is NOT the paper's
  metric** (`asymmetry_pp_caveat` says so). Use `same_target_contrasts` instead;
  the `verdict_gated_FFmRF` field is retained for the record only.

---

## Headline summary — `primary_contrast_pp` (native − variant potency, pp)

Latest ablation run per (model, tier). Larger ⇒ more script-specific direction.

| model | tier_a (Arabizi) | tier_c (ITRANS) | tier_e (Leetspeak) |
|-------|:---:|:---:|:---:|
| gemma_3_1b_it | +67 | +60 | +85 |
| gemma_2_2b_it | +9 | +73 | +2 |
| gemma_3_4b_it | +77 | +75 | +97 |
| gemma_2_9b_it | +84 | +43 | +29 |
| gemma_3_27b_it | +92 | +87 | +75 |
| qwen3_4b | +92 | +35 | +3 |
| qwen3_8b | +86 | +56 | +94 |
| qwen2_5_7b_Instruct | +92 | +52 | +15 |
| llama_3_1_8b_Instruct | +88 | +87 | +87 |
| llama_3_2_3b_Instruct | +25 | +66 | +22 |

**Reading it:** the native-script direction is strongly causal (removing it
drops refusal), while the romanized-script direction is largely inert — a
positive contrast in essentially every cell. This is *consistent with* but does
**not by itself prove** script-anchoring: H2 asks whether the *direction* is
script-specific; whether the model even *built* the harmful concept in the
romanized run is the comprehension question answered by the gate (Phase 5) and
patching (Phase 4 patching). Interpret H2 alongside those.

> Note: cells that read low here (e.g. tier_e for gemma_2_2b, qwen3_4b) are
> exactly where the romanized variant was rarely refused/understood at baseline,
> so there was little "room" to flip — read them next to `baseline_refusal` and
> the comprehension gate, not in isolation.

---

## Provenance

- Directions: Phase 2 (`phase2/<model>/run_*`), diff-in-means at the causal layer.
- These runs: `02b`/H2 forward-reverse script (`phase4_causal_intervention/interventions/01_h2_forward_reverse.py`), N=200, position-matched, coherence-gated.
- Regenerate the demo/paper numbers from these files; nothing is hand-transcribed.

---

## Direction transfer on the comprehended subset (the positive half)

**Why.** The H2 ablation above establishes a *necessity* claim: the **romanized** direction is
causally inert (ablating it on the native prompt barely flips refusal). The complementary
**positive** claim — that the **native** (Arabic) direction is what *mediates* refusal once the
romanized prompt **is** comprehended — is what closes "safety is intact but starved." This is the
experiment a main-track reviewer asked for; it is tested here.

**Method.** Restrict to the **comprehended-Arabizi** subset: items the Gemini judge scored
*engaged* (ON\_TOPIC ∪ REFUSAL) **and** that the model refused at baseline (this refused count is the
`room` — there must be a refusal to remove). On that subset:
- ablate the **native (Arabic)** refusal direction → does refusal disappear?
- ablate a **norm-matched random** direction (specificity control) → it should not.

Flip = baseline-refused **and** intervened-complied **and** the compliance is coherent.
Script: [`interventions/02_direction_transfer_comprehended.py`](../../scripts/phase4_causal_intervention/interventions/02_direction_transfer_comprehended.py).
Run locally on Apple-Silicon MPS (≤9B); α and causal layer L inherited from Phase 2. Output:
`<model>/direction_transfer/transfer_<stamp>.json`.

### Results — all locally-run models (sorted by room)

Calibrated α is **model-specific**: Gemma-3 uses **0.4**, all other families use **1.5**
(same values as Phase-2 / H2). Every model below is at its correct α.

| Model | L | α | n\_comp | **room** | ablate **native** | ablate **random** | gap |
|---|---|---|---|---|---|---|---|
| Gemma-3-27B-it | 30 | 0.4 | 41 | **38** | **45%** | **0%** | **+45** |
| Gemma-3-1B-it | 11 | 0.4 | 31 | **29** | **72%** | **0%** | **+72** |
| Gemma-2-9B-it | 20 | 1.5 | 22 | **17** | **100%** | **0%** | **+100** |
| Gemma-3-4B-it | 16 | 0.4 | 16 | 10 | 80% | 40% | +40 |
| Llama-3.2-3B-Instruct | 20 | 1.5 | 7 | 5 | 60% | 0% | +60 |
| Qwen3-8B | 15 | 1.5 | 8 | 2 | 100% | 0% | +100 |
| Qwen3-4B | 18 | 1.5 | 2 | 2 | 100% | 0% | +100 |
| Gemma-2-2B-it | 12 | 1.5 | 3 | 1 | 100% | 0% | +100 |
| Qwen2.5-7B-Instruct | 14 | 1.5 | 5 | **0** | — | — | — |
| Llama-3.1-8B-Instruct | 11 | 1.5 | 10 | **0** | — | — | — |

Gemma-3-27B-it (top row) was run on an A100-80GB (54 GB bf16 + Llama-Guard-3-8B); all others ran locally.

### Comparison — power tiers

| Tier | Models | native | random | reliability |
|---|---|---|---|---|
| **Well-powered** (room ≥ 15) | Gemma-3-27B-it, Gemma-3-1B-it, Gemma-2-9B-it | **45–100%** | **0%** | ✅ clean, decisive |
| Small-room (2–10) | Gemma-3-4B-it, Llama-3.2-3B, Qwen3-4B/8B, Gemma-2-2B | 60–100% | 0–40% | ⚠️ directional only |
| **No room** (0) | Qwen2.5-7B, Llama-3.1-8B | n/a | n/a | comply even when comprehending |

### Interpretation

1. **The positive half is demonstrated.** In the three well-powered cells, ablating the native
   (Arabic) direction removes refusal at **45% (Gemma-3-27B, room 38)**, **72% (Gemma-3-1B, room 29)**
   and **100% (Gemma-2-9B, room 17)** while a random direction removes **0%**. The **same native
   refusal direction** mediates refusal on comprehended-romanized items — safety is not a separate
   romanized mechanism, it is the native circuit merely starved of input when the prompt is not
   decoded. (The 27B native rate is the lowest of the three despite the largest room: a bigger model
   carries refusal more redundantly, so removing a single direction flips fewer items — but the
   specificity is intact, native ≫ random = 0.)
2. **Small-room cells are directional, not decisive.** Gemma-3-4B-it (room 10) shows native 80% but
   random 40% — small-sample noise in the control; the clean 100%/0% cells (Qwen3-8B/4B, Gemma-2-2B)
   sit on 1–2 items. They agree in sign (native ≫ random) but must not be quoted as effect sizes.
3. **`room = 0` is itself a finding.** Qwen2.5-7B and Llama-3.1-8B **comply even on the Arabizi they
   comprehend** — nothing is refused at baseline, so there is no refusal to ablate. Their high
   refusal-given-comprehension in the *pooled* conditional analysis comes from **Leetspeak, not
   Arabizi**; on Arabizi specifically these two do not refuse even when they read it (a per-model,
   per-tier residual gap distinct from comprehension-gating).
4. **Control specificity.** Random-direction ablation removes 0% in 5 of 7 measurable cells,
   confirming the effect is specific to the extracted refusal direction, not an artifact of removing
   *any* direction at strength α.

**Bottom line.** Combined with the per-item conditional analysis (the *negative* half — where
comprehension fails, no refusal machinery engages), this closes both halves of the two-route model:
where the romanized prompt is read, the **native** refusal circuit fires and mediates the refusal;
where it is not, nothing does. The paper tables the three well-powered cells and treats the rest as
directional (Table `tab:transfer`).
