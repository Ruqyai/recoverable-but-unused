# Phase 4 — H6: Activation Patching (the confirmatory test)

Complete for **all 10 models × 3 tiers**. H6 is the paper's **pre-registered
confirmatory** test of script-anchoring. It asks: if refusal is carried by a
script-specific mechanism, does **patching the native-script residual into the
romanized run specifically restore refusal**?

For each *(native, romanized)* pair where the romanized variant complied
("with room"), we interchange the residual at the assistant read-out position
(layer sweep) and measure how often refusal is restored — against three
controls that decide whether any restoration is **specific**.

---

## The four conditions (per layer)

| condition | what is patched in | reads as |
|-----------|--------------------|----------|
| **matched** (`recover_curve_A2B`) | the **paired** native residual | the effect under test |
| **self** (`controls_self`) | the variant's own residual | no-op sanity — must be ≈ 0 |
| **random** (`controls_random`) | an **unrelated** harmful native residual | if this restores too ⇒ effect is **non-specific** |
| **benign** (`controls_benign`) | a benign native residual | must be ≈ 0 (restoration needs a harm/decision state) |

**Headline metric: `matched − random`** at the peak layer. Pre-registered bar:
**≥ 25 pp AND** the harmful concept present in the variant run (logit lens) to
declare script-anchoring (S). `matched ≈ random` ⇒ **non-specific** ⇒ S not
declarable.

---

## Files (per model)

```
phase4_patching/<model>/
├── run_*/patching_results.json     # one per tier: recover curve + 4 controls
├── logitlens_*/logit_lens.json     # per-tier concept/refusal read-out (P vs S discriminator)
└── PROVENANCE_tier_e.md            # (qwen3_8b, llama_3_1_8b only) tier_e reused from sibling repo
```

### `patching_results.json` key fields
`pair`, `native`/`variant`, `n_pairs` (200), `n_room` (variants that complied →
patchable), `recover_curve_A2B` (matched flip-rate per layer),
`recover_peak_layer` / `recover_peak_rate`, and `controls_self` /
`controls_random` / `controls_benign` (per-layer curves).

### `logit_lens.json`
Per-layer read-out at position −1: does the **harmful concept** appear in the
variant's representation (`concept_present_in_variant_frac`) and where does
refusal emerge? This is the **P-vs-S discriminator** — patching non-specificity
alone can't tell "parse failure" from "generic harm-state injection"; the lens
concept signal does. (Concept legibility is scale-dependent — clean only in the
larger models.)

---

## Headline — `matched − random` at peak (pp), all 10 models

Pre-reg bar = **+25 pp**. `(matched vs random)` peak rates in brackets.

| model | tier_a (Arabizi) | tier_c (ITRANS) | tier_e (Leetspeak) |
|-------|:---:|:---:|:---:|
| gemma_3_1b_it | +17 (.89/.72) | +13 (.33/.20) | −3 (.57/.59) |
| gemma_2_2b_it | +8 (.73/.65) | +2 (.44/.42) | +6 (.79/.73) |
| gemma_3_4b_it | +3 (.88/.84) | −2 (.15/.17) | +6 (.21/.15) |
| gemma_2_9b_it | +3 (.97/.94) | +5 (.22/.17) | −10 (.80/.90) |
| gemma_3_27b_it | −8 (.92/1.0) | +6 (.48/.42) | +3 (.81/.77) |
| qwen3_4b | +6 (.32/.26) | +4 (.04/.00) | +0 (.87/.87) |
| qwen3_8b | +2 (.60/.57) | +2 (.16/.14) | −11 (.62/.73) |
| qwen2_5_7b_Instruct | +14 (.31/.17) | +4 (.04/.00) | −3 (.44/.47) |
| llama_3_1_8b_Instruct | +2 (.84/.82) | +0 (.00/.00) | −1 (.96/.97) |
| llama_3_2_3b_Instruct | −2 (.61/.63) | +4 (.04/.00) | +0 (1.0/1.0) |

**Verdict: REJECTED across all 10 models.** No cell clears the +25 pp bar — every
`matched − random` is small (≤ +17 pp, most < 10 pp, several negative). Patching
is **non-specific**: a random harmful native residual restores refusal about as
well as the correctly-paired one. Combined with `benign ≈ 0` (restoration needs a
harm/decision state, but any harm state will do) and `self ≈ 0` (sanity holds),
the read is: the read-out position carries a **transferable decision/harm-state**,
not a script-specific refusal that patching could localize. Script-anchoring (S)
is **not declarable**; single-position patching is a screen, not a localizer.

**tier_c (ITRANS) is the deepest failure** — matched peak is near-zero for
qwen/llama (0.00–0.16) and low for gemma (0.15–0.48): patching barely restores
refusal on romanized Hindi at all, consistent with the concept never being built
(comprehension gate).

---

## Provenance
- Script: `scripts/phase4_causal_intervention/patching/01_activation_patching.py` (+ `02_logit_lens.py`), N=200, position −1, layer sweep, four controls, seed 42.
- `tier_e` for **qwen3_8b** and **llama_3_1_8b_Instruct** is reused from the sibling repo (English/Leetspeak = canonical AdvBench, unchanged by the 2026-06-21 corpus correction) — see each model's `PROVENANCE_tier_e.md`. Their cloud reruns covered tier_a + tier_c only.
- H6 is confirmatory and **rejected**; report it as the null that inverts the script-anchoring reading toward comprehension-gating — not as a positive S result.
