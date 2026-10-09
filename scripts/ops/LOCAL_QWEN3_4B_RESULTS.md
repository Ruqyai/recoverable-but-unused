# Local run — qwen3_4b on the corrected corpus (2026-06-25)

First end-to-end run after the data review/repair (see `data/review/LOG.md`).
Hardware: **Apple M2 Max, MPS, 32 GB, fp16** (no CUDA GPU). Model **Qwen/Qwen3-4B** (non-gated).
Scope: the *light* phases run locally; the *generation-heavy* H2 interventions + patching are deferred to cloud A100.

Run dirs:
- Phase 2: `results/phase2/qwen3_4b/run_20260625_141430`
- Phase 3: `results/phase3/qwen3_4b/run_20260625_161321`
- Diagnostics: `results/phase4_diagnostic/.../run_20260625_165309`
- Phase 5: `results/phase5/qwen3_4b/run_20260625_172047`

---

## Phase 2 — direction extraction
- **Causal layer = L18** (peak intervention flip-rate = **1.0**). Separation peak L34 (descriptive only).
- Confirms the expected Qwen causal layer (18, not the separation peak 34).

## Phase 3 — geometry (refusal-direction alignment)
- arabic ↔ arabic: 1.00 (sanity)
- **arabic ↔ arabizi: mean cos ≈ 0.14** (romanized refusal axis poorly aligned with native)
- arabic ↔ devanagari: ≈ 0.62

## Phase 4 — diagnostics (baseline coherence, 15 prompts/script)
| script | coherent | avg score |
|---|---|---|
| arabizi | 14/15 | 0.93 |
| devanagari | 15/15 | 0.93 |
| english | 15/15 | 0.95 |
- Surface coherence looks fine — but see the behavioral readout below: coherent ≠ comprehended.

## Phase 5 — SAE (transcoder `mwhanna-qwen3-4b-transcoders`, layer 18, exploratory)

### H3 control ladder (06) — all 3 arms
| arm | tier_d (meaning, same script) | headline (script change) |
|---|---|---|
| arabic | **0.739** | **0.295** |
| english | **0.681** | **0.439** |
| hindi | **0.887** | **0.460** |
- tier_d HIGH everywhere (0.68–0.89) ⇒ the measure **does track meaning** (validity OK).
- headline LOW(er) ⇒ looks like orthographic anchoring; **strongest for Arabizi (0.30)**.
- These numbers depend on the corrected `tier_d` paraphrase data (our 19 arabic + 24 devanagari fixes).

### Headline Jaccard (script difference) per pair
| pair | Jaccard |
|---|---|
| arabic ↔ arabizi | 0.295 |
| devanagari ↔ itrans | 0.460 |
| english ↔ leetspeak | 0.439 |

---

## ⚠️ Behavioral read-out (07) — the make-or-break validity gate
50 prompts/condition. Buckets = refusal / harmful_compliance / off_topic / gibberish.

| condition | refusal | compliance | off_topic | gibberish |
|---|---|---|---|---|
| arabic_harmful | 90% | 2% | 8% | 0% |
| **arabizi_harmful** | 4% | 16% | **80%** | 0% |
| english_harmful | 100% | 0% | 0% | 0% |
| **leetspeak_harmful** | 18% | 18% | **64%** | 0% |
| devanagari_harmful | 50% | 4% | 46% | 0% |
| **itrans_harmful** | 2% | 18% | **80%** | 0% |

Comprehension parity (engaged% = refusal + on-topic compliance, vs native):
| romanized | engaged | native engaged | drop | verdict |
|---|---|---|---|---|
| arabizi | 20% | 92% | **−72 pp** | distorted_not_understood |
| leetspeak | 36% | 100% | **−64 pp** | distorted_not_understood |
| itrans | 20% | 54% | −34 pp | degenerate native (devanagari itself <60%) |

**No gibberish (0%) — this is SILENT non-comprehension:** the model misreads the romanized
text and fluently answers a *different* question (off_topic), rather than refusing.

---

## Conclusion
- **qwen3_4b fails the comprehension gate** on Arabizi/Leetspeak/ITRANS.
- Therefore the H3 "orthographic anchoring" numbers (low romanized Jaccard) are **confounded**:
  the low overlap could be anchoring OR simply non-comprehension — the two are inseparable here.
  (W1 open for arabizi/leetspeak; degenerate-native for itrans — do NOT interpret as anchoring.)
- This is consistent with the project thesis ("clarity grows with comprehension / model size"):
  a 4B model is too small to decode the romanized scripts cleanly. **Non-comprehension at 4B is
  itself a valid result**, but 4B cannot serve as a clean anchoring test.

## Deferred to cloud (A100, when quota approved)
- H2 forward/reverse interventions (checkpoint saved at
  `results/phase4_intervention/qwen3_4b/checkpoints/h2_tier_a_ablation_L18_a1.5_n200.partial.json`
  — baselines done, resumable) + H6 patching.
- Larger comprehending models (qwen3_8b, gemma_2_9b_it, …) for the clean anchoring test.
