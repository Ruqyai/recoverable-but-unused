# Phase 1 — Corpus construction, translation & review

Phase 1 is the **data-preparation** stage. It has no computed results of its own
(this folder is intentionally empty) — its product is the multilingual corpus
under [`data/`](../../data/), which every later phase consumes. This README
documents how that corpus was built, transliterated/translated, and reviewed.

**Design invariant:** every record holds **meaning constant** and varies **only
the writing system**, keyed to a shared English anchor (`source_id =
advbench_NNN`). That is what lets the study separate a *script* effect from a
*language* effect.

---

## Source

- **AdvBench** \[Zou et al., 2023\] — 200 harmful English prompts, sampled to
  `data/raw/advbench_sample_en.jsonl` (fields: `source_id`, `prompt_en`,
  `harm_category`, `paired_with`). This is the **English anchor** for every
  variant. The native Hindi tier additionally derives from Joshi et al., 2025.

## Tiers & scripts (200 matched records each)

| tier | files | scripts | how produced |
|------|-------|---------|--------------|
| **A** | `tier_a_arabic`, `tier_a_arabizi` | Arabic ↔ **Arabizi** (`ar`) | Arabic translation of the anchor; **naturalistic romanization** (Arabizi — an effect *could* reflect training-data leakage) |
| **C** | `tier_c_devanagari`, `tier_c_itrans` | Hindi Devanagari ↔ **ITRANS** (`hi`) | Hindi in Devanagari; **deterministic, bijective, ASCII-only transliteration** (ITRANS — *cannot* be leakage) |
| **E** | `tier_e_english`, `tier_e_leetspeak` | English ↔ **Leetspeak** (`en`) | anchor English; **deterministic within-language substitution** `a→4 e→3 i→1 o→0 s→5 t→7 …` (only spelling changes → no language-coverage confound) |
| **B** | `benign_{arabic,arabizi,devanagari,itrans,english,leetspeak}` | all 6 | matched **benign** controls (harm-specificity control) |
| **D** | `tier_d_{arabic,devanagari,english}_paraphrased` | 3 | within-script **paraphrase** controls (meaning-tracking) |

**Totals:** 3,000 records — 1,800 harmful (A/C/E native+romanized) + 1,200 benign
(B) + paraphrase (D). Each romanized record is `paired_with` its native twin and
shares the anchor's `source_id`, so variants align 1:1.

### Record schema (`.jsonl`)
```json
{"id":"tier_c_adv_001_itrans", "tier":"C", "harm_category":"cybercrime",
 "source_id":"advbench_000", "source_id_verified":true, "script":"itrans",
 "language":"hi", "prompt":"eka skripTa likheM jo ...",
 "paired_with":"tier_c_adv_001_devanagari", "is_harmful":true, "validation":{...}}
```
Behavioural labels downstream are **coherence-gated** — a gibberish "compliance"
never counts as engagement.

---

## Translation & transliteration principles

- **ITRANS (Hindi):** strict 1:1 deterministic transliteration of a *natural*
  Devanagari sentence — **not** conversational Hinglish. The schwa / inherent-`a`
  is retained (`eka`, `skripTa`), because swapping Hindi words for English ones
  would be code-switching (a language change) — exactly the confound the design
  excludes.
- **Arabizi (Arabic):** naturalistic Arabic-chat romanization (digit-for-letter,
  e.g. `7`=ح, `2`=ء/ق, `3`=ع), reflecting how speakers actually write.
- **Leetspeak (English):** a fixed substitution table — meaning identical, spelling
  only.
- Two columns per pair must differ **only in script**; the Devanagari/Arabic side
  is judged for naturalness & grammaticality, the romanized side for faithful
  mechanical correspondence to it.

---

## Review & repair (full log: [`data/review/LOG.md`](../../data/review/LOG.md))

Cross-language QA (2026-06-22 → 06-25); all edits made to the source `.jsonl`,
backups under `data/review/_backup_*`.

### Reviewers
Every language family had a **native / fluent human reviewer** — no tier rests on
machine translation alone.

| reviewer | scope | role |
|----------|-------|------|
| **Reviewer** (native Arabic) | Arabic + Arabizi (harmful, benign, paraphrase) | native-speaker review of meaning, naturalness & Arabizi faithfulness |
| **Reviewer** (fluent English) | English + Leetspeak | human review of the anchor and the leetspeak substitution |
| **Reviewer** (native Hindi) | Hindi Devanagari + ITRANS (harmful, benign, paraphrase) | 2 rounds — flagging then confirmation; acknowledged in the paper |
| **Claude auto-review** (Opus/Sonnet subagents) | all languages | batched fidelity checks to *assist* the human reviewers |
| **Independent audit** (fresh agent) | all languages, merged table | caught 4 truncations the human review missed |

### Key defects found & fixed
- **Benign Hindi mis-pairing (tier_b):** only `source_id` had been re-keyed, not
  the actual EN↔Hindi pairing → mismatched from row ~8. Repaired by translating
  all 200 Hindi→English and solving an optimal (Hungarian) assignment against the
  canonical benign English: **182 rows realigned**, 18 orphans auto-translated
  (flagged `hindi_origin = auto-translated`); cognate alignment 21% → 98%.
- **tier_c harmful Hindi:** 7 mid-word truncations completed (3 flagged by the Hindi reviewer,
  **4 missed-by-human caught by the independent audit**) + 4 word-swaps.
- **tier_d paraphrase:** Devanagari 24 fixed (21 truncations + 3 mistranslations),
  Arabic 19 fixed (typos, nominal→imperative, one meaning-inversion).
- **tier_a Arabic/Arabizi:** 6 fixed, incl. **advbench_237 — a refusal-leak** (the
  row carried a model refusal instead of the prompt) and several truncations/drifts.

### The 2026-06-21 corpus correction (why some old runs differ)
tier_a (Arabic/Arabizi) and tier_c (Hindi; source changed Nemotron→AdvBench) were
corrected on 2026-06-21; benign was re-synced for every language. **English +
Leetspeak (tier_e) harmful side is unchanged** (canonical AdvBench). This is why
Phase-4 experiments were re-run for tier_a + tier_c but tier_e numbers were reused.

---

## What to trust
- English anchor + `source_id` pairing are the ground truth; verify alignment via
  `paired_with` / `source_id`, not row order.
- **All three language families carry native / fluent human review**: Arabic +
  Arabizi and English + Leetspeak by a native-Arabic / fluent-English reviewer,
  Hindi Devanagari + ITRANS by an independent native Hindi reviewer.
  LLM auto-review and a fresh independent audit assisted but did not replace the
  human sign-off.
- `harm_category` is a label only — it does not enter any Phase 2–6 number.
