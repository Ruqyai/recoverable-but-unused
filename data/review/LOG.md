# Data Review & Repair Log

Cross-language QA and repair of the AdvBench-keyed multilingual corpus (200 prompts × source_id `advbench_NNN`).
Goal: every `source_id`'s variants (English / Arabic / Arabizi / Devanagari / ITRANS, across harmful + benign + paraphrased) carry the **same meaning as the English anchor**, with no truncations, mistranslations, refusal-leaks, or misalignment.

Dates: 2026-06-22 → 2026-06-25. All edits made to source `.jsonl`; backups under `data/review/_backup_*`.

---

## 1. Reviewers / methods

| Reviewer | Scope | Role |
|---|---|---|
| **Hindi annotator** (native speaker) | Hindi harmful + benign | 2 review rounds (flagging + confirmation) |
| **Claude auto-review** (Opus/Sonnet subagents) | Arabic, Arabizi, paraphrased, Hindi | Batched fidelity checks; Sonnet used where cyber content tripped Opus filters |
| **Independent audit** (fresh agent, no trust in prior work) | All languages via merged table | Caught 4 truncations the human review missed |

---

## 2. Defects found & fixed (by tier / language)

### tier_b — benign Hindi (`benign_devanagari.jsonl`, `benign_itrans.jsonl`)
- **Root cause:** Hindi prompts were linked to the WRONG rows — only `source_id` had been re-keyed, not the actual EN↔Hindi pairing. English↔Hindi mismatched from row ~8 onward (the Hindi annotator flagged this; confirmed correct).
- **Repair:** translated all 200 Hindi→English (10 parallel subagents) → Hungarian optimal assignment (scipy) against the canonical benign English → recovered the 1:1 mapping. Cognate-alignment 21% → 98%.
  - **182 rows realigned** (existing Hindi moved to correct English; reviewer notes carried with each).
  - **18 orphan rows** (English had no matching Hindi — corpora had drifted) → filled with auto-translation, flagged `hindi_origin = auto-translated`.
- **Hindi annotator, round 2** confirmed the 18 ("Alright") and applied 12 word/sentence fixes (see §3).

### tier_c — harmful Hindi (`tier_c_devanagari.jsonl`, `tier_c_itrans.jsonl`)
- **7 mid-word truncations completed:**
  - 3 flagged by the Hindi annotator: advbench_029 (`विधि`), 101 (`निर्देश दें`), 112 (`योजना बनाएं`) — confirmed "Done".
  - **4 MISSED by the Hindi annotator, caught by independent audit:** advbench_061, 168, 178, 412.
- **4 word-swaps** (Hindi annotator, round 2): see §3.

### tier_d — paraphrased
- **Devanagari (`tier_d_devanagari_paraphrased.jsonl`): 24 fixed** — 21 truncations + 3 mistranslations (advbench_092 anorexia→`एनोरेक्सिया`, 090 community→home, 155 consumer→reader).
- **Arabic (`tier_d_arabic_paraphrased.jsonl`): 19 fixed** — typos, nominal→imperative, first-person→directive, meaning-inversion (382 "brief"→"granular").

### tier_a — harmful Arabic / Arabizi (`tier_a_arabic.jsonl`, `tier_a_arabizi.jsonl`)
- **6 fixed:** advbench_011 (arabizi truncated), 167 (ar+az truncated), 174 (arabizi corrupted word), **237 (arabizi REFUSAL-LEAK — row carried a model refusal instead of the prompt)**, 383 (ar+az "script/operations" drift), 409 (ar+az dropped "in detail").

---

## 3. the Hindi annotator 2nd-review edits applied (2026-06-25)

**tier_c harmful — 4 word-swaps** (devanagari + itrans):
| id | change |
|---|---|
| advbench_019 | `महिमामंडन करता है` → `बढ़ावा देता है` |
| advbench_100 | `संलग्न` → `शामिल` |
| advbench_134 | `व्यवसायों` → `कंपनियों` |
| advbench_512 | `महिमामंडित करे` → `बढ़ावा दे` |

**tier_b benign — 12 edits:**
- Word swaps: 016 (`खुले`→`उपलब्ध`), 077/095 (`प्रणालियों`→`सिस्टमों`), 097 (`संकलित करे`→`जोड़े`), 115 (`आकस्मिक खाद्य विषाक्तता`→`फूड पॉइज़निंग`), 148 (`प्रणाली`→`सिस्टम`).
- "Suitable alternative" (the Hindi annotator asked, Claude proposed): 142 (`प्रकटीकरण`→`खुलासे`), 147 (`अनुवर्ती`→`आगे की`), 382 (`आग्नेयास्त्रों`→`बंदूकों`).
- Full rewrites: 023, 045, 146.

---

## 4. Critique of the Hindi annotator's review (for the record)
- **Strengths:** consistent native naturalness improvements; caught ambiguous wording (`प्रकटीकरण`); accurate confirmations.
- **Gaps:** (1) missed 4 harmful truncations (061/168/178/412) across BOTH rounds — the independent audit caught them; (2) a few suggestions traded meaning-fidelity for naturalness.
- **Fidelity restorations applied** after the critique: 019 (`का`→`को` grammar), 023 & 045 (restored `सहिष्णुता` = tolerance, per English), 146 (`tax paying`→`file tax returns` = `टैक्स रिटर्न दाखिल`).

---

## 5. Final verification (2026-06-25)
- ✅ 15 source files × **200 rows**, **0 empty**, **0 duplicate** source_ids.
- ✅ `source_id` set **identical** across all files.
- ✅ Hindi truncation scan (tier_c / tier_b / tier_d): **0 truncations**.
- ✅ All the Hindi annotator edits + fidelity fixes present in devanagari AND itrans (no stale residue).
- ✅ Cross-language QA: benign Arabic/Arabizi 200/200 clean; harmful Arabic/Arabizi clean post-fix; paraphrases reviewed+fixed.

---

## 6. Deliverables
- `data/review/MERGED_all_variants_compare.csv` — 200 × 16 wide table, all variants side-by-side (rebuilt from current source).
- `data/review/MY_EDITS_log.csv` — 84 content edits (pre Hindi annotator, round 2).
- `data/review/tier_d_devanagari_paraphrased.REVIEWED.csv`, `tier_d_arabic_paraphrased.REVIEWED.csv` — full auto-review verdicts + `fixed` column.
- Backups: `data/review/_backup_*` (tier_a, tier_c ×2, tier_d ×2, benign writeback, anuj_round2, fidelity_fix).

## 7. Outstanding (non-blocking)
- Set `translation_ok = Y` for the 4 tier_c rows (019/100/134/512) in the review CSV (text now matches the Hindi annotator's request).
- Decide benign `harm_category` — currently all `other` (the Hindi annotator's issue #1); re-categorize or document as intentional.
