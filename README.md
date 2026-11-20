# Recoverable but Unused

**Cross-Script Jailbreaks Are a Failure to Act on Meaning the Model Can Recover**

Ten open instruction-tuned models (Gemma, Qwen, Llama), three rewritings: Arabic to Arabizi,
Hindi Devanagari to ITRANS, English to Leetspeak (200 harmful requests per cell; 12,000 main-grid
responses).

Anonymized for review.

---

## Run the demo

Open `demo/index.html` in any browser. It works offline, `data.js` is pre-generated, and there is
no build step. Or serve it with `python3 -m http.server -d demo` and visit http://localhost:8000.

The page never displays a harmful request. Only the benign matched pairs appear in the interactive
panel, and the harmful sets are shown as aggregates.

---

## The finding

Instruction-tuned models refuse a harmful request in its native script and answer the same request
written in Latin letters. The usual reading is that safety training is tied to the script. A second
reading is that the model cannot read the rewritten text. Neither survives these measurements.

Two tests ask whether the model recovers the meaning, and neither lets it produce a response, so a
refusal cannot enter the measure.

- **Transliteration.** Asked to rewrite ITRANS as Devanagari across 200 items, five of six models
  score above 97% character-similarity accuracy.
- **Meaning identification (within-language and cross-language).** Shown the rewritten request and
  four near-paraphrase candidate meanings, models pick the right one in 42% to 97.5% of items,
  exceeding engagement in every interpretable cell by 22 to 81 percentage points, and by 15 to 54
  points across all ten cells when options are presented in English with no native script on screen.

Across all 12,000 main-grid responses (200 per cell), the model neither refuses nor complies:
specific usable harmful material appears in at most 4.5% of items in any rewritten cell (1.0%
overall, and at most 4.0% at 256 tokens across fifteen cells), while off-topic text is the largest
outcome on all three rewritings.

Making the recovery explicit restores refusal. Asking the model to rewrite the request and then
answer its own rewriting improves refusal in eleven of fifteen cells across Arabizi, ITRANS, and
Leetspeak with no retraining, closing up to 79% of the gap to the native-script ceiling ($r = 0.62$
with decoding quality).

---

## What is here

| Path | What it holds |
|---|---|
| `demo/` | The static page, its pre-generated data, and the script that builds it |
| `results/` | Every result file the paper's numbers are computed from |
| `scripts/` | The code that produced them, by phase |
| `data/` | The corpora: harmful and benign, in each script |

`demo/build_data.py` regenerates `demo/data.js` from `results/`. Nothing in the page is hand-typed.

The activation tensors are not shipped here. They are large and nothing in the paper is read off
them directly.

Responses judged to contain specific usable harmful content have their text withheld and replaced
with a marker (`redacted: true`). Their labels are kept, so every statistic in the paper can still
be recomputed from these files.

---

## Reproducing a number

Every table and figure in the paper is produced by the scripts in `scripts/ops/` from the files in
`results/`. The recovery tests are `07_forced_choice_reading.py` and `09_cross_language_reading.py`
in `scripts/phase5_sae_features/comprehension_validation/`, the harm judgments are `06_harm_judge.py`
and `06b_harm_judge_256.py`, the mitigation is `08_decode_then_refuse.py`, and the cross-native
direction transfer baseline is `scripts/phase4_causal_intervention/interventions/03_cross_native_transfer.py`.

---

## Pre-registration

The activation-patching test was pre-registered before the mechanistic analyses and is reported as
the null it returned. Every other analysis is post-hoc and labelled as such. The pre-registration is
linked from the demo's Pre-registration tab.
