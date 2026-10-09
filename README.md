# Recoverable but Unused

**Cross-Script Jailbreaks Are a Failure to Act on Meaning the Model Can Recover**

Ten open instruction-tuned models (Gemma, Qwen, Llama), three rewritings: Arabic to Arabizi,
Hindi Devanagari to ITRANS, English to Leetspeak.

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

- **Transliteration.** Asked to rewrite ITRANS as Devanagari, five of six models score above 97%
  character similarity.
- **Meaning identification.** Shown the rewritten request and four candidate meanings in the native
  script, models pick the right one in 42% to 97.5% of items, read from the output distribution.

Recovery exceeds engagement in every interpretable cell, by 22 to 81 percentage points.

The model neither refuses nor complies. Scoring every on-topic response for harmful content,
specific usable material appears in at most 4% of items in any cell, and is zero in 37 of 60 cells.
What replaces the lost refusals is off-topic text.

Making the recovery explicit restores refusal. Asking the model to rewrite the request and then
answer its own rewriting improves refusal in eight of ten cells with no retraining, closing up to
79% of the gap to the native-script ceiling.

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

Twenty-seven responses were judged to contain specific usable harmful content. Their text is
withheld and replaced with a marker. Their labels are kept, so every statistic in the paper can
still be recomputed from these files.

---

## Reproducing a number

Every table and figure in the paper is produced by the scripts in `scripts/ops/` from the files in
`results/`. The recovery test is `scripts/phase5_sae_features/comprehension_validation/07_forced_choice_reading.py`,
the harm judgment is `06_harm_judge.py` beside it, and the mitigation is `08_decode_then_refuse.py`.

---

## Pre-registration

The activation-patching test was pre-registered before the mechanistic analyses and is reported as
the null it returned. Every other analysis is post-hoc and labelled as such. The pre-registration is
linked from the demo's Pre-registration tab.
