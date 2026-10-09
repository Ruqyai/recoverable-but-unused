# Interactive demo — Script-Alignment Surface

A self-contained static demo of the research. **No server, no GPU, no build step.**
Every number is generated from `results/` by `build_data.py` — nothing is hand-typed.

> **Reviewer bundle (this repo):** the demo is **fully self-contained** — `data.js` is pre-generated and
> `index.html` opens offline with **no `results/` needed**. The heavy per-run
> `results/` artifacts are **not shipped here**; regenerating `data.js` requires the full
> `results/` tree (main repository / OSF, anonymized for review). Just open `index.html`.

## Run it

```bash
# open the page — works straight from the filesystem, nothing to build
open demo/index.html

# (only if you have the full results/ tree) regenerate the baked data:
python3 demo/build_data.py        # writes demo/data.js

# 2. open the page — works straight from the filesystem
open demo/index.html              # macOS
# or: python3 -m http.server -d demo  then visit http://localhost:8000
```

## What's inside

Tabs follow the paper's narrative: story → setup → questions → one result tab per hypothesis (H1→H6) → tools.

| Tab | What it shows | Source |
|-----|---------------|--------|
| **Overview** | Headline finding + the two mechanisms (S vs C) + contributions | `headline`, `framing`, `contributions` (paper §1) |
| **Models · Data · Experiments** | 9-model table (family/size/causal layer/SAE/role), 5 data tiers, 6 experiments + a live model×experiment coverage matrix | `models_detail` (scripts/models + phase2), `tiers` (paper §3.1), `experiments` (live globs) |
| **Hypotheses** | H1–H6 cards: question · method · models · rejection criterion · verdict | `hypotheses` |
| **H1 · Behavioural** | Native vs romanized refusal bars + behavioural heatmap (refusal / engaged / off-topic) | `comprehension` (behavioral_readout) |
| **H2 · Direction** | Same-target contrast vs the saturation-confounded forward−reverse index + per-model 2×2 transfer matrix | `results/phase4_intervention/.../h2_results.json` |
| **H3 · Features** | SAE Jaccard ladder per orthography arm (paraphrase / language / script / benign) vs 0.4/0.6 thresholds | `jaccard` + `h3_controls_*.json` |
| **H4 · Comprehension** | The gate scatter (comprehension × SAE overlap; survivor highlighted) + comprehension heatmap | `results/phase5/*/relevance_judged.json` (Gemini judge, 6 models) × `jaccard` |
| **H5 · Geometry** | Silhouette per script vs Wang 0.24 reference + cross-script direction cosine | `results/phase3/.../{silhouette,cosine_similarity}.json` |
| **H6 · Patching** | The confirmatory test: matched/random/self/benign recovery curves + verdict per model | `results/phase4_patching/` |
| **Playground** | Flip a real (benign, safe) prompt between scripts; see how a model behaves on the matched *harmful* set | `data/tier_b/*` (benign only) + behavioural/judge/jaccard overlays |
| **Related Work** | Prior-work cards (they show / we find) + a results-vs-prior-claims comparison table | `related_work`, `result_comparisons` (paper §2) |

## Safety

Harmful prompts are **never displayed**. The Playground exposes only Tier B benign matched pairs.
Aggregate statistics from the harmful experiments (refusal/comprehension rates) are shown as numbers, not text.

## Provenance

- Behavioral breakdown → `results/phase5/<model>/run_*/behavioral_readout.json`
- SAE Jaccard ladder → `results/phase5/<model>/run_*/h3_controls_*.json` + `jaccard.json`
- H2 direction ablation → `results/phase4_intervention/<model>/run_*/h2_results.json` (same-target contrast)
- H6 patching → `results/phase4_patching/<model>/run_*/patching_results.json`
- Clean-judge comprehension (§4.1) → frozen, load-bearing numbers curated directly in `build_data.py` (transcribed from the paper, source-attributed in code).

Re-run `build_data.py` whenever `results/` changes; the page picks up the new `data.js` on reload.
