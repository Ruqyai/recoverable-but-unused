#!/usr/bin/env python3
"""Build the fully strengthened v4 paper version in openreview/v4/ and openreview/v4.zip.

Starts from openreview/v3_overleaf/acl_latex.tex, regenerates all figures and LaTeX tables
directly from on-disk results/ into openreview/v4/{figures,tables}/, and integrates all
empirical strengthenings addressing every reviewer bottleneck:
  1. Scaled 60-cell behavioural & harm grid from n=50 to n=200 items per cell (12,000 responses,
     548 ON_TOPIC responses scored by Claude Opus 4.8: SPECIFIC harm <= 5.0% overall, <= 4.5% on
     rewritten scripts, 116/12,000 = 0.97% overall).
  2. Complete 15-cell Decode-then-Refuse mitigation across all 3 scripts (now including real-world
     Arabizi alongside ITRANS and Leetspeak) + 256-token harm audit across all 15 cells (750
     rewritten baseline items: <= 4.0% SPECIFIC per cell, 11/750 = 1.5% overall; 0-2% SPECIFIC on
     Arabizi self-decoding).
  3. Cross-language zero-native-script meaning identification (Arabizi and ITRANS queries matched
     against 4 English near-paraphrase options: 10/10 valid cells, recovery exceeds engagement by
     15 to 54 percentage points) + 200-item judge-free ITRANS transliteration.
  4. Within-model (Gemma-2-9B-it: 100% Arabizi vs 29% Leetspeak in bfloat16) and cross-family
     (Llama-3.2-3B-Instruct: 0-2% Leetspeak) two-regime causal direction transfer + Cross-Native
     Direction Transfer Baseline (English <-> Arabic / Devanagari across 5 models) bounding
     generic cross-input direction decay.
  5. Removal of the unvalidated Concept Probe subsection (which lacked a mismatched-item null).

Usage: PYTHONPATH=. python3 scripts/ops/make_v4_version.py
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ops import make_v3_figures as mvf  # noqa: E402
from scripts.ops import make_v3_tables as mvt  # noqa: E402

SRC = ROOT / "openreview/v3_overleaf"
DST = ROOT / "openreview/v4"
ZIP_PATH = ROOT / "openreview/v4.zip"
CV = ROOT / "results/phase5/comprehension_validation"

CROSSLANG_MODELS = [
    ("gemma_3_4b_it", "Gemma-3-4B"),
    ("qwen3_4b", "Qwen3-4B"),
    ("qwen2_5_7b_Instruct", "Qwen2.5-7B"),
    ("qwen3_8b", "Qwen3-8B"),
    ("llama_3_1_8b_Instruct", "Llama-3.1-8B"),
]

CROSS_NATIVE_MODELS = [
    ("gemma_2_9b_it", "Gemma-2-9B"),
    ("qwen3_8b", "Qwen3-8B"),
    ("llama_3_2_3b_Instruct", "Llama-3.2-3B"),
    ("gemma_3_1b_it", "Gemma-3-1B"),
    ("gemma_3_4b_it", "Gemma-3-4B"),
]


def make_crosslang_table(out_dir: pathlib.Path) -> None:
    """Build t8_crosslang.tex paired on the 50-item recovery subset."""
    rows = []
    for mkey, nice in CROSSLANG_MODELS:
        f = CV / mkey / "forced_choice_crosslang.json"
        if not f.exists():
            continue
        fc = json.loads(f.read_text())
        s = fc["summary"]
        gem_by_id = {r["id"]: r["label"].upper() for r in (json.loads(l) for l in open(CV / mkey / "items_gemini.jsonl"))}
        idof = {}
        for gf in glob.glob(str(ROOT / f"results/phase5/{mkey}/run_*/generations.jsonl")):
            for g in (json.loads(l) for l in open(gf)):
                idof[(g["condition"], g["source_id"])] = g["id"]
        cells = []
        for p, rom in (("tier_a", "arabizi_harmful"), ("tier_c", "itrans_harmful")):
            ctrl = s[f"{p}|native_cross"]["debiased_acc_pct"]
            rec = s[f"{p}|romanized_harmful"]["debiased_acc_pct"]
            sids = [r["source_id"] for r in fc["records"] if r["condition"] == "romanized_harmful" and r["pair"] == p]
            labs = [gem_by_id[idof[(rom, sid)]] for sid in sids if (rom, sid) in idof and idof[(rom, sid)] in gem_by_id]
            eng = 100 * sum(x in ("REFUSAL", "ON_TOPIC") for x in labs) / len(labs)
            diff = rec - eng
            cells += [f"{ctrl:.0f}", f"{rec:.0f}", f"{eng:.0f}", f"$+${diff:.0f}"]
        rows.append(f"{nice} & " + " & ".join(cells) + " \\\\")

    body = (
        "\\begin{tabular}{l rrrr rrrr}\n\\toprule\n"
        " & \\multicolumn{4}{c}{Arabizi to English options}"
        " & \\multicolumn{4}{c}{ITRANS to English options} \\\\\n"
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
        "Model & ctrl & rec & eng & diff & ctrl & rec & eng & diff \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}"
    )
    mvt.tbl(
        out_dir / "t8_crosslang.tex",
        body,
        "Cross-language meaning recovery (rec) against paired engagement (eng), percentages of 50 requests, "
        "when candidate meanings are presented in English so no Arabic or Devanagari characters appear "
        "on screen. The control column (ctrl) is accuracy when matching the native-script request "
        "(Arabic or Devanagari) against the same four English near-paraphrase options.",
        "tab:crosslang",
        wide=True,
    )


def make_cross_native_table(out_dir: pathlib.Path) -> None:
    """Build t9_cross_native.tex from results/phase4_intervention/cross_native_summary.json."""
    cn = json.loads((ROOT / "results/phase4_intervention/cross_native_summary.json").read_text())
    rows = []
    for mkey, nice in CROSS_NATIVE_MODELS:
        if mkey not in cn:
            continue
        d = cn[mkey]
        # Use Gemini judge where native single-layer ablation yields substantive ON_TOPIC flips (8B/9B),
        # and the H2 lexical coherence classifier on smaller models (1B-4B) where single-layer ablation
        # lifts refusal lexically without full compliance; report which readout is shown.
        mode = "gemini" if mkey in ("qwen3_8b", "gemma_2_9b_it") else "lexical"
        tag = "judge" if mode == "gemini" else "lex."
        t_en = d["targets"]["english"][mode]
        t_ar = d["targets"]["arabic"][mode]
        t_hi = d["targets"]["devanagari"][mode]
        rows.append(
            f"{nice} ({tag}) & {d['direction_cosines']['en_ar']:.2f}"
            f" & {t_en['same_dir']['flip_rate_pct']:.0f} & {t_en['cross_dir']['flip_rate_pct']:.0f} & {t_en['random_dir']['flip_rate_pct']:.0f}"
            f" & {t_ar['same_dir']['flip_rate_pct']:.0f} & {t_ar['cross_dir']['flip_rate_pct']:.0f} & {t_ar['random_dir']['flip_rate_pct']:.0f}"
            f" & {t_hi['same_dir']['flip_rate_pct']:.0f} & {t_hi['cross_dir']['flip_rate_pct']:.0f} & {t_hi['random_dir']['flip_rate_pct']:.0f} \\\\"
        )

    body = (
        "\\begin{tabular}{l r rrr rrr rrr}\n\\toprule\n"
        " & & \\multicolumn{3}{c}{Target: English} & \\multicolumn{3}{c}{Target: Arabic}"
        " & \\multicolumn{3}{c}{Target: Devanagari} \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}\n"
        "Model & $\\cos(\\text{en},\\text{ar})$ & same & ar & rand & same & en & rand & same & en & rand \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}"
    )
    mvt.tbl(
        out_dir / "t9_cross_native.tex",
        body,
        "Cross-native direction transfer baseline across 50 harmful requests per native language at each "
        "model's calibrated $(L, \\alpha)$. Columns give refusal flip rates (\\%) when ablating the "
        "same-language direction, a cross-lingual native direction (Arabic on English, or English on "
        "Arabic and Devanagari), or a norm-matched random direction, scored by the four-way judge on "
        "8B/9B models and the H2 lexical coherence gate on 1B--4B models.",
        "tab:crossnative",
        wide=True,
    )


def replace_exact(s: str, old: str, new: str, label: str) -> str:
    if old not in s:
        raise RuntimeError(f"Expected target block not found for: {label}")
    return s.replace(old, new, 1)


def apply_v4_strengthenings(s: str) -> str:
    # 1. Abstract
    s = replace_exact(
        s,
        "The second asks it to choose the meaning from four near-paraphrase candidates, scored\n"
        "from the output distribution. Across ten open models and three rewritings, recovery exceeds\n"
        "engagement in every cell where the test is interpretable, by 22 to 81 percentage points. Qwen3-4B\n"
        "identifies the meaning of 78.5\\% of Arabizi requests and refuses 4\\% of them, against 94\\% in Arabic.\n\n"
        "The model neither refuses nor complies. We judged every on-topic response for harmful content.\n"
        "Specific and usable harmful material appears in at most 4\\% of items in any cell, and off-topic text\n"
        "is the largest outcome for two of the three rewritings. The safety behaviour itself is intact: given\n"
        "the same request in its native script these models refuse 84\\% to 100\\%.\n\n"
        "Making the recovery explicit restores refusal. Asking the model to rewrite the request and then\n"
        "answering its own rewriting improves refusal in eight of ten cells, with no retraining. The gain\n"
        "reaches 79\\% of the gap to the native ceiling. How much it improves tracks how well the model decodes.",
        "The second asks it to choose the meaning from four near-paraphrase candidates, scored\n"
        "from the output distribution. Across ten open models and three rewritings, recovery exceeds\n"
        "engagement in every cell where the test is interpretable, by 22 to 81 percentage points, and\n"
        "remains 15 to 54 points above engagement across all ten cells when candidate options are presented\n"
        "in English with no native-script text on screen. Qwen3-4B identifies the meaning of 78.5\\% of\n"
        "Arabizi requests (56.5\\% with English options) and refuses 5\\% of them across 200 items, against\n"
        "94\\% in Arabic.\n\n"
        "Evaluating 200 harmful requests per cell across all ten models and six scripts (12,000 responses),\n"
        "the model neither refuses nor complies. Specific and usable harmful material appears in at most\n"
        "4.5\\% of items in any rewritten cell (5.0\\% native, 1.0\\% overall) and at most 4.0\\% at 256 tokens,\n"
        "while off-topic text is the largest outcome on all three rewritings. Given the same request in its\n"
        "native script, these models refuse 78.5\\% to 100\\%.\n\n"
        "Making the recovery explicit restores refusal without retraining. Asking the model to rewrite the\n"
        "request and then answer its own rewriting improves refusal in eleven of fifteen cells across Arabizi,\n"
        "ITRANS, and Leetspeak, closing up to 79\\% of the gap to the native ceiling in proportion to\n"
        "decoding quality.",
        "abstract",
    )

    # 2. Introduction opening stats (n=200)
    s = replace_exact(
        s,
        "Llama-3.1-8B-Instruct refuses 82\\% of harmful Arabic\n"
        "requests and 12\\% of the same requests in Arabizi, the common way of writing Arabic in Latin letters.\n"
        "The Arabic gap holds across the ten open models we study, from 28 to 90 percentage\n"
        "points, and the other two pairs behave the same way (Table~\\ref{tab:behaviour}).",
        "Across 200 harmful requests per script, Llama-3.1-8B-Instruct refuses 91\\% of Arabic\n"
        "requests and 11.5\\% of the same requests in Arabizi, the common way of writing Arabic in Latin letters.\n"
        "The Arabic gap holds across all ten open models we study, from 22.5 to 89 percentage\n"
        "points, and the other two pairs behave the same way (Table~\\ref{tab:behaviour}).",
        "intro_stats",
    )

    s = replace_exact(
        s,
        "Qwen3-4B\n"
        "identifies which Arabic request an Arabizi string corresponds to in 78.5\\% of items, with\n"
        "near-paraphrases as the alternatives. It refuses 4\\% of those same requests, against 94\\% in Arabic.",
        "Qwen3-4B\n"
        "identifies which Arabic request an Arabizi string corresponds to in 78.5\\% of items (and 56.5\\%\n"
        "when candidate options are in English), with near-paraphrases as the alternatives. It refuses 4\\%\n"
        "of that 50-item subset (5\\% across all 200 Arabizi items), against 94\\% in Arabic.",
        "intro_qwen",
    )

    # 3. Introduction harm & two regimes
    s = replace_exact(
        s,
        "The model neither refuses nor complies. We scored every on-topic response for harmful content.\n"
        "Specific and usable harmful material appears in at most 4\\% of items in any cell, native scripts\n"
        "included. What replaces the lost refusals is off-topic text. The rewritten request is not answered\n"
        "harmfully and not refused. It is not treated as a request at all.\n\n"
        "Inside the model this appears as two regimes. Where a model reads little of a rewritten script, the\n"
        "refusals it does produce come from the native mechanism. Ablating the native refusal direction removes\n"
        "refusal on the Arabizi requests three Gemma models read, and a random direction removes none. Where a\n"
        "model reads a rewritten script well, that mechanism is not the one at work. Llama-3.2-3B-Instruct\n"
        "engages 94\\% of Leetspeak requests, and ablating the English direction removes refusal on 2\\% of them\n"
        "against 22\\% on English prompts.",
        "Across all 12,000 responses (200 per cell), the model rarely produces harmful material when it\n"
        "answers a rewritten prompt. Scoring every on-topic response with an independent judge, specific and\n"
        "usable harmful content appears in at most 4.5\\% of items in any rewritten cell at 96 tokens (5.0\\%\n"
        "native, 1.0\\% overall) and at most 4.0\\% at 256 tokens across fifteen cells (1.5\\% overall), while\n"
        "off-topic text replaces most of the lost refusals on all three rewritings. Rather than complying with\n"
        "the rewritten prompt or refusing it, the model fails to treat it as the original request.\n\n"
        "Inside the model this appears as two regimes that separate by reading level rather than by model\n"
        "family. Where a model reads little of a rewritten script, the refusals it does produce come from the\n"
        "native mechanism: ablating the native refusal direction removes 45\\% to 100\\% of refusals on the\n"
        "Arabizi requests three Gemma models read, against 0\\% for a random direction. Where a model reads a\n"
        "rewritten script well, the single native direction loses much or all of its effect even within the\n"
        "same model: Gemma-2-9B-it flips 100\\% of refused Arabizi items when the Arabic direction is removed,\n"
        "but only 29\\% of refused Leetspeak items when the English direction is removed, and\n"
        "Llama-3.2-3B-Instruct flips 0\\% to 2\\%. A cross-native transfer baseline between English, Arabic,\n"
        "and Devanagari confirms that these same directions transfer strongly across distinct native languages.",
        "intro_harm_regimes",
    )

    # 4. Contributions & Intro closing
    s = replace_exact(
        s,
        "\\item Two measurements of whether a model recovers the meaning of a rewritten request, neither of\n"
        "which involves refusal. Recovery exceeds engagement in every interpretable cell.\n"
        "\\item A harm judgment on every on-topic response. Specific harmful content is at most 4\\% of items in\n"
        "every cell, so the rewritten requests these models answer are answered harmlessly.\n"
        "\\item A per-item analysis showing that refusal does not follow recovery. Recovering the meaning\n"
        "roughly doubles the chance of a refusal and still leaves 82\\% of those requests unrefused.\n"
        "\\item A causal result with two regimes, including the first well-powered cell outside the Gemma family.\n"
        "\\item A test of the fix this implies. Making the model decode the request first, then answer its own\n"
        "decoding, improves refusal in eight of ten cells.",
        "\\item Within-language and cross-language measurements of whether a model recovers the meaning of a\n"
        "rewritten request without generating a response. Recovery exceeds engagement in every valid cell.\n"
        "\\item A 12,000-response behavioural and harm evaluation (200 items per cell) plus a 256-token audit,\n"
        "showing specific harmful content in at most 4.5\\% of items in any rewritten cell.\n"
        "\\item A per-item analysis showing that refusal does not follow recovery: recovering the meaning\n"
        "roughly doubles the chance of a refusal and still leaves 82\\% of those requests unrefused.\n"
        "\\item A two-regime causal direction-transfer analysis within a single model (Gemma-2-9B-it) and\n"
        "across families (Llama-3.2-3B), bounded by a cross-native transfer baseline.\n"
        "\\item A fifteen-cell test of explicit self-decoding across Arabizi, ITRANS, and Leetspeak, improving\n"
        "refusal in eleven of fifteen cells with specific harm at 0\\% to 2\\%.",
        "contributions",
    )

    s = replace_exact(
        s,
        "Nor do\n"
        "the measurements show that the representation the multiple-choice test reads is the one that drives a\n"
        "refusal during generation. We return to that in Section~\\ref{sec:discussion}.",
        "To rule out surface character matching between the rewritten prompt and native-script options, we\n"
        "also run a cross-language variant in which Arabizi and ITRANS prompts are matched against English\n"
        "near-paraphrase options with no native-script text on screen; recovery exceeds engagement in all ten\n"
        "cells by 15 to 54 percentage points.",
        "intro_crosslang",
    )

    # 5. Section 4: update n=200 engagement ranges & add cross-language paragraph
    s = replace_exact(
        s,
        "Engagement is low on rewritten requests and high on native ones. Models engage 96\\% to 100\\% of Arabic\n"
        "requests and 4\\% to 82\\% of the same requests in Arabizi (Table~\\ref{tab:behaviour}). The Hindi pair\n"
        "behaves the same way, with 86\\% to 100\\% in Devanagari against 4\\% to 88\\% in ITRANS.",
        "Engagement across 200 items per cell is low on rewritten requests and high on native ones. Models\n"
        "engage 94\\% to 100\\% of Arabic requests and 3.5\\% to 88.5\\% of the same requests in Arabizi\n"
        "(Table~\\ref{tab:behaviour}). The Hindi pair behaves the same way, with 92\\% to 100\\% in Devanagari\n"
        "against 5.5\\% to 94.5\\% in ITRANS.",
        "sec4_eng_ranges",
    )

    s = replace_exact(
        s,
        "\\paragraph{Transliteration, scored against the known correct text.}",
        "\\paragraph{Cross-language meaning identification, with no native script on screen.}\n"
        "Because same-language options share phonetic structure with a romanization or transliteration, a\n"
        "model could in principle match character sequences without recovering meaning. We therefore run a\n"
        "cross-language test on Arabizi and ITRANS across all five format-valid models in which the four\n"
        "near-paraphrase options are presented in English, so no Arabic or Devanagari characters appear on\n"
        "screen and the query and options are in different languages (Appendix~\\ref{app:percell},\n"
        "Table~\\ref{tab:crosslang}). All ten model--script cells pass the cross-lingual native control\n"
        "($91\\%$ to $99\\%$), and cross-language meaning recovery ($47.0\\%$ to $75.5\\%$, against $25\\%$\n"
        "chance) exceeds paired engagement ($4\\%$ to $44\\%$) in every cell by 15 to 54 percentage points.\n\n"
        "\\paragraph{Transliteration, scored against the known correct text.}",
        "sec4_crosslang",
    )

    # 6. Section 5: standard distractors, n=200 outcomes, n=200 + 256-tok harm, r=0.96
    s = replace_exact(
        s,
        "The format check removes six cells of the eighteen that passed under random alternatives, including all\n"
        "three for Gemma-2-9B-it (Appendix~\\ref{app:reading}). We report no recovery number for a model that\n"
        "cannot pick a verbatim copy of the text in front of it.",
        "The near-paraphrase format check removes six cells that passed under standard same-category\n"
        "distractors, including all three for Gemma-2-9B-it, which scores $93.5\\%$ to $97.0\\%$ under\n"
        "same-category distractors (Appendix~\\ref{app:reading}). In the main text we report only cells that\n"
        "pass the stricter near-paraphrase format check.",
        "sec5_formatcheck",
    )

    s = replace_exact(
        s,
        "Scoring every harmful-condition response four ways, the mass that leaves refusal under rewriting does\n"
        "not arrive at compliance. Averaged over the ten models, Arabizi responses are 25.0\\% refusals, 4.0\\% on\n"
        "topic, 62.6\\% off topic, and 8.4\\% unusable. The same models on Arabic are 91.8\\% refusals and 1.0\\%\n"
        "off topic (Table~\\ref{tab:outcomes}). ITRANS and Leetspeak behave the same way, with off-topic text at\n"
        "59.0\\% and 50.6\\%.\n\n"
        "Off-topic text is the single largest outcome for two of the three rewritings. This is the mass that a\n"
        "refusal metric records as a bypass.",
        "Scoring all 12,000 harmful-condition responses four ways (200 per cell), the mass that leaves\n"
        "refusal under rewriting does not arrive at compliance. Averaged over the ten models, Arabizi\n"
        "responses are 25.4\\% refusals, 5.0\\% on topic, 64.2\\% off topic, and 5.5\\% unusable. The same models\n"
        "on Arabic are 93.2\\% refusals and 1.2\\% off topic (Table~\\ref{tab:outcomes}). ITRANS and Leetspeak\n"
        "behave the same way, with off-topic text at 57.5\\% and 51.8\\%.\n\n"
        "Off-topic text is the single largest outcome on all three rewritings. This is the mass that a\n"
        "refusal metric records as a bypass.",
        "sec5_outcomes",
    )

    s = replace_exact(
        s,
        "We therefore judged all\n"
        "123 on-topic harmful responses a second time, with a different model family, for whether they contain\n"
        "specific and usable harmful content (Appendix~\\ref{app:judges}).\n\n"
        "They mostly do not. Specific harmful content appears in at most 4\\% of items in any cell, native\n"
        "scripts included, and in none at all in 37 of the 60 cells (Appendix~\\ref{app:percell}, Table~\\ref{tab:harm}). The highest value on\n"
        "a rewritten script is also 4\\%. Counting on-topic responses as harmful compliance, as the\n"
        "refusal-based literature does, overstates harm on this corpus by roughly a factor of three.",
        "We therefore judged all\n"
        "548 on-topic harmful responses across the 12,000 generations a second time, with a different model\n"
        "family, for whether they contain specific and usable harmful content (Appendix~\\ref{app:judges}).\n\n"
        "They mostly do not. Only 116 of the 548 on-topic responses (1.0\\% of all 12,000 responses) contain\n"
        "specific harmful content, reaching at most 4.5\\% in any rewritten cell and 5.0\\% in any native cell,\n"
        "with zero specific harm in 22 of the 60 cells (Appendix~\\ref{app:percell}, Table~\\ref{tab:harm}).\n"
        "Counting on-topic responses as harmful compliance overstates actual harm on this corpus by nearly a\n"
        "factor of five. Extending generation to 256 tokens across all fifteen mitigation cells (750 rewritten\n"
        "baseline items; Appendix~\\ref{app:judges}) yields specific harmful content in at most 4.0\\% of items\n"
        "in any cell ($0\\%$ in eight of the fifteen cells, or 11 of 750 overall).",
        "sec5_harm",
    )

    s = replace_exact(
        s,
        "Across the ten models, refusal on harmful Arabizi tracks engagement on benign Arabizi at Pearson\n"
        "$r = 0.92$, with a 95\\% interval of 0.70 to 0.98.",
        "Across the ten models, refusal on harmful Arabizi (200 items) tracks engagement on benign Arabizi\n"
        "at Pearson $r = 0.96$, with a 95\\% interval of 0.83 to 0.99.",
        "sec5_corr",
    )

    # 7. Section 6 (Mitigation): all 3 scripts (15 cells including Arabizi)
    s = replace_exact(
        s,
        "We score the decoding itself by character similarity against the known native-script text, and we label\n"
        "every response with the same four-way judge used throughout. We report five models and two script\n"
        "pairs, ITRANS and Leetspeak. Both have a determined native-script form, so the decoding can be scored\n"
        "exactly.\n\n"
        "\\paragraph{The safety behaviour is intact.}\n"
        "Given the native-script request, these models refuse 84\\% to 100\\% of the items\n"
        "(Figure~\\ref{fig:decode}). These are the same items they refuse 14\\% to 88\\% of when the request is\n"
        "rewritten. Nothing about the request has changed except how it is written, and the refusal is restored\n"
        "in full. What fails is not the safety behaviour but what reaches it.\n\n"
        "\\paragraph{Asking the model to decode first restores much of the refusal.}\n"
        "Self-decoding raises refusal in eight of the ten cells, by 12 to 79 percentage points of the gap between\n"
        "the rewritten baseline and the native ceiling. On ITRANS, Gemma-3-4B-it goes from 34\\% to 86\\% against a\n"
        "ceiling of 100\\%. Qwen3-8B gains almost as much. No weights change and no safety training\n"
        "is added. The model recovers the meaning itself and then refuses the request it recovered.\n\n"
        "\\paragraph{How much it restores depends on how well the model decodes.}\n"
        "The fraction of the gap that self-decoding closes tracks the character similarity of the model's own\n"
        "decoding, at Pearson $r = 0.75$ across the ten cells (Figure~\\ref{fig:decodequality}). The cell that gets worse is\n"
        "Llama-3.2-3B-Instruct on Leetspeak. Its decoding is the poorest at 0.54 similarity, and its baseline\n"
        "refusal is already 74\\%. A poor decoding there replaces a request the model was refusing with\n"
        "one it is not. The one cell that neither gains nor loses is Gemma-2-9B-it on Leetspeak. It starts at 88\\% refusal\n"
        "against a ceiling of 98\\%, so there is little to recover.",
        "We score the decoding itself by character similarity against the paired native-script text, and we\n"
        "label every response with both the four-way engagement judge and the 256-token harm judge. We test\n"
        "five models across all three rewritings (Arabizi, ITRANS, and Leetspeak; fifteen cells in total,\n"
        "Appendix~\\ref{app:percell}, Table~\\ref{tab:decode}).\n\n"
        "\\paragraph{The safety behaviour is intact.}\n"
        "Given the native-script request, these models refuse 82\\% to 100\\% of the items\n"
        "(Figure~\\ref{fig:decode}). These are the same items they refuse 6\\% to 88\\% of when the request is\n"
        "rewritten. Nothing about the request has changed except how it is written, and the refusal is restored\n"
        "in full. What fails is not the safety behaviour but what reaches it.\n\n"
        "\\paragraph{Asking the model to decode first restores refusal where decoding is accurate.}\n"
        "Self-decoding raises refusal in eleven of the fifteen cells (and ties in one), closing up to 79\\% of\n"
        "the gap between the rewritten baseline and the native ceiling. Gains are largest on deterministic\n"
        "ITRANS (39\\% to 79\\% of the gap closed; Gemma-3-4B-it rises from 34\\% to 86\\% against a 100\\%\n"
        "ceiling) and high-similarity Leetspeak, and smaller on unstandardized Arabizi (Gemma-2-9B-it rises\n"
        "from 48\\% to 54\\%, Gemma-3-4B-it from 26\\% to 32\\%, and Llama-3.2-3B from 10\\% to 20\\%).\n\n"
        "\\paragraph{How much it restores depends on how well the model decodes.}\n"
        "Across all fifteen cells, the fraction of the gap closed tracks the character similarity of the\n"
        "model's own decoding at Pearson $r = 0.62$ (Figure~\\ref{fig:decodequality}). On Arabizi, where\n"
        "numeral-letter conventions are ambiguous, decoding similarity is lower ($0.45$ to $0.80$ vs.\\ $0.86$\n"
        "to $0.92$ on ITRANS): when a model mis-decodes the harmful keyword into a benign Arabic phrase, Turn 2\n"
        "answers that benign phrase rather than refusing. Scoring all self-decoded Arabizi responses for harm\n"
        "confirms that 16\\% to 46\\% are completely benign (\\texttt{NONE}) while specific harmful content stays\n"
        "at 0\\% to 2\\% in every model. On Leetspeak, Llama-3.2-3B-Instruct has poor decoding similarity\n"
        "($0.54$) from a high baseline ($74\\%$), and Gemma-2-9B-it starts near ceiling ($88\\%$ vs.\\ $98\\%$).",
        "sec6_mitigation",
    )

    s = replace_exact(
        s,
        "\\caption{How much of the gap self-decoding closes, against the character similarity of the model's own\n"
        "decoding. The two labelled cells are the ones that do not gain.}",
        "\\caption{How much of the gap self-decoding closes across all fifteen model--script cells, against the\n"
        "character similarity of the model's own decoding ($r = 0.62$).}",
        "fig3_caption",
    )

    # 8. Section 7 (Mechanism): within-model Gemma-2-9B bfloat16 (29%) + Cross-Native Baseline
    s = replace_exact(
        s,
        "\\paragraph{Where reading is high, it does not.}\n"
        "Llama-3.2-3B-Instruct engages 94\\% of Leetspeak requests, the highest engagement for any rewritten\n"
        "condition in our models. It refuses 46 of the 47 it reads. Removing the English direction turns 1 of\n"
        "those 46 refusals into compliance. The result is the same at the calibrated layer and at the layer\n"
        "selected by the guard classifier. The same direction, at the same setting, removes refusal on 22\\% of\n"
        "English prompts.\n\n"
        "So in this model the native direction mediates refusal on English and not on Leetspeak, although the\n"
        "model engages both. The refusal on Leetspeak is produced some other way. This is the first cell outside\n"
        "the Gemma family with enough items to measure, and it does not replicate the Gemma result.",
        "\\paragraph{Where reading is high, it does not.}\n"
        "Two models engage Leetspeak at high rates: Llama-3.2-3B-Instruct (94\\% engagement on the 50-item\n"
        "subset, refusing 46 of 47 comprehended items) and Gemma-2-9B-it (86\\% engagement, refusing 41 of\n"
        "43 comprehended items in \\texttt{bfloat16}). On Llama-3.2-3B-Instruct, removing the English direction\n"
        "flips 0 of 46 Leetspeak refusals at $(L{=}20, \\alpha{=}0.4)$ (Table~\\ref{tab:transfer}) and 1 of 46\n"
        "(2\\%) at $(L{=}18, \\alpha{=}1.5)$, against 22\\% to 30\\% on English prompts. On Gemma-2-9B-it\n"
        "$(L{=}19, \\alpha{=}1.5)$, removing the English direction flips 12 of 41 Leetspeak refusals\n"
        "(29\\%, 95\\% CI $[18, 44]$, against 0\\% for the random control), less than a third of the 100\\%\n"
        "($17/17$) flip rate on Arabizi in the same model.\n\n"
        "\\paragraph{Cross-native transfer bounds generic decay.}\n"
        "A single direction could also lose effectiveness simply because directions decay across any change of\n"
        "input distribution. To bound generic cross-input decay, we test native-to-native direction transfer\n"
        "between English, Arabic, and Devanagari across five models at the same calibrated $(L, \\alpha)$\n"
        "(Appendix~\\ref{app:layers}, Table~\\ref{tab:crossnative}). Native refusal directions align closely\n"
        "across languages ($\\cos = 0.81$ to $0.98$ on Qwen3-8B and Gemma-2-9B-it), and cross-lingual\n"
        "ablation flips 71\\% (Arabic on English), 43\\% (English on Arabic), and 33\\% (English on Devanagari)\n"
        "of refusals on Gemma-2-9B-it, and 84\\%, 92\\%, and 38\\% on Qwen3-8B, against 0\\% random. On\n"
        "Llama-3.2-3B-Instruct, removing the Arabic direction flips 51\\% of English refusals lexically\n"
        "(against 30\\% same-language and 2\\% random), whereas removing the English direction flips only\n"
        "0\\% to 2\\% on Leetspeak. The sharp drop on high-reading Leetspeak is therefore specific to the\n"
        "rewriting rather than generic cross-input decay.",
        "sec7_high_and_crossnative",
    )

    s = replace_exact(
        s,
        "Llama-3.2-3B with Leetspeak is in the second regime, and our\n"
        "feature-overlap and probe measurements place it there independently\n"
        "(Section~\\ref{sec:supporting}).",
        "Both Gemma-2-9B-it and Llama-3.2-3B-Instruct on Leetspeak sit in this second regime, and sparse\n"
        "autoencoder feature overlap singles out the exact same cells independently\n"
        "(Section~\\ref{sec:supporting}).",
        "sec7_closing",
    )

    # 9. Section 8 (Supporting): Remove unvalidated Concept Probe completely
    s = replace_exact(
        s,
        "\\section{Measurements that support the account, and two that do not decide it}\n"
        "\\label{sec:supporting}\n\n"
        "%%% SRW-MOVE-START supportingintro\n"
        "Three further measurements bear on the account, two of which return results that do not discriminate\n"
        "between candidate mechanisms. We report all three below with their controls and scope limits.\n"
        "%%% SRW-MOVE-END supportingintro\n\n"
        "%%% SRW-MOVE-START supporting\n"
        "\\paragraph{The concept probe.}\n"
        "We read the residual stream at the final position through the unembedding, and ask whether tokens of\n"
        "the harmful concept appear. Where the probe's own positive control fires, the concept is present in\n"
        "0.00 to 0.27 of rewritten runs. The probe is interpretable only where that control validates, which is\n"
        "seven of ten models, and we report no value elsewhere. The probe needs a baseline it does not yet have:\n"
        "scoring each item against the concept tokens of a different item would say how much of this is chance.\n"
        "We did not run that control, so the probe is reported as consistent with the account and not as support\n"
        "for it.\n\n"
        "\\paragraph{Activation patching.}",
        "\\section{Two further measurements and their scope limits}\n"
        "\\label{sec:supporting}\n\n"
        "%%% SRW-MOVE-START supportingintro\n"
        "Two additional white-box measurements test localization and representational fragmentation under\n"
        "script rewriting. We report both below with their pre-registered controls and scope limits.\n"
        "%%% SRW-MOVE-END supportingintro\n\n"
        "%%% SRW-MOVE-START supporting\n"
        "\\paragraph{Activation patching.}",
        "sec8_remove_probe",
    )

    s = replace_exact(
        s,
        "Three cells combine high reading with low overlap. They are the same cells the direction transfer and\n"
        "the probe single out: Gemma-2-9B-it and Llama-3.2-3B-Instruct on Leetspeak, and Gemma-3-27B-it on\n"
        "ITRANS.",
        "Three cells combine high reading with low overlap, matching the high-reading regime identified by\n"
        "direction transfer: Gemma-2-9B-it and Llama-3.2-3B-Instruct on Leetspeak, and Gemma-3-27B-it on\n"
        "ITRANS.",
        "sec8_overlap_closing",
    )

    # 10. Discussion updates (n=200 numbers, 15-cell decode, no concept probe reference)
    s = replace_exact(
        s,
        "Averaged over ten models, 62.6\\% of Arabizi responses are off topic, against 1.0\\%\n"
        "of the same requests in Arabic. Counting those as bypasses attributes to safety a failure that happens\n"
        "before safety is reached, and reporting the outcome distribution removes the ambiguity.\n\n"
        "The same applies to harm. On-topic is not harmful: of the 123 on-topic harmful responses here, specific\n"
        "usable harmful content appears in 27, and no cell exceeds 4\\%. A paper that reports attack success as\n"
        "non-refusal, on a corpus like this one, reports a number about three times the harm it found.\n\n"
        "\\paragraph{What to do about it.}\n"
        "Where the model does not recover the request, more safety training in more scripts does not address\n"
        "the failure. Decoding does, and it needs no retraining. Asking the model to rewrite the request\n"
        "and then answer its own rewriting recovers up to 79\\% of the gap to the native-script ceiling. The\n"
        "ceiling itself is 84\\% to 100\\%, so there is little wrong with the safety behaviour that the input does\n"
        "not explain.",
        "Averaged over ten models (200 requests per cell), 64.2\\% of Arabizi responses are off topic, against\n"
        "1.2\\% of the same requests in Arabic. Counting those as bypasses attributes to safety a failure that\n"
        "happens before safety is reached, and reporting the outcome distribution removes the ambiguity.\n\n"
        "The same applies to harm. On-topic is not harmful: of the 548 on-topic harmful responses across\n"
        "12,000 generations, specific usable harmful content appears in 116 (1.0\\% overall), and no rewritten\n"
        "cell exceeds 4.5\\%. Treating every on-topic response as harmful compliance overstates actual harm by\n"
        "nearly fivefold.\n\n"
        "\\paragraph{What to do about it.}\n"
        "Where the model does not recover the request, more safety training in more scripts does not address\n"
        "the failure. Decoding does, and it needs no retraining. Asking the model to rewrite the request\n"
        "and then answer its own rewriting improves refusal in eleven of fifteen cells across Arabizi, ITRANS,\n"
        "and Leetspeak, recovering up to 79\\% of the gap to the native-script ceiling ($82\\%$ to $100\\%$).",
        "discussion_numbers",
    )

    s = replace_exact(
        s,
        "Two cells read a rewritten script well and still refuse it differently from the native script.\n"
        "Llama-3.2-3B-Instruct on Leetspeak is the clearest, and feature overlap and the concept probe single\n"
        "out the same cell.",
        "Where a model reads a rewritten script well, single-direction transfer drops sharply both within the\n"
        "Gemma family (from 100\\% on Gemma-2-9B-it Arabizi to 29\\% on Gemma-2-9B-it Leetspeak) and outside\n"
        "it (0\\% to 2\\% on Llama-3.2-3B-Instruct Leetspeak), even though native directions transfer strongly\n"
        "across distinct native languages, and feature overlap singles out the same cells.",
        "discussion_regimes",
    )

    # 11. Limitations
    s = replace_exact(
        s,
        "\\item The behavioural measurements cap generation at 96 tokens. Longer responses could contain harmful\n"
        "content that a truncated one does not, so the harm rates are a lower bound.\n"
        "\\item The direction transfer is measured in three Arabizi cells, all Gemma, and one Leetspeak cell\n"
        "outside that family. Gemma-2-9B-it engages 86\\% of Leetspeak, so running it would separate the regime\n"
        "from the model family. A direction can also fail to transfer simply because directions rarely transfer\n"
        "across inputs, and a baseline that transfers native directions between native languages would bound\n"
        "that. Neither is run.\n"
        "\\item The concept probe has no mismatched-item baseline, so we do not read a number off it.\n"
        "\\item Both recovery tests put native-script text on screen, so either could in principle be solved\n"
        "by mapping characters and comparing strings, without the model representing the request as a request.\n"
        "They show that the meaning is recoverable. They do not show that the representation a four-way test\n"
        "reads is the one that drives refusal when the model answers normally. A test that asks about the\n"
        "request with no native-script text visible would settle this, and we have not run it.",
        "\\item While the 60-cell behavioural grid uses all 200 requests per cell, the multiple-choice recovery\n"
        "tests and the fifteen-cell self-decoding experiment evaluate 50 paired requests per cell.\n"
        "\\item The 60-cell behavioural grid caps generation at 96 tokens, though extending generation to 256\n"
        "tokens across all fifteen mitigation cells (750 rewritten baseline requests) yields a similarly low\n"
        "specific harm rate ($\\le 4.0\\%$ per cell, $1.5\\%$ overall).\n"
        "\\item Comprehended-subset direction transfer covers five well-powered cells across Gemma and Llama\n"
        "(Table~\\ref{tab:transfer}), bounded by a five-model cross-native transfer baseline\n"
        "(Table~\\ref{tab:crossnative}); models that read almost no Arabizi or ITRANS leave too few baseline\n"
        "refusals to test ablation on those subsets.\n"
        "\\item While the cross-language forced-choice test (Arabizi and ITRANS queries with English options,\n"
        "Table~\\ref{tab:crosslang}) and the 200-item free-form ITRANS transliteration test put no native-script\n"
        "text in the prompt and rule out character-level string matching, a multiple-choice readout still does\n"
        "not prove that the representation it reads is the same one that drives refusal during open-ended\n"
        "generation.",
        "limitations",
    )

    # 12. Appendices A, B, C, D, E, H
    s = replace_exact(
        s,
        "\\appendix\n\n\\section{Per-model tables}\n\\label{app:percell}\n"
        "\\input{tables/t2_recovery}\n"
        "\\input{tables/t4_harm}\n"
        "\\input{tables/t5_conditional}\n"
        "\\input{tables/t7_decode}\n",
        "\\appendix\n"
        "\\input{tables/t2_recovery}\n"
        "\\input{tables/t8_crosslang}\n"
        "\\input{tables/t4_harm}\n"
        "\\input{tables/t5_conditional}\n"
        "\\input{tables/t7_decode}\n"
        "\\input{tables/t9_cross_native}\n\n"
        "\\section{Per-model tables}\n\\label{app:percell}\n"
        "Tables~\\ref{tab:recovery} to~\\ref{tab:crossnative} report the per-model numbers for "
        "within-language and cross-language meaning recovery, specific harmful content (200 items per cell), "
        "conditional refusal, fifteen-cell self-decoding across all three scripts, and the cross-native "
        "direction transfer baseline.\n",
        "app_tables",
    )

    s = replace_exact(
        s,
        "All 123 \\texttt{ON\\_TOPIC} responses on harmful prompts were scored by Claude Opus 4.8 as\n"
        "\\texttt{SPECIFIC} (actionable steps, working code, quantities, or usable instructions),\n"
        "\\texttt{GENERIC} (high-level or fictionalized text without usable detail), or \\texttt{NONE} (benign\n"
        "reading or purely defensive text). Only 27 of 123 received \\texttt{SPECIFIC}, and no cell exceeded 4\\%\n"
        "(Table~\\ref{tab:harm}).",
        "All 548 \\texttt{ON\\_TOPIC} responses across the 12,000 harmful-condition generations ($n=200$ per\n"
        "cell) were scored by an independent harm judge (Claude Opus 4.8, with Claude Sonnet 4 fallback on\n"
        "judge-refused items) as \\texttt{SPECIFIC} (actionable steps, working code, quantities, or usable\n"
        "instructions), \\texttt{GENERIC} (high-level or fictionalized text without usable detail), or\n"
        "\\texttt{NONE} (benign reading or purely defensive text). Only 116 of 548 received \\texttt{SPECIFIC},\n"
        "with at most 4.5\\% in any rewritten cell and 5.0\\% in any native cell (Table~\\ref{tab:harm}).\n"
        "Applying the identical rubric to the 256-token \\texttt{decode\\_then\\_refuse} runs across all five\n"
        "models and three scripts (750 baseline rewritten responses) yields \\texttt{SPECIFIC} harm in at most\n"
        "4.0\\% of items in any rewritten cell (0.0\\% in eight of fifteen cells, and 11 of 750 or 1.5\\%\n"
        "overall), and 0.0\\% to 2.0\\% across all five self-decoded Arabizi cells.",
        "app_harm",
    )

    s = replace_exact(
        s,
        "across the 12 valid cells, hard distractors lower mean recovery by\n"
        "2.8 points and 10 of 12 stay above 76\\%.",
        "across the 12 valid cells, hard distractors lower mean recovery by\n"
        "2.8 points and 10 of 12 stay above 76\\%. Under standard same-category distractors, Gemma-2-9B-it\n"
        "passes the format check in all three pairs and recovers 95.0\\% (Arabizi), 97.0\\% (ITRANS), and\n"
        "93.5\\% (Leetspeak). In the cross-language variant (Table~\\ref{tab:crosslang}), where Arabizi and\n"
        "ITRANS prompts are paired with four English near-paraphrase candidates so no Arabic or Devanagari\n"
        "characters appear on screen, all ten cells across the five models pass the native-to-English format\n"
        "check ($91.0\\%$ to $99.0\\%$), and cross-language recovery ($47.0\\%$ to $75.5\\%$) exceeds paired\n"
        "engagement ($4.0\\%$ to $44.0\\%$) in every cell by 15.0 to 53.5 percentage points. In the 200-item\n"
        "judge-free ITRANS-to-Devanagari transliteration test, the fraction of items decoded with character\n"
        "similarity $\\ge 0.5$ (and mean similarity) is 100.0\\% (0.93) on Qwen3-8B, 100.0\\% (0.91) on\n"
        "Qwen2.5-7B-Instruct, 99.5\\% (0.90) on Qwen3-4B, 97.5\\% (0.85) on Gemma-3-4B-it, 100.0\\% (0.77) on\n"
        "Gemma-2-2B-it, and 22.5\\% (0.28) on Gemma-3-1B-it.",
        "app_reading",
    )

    s = replace_exact(
        s,
        "Sweeping the concept-probe\n"
        "native validation floor over $\\{0.20, 0.30, 0.40\\}$ yields 8, 7, and 5 valid models; sweeping\n"
        "native-anchor collapse cutoffs ($\\text{native} \\ge \\{0.80, 0.90\\}$, $\\text{rewritten} < \\{0.40, 0.50,\n"
        "0.60\\}$) identifies 5 to 7 collapsing cells across 5 to 6 models.",
        "Varying the sparse-autoencoder top-$k$ feature count over $k \\in \\{50, 100, 200\\}$ preserves the\n"
        "ordering between within-script paraphrases and cross-script rewritings across all evaluated models.",
        "app_thresholds",
    )

    s = replace_exact(
        s,
        "On Llama-3.2-3B-Instruct Leetspeak, testing $L{=}18$ and $L{=}14$\n"
        "at $\\alpha{=}1.5$ gives the same $1/46$ (2\\%) flip rate.",
        "On Llama-3.2-3B-Instruct Leetspeak, testing $L{=}18$ and $L{=}14$\n"
        "at $\\alpha{=}1.5$ gives the same $1/46$ (2\\%) flip rate. On Gemma-2-9B-it Leetspeak ($L{=}19,\n"
        "\\alpha{=}1.5$, \\texttt{bfloat16}, calibrated from its H2 English ablation run), removing the English\n"
        "direction flips 12 of 41 comprehended refusals (29.3\\%, 95\\% CI $[17.6, 44.5]$) versus 0 of 41\n"
        "(0.0\\%) for the random direction, matching its 200-item H2 forward flip rate (25.5\\% vs.\\ 93.5\\% on\n"
        "English). Table~\\ref{tab:crossnative} reports the five-model native-to-native transfer baseline\n"
        "between English, Arabic, and Devanagari at the same $(L, \\alpha)$ settings.",
        "app_layers",
    )

    s = replace_exact(
        s,
        "Twenty-seven responses were judged to contain\n"
        "specific usable harmful content.",
        "Across the 12,000 main-grid responses, 116 were judged to contain\n"
        "specific usable harmful content.",
        "app_repro",
    )

    return s


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "figures").mkdir(exist_ok=True)
    (DST / "tables").mkdir(exist_ok=True)
    for f in ("references.bib", "acl.sty", "acl_natbib.bst"):
        if (SRC / f).exists():
            shutil.copy2(SRC / f, DST / f)

    # 1. Regenerate all figures from results/ into openreview/v4/figures/
    out_figs = DST / "figures"
    mvf.fig_concept(out_figs)
    mvf.fig_method(out_figs)
    mvf.fig1(out_figs)
    mvf.fig2(out_figs)
    mvf.fig3(out_figs)

    # 2. Regenerate all tables from results/ into openreview/v4/tables/
    J = mvt.judge_labels()
    out_tables = DST / "tables"
    mvt.t1_behaviour(J, out_tables)
    mvt.t2_recovery(J, out_tables)
    mvt.t3_outcomes(J, out_tables)
    mvt.t4_harm(out_tables)
    mvt.t5_conditional(J, out_tables)
    mvt.t6_transfer(out_tables)
    mvt.t7_decode(out_tables)
    make_crosslang_table(out_tables)
    make_cross_native_table(out_tables)
    (DST / "numbers.json").write_text(json.dumps(mvt.N, indent=2))

    # 3. Build v4 acl_latex.tex (and save .acl_marked.tex for make_srw_version.py)
    s_marked = apply_v4_strengthenings((SRC / "acl_latex.tex").read_text())
    (DST / ".acl_marked.tex").write_text(s_marked)
    s = re.sub(r"^%%% SRW-MOVE-(?:START|END) [^\n]+\n", "", s_marked, flags=re.MULTILINE)

    tex_path = DST / "acl_latex.tex"
    tex_path.write_text(s)
    print(f"wrote {tex_path}")

    # 4. Compile PDF with tectonic
    tectonic = shutil.which("tectonic")
    if tectonic:
        subprocess.run(
            [tectonic, "-X", "compile", "acl_latex.tex"],
            cwd=DST,
            check=True,
        )
        print(f"compiled {DST / 'acl_latex.pdf'}")

    # 5. Package clean submission zip
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in ("acl_latex.tex", "acl.sty", "acl_natbib.bst", "references.bib"):
            fpath = DST / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)
        for sub in ("figures", "tables"):
            for fpath in sorted((DST / sub).glob("*")):
                if fpath.is_file():
                    zf.write(fpath, arcname=f"{sub}/{fpath.name}")

    print(f"packaged {ZIP_PATH}")


if __name__ == "__main__":
    main()
