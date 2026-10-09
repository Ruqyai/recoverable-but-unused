#!/usr/bin/env python3
"""Selection-confound control for the per-item conditional analysis (reviewer ask #2).

The conditional result (refusal returns within the judged-comprehended subset)
is only clean if the comprehended items are not systematically the EASIER prompts
(shorter / simpler), which a model might also be better aligned on. This script
tests that: for each romanized harmful condition, it compares prompt difficulty
between the comprehended and non-comprehended items and runs a Mann--Whitney U
test. No GPU; reads only generations.jsonl + items_gemini.jsonl.

Difficulty proxies (held-meaning-constant corpus, so these index surface load):
  - prompt length in characters
  - token-ish length (whitespace words)
  - a romanization-density proxy (digit/nonalpha ratio for Leetspeak/Arabizi)

Run:  python3 scripts/selection_confound_check.py
"""
from __future__ import annotations
import glob, json, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # comprehension_validation -> phase5_sae_features -> scripts -> repo root
MODELS = ["gemma_3_1b_it", "gemma_2_2b_it", "gemma_3_4b_it", "gemma_2_9b_it",
          "gemma_3_27b_it", "qwen3_4b", "qwen3_8b", "qwen2_5_7b_Instruct",
          "llama_3_1_8b_Instruct", "llama_3_2_3b_Instruct"]
ROM = ["arabizi_harmful", "leetspeak_harmful", "itrans_harmful"]
COMP = {"ON_TOPIC", "REFUSAL"}


def mann_whitney_u(a, b):
    """Two-sided Mann-Whitney U with normal approximation; returns (U, z, p)."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan"), float("nan")
    comb = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    # rank with ties -> average ranks
    ranks = [0.0] * len(comb)
    i = 0
    while i < len(comb):
        j = i
        while j + 1 < len(comb) and comb[j + 1][0] == comb[i][0]:
            j += 1
        r = (i + j) / 2 + 1  # average rank (1-based)
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    ra = sum(r for r, (_, g) in zip(ranks, comb) if g == 0)
    ua = ra - na * (na + 1) / 2
    u = ua
    mu = na * nb / 2
    sigma = math.sqrt(na * nb * (na + nb + 1) / 12)
    z = (u - mu) / sigma if sigma > 0 else float("nan")
    # two-sided normal p
    p = math.erfc(abs(z) / math.sqrt(2)) if not math.isnan(z) else float("nan")
    return u, z, p


def load(m):
    jf = glob.glob(str(ROOT / f"results/phase5/comprehension_validation/{m}/items_gemini.jsonl"))
    gf = sorted(glob.glob(str(ROOT / f"results/phase5/{m}/**/generations.jsonl"), recursive=True))
    if not jf or not gf:
        return None
    judge = {(r["condition"], r["id"]): r["label"].upper()
             for r in (json.loads(l) for l in open(jf[0]))}
    prompts = {(r["condition"], r["id"]): r.get("prompt", "")
               for r in (json.loads(l) for l in open(gf[-1]))}
    return judge, prompts


def difficulty(p):
    nonalpha = sum(1 for c in p if not c.isalpha() and not c.isspace())
    return {"chars": len(p), "words": len(p.split()),
            "nonalpha_ratio": nonalpha / len(p) if p else 0.0}


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("Selection-confound check: prompt difficulty, comprehended vs non-comprehended")
    emit("(Mann-Whitney U, two-sided; p>0.05 => no significant difficulty gap => confound not supported)\n")
    pooled_comp = defaultdict(list)
    pooled_not = defaultdict(list)
    n_sig = n_tot = 0
    for m in MODELS:
        d = load(m)
        if not d:
            continue
        judge, prompts = d
        for cond in ROM:
            comp_len, not_len = [], []
            for (c, i), lab in judge.items():
                if c != cond:
                    continue
                p = prompts.get((c, i), "")
                L = difficulty(p)["chars"]
                (comp_len if lab in COMP else not_len).append(L)
            if len(comp_len) >= 3 and len(not_len) >= 3:
                u, z, pv = mann_whitney_u(comp_len, not_len)
                n_tot += 1
                sig = pv < 0.05
                n_sig += sig
                pooled_comp[cond] += comp_len
                pooled_not[cond] += not_len
                flag = " <-- SIG" if sig else ""
                emit(f"{m:22} {cond:18} comp {sum(comp_len)/len(comp_len):5.0f}c (n={len(comp_len):2d}) "
                     f"vs not {sum(not_len)/len(not_len):5.0f}c (n={len(not_len):2d})  p={pv:.3f}{flag}")
    emit("\n=== POOLED across models per condition ===")
    for cond in ROM:
        c, n = pooled_comp[cond], pooled_not[cond]
        if len(c) >= 3 and len(n) >= 3:
            u, z, pv = mann_whitney_u(c, n)
            emit(f"{cond:18} comp mean {sum(c)/len(c):5.0f}c (n={len(c):3d}) vs "
                 f"not {sum(n)/len(n):5.0f}c (n={len(n):3d})  U-p={pv:.3f}  "
                 f"{'DIFFERENCE' if pv<0.05 else 'no sig. difference'}")
    emit(f"\nper-cell tests significant at p<0.05: {n_sig}/{n_tot}  "
         f"(low => comprehended items are NOT systematically shorter => selection confound not supported)")

    outdir = ROOT / "results" / "phase5" / "comprehension_validation" / "selection_confound"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "selection_confound.txt").write_text("\n".join(lines) + "\n")
    print(f"\n[saved] {outdir / 'selection_confound.txt'}")


if __name__ == "__main__":
    main()
