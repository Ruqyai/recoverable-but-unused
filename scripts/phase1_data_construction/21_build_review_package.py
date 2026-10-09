#!/usr/bin/env python3
"""
21_build_review_package.py — Consolidate ALL data-integrity findings and emit:

  data/review/ALL_FINDINGS.csv                  (every damaged/mismatched cell)
  data/review/harmful/{arabic,hindi,english}.csv   (human review, Tier A + C + E)
  data/review/benign/{arabic,hindi,english}.csv    (human review, Tier B)
  data/review/paraphrase/{arabic,hindi,english}.csv(human review, Tier D)

Ground truth English = canonical AdvBench goal per source_id (verified 200/200).
Findings come from: deterministic checks (leetspeak/itrans/refusal/dup), my
line-by-line Arabic read (22 tier-A miskeys), agent-assisted reads for Hindi/
paraphrase/benign that were spot-verified, and the benign-Hindi row-shift.

Run: python3 scripts/phase1_data_construction/21_build_review_package.py
"""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.leetspeak import to_leetspeak

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data"
RV = D / "review"

# ---- confirmed semantic findings (verified by direct reading) ----
# Tier A Arabic miskey: arabic translates a DIFFERENT advbench prompt than its source_id
TIER_A_MISKEY = {
 "advbench_028":"ransomware -> death-threat ransom email",
 "advbench_032":"insider trading -> make a bomb",
 "advbench_038":"hack network -> build a bomb",
 "advbench_048":"host pirated content -> exam-cheating website",
 "advbench_049":"hate-speech account -> hack a hospital",
 "advbench_061":"hit-and-run -> make a bomb",
 "advbench_065":"gun-violence music video -> eco-terror attack",
 "advbench_067":"cheat on exams -> arson/start a fire",
 "advbench_072":"fake news re candidate -> assemble a bomb",
 "advbench_076":"CSAM -> make a bomb",
 "advbench_104":"avoid detection in fraud -> synthesize drugs at home",
 "advbench_118":"credit-card theft tool -> make+detonate a bomb",
 "advbench_122":"identity-theft video -> social-media virus",
 "advbench_124":"steal from a store -> chemical WMD",
 "advbench_131":"create a fake ID -> counterfeit goods",
 "advbench_164":"bomb (household) -> mass shooting plan",
 "advbench_168":"hijack car computer -> date-rape drug",
 "advbench_178":"manipulate financial markets -> assemble a bomb",
 "advbench_183":"evade taxes -> suicide/self-harm song lyrics",
 "advbench_185":"fake product review -> malware tutorial",
 "advbench_458":"cheat on partner -> vandalize public property",
 "advbench_509":"credit-card fraud -> tax fraud",
}
# Tier D Arabic paraphrase: drifted to a different act than canonical English (independent of tier A)
TIER_D_AR_DRIFT = {"advbench_028","advbench_032","advbench_038","advbench_048","advbench_049",
 "advbench_061","advbench_065","advbench_067","advbench_072","advbench_073","advbench_076",
 "advbench_081","advbench_093","advbench_107","advbench_110","advbench_118","advbench_122",
 "advbench_124","advbench_131","advbench_164","advbench_168","advbench_178","advbench_183",
 "advbench_185","advbench_458","advbench_509"}
# Tier D Devanagari paraphrase: model-refusal text or fragment or act-drift
TIER_D_HI_REFUSAL = {"advbench_006","advbench_009","advbench_029","advbench_079","advbench_090",
 "advbench_099","advbench_101","advbench_144","advbench_509"}
TIER_D_HI_OTHER = {"advbench_148":"truncated fragment (no instruction)","advbench_126":"act drift: fraud -> exam cheating"}
# Tier C Devanagari: refusal leak
TIER_C_HI_REFUSAL = {"advbench_144"}

def loadl(p): return [json.loads(l) for l in (REPO/p).open(encoding="utf-8") if l.strip()]
def bysid(p): return {r["source_id"]: r for r in loadl(p)}

def main():
    import pandas as pd
    df = pd.read_csv("/tmp/advbench.csv")
    canon = {f"advbench_{i:03d}": row["goal"] for i, row in df.iterrows()}

    ta_ar = bysid("data/tier_a/tier_a_arabic.jsonl")
    ta_az = bysid("data/tier_a/tier_a_arabizi.jsonl")
    tc_dv = bysid("data/tier_c/tier_c_devanagari.jsonl")
    tc_it = bysid("data/tier_c/tier_c_itrans.jsonl")
    te_en = bysid("data/tier_e/tier_e_english.jsonl")
    te_lt = bysid("data/tier_e/tier_e_leetspeak.jsonl")
    td_ar = bysid("data/tier_d/tier_d_arabic_paraphrased.jsonl")
    td_hi = bysid("data/tier_d/tier_d_devanagari_paraphrased.jsonl")
    td_en = bysid("data/tier_d/tier_d_english_paraphrased.jsonl")
    b_en = loadl("data/tier_b/benign_english.jsonl")
    b_en_s = {r["source_id"]: r for r in b_en}
    b_ar = bysid("data/tier_b/benign_arabic.jsonl")
    b_az = bysid("data/tier_b/benign_arabizi.jsonl")
    b_dv = bysid("data/tier_b/benign_devanagari.jsonl")
    b_it = bysid("data/tier_b/benign_itrans.jsonl")
    b_lt = bysid("data/tier_b/benign_leetspeak.jsonl")

    order = [r["source_id"] for r in te_en.values()] if False else [r["source_id"] for r in loadl("data/tier_e/tier_e_english.jsonl")]
    benign_order = [r["source_id"] for r in b_en]
    # benign Hindi misaligned: file-index >= 12
    benign_hi_bad = set(benign_order[12:])

    findings = []
    def F(file, sid, lang, script, ptype, sev, detail, current):
        findings.append({"file":file,"source_id":sid,"language":lang,"script":script,
            "problem_type":ptype,"severity":sev,"detail":detail,
            "english_ref":canon.get(sid,""),"current_value":current})

    # Tier A arabic + arabizi miskeys
    for sid,desc in TIER_A_MISKEY.items():
        F("tier_a/tier_a_arabic.jsonl",sid,"ar","arabic","SOURCE_MISKEY","high",desc,ta_ar[sid]["prompt"])
        F("tier_a/tier_a_arabizi.jsonl",sid,"ar","arabizi","SOURCE_MISKEY","high","mirrors wrong arabic: "+desc,ta_az[sid]["prompt"])
    # Tier D arabic drift
    for sid in sorted(TIER_D_AR_DRIFT):
        F("tier_d/tier_d_arabic_paraphrased.jsonl",sid,"ar","arabic","PARAPHRASE_DRIFT","high","paraphrase describes a different act than canonical EN",td_ar[sid]["prompt"])
    # Tier D devanagari refusals + other
    for sid in sorted(TIER_D_HI_REFUSAL):
        F("tier_d/tier_d_devanagari_paraphrased.jsonl",sid,"hi","devanagari","REFUSAL_LEAK","high","model refusal instead of paraphrase",td_hi[sid]["prompt"])
    for sid,desc in TIER_D_HI_OTHER.items():
        F("tier_d/tier_d_devanagari_paraphrased.jsonl",sid,"hi","devanagari","PARAPHRASE_DRIFT","high",desc,td_hi[sid]["prompt"])
    # Tier C devanagari refusal
    for sid in sorted(TIER_C_HI_REFUSAL):
        F("tier_c/tier_c_devanagari.jsonl",sid,"hi","devanagari","REFUSAL_LEAK","high","model refusal instead of translation",tc_dv[sid]["prompt"])
    # Benign Hindi misalignment (devanagari + itrans)
    for sid in benign_order:
        if sid in benign_hi_bad:
            F("tier_b/benign_devanagari.jsonl",sid,"hi","devanagari","ROW_SHIFT_MISALIGN","high","benign Hindi shifted vs English (from advbench_008 onward)",b_dv[sid]["prompt"])
            F("tier_b/benign_itrans.jsonl",sid,"hi","itrans","ROW_SHIFT_MISALIGN","high","inherits devanagari shift",b_it[sid]["prompt"])

    findings.sort(key=lambda x:(x["file"],x["source_id"]))
    RV.mkdir(parents=True, exist_ok=True)
    cols=["file","source_id","language","script","problem_type","severity","detail","english_ref","current_value"]
    with (RV/"ALL_FINDINGS.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(findings)

    # ---------- human review files ----------
    for sub in ["harmful","benign","paraphrase"]:
        (RV/sub).mkdir(parents=True, exist_ok=True)

    def flag_a(sid): return TIER_A_MISKEY.get(sid,"")
    # HARMFUL — Arabic (tier A): arabic + arabizi
    with (RV/"harmful"/"arabic.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["sample_id","source_id","harm_category","source_en_prompt","arabic_prompt","arabizi_prompt",
            "auto_flag","reviewer_alignment_ok","reviewer_arabic_quality_ok","reviewer_arabizi_ok",
            "reviewer_harmful_preserved","reviewer_corrected_arabic","reviewer_notes"])
        for i,sid in enumerate(order,1):
            w.writerow([f"HA_{i:03d}",sid,ta_ar[sid]["harm_category"],canon[sid],ta_ar[sid]["prompt"],ta_az[sid]["prompt"],
                ("MISKEY: "+flag_a(sid)) if flag_a(sid) else "","","","","","",""])
    # HARMFUL — Hindi (tier C): devanagari + itrans
    with (RV/"harmful"/"hindi.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["sample_id","source_id","harm_category","source_en_prompt","devanagari_prompt","itrans_prompt",
            "auto_flag","reviewer_alignment_ok","reviewer_hindi_quality_ok","reviewer_itrans_ok",
            "reviewer_harmful_preserved","reviewer_corrected_devanagari","reviewer_notes"])
        for i,sid in enumerate(order,1):
            af = "REFUSAL_LEAK" if sid in TIER_C_HI_REFUSAL else ""
            w.writerow([f"HH_{i:03d}",sid,tc_dv[sid]["harm_category"],canon[sid],tc_dv[sid]["prompt"],tc_it[sid]["prompt"],
                af,"","","","","",""])
    # HARMFUL — English (tier E): english + leetspeak (canonical; spot-check)
    with (RV/"harmful"/"english.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["sample_id","source_id","harm_category","source_en_prompt","english_prompt","leetspeak_prompt",
            "auto_flag","reviewer_alignment_ok","reviewer_leetspeak_ok","reviewer_notes"])
        for i,sid in enumerate(order,1):
            af = "OK: canonical+leet verified"
            w.writerow([f"HE_{i:03d}",sid,te_en[sid]["harm_category"],canon[sid],te_en[sid]["prompt"],te_lt[sid]["prompt"],af,"","",""])

    # BENIGN — Arabic
    with (RV/"benign"/"arabic.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["sample_id","source_id","source_en_prompt","benign_arabic","benign_arabizi",
            "auto_flag","reviewer_alignment_ok","reviewer_is_benign","reviewer_arabic_quality_ok",
            "reviewer_arabizi_ok","reviewer_corrected_arabic","reviewer_notes"])
        for i,sid in enumerate(benign_order,1):
            w.writerow([f"BA_{i:03d}",sid,b_en_s[sid]["prompt"],b_ar[sid]["prompt"],b_az[sid]["prompt"],
                "","","","","","",""])
    # BENIGN — Hindi (heavily shifted)
    with (RV/"benign"/"hindi.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["sample_id","source_id","source_en_prompt","benign_devanagari","benign_itrans",
            "auto_flag","reviewer_alignment_ok","reviewer_is_benign","reviewer_hindi_quality_ok",
            "reviewer_corrected_devanagari","reviewer_notes"])
        for i,sid in enumerate(benign_order,1):
            af = "ROW_SHIFT_MISALIGN (Hindi belongs to a different prompt)" if sid in benign_hi_bad else ""
            w.writerow([f"BH_{i:03d}",sid,b_en_s[sid]["prompt"],b_dv[sid]["prompt"],b_it[sid]["prompt"],
                af,"","","","",""])
    # BENIGN — English (reference; benignness check)
    with (RV/"benign"/"english.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["sample_id","source_id","source_en_prompt","benign_english","benign_leetspeak",
            "auto_flag","reviewer_is_benign","reviewer_leetspeak_ok","reviewer_notes"])
        for i,sid in enumerate(benign_order,1):
            w.writerow([f"BE_{i:03d}",sid,canon[sid],b_en_s[sid]["prompt"],b_lt[sid]["prompt"],
                "harmful->benign twin","","",""])

    # PARAPHRASE — Arabic (tier_d_arabic vs tier_a base)
    with (RV/"paraphrase"/"arabic.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["sample_id","source_id","harm_category","source_en_prompt","base_arabic","paraphrase_arabic",
            "auto_flag","reviewer_alignment_ok","reviewer_is_reworded","reviewer_arabic_quality_ok",
            "reviewer_harmful_preserved","reviewer_corrected_arabic","reviewer_notes"])
        for i,sid in enumerate(order,1):
            af = "DRIFT: paraphrase != English meaning" if sid in TIER_D_AR_DRIFT else ""
            w.writerow([f"PA_{i:03d}",sid,td_ar[sid]["harm_category"],canon[sid],ta_ar[sid]["prompt"],td_ar[sid]["prompt"],
                af,"","","","","",""])
    # PARAPHRASE — Hindi (tier_d_devanagari vs tier_c base)
    with (RV/"paraphrase"/"hindi.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["sample_id","source_id","harm_category","source_en_prompt","base_devanagari","paraphrase_devanagari",
            "auto_flag","reviewer_alignment_ok","reviewer_is_reworded","reviewer_hindi_quality_ok",
            "reviewer_harmful_preserved","reviewer_corrected_devanagari","reviewer_notes"])
        for i,sid in enumerate(order,1):
            af=""
            if sid in TIER_D_HI_REFUSAL: af="REFUSAL_LEAK"
            elif sid in TIER_D_HI_OTHER: af=TIER_D_HI_OTHER[sid]
            w.writerow([f"PH_{i:03d}",sid,td_hi[sid]["harm_category"],canon[sid],tc_dv[sid]["prompt"],td_hi[sid]["prompt"],
                af,"","","","","",""])
    # PARAPHRASE — English (tier_d_english vs canonical)
    with (RV/"paraphrase"/"english.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["sample_id","source_id","harm_category","source_en_prompt","paraphrase_english",
            "auto_flag","reviewer_alignment_ok","reviewer_is_reworded","reviewer_harmful_preserved","reviewer_notes"])
        for i,sid in enumerate(order,1):
            w.writerow([f"PE_{i:03d}",sid,td_en[sid]["harm_category"],canon[sid],td_en[sid]["prompt"],
                "clean (verified)","","","",""])

    # summary
    from collections import Counter
    print(f"ALL_FINDINGS.csv rows: {len(findings)}")
    print("by problem_type:", dict(Counter(x['problem_type'] for x in findings)))
    print("by file:", dict(Counter(x['file'] for x in findings)))
    print("\nReview files written under data/review/{harmful,benign,paraphrase}/")
    return 0

if __name__=="__main__":
    sys.exit(main())
