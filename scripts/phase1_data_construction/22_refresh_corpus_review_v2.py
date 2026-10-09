#!/usr/bin/env python3
"""
22_refresh_corpus_review_v2.py — Refresh data/corpus_review_v2/*.csv from the CURRENT
(post-fix) data files, preserving each file's existing row order, sample_id scheme and
column schema. Only the content columns (english_reference / prompts / harm_category) are
re-pulled from disk; reviewer_* columns are emitted empty for the human pass.

Sources (current, post-fix):
  harmful  EN/Leet = tier_e_english / tier_e_leetspeak   (english_reference = canonical AdvBench)
           AR/Azi  = tier_a_arabic  / tier_a_arabizi
           DV/ITR  = tier_c_devanagari / tier_c_itrans
  benign   EN/Leet = tier_b/benign_english / benign_leetspeak  (english_reference = benign_english)
           AR/Azi  = benign_arabic / benign_arabizi
           DV/ITR  = benign_devanagari / benign_itrans

Run: python3 scripts/phase1_data_construction/22_refresh_corpus_review_v2.py
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
V2 = REPO / "data" / "corpus_review_v2"

def bysid(p): return {json.loads(l)["source_id"]: json.loads(l)
                      for l in (REPO/p).open(encoding="utf-8") if l.strip()}

def order_of(csv_path: Path):
    with csv_path.open(encoding="utf-8") as f:
        return [(r["sample_id"], r["source_id"]) for r in csv.DictReader(f)]

def write_csv(path: Path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def main():
    canon = {f"advbench_{i:03d}": row["goal"]
             for i, row in pd.read_csv("/tmp/advbench.csv").iterrows()}
    te = bysid("data/tier_e/tier_e_english.jsonl"); tl = bysid("data/tier_e/tier_e_leetspeak.jsonl")
    ta = bysid("data/tier_a/tier_a_arabic.jsonl");  taz = bysid("data/tier_a/tier_a_arabizi.jsonl")
    tcd = bysid("data/tier_c/tier_c_devanagari.jsonl"); tci = bysid("data/tier_c/tier_c_itrans.jsonl")
    be = bysid("data/tier_b/benign_english.jsonl");  bl = bysid("data/tier_b/benign_leetspeak.jsonl")
    ba = bysid("data/tier_b/benign_arabic.jsonl");   baz = bysid("data/tier_b/benign_arabizi.jsonl")
    bd = bysid("data/tier_b/benign_devanagari.jsonl"); bi = bysid("data/tier_b/benign_itrans.jsonl")

    def hc(d, sid): return d[sid].get("harm_category", "other")

    # ---- harmful ----
    o = order_of(V2/"harmful"/"english_review.csv")
    write_csv(V2/"harmful"/"english_review.csv",
        ["sample_id","source_id","harm_category","english_prompt","leetspeak_prompt",
         "reviewer_english_natural_ok","reviewer_leetspeak_natural_ok","reviewer_semantic_match",
         "reviewer_harmful_preserved","reviewer_notes"],
        [[sid_s, sid, hc(te,sid), te[sid]["prompt"], tl[sid]["prompt"], "","","","",""] for sid_s,sid in o])

    o = order_of(V2/"harmful"/"arabic_review.csv")
    write_csv(V2/"harmful"/"arabic_review.csv",
        ["sample_id","source_id","harm_category","english_reference","arabic_prompt","arabizi_prompt",
         "reviewer_arabic_natural_ok","reviewer_arabizi_natural_ok","reviewer_matches_english",
         "reviewer_harmful_preserved","reviewer_notes"],
        [[sid_s, sid, hc(ta,sid), canon[sid], ta[sid]["prompt"], taz[sid]["prompt"], "","","","",""] for sid_s,sid in o])

    o = order_of(V2/"harmful"/"hindi_review.csv")
    write_csv(V2/"harmful"/"hindi_review.csv",
        ["sample_id","source_id","harm_category","english_reference","devanagari_prompt","itrans_prompt",
         "reviewer_devanagari_natural_ok","reviewer_itrans_natural_ok","reviewer_matches_english",
         "reviewer_harmful_preserved","reviewer_notes"],
        [[sid_s, sid, hc(tcd,sid), canon[sid], tcd[sid]["prompt"], tci[sid]["prompt"], "","","","",""] for sid_s,sid in o])

    # ---- benign ----
    o = order_of(V2/"benign"/"english_review.csv")
    write_csv(V2/"benign"/"english_review.csv",
        ["sample_id","source_id","harm_category","english_prompt","leetspeak_prompt",
         "reviewer_english_natural_ok","reviewer_leetspeak_natural_ok","reviewer_semantic_match",
         "reviewer_is_benign","reviewer_notes"],
        [[sid_s, sid, hc(be,sid), be[sid]["prompt"], bl[sid]["prompt"], "","","","",""] for sid_s,sid in o])

    o = order_of(V2/"benign"/"arabic_review.csv")
    write_csv(V2/"benign"/"arabic_review.csv",
        ["sample_id","source_id","harm_category","english_reference","arabic_prompt","arabizi_prompt",
         "reviewer_arabic_natural_ok","reviewer_arabizi_natural_ok","reviewer_matches_english",
         "reviewer_is_benign","reviewer_notes"],
        [[sid_s, sid, hc(ba,sid), be[sid]["prompt"], ba[sid]["prompt"], baz[sid]["prompt"], "","","","",""] for sid_s,sid in o])

    o = order_of(V2/"benign"/"hindi_review.csv")
    write_csv(V2/"benign"/"hindi_review.csv",
        ["sample_id","source_id","harm_category","english_reference","devanagari_prompt","itrans_prompt",
         "reviewer_devanagari_natural_ok","reviewer_itrans_natural_ok","reviewer_matches_english",
         "reviewer_is_benign","reviewer_notes"],
        [[sid_s, sid, hc(bd,sid), be[sid]["prompt"], bd[sid]["prompt"], bi[sid]["prompt"], "","","","",""] for sid_s,sid in o])

    print("Refreshed 6 CSVs in data/corpus_review_v2/ from current data.")

if __name__ == "__main__":
    main()
