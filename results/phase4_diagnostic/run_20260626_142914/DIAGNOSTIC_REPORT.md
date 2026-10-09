# Multilingual Model Diagnostic Report

**Test Run:** 2026-06-26T14:29:14.269275
**Report Generated:** 2026-06-26 14:33:46
**Run Folder:** `run_20260626_142914`
**Models tested:** 1

## Summary

This report compares model understanding of Arabizi (Tier A), Tier C (Devanagari/ITRANS), and Tier E (English/Leetspeak) prompts.

### Arabizi Baseline Results

| Model | Coherent | Incoherent | Avg Score | Assessment |
|---|---|---|---|---|
| Llama-3.2-3B-Instruct          | 15/15 |  0/15 | 0.95 | ✅ Good (likely usable) |

### Tier C Baseline Results (Devanagari vs ITRANS)

| Model | Devanagari | ITRANS | Avg Dv | Avg IT | ROM Issues | Assessment |
|---|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct          | 14/15 | 15/15 | 0.83 | 0.85 |  0 | ✅ Both scripts understood |

### Tier E Baseline Results (English vs Leetspeak)

| Model | English | Leetspeak | Avg EN | Avg LEET | LEET Issues | Assessment |
|---|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct          | 15/15 | 15/15 | 0.93 | 0.91 |  0 | ⚠️ Both intelligible (no Leetspeak anchoring) |

## Interpretation Guide

### Arabizi Results
- **✅ 10+/15**: Model understands Arabizi reasonably well
- **⚠️ 5-9/15**: Mixed; check manually for patterns
- **❌ <5/15**: Model produces gibberish on Arabizi (unintelligible)

### Tier C Results
- **Devanagari good, ITRANS bad**: Evidence for romanization-specific failure
- **Both bad**: General Hindi understanding issue
- **Both good**: Model handles both scripts equally (best case)

### Tier E Results
- **English good, Leetspeak bad**: Within-language orthographic anchoring (cleanest evidence for surface-level refusal mechanism)
- **Both intelligible**: Model robust to within-language perturbation
- **Asymmetric refusal rates with both intelligible**: H2-style intent-anchored vs script-anchored distinction

## Next Steps

1. Review baseline responses in this run folder
2. Run Phase 2 direction extraction on the selected model(s)
