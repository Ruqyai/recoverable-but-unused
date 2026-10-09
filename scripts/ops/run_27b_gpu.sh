#!/usr/bin/env bash
# Run the two pending Gemma-3-27B-it tasks on a CUDA GPU (Vertex AI).
#
#   Task 1: direction transfer on the comprehended-Arabizi subset
#           -> adds a THIRD well-powered cell to the paper's tab:transfer (room ~35)
#   Task 2: H3 SAE-feature control arms (English + Hindi)
#           -> closes the paper's one coverage gap (Arabic arm already done)
#
# 27B needs bfloat16 (~54 GB); with the guard/SAE loaded, use an 80 GB GPU (A100-80G / H100).
# Prereqs that must be present in results/ on this machine (sync from HF/Drive if missing):
#   results/phase2/gemma_3_27b_it/run_*/directions/arabic.pt
#   results/phase5/comprehension_validation/gemma_3_27b_it/items_gemini.jsonl
#   results/phase5/gemma_3_27b_it/run_*/sae_meta.json  (+ the phase-5 activations it points to)
#
# Usage:  bash scripts/ops/run_27b_gpu.sh 2>&1 | tee /tmp/run_27b.log
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
export PYTHONPATH=.
MODEL=gemma_3_27b_it
DT=scripts/phase4_causal_intervention/interventions/02_direction_transfer_comprehended.py
H3=scripts/phase5_sae_features/06_h3_controls.py

echo "=============================================================="
echo " Gemma-3-27B-it GPU batch — $(date -u '+%Y-%m-%d %H:%M UTC' 2>/dev/null || echo now)"
echo "=============================================================="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
else
  echo "!! nvidia-smi not found — are you on the GPU instance?"; exit 1
fi
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available — aborting (27B OOMs on CPU/MPS)."
free,total=torch.cuda.mem_get_info()
gb=lambda b: round(b/1e9,1)
print(f"  CUDA ok: {torch.cuda.get_device_name(0)}  free {gb(free)} / {gb(total)} GB")
assert total/1e9 >= 60, f"GPU has only {gb(total)} GB; 27B bf16 needs ~54 GB + headroom (use 80 GB)."
PY
[ $? -ne 0 ] && { echo "aborting"; exit 1; }

run () {  # label + command...
  local label="$1"; shift
  echo; echo "---- [$label] $(date -u '+%H:%M:%S' 2>/dev/null) ----"; echo "  $*"
  if "$@"; then echo "  [$label] DONE"; else echo "  [$label] FAILED (exit $?) — continuing"; fi
}

# ---- Task 1: direction transfer (alpha 0.4 = correct for Gemma-3) ----
run "transfer" python3 "$DT" --model "$MODEL" --alpha 0.4 --n 50 --guard

# ---- Task 2: H3 control arms (Arabic already present) ----
run "h3-english" python3 "$H3" --model "$MODEL" --arm english --prefer-device cuda
run "h3-hindi"   python3 "$H3" --model "$MODEL" --arm hindi   --prefer-device cuda

echo; echo "=============================================================="
echo " OUTPUTS"
ls -1 results/phase4_intervention/$MODEL/direction_transfer/transfer_*.json 2>/dev/null | tail -1 | sed 's/^/  transfer:  /' || echo "  transfer:  (not written)"
ls -1 results/phase5/$MODEL/run_*/h3_controls_english.json 2>/dev/null | tail -1 | sed 's/^/  h3-eng:    /' || echo "  h3-eng:    (not written)"
ls -1 results/phase5/$MODEL/run_*/h3_controls_hindi.json   2>/dev/null | tail -1 | sed 's/^/  h3-hindi:  /' || echo "  h3-hindi:  (not written)"
echo "=============================================================="
echo "Next: copy those JSONs back, then locally re-run demo/build_data.py + make_paper_figures.py"
echo "and I will fold the 27B numbers into tab:transfer, fig7, and the docx."
