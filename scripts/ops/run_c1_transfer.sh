#!/usr/bin/env bash
# C1 — well-powered CROSS-FAMILY direction transfer on the comprehended subset.
#
# The paper's positive causal half (native direction mediates comprehended-romanized
# refusal) is three Gemma cells because Arabizi is too poorly comprehended by Qwen/Llama
# to leave any "room" (comprehended-and-refused items). This runs the transfer on the
# romanization each family DOES read, giving well-powered non-Gemma cells:
#   - Leetspeak (native=English): Llama-3.2-3B comprehends 94% -> room ~40 (WELL-POWERED).
#   - ITRANS (native=Devanagari): for models that read Hindi transliteration.
# Qwen decodes no romanization well (all <28%), so it stays underpowered by nature —
# itself a reportable finding, not a bug.
#
# Env: MODELS (space-separated keys), PAIRS (tier_e=Leetspeak, tier_c=ITRANS, tier_a=Arabizi),
#      N (items, default 200 -> uses the full comprehended subset), DEVICE (default cuda).
# Results: results/phase4_intervention/<model>/direction_transfer/transfer_<pair>_<stamp>.json
#          (synced to $BUCKET/results by the Vertex entrypoint's EXIT trap).
#
# Local smoke (CPU, tiny): MODELS=llama_3_2_3b_Instruct PAIRS=tier_e N=4 DEVICE=cpu \
#   bash scripts/ops/run_c1_transfer.sh
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
export PYTHONPATH="${PYTHONPATH:-$PWD}"
PY="${PY:-python3}"

MODELS="${MODELS:?set MODELS='llama_3_2_3b_Instruct llama_3_1_8b_Instruct ...'}"
PAIRS="${PAIRS:-tier_e}"            # default: Leetspeak (the well-powered cross-family lever)
N="${N:-200}"
GUARD="${GUARD:-1}"                 # 1 = use the guard classifier for refusal (matches paper H2)
SCRIPT=scripts/phase4_causal_intervention/interventions/02_direction_transfer_comprehended.py

guard_flag=""
[[ "$GUARD" == "1" ]] && guard_flag="--guard"

overall=0
for m in $MODELS; do
  for pair in $PAIRS; do
    echo "[c1] model=$m pair=$pair n=$N guard=$GUARD"
    if ! "$PY" "$SCRIPT" --model "$m" --pair "$pair" --n "$N" $guard_flag; then
      echo "[c1] WARN failed: $m x $pair"; overall=1
    fi
  done
done
echo "[c1] done (overall_exit=$overall)"
exit "$overall"
