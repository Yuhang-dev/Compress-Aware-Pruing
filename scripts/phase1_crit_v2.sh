#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TARGET_SUFFIXES="${TARGET_SUFFIXES:-o_proj down_proj}"
SAFE_LIMIT="${SAFE_LIMIT:-32}"
UTILITY_LIMIT="${UTILITY_LIMIT:-32}"
RUN_ABLATION="${RUN_ABLATION:-0}"
ABLATION_CANDIDATES="${ABLATION_CANDIDATES:-3}"
CONTEXT_LEN="${CONTEXT_LEN:-1024}"
STRIDE="${STRIDE:-512}"
SAMPLE_WINDOWS="${SAMPLE_WINDOWS:-128}"
SEED="${SEED:-0}"

ABLATION_ARGS=()
if [[ "$RUN_ABLATION" == "1" ]]; then
  ABLATION_ARGS=(
    --run-ablation
    --ablation-candidates "$ABLATION_CANDIDATES"
    --ppl-context-len "$CONTEXT_LEN"
    --ppl-stride "$STRIDE"
    --ppl-sample-windows "$SAMPLE_WINDOWS"
    --ppl-window-index-file "results/phase1_v2/ppl_windows_wikitext2_seed${SEED}.json"
  )
fi

python -m casafety.crit_selection_v2 \
  --config configs/base.yaml \
  --model "$MODEL" \
  --target-suffixes $TARGET_SUFFIXES \
  --harmful-limit "$SAFE_LIMIT" \
  --utility-limit "$UTILITY_LIMIT" \
  --seed "$SEED" \
  --output-dir results/phase1_v2 \
  "${ABLATION_ARGS[@]}" \
  --local-files-only
