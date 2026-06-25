#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TARGET_SUFFIXES="${TARGET_SUFFIXES:-o_proj down_proj}"
SAFE_LIMIT="${SAFE_LIMIT:-32}"
UTILITY_LIMIT="${UTILITY_LIMIT:-32}"
UTILITY_FILE="${UTILITY_FILE:-}"
UTILITY_DATASET="${UTILITY_DATASET:-yahma/alpaca-cleaned}"
UTILITY_CONFIG="${UTILITY_CONFIG:-}"
UTILITY_SPLIT="${UTILITY_SPLIT:-train}"
RUN_ABLATION="${RUN_ABLATION:-0}"
ABLATION_CANDIDATES="${ABLATION_CANDIDATES:-3}"
CONTEXT_LEN="${CONTEXT_LEN:-1024}"
STRIDE="${STRIDE:-512}"
SAMPLE_WINDOWS="${SAMPLE_WINDOWS:-128}"
SEED="${SEED:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

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

UTILITY_ARGS=()
if [[ -n "$UTILITY_FILE" ]]; then
  UTILITY_ARGS=(--utility-file "$UTILITY_FILE")
else
  UTILITY_ARGS=(--utility-dataset "$UTILITY_DATASET" --utility-split "$UTILITY_SPLIT")
  if [[ -n "$UTILITY_CONFIG" ]]; then
    UTILITY_ARGS+=(--utility-config "$UTILITY_CONFIG")
  fi
fi

LOCAL_ARGS=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_ARGS=(--local-files-only)
fi

python -m casafety.crit_selection_v2 \
  --config configs/base.yaml \
  --model "$MODEL" \
  --target-suffixes $TARGET_SUFFIXES \
  --harmful-limit "$SAFE_LIMIT" \
  --utility-limit "$UTILITY_LIMIT" \
  --seed "$SEED" \
  --output-dir results/phase1_v2 \
  "${UTILITY_ARGS[@]}" \
  "${ABLATION_ARGS[@]}" \
  "${LOCAL_ARGS[@]}"
