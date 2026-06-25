#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase1_v2}"
TARGET_SUFFIXES="${TARGET_SUFFIXES:-o_proj down_proj}"
P_SAFE="${P_SAFE:-0.01 0.02 0.03}"
P_UTIL="${P_UTIL:-0.01 0.03 0.05}"
LAMBDA_VALUES="${LAMBDA_VALUES:-0.5 1.0 2.0}"
SCORE_TYPES="${SCORE_TYPES:-snip grad norm_snip}"
SELECTORS="${SELECTORS:-wei_setdiff ratio penalty}"
SAFE_LIMIT="${SAFE_LIMIT:-32}"
UTILITY_LIMIT="${UTILITY_LIMIT:-32}"
UTILITY_FILE="${UTILITY_FILE:-}"
UTILITY_DATASET="${UTILITY_DATASET:-yahma/alpaca-cleaned}"
UTILITY_CONFIG="${UTILITY_CONFIG:-}"
UTILITY_SPLIT="${UTILITY_SPLIT:-train}"
RUN_ABLATION="${RUN_ABLATION:-0}"
ABLATION_CANDIDATES="${ABLATION_CANDIDATES:-3}"
RUN_PPL_MATCHED_RANDOM="${RUN_PPL_MATCHED_RANDOM:-0}"
PPL_MATCHED_RANDOM_CANDIDATES="${PPL_MATCHED_RANDOM_CANDIDATES:-3}"
PPL_MATCHED_RANDOM_MULTIPLIERS="${PPL_MATCHED_RANDOM_MULTIPLIERS:-1 2 4 8}"
RUN_WANDA_REMOVED_PROBE="${RUN_WANDA_REMOVED_PROBE:-0}"
WANDA_REMOVED_SPARSITY="${WANDA_REMOVED_SPARSITY:-0.50}"
CONTEXT_LEN="${CONTEXT_LEN:-1024}"
STRIDE="${STRIDE:-512}"
SAMPLE_WINDOWS="${SAMPLE_WINDOWS:-128}"
SEED="${SEED:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
PPL_WINDOW_INDEX_FILE="${PPL_WINDOW_INDEX_FILE:-results/phase1_v2/ppl_windows_wikitext2_seed${SEED}.json}"

ABLATION_ARGS=()
if [[ "$RUN_ABLATION" == "1" ]]; then
  ABLATION_ARGS=(
    --run-ablation
    --ablation-candidates "$ABLATION_CANDIDATES"
    --ppl-context-len "$CONTEXT_LEN"
    --ppl-stride "$STRIDE"
    --ppl-sample-windows "$SAMPLE_WINDOWS"
    --ppl-window-index-file "$PPL_WINDOW_INDEX_FILE"
  )
  if [[ "$RUN_PPL_MATCHED_RANDOM" == "1" ]]; then
    ABLATION_ARGS+=(
      --run-ppl-matched-random
      --ppl-matched-random-candidates "$PPL_MATCHED_RANDOM_CANDIDATES"
      --ppl-matched-random-multipliers $PPL_MATCHED_RANDOM_MULTIPLIERS
    )
  fi
  if [[ "$RUN_WANDA_REMOVED_PROBE" == "1" ]]; then
    ABLATION_ARGS+=(
      --run-wanda-removed-probe
      --wanda-removed-sparsity "$WANDA_REMOVED_SPARSITY"
    )
  fi
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
  --p-safe $P_SAFE \
  --p-util $P_UTIL \
  --lambda-values $LAMBDA_VALUES \
  --score-types $SCORE_TYPES \
  --selectors $SELECTORS \
  --harmful-limit "$SAFE_LIMIT" \
  --utility-limit "$UTILITY_LIMIT" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR" \
  "${UTILITY_ARGS[@]}" \
  "${ABLATION_ARGS[@]}" \
  "${LOCAL_ARGS[@]}"
