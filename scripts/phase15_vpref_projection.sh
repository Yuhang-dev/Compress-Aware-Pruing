#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase15_vpref_projection}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/vpref_projection}"
LAYERS="${LAYERS:-8,12,16,20,24,28}"
KR="${KR:-1 4 8}"
SEED="${SEED:-0}"
DIRECTION_LIMIT="${DIRECTION_LIMIT:-256}"
EVAL_LIMIT="${EVAL_LIMIT:-128}"
HARM_EVAL_OFFSET="${HARM_EVAL_OFFSET:-0}"
BENIGN_EVAL_OFFSET="${BENIGN_EVAL_OFFSET:-0}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NULL_DIRECTIONS="${NULL_DIRECTIONS:-50}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
VALIDATION_ALPHAS="${VALIDATION_ALPHAS:-2 4 8}"
VALIDATION_MAX_NEW_TOKENS="${VALIDATION_MAX_NEW_TOKENS:-128}"
VALIDATION_MIN_DELTA="${VALIDATION_MIN_DELTA:-0.1}"
CONTROL_DIRECTIONS="${CONTROL_DIRECTIONS:-2}"
CONTROL_MIN_GROUP="${CONTROL_MIN_GROUP:-16}"
CONTROL_COLLAPSE_RATIO="${CONTROL_COLLAPSE_RATIO:-0.5}"
PROJECTION_NEIGHBOR_RADIUS="${PROJECTION_NEIGHBOR_RADIUS:-0}"
HARMFUL_DATASET="${HARMFUL_DATASET:-walledai/AdvBench}"
HARMFUL_CONFIG="${HARMFUL_CONFIG:-}"
HARMFUL_SPLIT="${HARMFUL_SPLIT:-train}"
HARMFUL_COLUMN="${HARMFUL_COLUMN:-auto}"
HARMFUL_FILE="${HARMFUL_FILE:-}"
BENIGN_DATASET="${BENIGN_DATASET:-yahma/alpaca-cleaned}"
BENIGN_CONFIG="${BENIGN_CONFIG:-}"
BENIGN_SPLIT="${BENIGN_SPLIT:-train}"
BENIGN_COLUMN="${BENIGN_COLUMN:-instruction}"
BENIGN_FILE="${BENIGN_FILE:-}"
JUDGE="${JUDGE:-llamaguard}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

DATA_ARGS=()
if [[ -n "$HARMFUL_FILE" ]]; then
  DATA_ARGS+=(--harmful-file "$HARMFUL_FILE")
else
  DATA_ARGS+=(--harmful-dataset "$HARMFUL_DATASET" --harmful-split "$HARMFUL_SPLIT" --harmful-column "$HARMFUL_COLUMN")
  if [[ -n "$HARMFUL_CONFIG" ]]; then
    DATA_ARGS+=(--harmful-config "$HARMFUL_CONFIG")
  fi
fi
if [[ -n "$BENIGN_FILE" ]]; then
  DATA_ARGS+=(--benign-file "$BENIGN_FILE")
else
  DATA_ARGS+=(--benign-dataset "$BENIGN_DATASET" --benign-split "$BENIGN_SPLIT" --benign-column "$BENIGN_COLUMN")
  if [[ -n "$BENIGN_CONFIG" ]]; then
    DATA_ARGS+=(--benign-config "$BENIGN_CONFIG")
  fi
fi

LOCAL_ARGS=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_ARGS=(--local-files-only)
fi

VALIDATION_ARGS=(--validation-alphas $VALIDATION_ALPHAS)
if [[ "$RUN_VALIDATION" == "1" ]]; then
  VALIDATION_ARGS+=(--run-validation)
else
  VALIDATION_ARGS+=(--no-run-validation)
fi

python -m casafety.vpref_projection \
  --config configs/base.yaml \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --artifact-dir "$ARTIFACT_DIR" \
  --layers "$LAYERS" \
  --kr $KR \
  --seed "$SEED" \
  --direction-limit "$DIRECTION_LIMIT" \
  --eval-limit "$EVAL_LIMIT" \
  --harm-eval-offset "$HARM_EVAL_OFFSET" \
  --benign-eval-offset "$BENIGN_EVAL_OFFSET" \
  --max-length "$MAX_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --null-directions "$NULL_DIRECTIONS" \
  --validation-max-new-tokens "$VALIDATION_MAX_NEW_TOKENS" \
  --validation-min-delta "$VALIDATION_MIN_DELTA" \
  --control-directions "$CONTROL_DIRECTIONS" \
  --control-min-group "$CONTROL_MIN_GROUP" \
  --control-collapse-ratio "$CONTROL_COLLAPSE_RATIO" \
  --projection-neighbor-radius "$PROJECTION_NEIGHBOR_RADIUS" \
  --judge "$JUDGE" \
  "${DATA_ARGS[@]}" \
  "${VALIDATION_ARGS[@]}" \
  "${LOCAL_ARGS[@]}"
