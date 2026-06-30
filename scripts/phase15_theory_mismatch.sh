#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase15_mismatch}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/vpref_projection}"
CRIT_OUTPUT_DIR="${CRIT_OUTPUT_DIR:-results/phase1_v2}"
CRIT_CANDIDATE="${CRIT_CANDIDATE:-wei_setdiff__score-grad__ps-0.01__pu-0.05}"
CRIT_SET_PATH="${CRIT_SET_PATH:-}"
LAYERS="${LAYERS:-24,28}"
TARGET_SUFFIXES="${TARGET_SUFFIXES:-o_proj down_proj}"
HARMFUL_FILE="${HARMFUL_FILE:-}"
HARMFUL_DATASET="${HARMFUL_DATASET:-walledai/AdvBench}"
HARMFUL_CONFIG="${HARMFUL_CONFIG:-}"
HARMFUL_SPLIT="${HARMFUL_SPLIT:-train}"
HARMFUL_COLUMN="${HARMFUL_COLUMN:-auto}"
HARMFUL_OFFSET="${HARMFUL_OFFSET:-0}"
HARMFUL_LIMIT="${HARMFUL_LIMIT:-128}"
CALIB_FILE="${CALIB_FILE:-}"
CALIB_LIMIT="${CALIB_LIMIT:-128}"
CALIB_MAX_LENGTH="${CALIB_MAX_LENGTH:-1024}"
PPL_DATASET="${PPL_DATASET:-Salesforce/wikitext}"
PPL_DATASET_CONFIG="${PPL_DATASET_CONFIG:-wikitext-2-raw-v1}"
PPL_SPLIT="${PPL_SPLIT:-test}"
PPL_CONTEXT_LEN="${PPL_CONTEXT_LEN:-1024}"
PPL_STRIDE="${PPL_STRIDE:-512}"
PPL_SAMPLE_WINDOWS="${PPL_SAMPLE_WINDOWS:-128}"
WINDOW_INDEX_FILE="${WINDOW_INDEX_FILE:-results/phase1_v2/ppl_windows_wikitext2_seed0.json}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
SPARSITIES="${SPARSITIES:-0.45,0.50,0.55}"
Q_SWEEP="${Q_SWEEP:-0.001,0.002,0.005,0.01}"
CORRELATION_SAMPLE="${CORRELATION_SAMPLE:-1000000}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-100}"
BOOTSTRAP_SAMPLE_SIZE="${BOOTSTRAP_SAMPLE_SIZE:-50000}"
PERCENTILE_CI_SAMPLE="${PERCENTILE_CI_SAMPLE:-200000}"
DECISION_LAYER="${DECISION_LAYER:--1}"
DECISION_CUT_BASE_TOLERANCE="${DECISION_CUT_BASE_TOLERANCE:-0.02}"
SEED="${SEED:-0}"
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

if [[ -n "$CALIB_FILE" ]]; then
  DATA_ARGS+=(--calib-file "$CALIB_FILE")
fi

CRIT_ARGS=(--crit-output-dir "$CRIT_OUTPUT_DIR" --crit-candidate "$CRIT_CANDIDATE")
if [[ -n "$CRIT_SET_PATH" ]]; then
  CRIT_ARGS=(--crit-set-path "$CRIT_SET_PATH")
fi

LOCAL_ARGS=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_ARGS=(--local-files-only)
else
  LOCAL_ARGS=(--no-local-files-only)
fi

python -m casafety.theory_mismatch \
  --config configs/base.yaml \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --artifact-dir "$ARTIFACT_DIR" \
  "${CRIT_ARGS[@]}" \
  --layers "$LAYERS" \
  --target-suffixes $TARGET_SUFFIXES \
  --harmful-offset "$HARMFUL_OFFSET" \
  --harmful-limit "$HARMFUL_LIMIT" \
  --calib-limit "$CALIB_LIMIT" \
  --calib-max-length "$CALIB_MAX_LENGTH" \
  --ppl-dataset "$PPL_DATASET" \
  --ppl-dataset-config "$PPL_DATASET_CONFIG" \
  --ppl-split "$PPL_SPLIT" \
  --ppl-context-len "$PPL_CONTEXT_LEN" \
  --ppl-stride "$PPL_STRIDE" \
  --ppl-sample-windows "$PPL_SAMPLE_WINDOWS" \
  --window-index-file "$WINDOW_INDEX_FILE" \
  --max-length "$MAX_LENGTH" \
  --sparsities "$SPARSITIES" \
  --q-sweep "$Q_SWEEP" \
  --correlation-sample "$CORRELATION_SAMPLE" \
  --bootstrap-reps "$BOOTSTRAP_REPS" \
  --bootstrap-sample-size "$BOOTSTRAP_SAMPLE_SIZE" \
  --percentile-ci-sample "$PERCENTILE_CI_SAMPLE" \
  --decision-layer "$DECISION_LAYER" \
  --decision-cut-base-tolerance "$DECISION_CUT_BASE_TOLERANCE" \
  --seed "$SEED" \
  "${DATA_ARGS[@]}" \
  "${LOCAL_ARGS[@]}"
