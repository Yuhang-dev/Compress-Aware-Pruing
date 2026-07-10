#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-run}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_lodo_remar}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/phase2_lodo_remar}"
CELL_INDEX="${CELL_INDEX:-0}"

ARGS=(
  --mode "$MODE"
  --model "${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
  --output-dir "$OUTPUT_DIR"
  --artifact-dir "$ARTIFACT_DIR"
  --layers "${LAYERS:-24,28,32}"
  --heldout-datasets "${HELDOUT_DATASETS:-advbench harmbench strongreject}"
  --cell-index "$CELL_INDEX"
  --seed "${SEED:-0}"
  --direction-limit "${DIRECTION_LIMIT:-64}"
  --fit-limit "${FIT_LIMIT:-128}"
  --eval-limit "${EVAL_LIMIT:-128}"
  --benign-direction-limit "${BENIGN_DIRECTION_LIMIT:-64}"
  --benign-fit-limit "${BENIGN_FIT_LIMIT:-128}"
  --benign-eval-limit "${BENIGN_EVAL_LIMIT:-128}"
  --max-length "${MAX_LENGTH:-512}"
  --calib-max-length "${CALIB_MAX_LENGTH:-256}"
  --max-new-tokens "${MAX_NEW_TOKENS:-128}"
  --benign-max-new-tokens "${BENIGN_MAX_NEW_TOKENS:-128}"
  --response-ppl-threshold "${RESPONSE_PPL_THRESHOLD:-100}"
  --judge "${JUDGE:-llamaguard}"
  --judge-max-new-tokens "${JUDGE_MAX_NEW_TOKENS:-16}"
  --target-margin "${TARGET_MARGIN:-20}"
  --lambda-benign "${LAMBDA_BENIGN:-20}"
  --ridge-mu "${RIDGE_MU:-0.01}"
  --delta-max "${DELTA_MAX:-50}"
  --harmbench-config "${HARMBENCH_CONFIG:-standard}"
)

if [[ -n "${BENIGN_FILE:-}" ]]; then ARGS+=(--benign-file "$BENIGN_FILE"); fi
if [[ "${LOCAL_FILES_ONLY:-1}" == "1" ]]; then ARGS+=(--local-files-only); fi
if [[ "${INCLUDE_RANDOM_CONTROL:-0}" == "1" ]]; then ARGS+=(--include-random-control); fi

python -m casafety.lodo_remar "${ARGS[@]}"
