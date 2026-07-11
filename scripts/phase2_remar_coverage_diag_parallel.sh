#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_oracle_hs_and_coverage}"
SHARD_DIR="${SHARD_DIR:-$OUTPUT_DIR/coverage_shards}"
LOG_DIR="${LOG_DIR:-logs/phase2_oracle_hs_and_coverage/coverage}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
mkdir -p "$OUTPUT_DIR" "$SHARD_DIR" "$LOG_DIR"

common=(
  OUTPUT_DIR="$OUTPUT_DIR"
  SHARD_DIR="$SHARD_DIR"
  SOLVE_ARTIFACT="${SOLVE_ARTIFACT:-artifacts/phase2_oracle_hs_and_coverage/remar_solve.pt}"
  MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
  LAYERS="${LAYERS:-24,28,32}"
  ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/vpref_projection}"
  MARGIN_DIR="${MARGIN_DIR:-results/phase15_margin_calib}"
  BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}"
  EVAL_LIMIT="${EVAL_LIMIT:-128}"
  ADVBENCH_EVAL_OFFSET="${ADVBENCH_EVAL_OFFSET:-128}"
  OOD_EVAL_OFFSET="${OOD_EVAL_OFFSET:-0}"
  DECODE_K="${DECODE_K:-32}"
  EPSILON="${EPSILON:-0.5}"
  ETA="${ETA:-1.0}"
  TARGET_MARGIN="${TARGET_MARGIN:-20}"
  LAMBDA_BENIGN="${LAMBDA_BENIGN:-20}"
  FIT_LIMIT="${FIT_LIMIT:-128}"
  BENIGN_FIT_LIMIT="${BENIGN_FIT_LIMIT:-128}"
  MAX_LENGTH="${MAX_LENGTH:-512}"
  CALIB_MAX_LENGTH="${CALIB_MAX_LENGTH:-256}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
  RESPONSE_PPL_THRESHOLD="${RESPONSE_PPL_THRESHOLD:-100}"
  JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-16}"
  CONDITIONAL_MIN_GAP="${CONDITIONAL_MIN_GAP:-1.0}"
  TEMPORAL_DROP_THRESHOLD="${TEMPORAL_DROP_THRESHOLD:-0.20}"
  TEMPORAL_CONTRAST_THRESHOLD="${TEMPORAL_CONTRAST_THRESHOLD:-0.10}"
  MIN_OUTCOME_N="${MIN_OUTCOME_N:-20}"
  SEED="${SEED:-0}"
  HARMBENCH_CONFIG="${HARMBENCH_CONFIG:-standard}"
  LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
)

echo "[coverage] preparing one frozen tm20_lb20 solve"
env "${common[@]}" MODE=prepare \
  bash scripts/phase2_remar_coverage_diag.sh > "$LOG_DIR/prepare.log" 2>&1

datasets=(advbench harmbench strongreject)
pids=()
labels=()
for dataset in "${datasets[@]}"; do
  while (( ${#pids[@]} >= MAX_PARALLEL )); do
    if wait "${pids[0]}"; then
      echo "[coverage] completed ${labels[0]}"
    else
      echo "[coverage] failed ${labels[0]}" >&2
      exit 1
    fi
    pids=("${pids[@]:1}")
    labels=("${labels[@]:1}")
  done
  echo "[coverage] launching dataset=$dataset"
  env "${common[@]}" MODE=cell EVAL_DATASET="$dataset" \
    bash scripts/phase2_remar_coverage_diag.sh > "$LOG_DIR/${dataset}.log" 2>&1 &
  pids+=("$!")
  labels+=("dataset=$dataset")
done

for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[coverage] completed ${labels[$index]}"
  else
    echo "[coverage] failed ${labels[$index]}" >&2
    exit 1
  fi
done

env "${common[@]}" MODE=merge \
  bash scripts/phase2_remar_coverage_diag.sh
