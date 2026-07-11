#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_oracle_hs_coverage_v2}"
SHARD_DIR="${SHARD_DIR:-$OUTPUT_DIR/coverage_v2_shards}"
LOG_DIR="${LOG_DIR:-logs/phase2_oracle_hs_coverage_v2/coverage}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
mkdir -p "$OUTPUT_DIR" "$SHARD_DIR" "$LOG_DIR"

common=(
  OUTPUT_DIR="$OUTPUT_DIR"
  SHARD_DIR="$SHARD_DIR"
  CANONICAL_ARTIFACT="${CANONICAL_ARTIFACT:-artifacts/phase2_oracle_hs_coverage_v2/canonical_remar.pt}"
  MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
  BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}"
  EVAL_LIMIT="${EVAL_LIMIT:-128}"
  DECODE_K="${DECODE_K:-32}"
  EPSILON="${EPSILON:-0.5}"
  MAX_LENGTH="${MAX_LENGTH:-512}"
  CALIB_MAX_LENGTH="${CALIB_MAX_LENGTH:-256}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
  RESPONSE_PPL_THRESHOLD="${RESPONSE_PPL_THRESHOLD:-100}"
  JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-16}"
  PERMUTATIONS="${PERMUTATIONS:-5000}"
  PREFILL_MARGIN_FRACTION="${PREFILL_MARGIN_FRACTION:-0.95}"
  TEMPORAL_GAP_MIN="${TEMPORAL_GAP_MIN:-1.0}"
  TEMPORAL_P_MAX="${TEMPORAL_P_MAX:-0.05}"
  MIN_OUTCOME_N="${MIN_OUTCOME_N:-20}"
  SEED="${SEED:-0}"
  HARMBENCH_CONFIG="${HARMBENCH_CONFIG:-standard}"
  LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
)

datasets=(advbench harmbench strongreject)
pids=()
labels=()
for dataset in "${datasets[@]}"; do
  while (( ${#pids[@]} >= MAX_PARALLEL )); do
    wait "${pids[0]}" || { echo "[coverage-v2] failed ${labels[0]}" >&2; exit 1; }
    pids=("${pids[@]:1}"); labels=("${labels[@]:1}")
  done
  echo "[coverage-v2] launching $dataset"
  env "${common[@]}" MODE=cell EVAL_DATASET="$dataset" \
    bash scripts/phase2_remar_coverage_v2.sh > "$LOG_DIR/${dataset}.log" 2>&1 &
  pids+=("$!"); labels+=("$dataset")
done
for index in "${!pids[@]}"; do
  wait "${pids[$index]}" || { echo "[coverage-v2] failed ${labels[$index]}" >&2; exit 1; }
done
env "${common[@]}" MODE=merge bash scripts/phase2_remar_coverage_v2.sh

if [[ "${SHUTDOWN:-0}" == "1" ]]; then /usr/bin/shutdown; fi
