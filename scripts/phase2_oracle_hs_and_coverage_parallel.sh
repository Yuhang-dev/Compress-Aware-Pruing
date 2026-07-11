#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_oracle_hs_and_coverage}"
LOG_DIR="${LOG_DIR:-logs/phase2_oracle_hs_and_coverage}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[joint] Part 1: AdvBench high-sparsity adaptive oracle"
env \
  OUTPUT_DIR="$OUTPUT_DIR" \
  SHARD_DIR="$OUTPUT_DIR/part1_shards" \
  LOG_DIR="$LOG_DIR/part1" \
  MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
  BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}" \
  EVAL_LIMIT="${PART1_EVAL_LIMIT:-128}" \
  BENIGN_EVAL_LIMIT="${PART1_BENIGN_EVAL_LIMIT:-128}" \
  SPARSITIES="${SPARSITIES:-0.40,0.45,0.50,0.55}" \
  MAX_PARALLEL="${PART1_MAX_PARALLEL:-4}" \
  BETA="${BETA:-0.5}" \
  EPSILON="${EPSILON:-0.5}" \
  LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}" \
  SHUTDOWN=0 \
  bash scripts/phase2_adaptive_oracle_advbench_sparsity_parallel.sh

echo "[joint] Part 2: ReMaR conditional/temporal coverage"
env \
  OUTPUT_DIR="$OUTPUT_DIR" \
  SHARD_DIR="$OUTPUT_DIR/coverage_shards" \
  LOG_DIR="$LOG_DIR/coverage" \
  MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
  BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}" \
  EVAL_LIMIT="${PART2_EVAL_LIMIT:-128}" \
  DECODE_K="${DECODE_K:-32}" \
  MAX_PARALLEL="${PART2_MAX_PARALLEL:-3}" \
  EPSILON="${EPSILON:-0.5}" \
  TARGET_MARGIN="${TARGET_MARGIN:-20}" \
  LAMBDA_BENIGN="${LAMBDA_BENIGN:-20}" \
  LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}" \
  SHUTDOWN=0 \
  bash scripts/phase2_remar_coverage_diag_parallel.sh

if [[ "${SHUTDOWN:-0}" == "1" ]]; then
  /usr/bin/shutdown
fi
