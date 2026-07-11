#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_oracle_hs_coverage_v2}"
LOG_DIR="${LOG_DIR:-logs/phase2_oracle_hs_coverage_v2}"
CANONICAL_ARTIFACT="${CANONICAL_ARTIFACT:-artifacts/phase2_oracle_hs_coverage_v2/canonical_remar.pt}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[v2] reconstructing and validating canonical ReMaR"
env \
  OUTPUT_DIR="$OUTPUT_DIR" \
  CANONICAL_ARTIFACT="$CANONICAL_ARTIFACT" \
  BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}" \
  LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}" \
  bash scripts/phase2_export_canonical_remar.sh > "$LOG_DIR/canonical.log" 2>&1

echo "[v2] Part 1 one-sided floor oracle"
env \
  OUTPUT_DIR="$OUTPUT_DIR" \
  SHARD_DIR="$OUTPUT_DIR/part1_shards" \
  LOG_DIR="$LOG_DIR/part1" \
  BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}" \
  EVAL_LIMIT="${PART1_EVAL_LIMIT:-128}" \
  BENIGN_EVAL_LIMIT="${PART1_BENIGN_EVAL_LIMIT:-128}" \
  SPARSITIES="${SPARSITIES:-0.40,0.45,0.50,0.55}" \
  MAX_PARALLEL="${PART1_MAX_PARALLEL:-4}" \
  ONE_SIDED_ADAPTIVE=1 \
  LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}" \
  SHUTDOWN=0 \
  bash scripts/phase2_adaptive_oracle_advbench_sparsity_parallel.sh

echo "[v2] Part 2 position-matched coverage"
env \
  OUTPUT_DIR="$OUTPUT_DIR" \
  SHARD_DIR="$OUTPUT_DIR/coverage_v2_shards" \
  LOG_DIR="$LOG_DIR/coverage" \
  CANONICAL_ARTIFACT="$CANONICAL_ARTIFACT" \
  BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}" \
  EVAL_LIMIT="${PART2_EVAL_LIMIT:-128}" \
  DECODE_K="${DECODE_K:-32}" \
  MAX_PARALLEL="${PART2_MAX_PARALLEL:-3}" \
  LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}" \
  SHUTDOWN=0 \
  bash scripts/phase2_remar_coverage_v2_parallel.sh

if [[ "${SHUTDOWN:-0}" == "1" ]]; then /usr/bin/shutdown; fi
