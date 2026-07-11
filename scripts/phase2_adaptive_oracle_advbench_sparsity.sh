#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-merge}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_adaptive_oracle_advbench_sparsity}"
SHARD_DIR="${SHARD_DIR:-$OUTPUT_DIR/shards}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/vpref_projection}"
MARGIN_DIR="${MARGIN_DIR:-results/phase15_margin_calib}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-results/phase15_vpref_projection/vpref_manifest.json}"
LAYERS="${LAYERS:-24,28,32}"
SPARSITY="${SPARSITY:-0.50}"
SPARSITIES="${SPARSITIES:-0.40,0.45,0.50,0.55}"
BETA="${BETA:-0.5}"
EPSILON="${EPSILON:-0.5}"
SEED="${SEED:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-128}"
BENIGN_EVAL_LIMIT="${BENIGN_EVAL_LIMIT:-128}"
MAX_LENGTH="${MAX_LENGTH:-512}"
CALIB_MAX_LENGTH="${CALIB_MAX_LENGTH:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
BENIGN_MAX_NEW_TOKENS="${BENIGN_MAX_NEW_TOKENS:-128}"
RESPONSE_PPL_THRESHOLD="${RESPONSE_PPL_THRESHOLD:-100}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-16}"
ORACLE_MIN_COHERENCE="${ORACLE_MIN_COHERENCE:-0.95}"
MAX_NEGATIVE_MARGIN="${MAX_NEGATIVE_MARGIN:-0.05}"
DENSE_ASR_TOLERANCE="${DENSE_ASR_TOLERANCE:-0.03}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}"

mkdir -p "$OUTPUT_DIR" "$SHARD_DIR"

args=(
  --mode "$MODE"
  --model "$MODEL"
  --output-dir "$OUTPUT_DIR"
  --shard-dir "$SHARD_DIR"
  --artifact-dir "$ARTIFACT_DIR"
  --margin-dir "$MARGIN_DIR"
  --split-manifest "$SPLIT_MANIFEST"
  --layers "$LAYERS"
  --sparsity "$SPARSITY"
  --sparsities "$SPARSITIES"
  --beta "$BETA"
  --epsilon "$EPSILON"
  --seed "$SEED"
  --eval-limit "$EVAL_LIMIT"
  --benign-eval-limit "$BENIGN_EVAL_LIMIT"
  --max-length "$MAX_LENGTH"
  --calib-max-length "$CALIB_MAX_LENGTH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --benign-max-new-tokens "$BENIGN_MAX_NEW_TOKENS"
  --response-ppl-threshold "$RESPONSE_PPL_THRESHOLD"
  --judge-max-new-tokens "$JUDGE_MAX_NEW_TOKENS"
  --oracle-min-coherence "$ORACLE_MIN_COHERENCE"
  --max-negative-margin "$MAX_NEGATIVE_MARGIN"
  --dense-asr-tolerance "$DENSE_ASR_TOLERANCE"
)

if [[ -n "${JUDGE_MODEL:-}" ]]; then args+=(--judge-model "$JUDGE_MODEL"); fi
if [[ -n "$BENIGN_FILE" ]]; then args+=(--benign-file "$BENIGN_FILE"); fi
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then args+=(--local-files-only); fi

python -m casafety.adaptive_oracle_sparsity "${args[@]}"
