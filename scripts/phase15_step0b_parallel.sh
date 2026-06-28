#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
SHARD_ROOT="${SHARD_ROOT:-results/phase15_step0b_parallel}"
LOG_DIR="${LOG_DIR:-logs}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/vpref_projection}"
MANIFEST="${MANIFEST:-results/phase15_vpref_projection/vpref_manifest.json}"
BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}"
EVAL_LIMIT="${EVAL_LIMIT:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-16}"
STEP0B_WINDOWS="${STEP0B_WINDOWS:-32}"
STEP0B_PROGRESS_EVERY="${STEP0B_PROGRESS_EVERY:-20}"
SEED="${SEED:-0}"
DIRECTION_LIMIT="${DIRECTION_LIMIT:-256}"
HARM_EVAL_OFFSET="${HARM_EVAL_OFFSET:-0}"
BENIGN_EVAL_OFFSET="${BENIGN_EVAL_OFFSET:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
PREPARE_INPUTS="${PREPARE_INPUTS:-1}"
WAIT_AND_MERGE="${WAIT_AND_MERGE:-0}"
MERGED_OUTPUT_DIR="${MERGED_OUTPUT_DIR:-results/phase15_step0b_parallel_merged}"
STEP0B_SHARDS="${STEP0B_SHARDS:-lg28_k1 lg28_k4 lg242832_k1 lg242832_k4}"

mkdir -p "$SHARD_ROOT" "$LOG_DIR"

COMMON_ENV=(
  MODEL="$MODEL"
  ARTIFACT_DIR="$ARTIFACT_DIR"
  MANIFEST="$MANIFEST"
  BENIGN_FILE="$BENIGN_FILE"
  EVAL_LIMIT="$EVAL_LIMIT"
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
  JUDGE_MAX_NEW_TOKENS="$JUDGE_MAX_NEW_TOKENS"
  STEP0B_WINDOWS="$STEP0B_WINDOWS"
  STEP0B_PROGRESS_EVERY="$STEP0B_PROGRESS_EVERY"
  SEED="$SEED"
  DIRECTION_LIMIT="$DIRECTION_LIMIT"
  HARM_EVAL_OFFSET="$HARM_EVAL_OFFSET"
  BENIGN_EVAL_OFFSET="$BENIGN_EVAL_OFFSET"
  LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY"
)

LOCAL_ARGS=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_ARGS+=(--local-files-only)
fi

if [[ "$PREPARE_INPUTS" == "1" ]]; then
  echo "[step0b-parallel] preparing shared direction inputs"
  env "${COMMON_ENV[@]}" \
    OUTPUT_DIR="$SHARD_ROOT/_prepare" \
    STEP0B_LAYER_GROUPS="28;24,28,32" \
    STEP0B_KR="1 4" \
    python -m casafety.step0_restore_s \
      --step0b \
      --step0b-prepare-only \
      --config configs/base.yaml \
      --model "$MODEL" \
      --output-dir "$SHARD_ROOT/_prepare" \
      --artifact-dir "$ARTIFACT_DIR" \
      --manifest "$MANIFEST" \
      --seed "$SEED" \
      --direction-limit "$DIRECTION_LIMIT" \
      --eval-limit "$EVAL_LIMIT" \
      --harm-eval-offset "$HARM_EVAL_OFFSET" \
      --benign-eval-offset "$BENIGN_EVAL_OFFSET" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --judge-max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
      --step0b-layer-groups "28;24,28,32" \
      --step0b-kr 1 4 \
      --step0b-windows $STEP0B_WINDOWS \
      --benign-file "$BENIGN_FILE" \
      "${LOCAL_ARGS[@]}"
fi

PIDS=()
DIRS=()

run_shard() {
  local layer_group="$1"
  local kr="$2"
  local tag="$3"
  local out_dir="$SHARD_ROOT/$tag"
  local log_file="$LOG_DIR/$tag.log"
  mkdir -p "$out_dir"
  echo "[step0b-parallel] launching $tag layer_group=$layer_group k=$kr out=$out_dir log=$log_file"
  (
    env "${COMMON_ENV[@]}" \
      OUTPUT_DIR="$out_dir" \
      STEP0B_LAYER_GROUPS="$layer_group" \
      STEP0B_KR="$kr" \
      bash scripts/phase15_step0b_restore_s.sh
  ) > "$log_file" 2>&1 &
  PIDS+=("$!")
  DIRS+=("$out_dir")
}

should_run_shard() {
  local tag="$1"
  [[ " $STEP0B_SHARDS " == *" $tag "* ]]
}

if should_run_shard "lg28_k1"; then
  run_shard "28" "1" "lg28_k1"
fi
if should_run_shard "lg28_k4"; then
  run_shard "28" "4" "lg28_k4"
fi
if should_run_shard "lg242832_k1"; then
  run_shard "24,28,32" "1" "lg242832_k1"
fi
if should_run_shard "lg242832_k4"; then
  run_shard "24,28,32" "4" "lg242832_k4"
fi

if [[ "${#PIDS[@]}" == "0" ]]; then
  echo "[step0b-parallel] no shards selected by STEP0B_SHARDS=$STEP0B_SHARDS" >&2
  exit 1
fi

echo "[step0b-parallel] pids: ${PIDS[*]}"
echo "[step0b-parallel] logs: $LOG_DIR/lg*.log"
echo "[step0b-parallel] merge after completion:"
echo "  SHARD_ROOT=$SHARD_ROOT MERGED_OUTPUT_DIR=$MERGED_OUTPUT_DIR bash scripts/phase15_step0b_merge.sh"

if [[ "$WAIT_AND_MERGE" == "1" ]]; then
  status=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    echo "[step0b-parallel] at least one shard failed" >&2
    exit "$status"
  fi
  SHARD_ROOT="$SHARD_ROOT" MERGED_OUTPUT_DIR="$MERGED_OUTPUT_DIR" bash scripts/phase15_step0b_merge.sh
fi
