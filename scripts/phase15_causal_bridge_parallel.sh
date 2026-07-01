#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase15_causal_bridge}"
SHARD_ROOT="${SHARD_ROOT:-results/phase15_causal_bridge_shards}"
LOG_DIR="${LOG_DIR:-logs}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
PREPARE_INDEX="${PREPARE_INDEX:-1}"
WAIT_AND_MERGE="${WAIT_AND_MERGE:-0}"
CAUSAL_BRIDGE_PROGRESS_EVERY="${CAUSAL_BRIDGE_PROGRESS_EVERY:-20}"
MASTER_OUTPUT_DIR="$OUTPUT_DIR"
MASTER_INDEX_SET_PATH="${INDEX_SET_PATH:-$MASTER_OUTPUT_DIR/index_sets.pt}"

export MODEL OUTPUT_DIR SHARD_ROOT LOG_DIR MAX_PARALLEL PREPARE_INDEX WAIT_AND_MERGE
export CAUSAL_BRIDGE_PROGRESS_EVERY
export CRIT_OUTPUT_DIR CRIT_CANDIDATE CRIT_SET_PATH TARGET_SUFFIXES SPARSITIES SEED
export PREP_WORKERS MAGMATCH_BINS HARMFUL_FILE HARMFUL_DATASET HARMFUL_CONFIG HARMFUL_SPLIT HARMFUL_COLUMN
export HARMFUL_OFFSET HARMFUL_LIMIT CALIB_FILE CALIB_LIMIT CALIB_MAX_LENGTH MAX_NEW_TOKENS
export RESPONSE_PPL_THRESHOLD JUDGE JUDGE_MODEL JUDGE_MAX_NEW_TOKENS
export PPL_DATASET PPL_DATASET_CONFIG PPL_SPLIT PPL_CONTEXT_LEN PPL_STRIDE PPL_SAMPLE_WINDOWS
export PPL_WINDOW_INDEX_FILE SKIP_PPL SAVE_RAW_TEXT LOCAL_FILES_ONLY DETERMINISTIC EVAL_LIMIT

mkdir -p "$OUTPUT_DIR" "$SHARD_ROOT" "$LOG_DIR"

if [[ "$PREPARE_INDEX" == "1" ]]; then
  echo "[causal-bridge-parallel] preparing frozen index sets"
  MODE=prepare \
  MODEL="$MODEL" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  bash scripts/phase15_causal_bridge.sh
fi

PIDS=()
for (( shard=0; shard<MAX_PARALLEL; shard++ )); do
  out_dir="$SHARD_ROOT/shard_${shard}_of_${MAX_PARALLEL}"
  log_file="$LOG_DIR/causal_bridge_shard_${shard}_of_${MAX_PARALLEL}.log"
  mkdir -p "$out_dir"
  echo "[causal-bridge-parallel] launching shard=$shard/$MAX_PARALLEL out=$out_dir log=$log_file"
  (
    MODE=eval \
    MODEL="$MODEL" \
    OUTPUT_DIR="$out_dir" \
    INDEX_SET_PATH="$MASTER_INDEX_SET_PATH" \
    SHARD_INDEX="$shard" \
    NUM_SHARDS="$MAX_PARALLEL" \
    bash scripts/phase15_causal_bridge.sh
  ) > "$log_file" 2>&1 &
  PIDS+=("$!")
done

echo "[causal-bridge-parallel] pids: ${PIDS[*]}"
echo "[causal-bridge-parallel] logs: $LOG_DIR/causal_bridge_shard_*_of_${MAX_PARALLEL}.log"
echo "[causal-bridge-parallel] merge after completion:"
echo "  SHARD_ROOT=$SHARD_ROOT OUTPUT_DIR=$OUTPUT_DIR bash scripts/phase15_causal_bridge_merge.sh"

if [[ "$WAIT_AND_MERGE" == "1" ]]; then
  status=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    echo "[causal-bridge-parallel] at least one shard failed" >&2
    exit "$status"
  fi
  SHARD_ROOT="$SHARD_ROOT" OUTPUT_DIR="$OUTPUT_DIR" bash scripts/phase15_causal_bridge_merge.sh
fi
