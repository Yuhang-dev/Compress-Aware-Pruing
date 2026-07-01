#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase15_margin_calib}"
SHARD_ROOT="${SHARD_ROOT:-results/phase15_margin_calib_shards}"
LOG_DIR="${LOG_DIR:-logs}"
CONDITIONS="${CONDITIONS:-dense wanda_40 wanda_45 wanda_50 wanda_55}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
WAIT_AND_MERGE="${WAIT_AND_MERGE:-0}"

mkdir -p "$OUTPUT_DIR" "$SHARD_ROOT" "$LOG_DIR"

export MODEL OUTPUT_DIR SHARD_ROOT LOG_DIR CONDITIONS MAX_PARALLEL WAIT_AND_MERGE
export ARTIFACT_DIR LAYERS KR HARMFUL_FILE HARMFUL_DATASET HARMFUL_CONFIG HARMFUL_SPLIT HARMFUL_COLUMN
export HARMFUL_OFFSET HARMFUL_LIMIT EVAL_LIMIT MAX_LENGTH CALIB_MAX_LENGTH MAX_NEW_TOKENS
export RESPONSE_PPL_THRESHOLD JUDGE JUDGE_MODEL JUDGE_MAX_NEW_TOKENS
export RESTORE_S_RESIDUAL_COUNTS RESTORE_S_RESIDUAL_DENOMINATOR AUC_THRESHOLD READOUT_SHARE_THRESHOLD
export LOCAL_FILES_ONLY MARGIN_PROGRESS_EVERY

PIDS=()
TAGS=()
status=0

launch_condition() {
  local condition="$1"
  local out_dir="$SHARD_ROOT/$condition"
  local log_file="$LOG_DIR/margin_calib_${condition}.log"
  mkdir -p "$out_dir"
  echo "[margin-calib-parallel] launching condition=$condition out=$out_dir log=$log_file"
  (
    MODE=eval \
    CONDITIONS="$condition" \
    OUTPUT_DIR="$out_dir" \
    bash scripts/phase15_margin_calibration.sh
  ) > "$log_file" 2>&1 &
  PIDS+=("$!")
  TAGS+=("$condition")
}

reap_one() {
  local pid
  if wait -n; then
    return 0
  fi
  status=1
  return 0
}

for condition in $CONDITIONS; do
  while [[ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]]; do
    reap_one
    # Drop finished pids from the active list.
    next_pids=()
    next_tags=()
    for idx in "${!PIDS[@]}"; do
      if kill -0 "${PIDS[$idx]}" 2>/dev/null; then
        next_pids+=("${PIDS[$idx]}")
        next_tags+=("${TAGS[$idx]}")
      fi
    done
    PIDS=("${next_pids[@]}")
    TAGS=("${next_tags[@]}")
  done
  launch_condition "$condition"
done

echo "[margin-calib-parallel] active pids: ${PIDS[*]}"
echo "[margin-calib-parallel] logs: $LOG_DIR/margin_calib_*.log"
echo "[margin-calib-parallel] merge after completion:"
echo "  SHARD_ROOT=$SHARD_ROOT OUTPUT_DIR=$OUTPUT_DIR bash scripts/phase15_margin_calibration_merge.sh"

if [[ "$WAIT_AND_MERGE" == "1" ]]; then
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    echo "[margin-calib-parallel] at least one shard failed" >&2
    exit "$status"
  fi
  SHARD_ROOT="$SHARD_ROOT" OUTPUT_DIR="$OUTPUT_DIR" bash scripts/phase15_margin_calibration_merge.sh
else
  if [[ "$status" != "0" ]]; then
    echo "[margin-calib-parallel] at least one early shard failed" >&2
    exit "$status"
  fi
fi
