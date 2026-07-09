#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_ood_direction_eval}"
SHARD_ROOT="${SHARD_ROOT:-results/phase2_ood_direction_eval_shards}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/phase2_ood_direction_eval}"
LOG_DIR="${LOG_DIR:-logs}"
CONDITIONS="${CONDITIONS:-dense wanda_50}"
EVAL_DATASETS="${EVAL_DATASETS:-advbench harmbench strongreject}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

mkdir -p "$OUTPUT_DIR" "$SHARD_ROOT" "$LOG_DIR"

echo "[ood-direction-parallel] preparing shared directions"
PREPARE_ONLY=1 \
OUTPUT_DIR="$OUTPUT_DIR/_prepare" \
ARTIFACT_DIR="$ARTIFACT_DIR" \
CONDITIONS="dense" \
EVAL_DATASETS="$EVAL_DATASETS" \
  bash scripts/phase2_ood_direction_eval.sh

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    sleep 5
  done
}

launch_cell() {
  local condition="$1"
  local dataset="$2"
  local tag="${condition}_${dataset}"
  local out_dir="$SHARD_ROOT/$tag"
  local log_file="$LOG_DIR/ood_direction_${tag}.log"
  mkdir -p "$out_dir"
  echo "[ood-direction-parallel] launching condition=$condition dataset=$dataset out=$out_dir log=$log_file"
  USE_PREBUILT_DIRECTIONS=1 \
  OUTPUT_DIR="$out_dir" \
  ARTIFACT_DIR="$ARTIFACT_DIR" \
  CONDITIONS="$condition" \
  EVAL_DATASETS="$dataset" \
    bash scripts/phase2_ood_direction_eval.sh >"$log_file" 2>&1 &
}

for condition in $CONDITIONS; do
  for dataset in $EVAL_DATASETS; do
    wait_for_slot
    launch_cell "$condition" "$dataset"
  done
done

wait

python -m casafety.ood_direction_eval \
  --mode merge \
  --shard-root "$SHARD_ROOT" \
  --output-dir "$OUTPUT_DIR"

if [[ "${SHUTDOWN:-0}" == "1" ]]; then
  /usr/bin/shutdown
fi
