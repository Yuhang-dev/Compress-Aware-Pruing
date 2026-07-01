#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase15_margin_calib}"
SHARD_ROOT="${SHARD_ROOT:-results/phase15_margin_calib_shards}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/vpref_projection}"
LAYERS="${LAYERS:-24,28,32}"
KR="${KR:-1}"
RESTORE_S_RESIDUAL_COUNTS="${RESTORE_S_RESIDUAL_COUNTS:-wanda_40:0 wanda_45:0 wanda_50:5 wanda_55:9}"
RESTORE_S_RESIDUAL_DENOMINATOR="${RESTORE_S_RESIDUAL_DENOMINATOR:-128}"
AUC_THRESHOLD="${AUC_THRESHOLD:-0.85}"
READOUT_SHARE_THRESHOLD="${READOUT_SHARE_THRESHOLD:-0.85}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

MERGE_DIRS=()
for dir in "$SHARD_ROOT"/*; do
  if [[ -f "$dir/margin_points.csv" ]]; then
    MERGE_DIRS+=("$dir")
  fi
done

if [[ "${#MERGE_DIRS[@]}" == "0" ]]; then
  echo "[margin-calib-merge] no margin_points.csv files found under $SHARD_ROOT" >&2
  exit 1
fi

ARGS=(
  --mode analyze
  --config configs/base.yaml
  --model "$MODEL"
  --output-dir "$OUTPUT_DIR"
  --artifact-dir "$ARTIFACT_DIR"
  --layers "$LAYERS"
  --kr "$KR"
  --restore-s-residual-counts "$RESTORE_S_RESIDUAL_COUNTS"
  --restore-s-residual-denominator "$RESTORE_S_RESIDUAL_DENOMINATOR"
  --auc-threshold "$AUC_THRESHOLD"
  --readout-share-threshold "$READOUT_SHARE_THRESHOLD"
  --merge-dirs "${MERGE_DIRS[@]}"
)

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  ARGS+=(--local-files-only)
else
  ARGS+=(--no-local-files-only)
fi

python -m casafety.margin_calibration "${ARGS[@]}"
