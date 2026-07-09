#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/base.yaml}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_ood_direction_eval}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/phase2_ood_direction_eval}"
EXISTING_ARTIFACT_DIR="${EXISTING_ARTIFACT_DIR:-artifacts/vpref_projection}"
EXISTING_MANIFEST="${EXISTING_MANIFEST:-results/phase15_vpref_projection/vpref_manifest.json}"
LAYERS="${LAYERS:-24,28,32}"
CONDITIONS="${CONDITIONS:-dense wanda_50}"
DIRECTIONS="${DIRECTIONS:-adv_existing mixed_scans64 ensemble_avg}"
EVAL_DATASETS="${EVAL_DATASETS:-advbench harmbench strongreject}"
SEED="${SEED:-0}"
DIRECTION_LIMIT="${DIRECTION_LIMIT:-64}"
MIXED_TOTAL_HARMFUL="${MIXED_TOTAL_HARMFUL:-64}"
EVAL_LIMIT="${EVAL_LIMIT:-128}"
EVAL_OFFSET="${EVAL_OFFSET:-0}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RESPONSE_PPL_THRESHOLD="${RESPONSE_PPL_THRESHOLD:-100}"
CALIB_MAX_LENGTH="${CALIB_MAX_LENGTH:-256}"
JUDGE="${JUDGE:-llamaguard}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-32}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
ALLOW_REBUILD_ADV_DIRECTION="${ALLOW_REBUILD_ADV_DIRECTION:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
USE_PREBUILT_DIRECTIONS="${USE_PREBUILT_DIRECTIONS:-0}"

ADVBENCH_DATASET="${ADVBENCH_DATASET:-walledai/AdvBench}"
ADVBENCH_CONFIG="${ADVBENCH_CONFIG:-}"
ADVBENCH_SPLIT="${ADVBENCH_SPLIT:-train}"
ADVBENCH_COLUMN="${ADVBENCH_COLUMN:-auto}"
HARMBENCH_DATASET="${HARMBENCH_DATASET:-walledai/HarmBench}"
HARMBENCH_CONFIG="${HARMBENCH_CONFIG:-}"
HARMBENCH_SPLIT="${HARMBENCH_SPLIT:-train}"
HARMBENCH_COLUMN="${HARMBENCH_COLUMN:-auto}"
STRONGREJECT_DATASET="${STRONGREJECT_DATASET:-walledai/StrongREJECT}"
STRONGREJECT_CONFIG="${STRONGREJECT_CONFIG:-}"
STRONGREJECT_SPLIT="${STRONGREJECT_SPLIT:-train}"
STRONGREJECT_COLUMN="${STRONGREJECT_COLUMN:-auto}"

BENIGN_FILE="${BENIGN_FILE:-}"
BENIGN_DATASET="${BENIGN_DATASET:-yahma/alpaca-cleaned}"
BENIGN_CONFIG="${BENIGN_CONFIG:-}"
BENIGN_SPLIT="${BENIGN_SPLIT:-train}"
BENIGN_COLUMN="${BENIGN_COLUMN:-instruction}"

LOCAL_ARGS=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_ARGS+=(--local-files-only)
fi
if [[ "$ALLOW_REBUILD_ADV_DIRECTION" == "1" ]]; then
  LOCAL_ARGS+=(--allow-rebuild-adv-direction)
fi
if [[ "$PREPARE_ONLY" == "1" ]]; then
  LOCAL_ARGS+=(--prepare-only)
fi
if [[ "$USE_PREBUILT_DIRECTIONS" == "1" ]]; then
  LOCAL_ARGS+=(--use-prebuilt-directions)
fi

JUDGE_ARGS=()
if [[ -n "$JUDGE_MODEL" ]]; then
  JUDGE_ARGS+=(--judge-model "$JUDGE_MODEL")
fi

DATA_ARGS=()
if [[ -n "$ADVBENCH_CONFIG" ]]; then DATA_ARGS+=(--advbench-config "$ADVBENCH_CONFIG"); fi
if [[ -n "$HARMBENCH_CONFIG" ]]; then DATA_ARGS+=(--harmbench-config "$HARMBENCH_CONFIG"); fi
if [[ -n "$STRONGREJECT_CONFIG" ]]; then DATA_ARGS+=(--strongreject-config "$STRONGREJECT_CONFIG"); fi
if [[ -n "$BENIGN_FILE" ]]; then
  DATA_ARGS+=(--benign-file "$BENIGN_FILE")
else
  DATA_ARGS+=(--benign-dataset "$BENIGN_DATASET" --benign-split "$BENIGN_SPLIT" --benign-column "$BENIGN_COLUMN")
  if [[ -n "$BENIGN_CONFIG" ]]; then DATA_ARGS+=(--benign-config "$BENIGN_CONFIG"); fi
fi

python -m casafety.ood_direction_eval \
  --config "$CONFIG" \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --artifact-dir "$ARTIFACT_DIR" \
  --existing-artifact-dir "$EXISTING_ARTIFACT_DIR" \
  --existing-manifest "$EXISTING_MANIFEST" \
  --layers "$LAYERS" \
  --conditions "$CONDITIONS" \
  --directions "$DIRECTIONS" \
  --eval-datasets "$EVAL_DATASETS" \
  --seed "$SEED" \
  --direction-limit "$DIRECTION_LIMIT" \
  --mixed-total-harmful "$MIXED_TOTAL_HARMFUL" \
  --eval-limit "$EVAL_LIMIT" \
  --eval-offset "$EVAL_OFFSET" \
  --max-length "$MAX_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --response-ppl-threshold "$RESPONSE_PPL_THRESHOLD" \
  --calib-max-length "$CALIB_MAX_LENGTH" \
  --judge "$JUDGE" \
  --judge-max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
  --advbench-dataset "$ADVBENCH_DATASET" \
  --advbench-split "$ADVBENCH_SPLIT" \
  --advbench-column "$ADVBENCH_COLUMN" \
  --harmbench-dataset "$HARMBENCH_DATASET" \
  --harmbench-split "$HARMBENCH_SPLIT" \
  --harmbench-column "$HARMBENCH_COLUMN" \
  --strongreject-dataset "$STRONGREJECT_DATASET" \
  --strongreject-split "$STRONGREJECT_SPLIT" \
  --strongreject-column "$STRONGREJECT_COLUMN" \
  "${JUDGE_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${LOCAL_ARGS[@]}"
