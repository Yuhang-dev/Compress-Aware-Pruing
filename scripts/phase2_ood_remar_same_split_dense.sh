#!/usr/bin/env bash
set -euo pipefail

EVAL_DATASET="${EVAL_DATASET:?Set EVAL_DATASET to harmbench or strongreject}"
CONFIG="${CONFIG:-configs/base.yaml}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/vpref_projection}"
MARGIN_DIR="${MARGIN_DIR:-results/phase15_margin_calib}"
DIRECT_DIR="${DIRECT_DIR:-results/phase2_ood_remar_${EVAL_DATASET}_w50}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_ood_remar_${EVAL_DATASET}_w50_strict}"
DENSE_DIR="$OUTPUT_DIR/dense_run"

case "$EVAL_DATASET" in
  harmbench)
    HARMFUL_EVAL_DATASET="walledai/HarmBench"
    HARMFUL_EVAL_CONFIG="${HARMBENCH_CONFIG:-standard}"
    ;;
  strongreject)
    HARMFUL_EVAL_DATASET="walledai/StrongREJECT"
    HARMFUL_EVAL_CONFIG="${STRONGREJECT_CONFIG:-}"
    ;;
  *)
    echo "Unsupported EVAL_DATASET=$EVAL_DATASET" >&2
    exit 2
    ;;
esac

if [[ ! -f "$DIRECT_DIR/repair_details.csv" || ! -f "$DIRECT_DIR/repair_benign_details.csv" ]]; then
  echo "Missing direct ReMaR details under $DIRECT_DIR" >&2
  exit 2
fi

run_args=(
  CONFIG="$CONFIG"
  MODEL="$MODEL"
  OUTPUT_DIR="$DENSE_DIR"
  ARTIFACT_DIR="$ARTIFACT_DIR"
  MARGIN_DIR="$MARGIN_DIR"
  CONDITIONS="dense"
  REPAIR_MODES="pruned"
  FIT_LIMIT=128
  EVAL_LIMIT=128
  HARMFUL_FIT_OFFSET=0
  HARMFUL_EVAL_OFFSET=0
  HARMFUL_EVAL_DATASET="$HARMFUL_EVAL_DATASET"
  HARMFUL_EVAL_CONFIG="$HARMFUL_EVAL_CONFIG"
  HARMFUL_EVAL_SPLIT="train"
  HARMFUL_EVAL_COLUMN="auto"
  BENIGN_FILE="$BENIGN_FILE"
  BENIGN_FIT_LIMIT=128
  BENIGN_EVAL_LIMIT=128
  BENIGN_FIT_OFFSET=0
  BENIGN_EVAL_OFFSET=128
  RESPONSE_PPL_THRESHOLD=100
  SKIP_PPL=1
)
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  run_args+=(LOCAL_FILES_ONLY=1)
fi

echo "[same-split-ood] generating dense baseline for $EVAL_DATASET"
env "${run_args[@]}" bash scripts/phase15_closed_form_readout_repair.sh

python -m casafety.ood_remar_same_split \
  --eval-dataset "$EVAL_DATASET" \
  --direct-dir "$DIRECT_DIR" \
  --dense-dir "$DENSE_DIR" \
  --output-dir "$OUTPUT_DIR"
