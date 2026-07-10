#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_ood_residual_followup}"
LOG_DIR="${LOG_DIR:-logs/phase2_ood_residual_followup}"
mkdir -p "$LOG_DIR"

pids=()
for dataset in harmbench strongreject; do
  echo "[ood-followup] launching adaptive oracle: $dataset"
  env \
    MODE=adaptive-oracle EVAL_DATASET="$dataset" \
    OUTPUT_DIR="$OUTPUT_DIR" BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}" \
    EVAL_LIMIT="${EVAL_LIMIT:-200}" ORACLE_EPSILONS="${ORACLE_EPSILONS:-0.5,2.0}" \
    LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}" \
    bash scripts/phase2_ood_residual_followup.sh > "$LOG_DIR/adaptive_oracle_${dataset}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
exit "$status"
