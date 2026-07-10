#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_ood_residual_followup}"
LOG_DIR="${LOG_DIR:-logs/phase2_ood_residual_followup}"
RANK2_ARTIFACT="${RANK2_ARTIFACT:-artifacts/phase2_ood_residual_followup/rank2_payload.pt}"
mkdir -p "$LOG_DIR"

common=(
  OUTPUT_DIR="$OUTPUT_DIR"
  RANK2_ARTIFACT="$RANK2_ARTIFACT"
  BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}"
  LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
  RANK2_TRAIN_LIMIT="${RANK2_TRAIN_LIMIT:-128}"
  RANK2_EVAL_LIMIT="${RANK2_EVAL_LIMIT:-72}"
  RANK2_TARGET_MARGIN="${RANK2_TARGET_MARGIN:-2.0}"
)

echo "[ood-followup] preparing frozen rank-2 payload"
env "${common[@]}" MODE=rank2-prepare \
  bash scripts/phase2_ood_residual_followup.sh > "$LOG_DIR/rank2_prepare.log" 2>&1

pids=()
for eta in ${RANK2_ETAS:-0 0.25 0.5 1.0}; do
  tag="${eta//./p}"
  echo "[ood-followup] launching rank2 eta=$eta"
  env "${common[@]}" MODE=rank2-cell RANK2_ETA="$eta" \
    bash scripts/phase2_ood_residual_followup.sh > "$LOG_DIR/rank2_eta${tag}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
if [[ "$status" != "0" ]]; then exit "$status"; fi

echo "[ood-followup] launching matched random rank2 eta=1.0"
env "${common[@]}" MODE=rank2-cell RANK2_ETA=1.0 RANK2_RANDOM_DIRECTION=1 \
  bash scripts/phase2_ood_residual_followup.sh > "$LOG_DIR/rank2_random_eta1.log" 2>&1

env "${common[@]}" MODE=rank2-merge bash scripts/phase2_ood_residual_followup.sh
if [[ "${SHUTDOWN:-0}" == "1" ]]; then /usr/bin/shutdown; fi
