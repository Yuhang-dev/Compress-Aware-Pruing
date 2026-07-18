#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_oracle_policy_distill_min_margin_sequential}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/phase2_oracle_policy_distill_min_margin_sequential}"
SHARD_DIR="${SHARD_DIR:-$OUTPUT_DIR/shards}"
LOG_DIR="${LOG_DIR:-logs/phase2_oracle_policy_distill_min_margin_sequential/eval}"
mkdir -p "$SHARD_DIR" "$LOG_DIR"

common=(
  OUTPUT_DIR="$OUTPUT_DIR"
  ARTIFACT_DIR="$ARTIFACT_DIR"
  SHARD_DIR="$SHARD_DIR"
  BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}"
  LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
)

pids=()
labels=()
for dataset in advbench harmbench strongreject; do
  env "${common[@]}" MODE=harmful-cell EVAL_DATASET="$dataset" \
    bash scripts/phase2_oracle_policy_distill.sh > "$LOG_DIR/$dataset.log" 2>&1 &
  pids+=("$!"); labels+=("$dataset")
done
env "${common[@]}" MODE=benign bash scripts/phase2_oracle_policy_distill.sh \
  > "$LOG_DIR/benign.log" 2>&1 &
pids+=("$!"); labels+=("benign")

for index in "${!pids[@]}"; do
  wait "${pids[$index]}" || { echo "[policy-distill] failed ${labels[$index]}" >&2; exit 1; }
done
env "${common[@]}" MODE=ppl bash scripts/phase2_oracle_policy_distill.sh \
  > "$LOG_DIR/ppl.log" 2>&1
env "${common[@]}" MODE=merge bash scripts/phase2_oracle_policy_distill.sh \
  > "$LOG_DIR/merge.log" 2>&1

if [[ "${SHUTDOWN:-0}" == "1" ]]; then /usr/bin/shutdown; fi
