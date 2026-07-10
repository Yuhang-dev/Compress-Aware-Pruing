#!/usr/bin/env bash
set -euo pipefail

EVAL_DATASETS="${EVAL_DATASETS:-harmbench strongreject}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LOG_DIR="${LOG_DIR:-logs/phase2_ood_remar_same_split_dense}"
mkdir -p "$LOG_DIR"

pids=()
for dataset in $EVAL_DATASETS; do
  while (( ${#pids[@]} >= MAX_PARALLEL )); do
    wait -n
    live=()
    for pid in "${pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && live+=("$pid")
    done
    pids=("${live[@]}")
  done
  echo "[same-split-ood] launching $dataset"
  EVAL_DATASET="$dataset" bash scripts/phase2_ood_remar_same_split_dense.sh \
    >"$LOG_DIR/${dataset}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
