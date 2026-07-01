#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/base.yaml}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
CONDITIONS="${CONDITIONS:-wanda_45 wanda_50}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_readout_repair}"
SHARD_ROOT="${SHARD_ROOT:-results/phase2_readout_repair_shards}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LOG_DIR="${LOG_DIR:-logs}"
PPL_MAX_DELTA="${PPL_MAX_DELTA:-1.0}"
BENIGN_REFUSAL_MAX_DELTA="${BENIGN_REFUSAL_MAX_DELTA:-0.05}"

mkdir -p "$SHARD_ROOT" "$LOG_DIR"

pids=()
active=0
for condition in $CONDITIONS; do
  shard_dir="$SHARD_ROOT/$condition"
  mkdir -p "$shard_dir"
  log_file="$LOG_DIR/readout_repair_${condition}.log"
  echo "[readout-repair-parallel] launching condition=$condition out=$shard_dir log=$log_file"
  CONDITIONS="$condition" OUTPUT_DIR="$shard_dir" PPL_WINDOW_INDEX_FILE="$shard_dir/ppl_windows_wikitext2_seed0.json" \
    bash scripts/phase15_closed_form_readout_repair.sh >"$log_file" 2>&1 &
  pids+=("$!")
  active=$((active + 1))
  if (( active >= MAX_PARALLEL )); then
    wait -n
    active=$((active - 1))
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

python -m casafety.closed_form_readout_repair \
  --mode merge \
  --config "$CONFIG" \
  --model "$MODEL" \
  --shard-root "$SHARD_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --ppl-max-delta "$PPL_MAX_DELTA" \
  --benign-refusal-max-delta "$BENIGN_REFUSAL_MAX_DELTA"
