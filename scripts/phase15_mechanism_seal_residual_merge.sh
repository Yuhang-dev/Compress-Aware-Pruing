#!/usr/bin/env bash
set -euo pipefail

SHARD_ROOT="${SHARD_ROOT:-results/phase15_mechanism_seal_residual_sweep_shards}"
WANDA_CONFIGS="${WANDA_CONFIGS:-wanda_40:0.40 wanda_45:0.45 wanda_50:0.50 wanda_55:0.55}"
SEAL_OUTPUT="${SEAL_OUTPUT:-results/phase15_vpref_projection/mechanism_seal_residual_sweep.csv}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
LAYER_GROUP="${LAYER_GROUP:-24,28,32}"
WINDOW="${WINDOW:-32}"
KR="${KR:-1}"
STEERING_MODE="${STEERING_MODE:-norm_relative}"
TARGET_KIND="${TARGET_KIND:-strong}"
BETA_VALUE="${BETA_VALUE:-0.25}"
COHERENT_MIN="${COHERENT_MIN:-0.85}"

DIRS=()
for spec in $WANDA_CONFIGS; do
  tag="${spec%%:*}"
  dir="$SHARD_ROOT/$tag"
  if [[ ! -f "$dir/step0b_restore_s.csv" ]]; then
    echo "[mechanism-seal] missing shard summary: $dir/step0b_restore_s.csv" >&2
    exit 1
  fi
  if [[ ! -f "$dir/step0b_restore_s_details.csv" ]]; then
    echo "[mechanism-seal] missing shard details: $dir/step0b_restore_s_details.csv" >&2
    exit 1
  fi
  DIRS+=("$dir")
done

python -m casafety.mechanism_seal residual-sweep \
  --input-dirs "${DIRS[@]}" \
  --output "$SEAL_OUTPUT" \
  --model "$MODEL" \
  --layer-group "$LAYER_GROUP" \
  --window "$WINDOW" \
  --k-r "$KR" \
  --steering-mode "$STEERING_MODE" \
  --target-kind "$TARGET_KIND" \
  --beta-value "$BETA_VALUE" \
  --coherent-min "$COHERENT_MIN"
