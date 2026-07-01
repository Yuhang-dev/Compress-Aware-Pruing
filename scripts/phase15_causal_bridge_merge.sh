#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/phase15_causal_bridge}"
SHARD_ROOT="${SHARD_ROOT:-results/phase15_causal_bridge_shards}"
MIN_GAP="${MIN_GAP:-0.03}"
SPECIFICITY_MARGIN="${SPECIFICITY_MARGIN:-0.03}"

python -m casafety.causal_bridge \
  --mode merge \
  --output-dir "$OUTPUT_DIR" \
  --shard-root "$SHARD_ROOT" \
  --min-gap "$MIN_GAP" \
  --specificity-margin "$SPECIFICITY_MARGIN"
