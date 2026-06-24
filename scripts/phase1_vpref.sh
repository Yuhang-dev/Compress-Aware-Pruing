#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
LAYERS="${LAYERS:-auto}"
TARGET_SUFFIXES="${TARGET_SUFFIXES:-o_proj down_proj}"
RUN_ABLATION="${RUN_ABLATION:-0}"
ABLATION_ARGS=()
if [[ "$RUN_ABLATION" == "1" ]]; then
  ABLATION_ARGS=(
    --run-ablation
    --ablation-output results/crit_ablation.csv
    --ablation-details-output results/crit_ablation_details.csv
  )
fi

python -m casafety.vpref \
  --config configs/base.yaml \
  --model "$MODEL" \
  --layers "$LAYERS" \
  --output results/vpref_validation.csv \
  --local-files-only

python -m casafety.mechanism_diagnosis \
  --config configs/base.yaml \
  --model "$MODEL" \
  --target-suffixes $TARGET_SUFFIXES \
  --output results/mechanism_diagnosis.csv \
  --crit-localization-output results/crit_localization.csv \
  "${ABLATION_ARGS[@]}" \
  --local-files-only
