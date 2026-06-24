#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TARGET_SUFFIXES="${TARGET_SUFFIXES:-o_proj down_proj}"
SPARSITIES="${SPARSITIES:-0.45 0.50}"
LAYERS="${LAYERS:-24 28 32 34 35}"
KR="${KR:-1 4 8}"

python -m casafety.phase1_followup_diagnostics \
  --config configs/base.yaml \
  --model "$MODEL" \
  --target-suffixes $TARGET_SUFFIXES \
  --sparsities $SPARSITIES \
  --layers $LAYERS \
  --kr $KR \
  --survival-output results/phase1_crit_wanda_survival.csv \
  --a2-output results/phase1_a2_channel_diagnostic.csv \
  --local-files-only
