#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
CONTEXT_LEN="${CONTEXT_LEN:-1024}"
STRIDE="${STRIDE:-512}"
SAMPLE_WINDOWS="${SAMPLE_WINDOWS:-128}"
SEED="${SEED:-0}"
SPARSITIES="${SPARSITIES:-0.45 0.50}"
PRUNERS="${PRUNERS:-wanda}"

python -m casafety.ppl_eval_v2 \
  --config configs/base.yaml \
  --model "$MODEL" \
  --context-len "$CONTEXT_LEN" \
  --stride "$STRIDE" \
  --sample-windows "$SAMPLE_WINDOWS" \
  --seed "$SEED" \
  --sparsities $SPARSITIES \
  --pruners $PRUNERS \
  --window-index-file "results/phase1_v2/ppl_windows_wikitext2_seed${SEED}.json" \
  --output results/phase1_v2/ppl_v2.csv \
  --local-files-only
