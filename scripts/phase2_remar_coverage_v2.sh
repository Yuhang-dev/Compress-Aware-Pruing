#!/usr/bin/env bash
set -euo pipefail

args=(
  --mode "${MODE:-merge}"
  --model "${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
  --output-dir "${OUTPUT_DIR:-results/phase2_oracle_hs_coverage_v2}"
  --shard-dir "${SHARD_DIR:-results/phase2_oracle_hs_coverage_v2/coverage_v2_shards}"
  --canonical-artifact "${CANONICAL_ARTIFACT:-artifacts/phase2_oracle_hs_coverage_v2/canonical_remar.pt}"
  --eval-dataset "${EVAL_DATASET:-advbench}"
  --eval-limit "${EVAL_LIMIT:-128}"
  --advbench-eval-offset "${ADVBENCH_EVAL_OFFSET:-128}"
  --ood-eval-offset "${OOD_EVAL_OFFSET:-0}"
  --decode-k "${DECODE_K:-32}"
  --epsilon "${EPSILON:-0.5}"
  --max-length "${MAX_LENGTH:-512}"
  --calib-max-length "${CALIB_MAX_LENGTH:-256}"
  --max-new-tokens "${MAX_NEW_TOKENS:-128}"
  --response-ppl-threshold "${RESPONSE_PPL_THRESHOLD:-100}"
  --judge-max-new-tokens "${JUDGE_MAX_NEW_TOKENS:-16}"
  --permutations "${PERMUTATIONS:-5000}"
  --prefill-margin-fraction "${PREFILL_MARGIN_FRACTION:-0.95}"
  --temporal-gap-min "${TEMPORAL_GAP_MIN:-1.0}"
  --temporal-p-max "${TEMPORAL_P_MAX:-0.05}"
  --min-outcome-n "${MIN_OUTCOME_N:-20}"
  --seed "${SEED:-0}"
  --harmbench-config "${HARMBENCH_CONFIG:-standard}"
)
if [[ -n "${JUDGE_MODEL:-}" ]]; then args+=(--judge-model "$JUDGE_MODEL"); fi
if [[ -n "${BENIGN_FILE:-}" ]]; then args+=(--benign-file "$BENIGN_FILE"); fi
if [[ "${LOCAL_FILES_ONLY:-1}" == "1" ]]; then args+=(--local-files-only); fi

python -m casafety.remar_coverage_v2 "${args[@]}"
