#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:?Set MODE to threshold, representation, oracle, or merge}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_ood_residual_diag}"

ARGS=(
  --mode "$MODE"
  --model "${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
  --output-dir "$OUTPUT_DIR"
  --artifact-dir "${ARTIFACT_DIR:-artifacts/vpref_projection}"
  --margin-dir "${MARGIN_DIR:-results/phase15_margin_calib}"
  --layers "${LAYERS:-24,28,32}"
  --seed "${SEED:-0}"
  --max-length "${MAX_LENGTH:-512}"
  --calib-max-length "${CALIB_MAX_LENGTH:-256}"
  --max-new-tokens "${MAX_NEW_TOKENS:-128}"
  --response-ppl-threshold "${RESPONSE_PPL_THRESHOLD:-100}"
  --judge-max-new-tokens "${JUDGE_MAX_NEW_TOKENS:-16}"
  --strict-eval-offset "${STRICT_EVAL_OFFSET:-0}"
  --strict-eval-limit "${STRICT_EVAL_LIMIT:-128}"
  --residual-eval-offset "${RESIDUAL_EVAL_OFFSET:-0}"
  --residual-eval-limit "${RESIDUAL_EVAL_LIMIT:-256}"
  --fit-limit "${FIT_LIMIT:-128}"
  --advbench-fit-offset "${ADVBENCH_FIT_OFFSET:-0}"
  --benign-fit-limit "${BENIGN_FIT_LIMIT:-128}"
  --benign-fit-offset "${BENIGN_FIT_OFFSET:-0}"
  --benign-eval-limit "${BENIGN_EVAL_LIMIT:-128}"
  --benign-eval-offset "${BENIGN_EVAL_OFFSET:-128}"
  --target-margin "${TARGET_MARGIN:-20}"
  --lambda-benign "${LAMBDA_BENIGN:-20}"
  --ridge-mu "${RIDGE_MU:-0.01}"
  --delta-max "${DELTA_MAX:-50}"
  --oracle-betas "${ORACLE_BETAS:-0.25,0.5}"
  --oracle-min-coherence "${ORACLE_MIN_COHERENCE:-0.95}"
  --r2-folds "${R2_FOLDS:-5}"
  --r2-auc-threshold "${R2_AUC_THRESHOLD:-0.70}"
  --random-draws "${RANDOM_DRAWS:-200}"
  --min-comply "${MIN_COMPLY:-30}"
  --harmbench-config "${HARMBENCH_CONFIG:-standard}"
)

if [[ -n "${BENIGN_FILE:-}" ]]; then ARGS+=(--benign-file "$BENIGN_FILE"); fi
if [[ -n "${EVAL_DATASET:-}" ]]; then ARGS+=(--eval-dataset "$EVAL_DATASET"); fi
if [[ "${LOCAL_FILES_ONLY:-1}" == "1" ]]; then ARGS+=(--local-files-only); fi

python -m casafety.ood_residual_diag "${ARGS[@]}"
