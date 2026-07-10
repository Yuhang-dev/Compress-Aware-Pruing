#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:?Set MODE}"
ARGS=(
  --mode "$MODE"
  --model "${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
  --output-dir "${OUTPUT_DIR:-results/phase2_ood_residual_followup}"
  --artifact-dir "${ARTIFACT_DIR:-artifacts/vpref_projection}"
  --margin-dir "${MARGIN_DIR:-results/phase15_margin_calib}"
  --rank2-artifact "${RANK2_ARTIFACT:-artifacts/phase2_ood_residual_followup/rank2_payload.pt}"
  --diagnosis-decision "${DIAGNOSIS_DECISION:-results/phase2_ood_residual_diag_n200/ood_residual_decision.json}"
  --layers "${LAYERS:-24,28,32}"
  --seed "${SEED:-0}"
  --eval-offset "${EVAL_OFFSET:-0}"
  --eval-limit "${EVAL_LIMIT:-200}"
  --max-length "${MAX_LENGTH:-512}"
  --calib-max-length "${CALIB_MAX_LENGTH:-256}"
  --max-new-tokens "${MAX_NEW_TOKENS:-128}"
  --benign-max-new-tokens "${BENIGN_MAX_NEW_TOKENS:-128}"
  --response-ppl-threshold "${RESPONSE_PPL_THRESHOLD:-100}"
  --judge-max-new-tokens "${JUDGE_MAX_NEW_TOKENS:-16}"
  --oracle-epsilons "${ORACLE_EPSILONS:-0.5,2.0}"
  --oracle-arm-group "${ORACLE_ARM_GROUP:-all}"
  --oracle-min-coherence "${ORACLE_MIN_COHERENCE:-0.95}"
  --max-negative-margin "${MAX_NEGATIVE_MARGIN:-0.02}"
  --fit-limit "${FIT_LIMIT:-128}"
  --benign-fit-limit "${BENIGN_FIT_LIMIT:-128}"
  --benign-eval-limit "${BENIGN_EVAL_LIMIT:-128}"
  --target-margin "${TARGET_MARGIN:-20}"
  --lambda-benign "${LAMBDA_BENIGN:-20}"
  --rank2-train-offset "${RANK2_TRAIN_OFFSET:-0}"
  --rank2-train-limit "${RANK2_TRAIN_LIMIT:-128}"
  --rank2-eval-offset "${RANK2_EVAL_OFFSET:-128}"
  --rank2-eval-limit "${RANK2_EVAL_LIMIT:-72}"
  --rank2-min-comply "${RANK2_MIN_COMPLY:-30}"
  --rank2-target-margin "${RANK2_TARGET_MARGIN:-2.0}"
  --rank2-eta "${RANK2_ETA:-0.0}"
  --harmbench-config "${HARMBENCH_CONFIG:-standard}"
)
if [[ -n "${EVAL_DATASET:-}" ]]; then ARGS+=(--eval-dataset "$EVAL_DATASET"); fi
if [[ -n "${BENIGN_FILE:-}" ]]; then ARGS+=(--benign-file "$BENIGN_FILE"); fi
if [[ "${LOCAL_FILES_ONLY:-1}" == "1" ]]; then ARGS+=(--local-files-only); fi
if [[ "${RANK2_RANDOM_DIRECTION:-0}" == "1" ]]; then ARGS+=(--rank2-random-direction); fi

python -m casafety.ood_residual_followup "${ARGS[@]}"
