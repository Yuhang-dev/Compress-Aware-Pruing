#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_ood_residual_diag}"
LOG_DIR="${LOG_DIR:-logs/phase2_ood_residual_diag}"
mkdir -p "$LOG_DIR"

shared=(
  OUTPUT_DIR="$OUTPUT_DIR"
  MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
  ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/vpref_projection}"
  MARGIN_DIR="${MARGIN_DIR:-results/phase15_margin_calib}"
  LAYERS="${LAYERS:-24,28,32}"
  SEED="${SEED:-0}"
  MAX_LENGTH="${MAX_LENGTH:-512}"
  CALIB_MAX_LENGTH="${CALIB_MAX_LENGTH:-256}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
  RESPONSE_PPL_THRESHOLD="${RESPONSE_PPL_THRESHOLD:-100}"
  JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-16}"
  STRICT_EVAL_LIMIT="${STRICT_EVAL_LIMIT:-128}"
  RESIDUAL_EVAL_LIMIT="${RESIDUAL_EVAL_LIMIT:-256}"
  FIT_LIMIT="${FIT_LIMIT:-128}"
  BENIGN_FIT_LIMIT="${BENIGN_FIT_LIMIT:-128}"
  BENIGN_EVAL_LIMIT="${BENIGN_EVAL_LIMIT:-128}"
  TARGET_MARGIN="${TARGET_MARGIN:-20}"
  LAMBDA_BENIGN="${LAMBDA_BENIGN:-20}"
  ORACLE_BETAS="${ORACLE_BETAS:-0.25,0.5}"
  HARMBENCH_CONFIG="${HARMBENCH_CONFIG:-standard}"
)
if [[ -n "${BENIGN_FILE:-}" ]]; then shared+=(BENIGN_FILE="$BENIGN_FILE"); fi
if [[ "${LOCAL_FILES_ONLY:-1}" == "1" ]]; then shared+=(LOCAL_FILES_ONLY=1); fi

launch () {
  local tag="$1"; shift
  echo "[ood-residual-parallel] launching $tag" >&2
  env "${shared[@]}" "$@" > "$LOG_DIR/$tag.log" 2>&1 &
  LAST_PID=$!
}

pids=()
launch threshold MODE=threshold bash scripts/phase2_ood_residual_diag.sh
pids+=("$LAST_PID")
launch representation MODE=representation bash scripts/phase2_ood_residual_diag.sh
pids+=("$LAST_PID")
launch oracle_harmbench MODE=oracle EVAL_DATASET=harmbench bash scripts/phase2_ood_residual_diag.sh
pids+=("$LAST_PID")
launch oracle_strongreject MODE=oracle EVAL_DATASET=strongreject bash scripts/phase2_ood_residual_diag.sh
pids+=("$LAST_PID")
echo "[ood-residual-parallel] pids: ${pids[*]}"

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [[ "$status" != "0" ]]; then
  echo "[ood-residual-parallel] a Phase A job failed; inspect $LOG_DIR" >&2
  exit "$status"
fi
env "${shared[@]}" MODE=merge bash scripts/phase2_ood_residual_diag.sh
if [[ "${SHUTDOWN:-0}" == "1" ]]; then /usr/bin/shutdown; fi
