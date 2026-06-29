#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
SHARD_ROOT="${SHARD_ROOT:-results/phase15_mechanism_seal_residual_sweep_shards}"
LOG_DIR="${LOG_DIR:-logs}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/vpref_projection}"
MANIFEST="${MANIFEST:-results/phase15_vpref_projection/vpref_manifest.json}"
BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}"
WANDA_CONFIGS="${WANDA_CONFIGS:-wanda_40:0.40 wanda_45:0.45 wanda_50:0.50 wanda_55:0.55}"
EVAL_LIMIT="${EVAL_LIMIT:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-16}"
STEP0B_WINDOWS="${STEP0B_WINDOWS:-32}"
STEP0B_LAYER_GROUPS="${STEP0B_LAYER_GROUPS:-24,28,32}"
STEP0B_MEASURE_LAYERS="${STEP0B_MEASURE_LAYERS:-32,35}"
STEP0B_KR="${STEP0B_KR:-1}"
STEP0B_MODES="${STEP0B_MODES:-norm_relative}"
STEP0B_TARGET_KINDS="${STEP0B_TARGET_KINDS:-zero strong}"
STEP0B_NORM_RELATIVE_STRONG="${STEP0B_NORM_RELATIVE_STRONG:-0.25}"
STEP0B_PROGRESS_EVERY="${STEP0B_PROGRESS_EVERY:-50}"
SEED="${SEED:-0}"
DIRECTION_LIMIT="${DIRECTION_LIMIT:-256}"
HARM_EVAL_OFFSET="${HARM_EVAL_OFFSET:-0}"
BENIGN_EVAL_OFFSET="${BENIGN_EVAL_OFFSET:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
PREPARE_INPUTS="${PREPARE_INPUTS:-1}"
WAIT_AND_SUMMARIZE="${WAIT_AND_SUMMARIZE:-0}"
SEAL_OUTPUT="${SEAL_OUTPUT:-results/phase15_vpref_projection/mechanism_seal_residual_sweep.csv}"

mkdir -p "$SHARD_ROOT" "$LOG_DIR"

COMMON_ENV=(
  MODEL="$MODEL"
  ARTIFACT_DIR="$ARTIFACT_DIR"
  MANIFEST="$MANIFEST"
  BENIGN_FILE="$BENIGN_FILE"
  EVAL_LIMIT="$EVAL_LIMIT"
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
  JUDGE_MAX_NEW_TOKENS="$JUDGE_MAX_NEW_TOKENS"
  STEP0B_WINDOWS="$STEP0B_WINDOWS"
  STEP0B_LAYER_GROUPS="$STEP0B_LAYER_GROUPS"
  STEP0B_MEASURE_LAYERS="$STEP0B_MEASURE_LAYERS"
  STEP0B_KR="$STEP0B_KR"
  STEP0B_MODES="$STEP0B_MODES"
  STEP0B_TARGET_KINDS="$STEP0B_TARGET_KINDS"
  STEP0B_NORM_RELATIVE_STRONG="$STEP0B_NORM_RELATIVE_STRONG"
  STEP0B_PROGRESS_EVERY="$STEP0B_PROGRESS_EVERY"
  SEED="$SEED"
  DIRECTION_LIMIT="$DIRECTION_LIMIT"
  HARM_EVAL_OFFSET="$HARM_EVAL_OFFSET"
  BENIGN_EVAL_OFFSET="$BENIGN_EVAL_OFFSET"
  LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY"
)

LOCAL_ARGS=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_ARGS+=(--local-files-only)
fi

if [[ "$PREPARE_INPUTS" == "1" ]]; then
  echo "[mechanism-seal] preparing shared Step0b direction inputs"
  env "${COMMON_ENV[@]}" \
    OUTPUT_DIR="$SHARD_ROOT/_prepare" \
    python -m casafety.step0_restore_s \
      --step0b \
      --step0b-prepare-only \
      --config configs/base.yaml \
      --model "$MODEL" \
      --output-dir "$SHARD_ROOT/_prepare" \
      --artifact-dir "$ARTIFACT_DIR" \
      --manifest "$MANIFEST" \
      --seed "$SEED" \
      --direction-limit "$DIRECTION_LIMIT" \
      --eval-limit "$EVAL_LIMIT" \
      --harm-eval-offset "$HARM_EVAL_OFFSET" \
      --benign-eval-offset "$BENIGN_EVAL_OFFSET" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --judge-max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
      --step0b-layer-groups "$STEP0B_LAYER_GROUPS" \
      --step0b-measure-layers "$STEP0B_MEASURE_LAYERS" \
      --step0b-kr $STEP0B_KR \
      --step0b-windows $STEP0B_WINDOWS \
      --step0b-modes $STEP0B_MODES \
      --step0b-target-kinds $STEP0B_TARGET_KINDS \
      --step0b-norm-relative-strong "$STEP0B_NORM_RELATIVE_STRONG" \
      --benign-file "$BENIGN_FILE" \
      "${LOCAL_ARGS[@]}"
fi

PIDS=()
DIRS=()

for spec in $WANDA_CONFIGS; do
  tag="${spec%%:*}"
  out_dir="$SHARD_ROOT/$tag"
  log_file="$LOG_DIR/mechanism_seal_residual_${tag}.log"
  mkdir -p "$out_dir"
  echo "[mechanism-seal] launching residual shard $tag config=$spec out=$out_dir log=$log_file"
  (
    env "${COMMON_ENV[@]}" \
      OUTPUT_DIR="$out_dir" \
      STEP0B_CONFIGS="$spec" \
      bash scripts/phase15_step0b_restore_s.sh
  ) > "$log_file" 2>&1 &
  PIDS+=("$!")
  DIRS+=("$out_dir")
done

echo "[mechanism-seal] pids: ${PIDS[*]}"
echo "[mechanism-seal] logs: $LOG_DIR/mechanism_seal_residual_*.log"
echo "[mechanism-seal] summarize after completion:"
echo "  SHARD_ROOT=$SHARD_ROOT SEAL_OUTPUT=$SEAL_OUTPUT bash scripts/phase15_mechanism_seal_residual_merge.sh"

if [[ "$WAIT_AND_SUMMARIZE" == "1" ]]; then
  status=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    echo "[mechanism-seal] at least one residual shard failed" >&2
    exit "$status"
  fi
  SHARD_ROOT="$SHARD_ROOT" WANDA_CONFIGS="$WANDA_CONFIGS" SEAL_OUTPUT="$SEAL_OUTPUT" \
    bash scripts/phase15_mechanism_seal_residual_merge.sh
fi
