#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
MODEL_TAG="${MODEL_TAG:-llama2_7b_chat}"
CONDITION="${CONDITION:-wanda_50}"
CONFIG="${CONFIG:-configs/base.yaml}"
BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/phase2_llama}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/phase2_llama}"
PROJECTION_OUTPUT_DIR="${PROJECTION_OUTPUT_DIR:-$OUTPUT_ROOT/${MODEL_TAG}_vpref_projection}"
PROJECTION_ARTIFACT_DIR="${PROJECTION_ARTIFACT_DIR:-$ARTIFACT_ROOT/${MODEL_TAG}_vpref_projection}"
MARGIN_OUTPUT_DIR="${MARGIN_OUTPUT_DIR:-$OUTPUT_ROOT/${MODEL_TAG}_${CONDITION}_margin_calib}"
REPAIR_OUTPUT_DIR="${REPAIR_OUTPUT_DIR:-$OUTPUT_ROOT/${MODEL_TAG}_${CONDITION}_readout_repair}"
REPAIR_SHARD_ROOT="${REPAIR_SHARD_ROOT:-${REPAIR_OUTPUT_DIR}_shards}"

PROJECTION_LAYERS="${PROJECTION_LAYERS:-4,8,12,16,20,24,28}"
PROJECTION_NEIGHBOR_RADIUS="${PROJECTION_NEIGHBOR_RADIUS:-4}"
KR="${KR:-1}"
DIRECTION_LIMIT="${DIRECTION_LIMIT:-256}"
PROJECTION_EVAL_LIMIT="${PROJECTION_EVAL_LIMIT:-128}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
PROJECTION_MAX_NEW_TOKENS="${PROJECTION_MAX_NEW_TOKENS:-128}"
NULL_DIRECTIONS="${NULL_DIRECTIONS:-20}"
CONTROL_DIRECTIONS="${CONTROL_DIRECTIONS:-2}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
VALIDATION_ALPHAS="${VALIDATION_ALPHAS:-2 4 8}"
VALIDATION_MAX_NEW_TOKENS="${VALIDATION_MAX_NEW_TOKENS:-96}"

RUN_PROJECTION="${RUN_PROJECTION:-1}"
RUN_MARGIN="${RUN_MARGIN:-1}"
RUN_REPAIR="${RUN_REPAIR:-1}"
RESTORE_ORACLE_MIN_COHERENCE="${RESTORE_ORACLE_MIN_COHERENCE:-0.95}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

mkdir -p "$OUTPUT_ROOT" "$ARTIFACT_ROOT" logs

if [[ "$RUN_PROJECTION" == "1" ]]; then
  echo "[llama-pipeline] running vpref projection model=$MODEL condition=$CONDITION"
  MODEL="$MODEL" \
  OUTPUT_DIR="$PROJECTION_OUTPUT_DIR" \
  ARTIFACT_DIR="$PROJECTION_ARTIFACT_DIR" \
  LAYERS="$PROJECTION_LAYERS" \
  KR="$KR" \
  DIRECTION_LIMIT="$DIRECTION_LIMIT" \
  EVAL_LIMIT="$PROJECTION_EVAL_LIMIT" \
  MAX_LENGTH="$MAX_LENGTH" \
  MAX_NEW_TOKENS="$PROJECTION_MAX_NEW_TOKENS" \
  NULL_DIRECTIONS="$NULL_DIRECTIONS" \
  CONTROL_DIRECTIONS="$CONTROL_DIRECTIONS" \
  PROJECTION_NEIGHBOR_RADIUS="$PROJECTION_NEIGHBOR_RADIUS" \
  RUN_VALIDATION="$RUN_VALIDATION" \
  VALIDATION_ALPHAS="$VALIDATION_ALPHAS" \
  VALIDATION_MAX_NEW_TOKENS="$VALIDATION_MAX_NEW_TOKENS" \
  BENIGN_FILE="$BENIGN_FILE" \
  LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY" \
    bash scripts/phase15_vpref_projection.sh
fi

MANIFEST="$PROJECTION_OUTPUT_DIR/vpref_manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "[llama-pipeline] missing manifest: $MANIFEST" >&2
  exit 1
fi

REPAIR_LAYERS="$(
  python - <<PY
import json
with open("$MANIFEST", "r", encoding="utf-8") as f:
    manifest = json.load(f)
layers = manifest.get("projection_layers") or manifest.get("layers_sweep")
if not layers:
    raise SystemExit("manifest has no projection_layers")
print(",".join(str(int(x)) for x in layers))
PY
)"
echo "[llama-pipeline] repair layers from manifest: $REPAIR_LAYERS"

if [[ "$RUN_MARGIN" == "1" ]]; then
  echo "[llama-pipeline] running margin calibration model=$MODEL condition=$CONDITION"
  LOCAL_ARGS=()
  if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    LOCAL_ARGS=(--local-files-only)
  else
    LOCAL_ARGS=(--no-local-files-only)
  fi
  python -m casafety.margin_calibration \
    --mode eval_analyze \
    --config "$CONFIG" \
    --model "$MODEL" \
    --output-dir "$MARGIN_OUTPUT_DIR" \
    --artifact-dir "$PROJECTION_ARTIFACT_DIR" \
    --layers "$REPAIR_LAYERS" \
    --conditions "dense $CONDITION" \
    --harmful-limit 128 \
    --max-length "$MAX_LENGTH" \
    --max-new-tokens 128 \
    --response-ppl-threshold 100 \
    --judge llamaguard \
    "${LOCAL_ARGS[@]}"
fi

if [[ "$RUN_REPAIR" == "1" ]]; then
  echo "[llama-pipeline] running readout repair model=$MODEL condition=$CONDITION"
  MODEL="$MODEL" \
  ARTIFACT_DIR="$PROJECTION_ARTIFACT_DIR" \
  MARGIN_DIR="$MARGIN_OUTPUT_DIR" \
  LAYERS="$REPAIR_LAYERS" \
  BENIGN_FILE="$BENIGN_FILE" \
  RESPONSE_PPL_THRESHOLD=100 \
  CONDITION="$CONDITION" \
  OUTPUT_DIR="$REPAIR_OUTPUT_DIR" \
  SHARD_ROOT="$REPAIR_SHARD_ROOT" \
  RESTORE_ORACLE_MIN_COHERENCE="$RESTORE_ORACLE_MIN_COHERENCE" \
  LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY" \
  MAX_PARALLEL="$MAX_PARALLEL" \
    bash scripts/phase15_closed_form_readout_repair_cell_parallel.sh
fi

if [[ "${SHUTDOWN:-0}" == "1" ]]; then
  /usr/bin/shutdown
fi
