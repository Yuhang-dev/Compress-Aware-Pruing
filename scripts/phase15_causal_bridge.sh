#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase15_causal_bridge}"
INDEX_SET_PATH="${INDEX_SET_PATH:-$OUTPUT_DIR/index_sets.pt}"
CRIT_OUTPUT_DIR="${CRIT_OUTPUT_DIR:-results/phase1_v2}"
CRIT_CANDIDATE="${CRIT_CANDIDATE:-wei_setdiff__score-grad__ps-0.01__pu-0.05}"
CRIT_SET_PATH="${CRIT_SET_PATH:-}"
TARGET_SUFFIXES="${TARGET_SUFFIXES:-o_proj down_proj}"
SPARSITIES="${SPARSITIES:-0.45,0.50}"
SEED="${SEED:-0}"
PREP_WORKERS="${PREP_WORKERS:-16}"
MAGMATCH_BINS="${MAGMATCH_BINS:-20}"
HARMFUL_FILE="${HARMFUL_FILE:-}"
HARMFUL_DATASET="${HARMFUL_DATASET:-walledai/AdvBench}"
HARMFUL_CONFIG="${HARMFUL_CONFIG:-}"
HARMFUL_SPLIT="${HARMFUL_SPLIT:-train}"
HARMFUL_COLUMN="${HARMFUL_COLUMN:-auto}"
HARMFUL_OFFSET="${HARMFUL_OFFSET:-0}"
HARMFUL_LIMIT_WAS_SET="${HARMFUL_LIMIT+x}"
HARMFUL_LIMIT="${HARMFUL_LIMIT:-128}"
if [[ -n "${EVAL_LIMIT:-}" && -z "$HARMFUL_LIMIT_WAS_SET" ]]; then
  HARMFUL_LIMIT="$EVAL_LIMIT"
fi
CALIB_FILE="${CALIB_FILE:-}"
CALIB_LIMIT="${CALIB_LIMIT:-128}"
CALIB_MAX_LENGTH="${CALIB_MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RESPONSE_PPL_THRESHOLD="${RESPONSE_PPL_THRESHOLD:-100.0}"
JUDGE="${JUDGE:-llamaguard}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-16}"
PPL_DATASET="${PPL_DATASET:-Salesforce/wikitext}"
PPL_DATASET_CONFIG="${PPL_DATASET_CONFIG:-wikitext-2-raw-v1}"
PPL_SPLIT="${PPL_SPLIT:-test}"
PPL_CONTEXT_LEN="${PPL_CONTEXT_LEN:-1024}"
PPL_STRIDE="${PPL_STRIDE:-512}"
PPL_SAMPLE_WINDOWS="${PPL_SAMPLE_WINDOWS:-128}"
PPL_WINDOW_INDEX_FILE="${PPL_WINDOW_INDEX_FILE:-results/phase1_v2/ppl_windows_wikitext2_seed0.json}"
SKIP_PPL="${SKIP_PPL:-0}"
SAVE_RAW_TEXT="${SAVE_RAW_TEXT:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
DETERMINISTIC="${DETERMINISTIC:-0}"
MODE="${MODE:-eval}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
CELLS="${CELLS:-}"

COMMON_ARGS=(
  --config configs/base.yaml
  --model "$MODEL"
  --output-dir "$OUTPUT_DIR"
  --index-set-path "$INDEX_SET_PATH"
  --crit-output-dir "$CRIT_OUTPUT_DIR"
  --crit-candidate "$CRIT_CANDIDATE"
  --target-suffixes $TARGET_SUFFIXES
  --sparsities "$SPARSITIES"
  --seed "$SEED"
  --prep-workers "$PREP_WORKERS"
  --magmatch-bins "$MAGMATCH_BINS"
  --harmful-offset "$HARMFUL_OFFSET"
  --harmful-limit "$HARMFUL_LIMIT"
  --calib-limit "$CALIB_LIMIT"
  --calib-max-length "$CALIB_MAX_LENGTH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --response-ppl-threshold "$RESPONSE_PPL_THRESHOLD"
  --judge "$JUDGE"
  --judge-max-new-tokens "$JUDGE_MAX_NEW_TOKENS"
  --ppl-dataset "$PPL_DATASET"
  --ppl-dataset-config "$PPL_DATASET_CONFIG"
  --ppl-split "$PPL_SPLIT"
  --ppl-context-len "$PPL_CONTEXT_LEN"
  --ppl-stride "$PPL_STRIDE"
  --ppl-sample-windows "$PPL_SAMPLE_WINDOWS"
  --ppl-window-index-file "$PPL_WINDOW_INDEX_FILE"
)

if [[ -n "$CRIT_SET_PATH" ]]; then
  COMMON_ARGS+=(--crit-set-path "$CRIT_SET_PATH")
fi
if [[ -n "$HARMFUL_FILE" ]]; then
  COMMON_ARGS+=(--harmful-file "$HARMFUL_FILE")
else
  COMMON_ARGS+=(--harmful-dataset "$HARMFUL_DATASET" --harmful-split "$HARMFUL_SPLIT" --harmful-column "$HARMFUL_COLUMN")
  if [[ -n "$HARMFUL_CONFIG" ]]; then
    COMMON_ARGS+=(--harmful-config "$HARMFUL_CONFIG")
  fi
fi
if [[ -n "$CALIB_FILE" ]]; then
  COMMON_ARGS+=(--calib-file "$CALIB_FILE")
fi
if [[ -n "$JUDGE_MODEL" ]]; then
  COMMON_ARGS+=(--judge-model "$JUDGE_MODEL")
fi
if [[ "$SKIP_PPL" == "1" ]]; then
  COMMON_ARGS+=(--skip-ppl)
fi
if [[ "$SAVE_RAW_TEXT" == "1" ]]; then
  COMMON_ARGS+=(--save-raw-text)
fi
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  COMMON_ARGS+=(--local-files-only)
else
  COMMON_ARGS+=(--no-local-files-only)
fi
if [[ "$DETERMINISTIC" == "1" ]]; then
  COMMON_ARGS+=(--deterministic)
else
  COMMON_ARGS+=(--no-deterministic)
fi

if [[ "$MODE" == "prepare" ]]; then
  python -m casafety.causal_bridge --mode prepare "${COMMON_ARGS[@]}"
elif [[ "$MODE" == "eval" ]]; then
  EVAL_ARGS=(--shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS")
  if [[ -n "$CELLS" ]]; then
    EVAL_ARGS+=(--cells $CELLS)
  fi
  python -m casafety.causal_bridge --mode eval "${COMMON_ARGS[@]}" "${EVAL_ARGS[@]}"
else
  echo "Unsupported MODE=$MODE. Use prepare or eval." >&2
  exit 1
fi
