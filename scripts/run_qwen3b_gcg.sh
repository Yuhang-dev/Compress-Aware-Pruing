#!/usr/bin/env bash
set -euo pipefail

# Shared launcher for the Qwen2.5-3B GCG table.  Pass MODEL to select the
# dense/pruned/repaired checkpoint on the remote host.
ROOT="${ROOT:-results/gcg_qwen3b}"
MODEL="${MODEL:?set MODEL to a HF id or local checkpoint}"
ARM="${ARM:?set ARM=dense|wanda50|remar}"
LIMIT="${LIMIT:-32}"
STEPS="${STEPS:-250}"
SEARCH_WIDTH="${SEARCH_WIDTH:-64}"
TOPK="${TOPK:-64}"
JUDGE="${JUDGE:-llamaguard}"
JUDGE_MODEL="${JUDGE_MODEL:-meta-llama/Llama-Guard-3-8B}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
PYTHON="${PYTHON:-/root/miniconda3/envs/pbp/bin/python}"

common=(
  --arm "$ARM" --model "$MODEL" --output-dir "$ROOT/$MODE/$ARM"
  --limit "$LIMIT" --steps "$STEPS" --search-width "$SEARCH_WIDTH" --topk "$TOPK"
  --judge "$JUDGE" --judge-model "$JUDGE_MODEL" --shuffle --seed "${SEED:-0}"
)
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then common+=(--local-files-only); fi

case "${MODE:?set MODE=optimize|standard|adaptive}" in
  optimize)
    "$PYTHON" scripts/run_qwen3b_gcg.py --mode optimize "${common[@]}"
    ;;
  standard)
    # Standard = suffixes optimized on Dense, then evaluated on each arm.
    suffix_file="${SUFFIX_FILE:?set SUFFIX_FILE to the Dense suffixes.jsonl}"
    "$PYTHON" scripts/run_qwen3b_gcg.py --mode evaluate "${common[@]}" --suffix-file "$suffix_file"
    ;;
  adaptive)
    "$PYTHON" scripts/run_qwen3b_gcg.py --mode adaptive "${common[@]}"
    ;;
  *) echo "MODE must be optimize, standard, or adaptive" >&2; exit 2 ;;
esac
