#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/phase2_lodo_remar}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/phase2_lodo_remar}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
LOG_DIR="${LOG_DIR:-logs/phase2_lodo_remar}"
mkdir -p "$LOG_DIR"

MODE=prepare OUTPUT_DIR="$OUTPUT_DIR" ARTIFACT_DIR="$ARTIFACT_DIR" bash scripts/phase2_lodo_remar.sh \
  > "$LOG_DIR/prepare.log" 2>&1

CELL_COUNT=$(OUTPUT_DIR="$OUTPUT_DIR" python - <<'PY'
import json
import os
with open(os.path.join(os.environ["OUTPUT_DIR"], "lodo_manifest.json"), encoding="utf-8") as f:
    print(len(json.load(f)["cells"]))
PY
)

pids=()
for ((cell=0; cell<CELL_COUNT; cell++)); do
  while (( ${#pids[@]} >= MAX_PARALLEL )); do
    wait -n
    next=()
    for pid in "${pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && next+=("$pid")
    done
    pids=("${next[@]}")
  done
  echo "[lodo-remar] launching cell=$cell/$CELL_COUNT"
  MODE=run CELL_INDEX="$cell" OUTPUT_DIR="$OUTPUT_DIR" ARTIFACT_DIR="$ARTIFACT_DIR" \
    bash scripts/phase2_lodo_remar.sh > "$LOG_DIR/cell_${cell}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

MODE=merge OUTPUT_DIR="$OUTPUT_DIR" ARTIFACT_DIR="$ARTIFACT_DIR" bash scripts/phase2_lodo_remar.sh \
  > "$LOG_DIR/merge.log" 2>&1
