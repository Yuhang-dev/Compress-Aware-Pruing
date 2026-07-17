#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/cap}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/fig10_paired_margins_real}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-20000}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-0}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

exec 9>"$OUTPUT_DIR/.run.lock"
if ! flock -n 9; then
  echo "[fig10] another collector owns $OUTPUT_DIR/.run.lock" >&2
  exit 1
fi

for required in \
  artifacts/phase2_oracle_policy_distill/pruned_model/config.json \
  artifacts/phase2_oracle_policy_distill/adv_decode.pt \
  results/phase2_oracle_policy_distill/prepare_manifest.json; do
  if [[ ! -f "$required" ]]; then
    echo "[fig10] missing required file: $required" >&2
    exit 1
  fi
done

wait_for_gpu_memory() {
  local free_mb
  while true; do
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= GPU_MIN_FREE_MB )); then
      echo "[fig10] GPU preflight passed: ${free_mb} MiB free"
      return
    fi
    if [[ "$WAIT_FOR_GPU" != "1" ]]; then
      echo "[fig10] only ${free_mb:-unknown} MiB free; require ${GPU_MIN_FREE_MB} MiB" >&2
      exit 75
    fi
    echo "[fig10] waiting for GPU: ${free_mb:-unknown}/${GPU_MIN_FREE_MB} MiB free"
    sleep "$GPU_POLL_SECONDS"
  done
}

wait_for_gpu_memory

local_files_flag="--local-files-only"
if [[ "${LOCAL_FILES_ONLY:-1}" == "0" ]]; then
  local_files_flag="--no-local-files-only"
fi

"$PYTHON_BIN" -m casafety.fig10_paired_margins \
  --dense-model "${MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
  --pruned-model-dir artifacts/phase2_oracle_policy_distill/pruned_model \
  --checkpoint-manifest results/phase2_oracle_policy_distill/prepare_manifest.json \
  --repair-artifact artifacts/phase2_oracle_policy_distill/adv_decode.pt \
  --repair-manifest results/phase2_oracle_policy_distill/prepare_manifest.json \
  --dataset "${DATASET:-walledai/AdvBench}" \
  --dataset-split "${DATASET_SPLIT:-train}" \
  --dataset-column "${DATASET_COLUMN:-auto}" \
  --offset "${OFFSET:-128}" \
  --limit "${LIMIT:-128}" \
  --layers "${LAYERS:-24,28,32}" \
  --eta "${ETA:-1.0}" \
  --max-length "${MAX_LENGTH:-1024}" \
  --dtype "${DTYPE:-bfloat16}" \
  --verify-checkpoint "${VERIFY_CHECKPOINT:-config}" \
  --output-dir "$OUTPUT_DIR" \
  "$local_files_flag" \
  --resume

echo "[fig10] paired data: $OUTPUT_DIR/fig10_paired_margins.csv"
echo "[fig10] quadrant counts: $OUTPUT_DIR/fig10_quadrant_counts.csv"
