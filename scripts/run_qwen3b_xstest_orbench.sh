#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/cap}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/phase2_qwen3b_xstest_orbench}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/phase2_qwen3b_xstest_orbench}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CONDITIONS="${CONDITIONS:-dense,pruned,remar}"

mkdir -p "$DATA_ROOT/xstest" "$DATA_ROOT/or-bench" "$OUTPUT_DIR" "$LOG_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"

exec 9>"$OUTPUT_DIR/.run.lock"
if ! flock -n 9; then
  echo "[xstest-orbench] another evaluator owns $OUTPUT_DIR/.run.lock" >&2
  exit 1
fi

download() {
  local url="$1"
  local destination="$2"
  if [[ -s "$destination" ]]; then
    echo "[xstest-orbench] reuse $destination"
    return
  fi
  local temporary="${destination}.part"
  rm -f "$temporary"
  curl --location --fail --retry 5 --retry-delay 3 \
    "$url" --output "$temporary"
  mv "$temporary" "$destination"
  echo "[xstest-orbench] downloaded $destination"
}

download \
  "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv" \
  "$DATA_ROOT/xstest/xstest_prompts.csv"
download \
  "https://huggingface.co/datasets/bench-llm/or-bench/resolve/main/or-bench-hard-1k.csv" \
  "$DATA_ROOT/or-bench/or-bench-hard-1k.csv"
download \
  "https://huggingface.co/datasets/bench-llm/or-bench/resolve/main/or-bench-toxic.csv" \
  "$DATA_ROOT/or-bench/or-bench-toxic.csv"

for required in \
  artifacts/phase2_oracle_policy_distill/pruned_model/config.json \
  artifacts/phase2_oracle_policy_distill/adv_decode.pt \
  results/phase2_oracle_policy_distill/prepare_manifest.json; do
  if [[ ! -f "$required" ]]; then
    echo "[xstest-orbench] missing required file: $required" >&2
    exit 1
  fi
done

"$PYTHON_BIN" -m casafety.xstest_orbench_eval \
  --conditions "$CONDITIONS" \
  --dense-model "${MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
  --pruned-model-dir artifacts/phase2_oracle_policy_distill/pruned_model \
  --checkpoint-manifest results/phase2_oracle_policy_distill/prepare_manifest.json \
  --repair-artifact artifacts/phase2_oracle_policy_distill/adv_decode.pt \
  --repair-manifest results/phase2_oracle_policy_distill/prepare_manifest.json \
  --expected-repair-layers 24,28,32 \
  --eta 1.0 \
  --xstest-file "$DATA_ROOT/xstest/xstest_prompts.csv" \
  --orbench-hard-file "$DATA_ROOT/or-bench/or-bench-hard-1k.csv" \
  --orbench-toxic-file "$DATA_ROOT/or-bench/or-bench-toxic.csv" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --max-input-tokens "${MAX_INPUT_TOKENS:-512}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
  --verify-checkpoint "${VERIFY_CHECKPOINT:-config}" \
  --local-files-only \
  --resume

echo "[xstest-orbench] result: $OUTPUT_DIR/comparison.csv"
