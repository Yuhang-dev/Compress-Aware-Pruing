#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${CAP_ROOT:-/root/autodl-tmp/cap}"
PYTHON="${PYTHON:-/root/miniconda3/envs/pbp/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/phase4_safety_sft_qwen3b_wanda50_saferpaca500}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/wanda50_model_zoo/qwen2.5-3b-instruct/pruned_model}"
CHECKPOINT_MANIFEST="${CHECKPOINT_MANIFEST:-results/wanda50_model_zoo/qwen2.5-3b-instruct/manifest.json}"
TRAINING_FILE="${TRAINING_FILE:-data/saferpaca_Instructions_500.json}"
SAFETY_REFERENCE_FILE="${SAFETY_REFERENCE_FILE:-data/safety_only_data_Instructions.json}"
BENIGN_FILE="${BENIGN_FILE:-data/alpaca_cleaned_train.jsonl}"
PPL_WINDOWS="${PPL_WINDOWS:-results/phase1_v2/ppl_windows_wikitext2_seed0.json}"
ALLOW_DATA_DOWNLOAD="${ALLOW_DATA_DOWNLOAD:-0}"

cd "$ROOT"
mkdir -p logs "$(dirname "$CHECKPOINT_DIR")" "$(dirname "$CHECKPOINT_MANIFEST")"

export CAP_ROOT="$ROOT"
export CASAFETY_ROOT="$ROOT"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/torch_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip_cache}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$ALLOW_DATA_DOWNLOAD" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
else
  unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE || true
fi

"$PYTHON" -m pip install -e . --no-deps

missing=0
for path in "$TRAINING_FILE" "$SAFETY_REFERENCE_FILE" "$BENIGN_FILE" "$PPL_WINDOWS"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    missing=1
  fi
done
if [[ "$missing" == "1" ]]; then
  echo "Download the two pinned Safety-Tuned files before starting the GPU run." >&2
  exit 2
fi

if [[ -d "$CHECKPOINT_DIR" && -f "$CHECKPOINT_MANIFEST" ]]; then
  echo "[safety-sft] reuse Wanda-50 checkpoint: $CHECKPOINT_DIR"
elif [[ ! -e "$CHECKPOINT_DIR" && ! -e "$CHECKPOINT_MANIFEST" ]]; then
  "$PYTHON" -m casafety.pruned_checkpoint \
    --model Qwen/Qwen2.5-3B-Instruct \
    --pruner wanda \
    --sparsity 0.50 \
    --condition wanda_50 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --manifest "$CHECKPOINT_MANIFEST" \
    --calib-max-length 256 \
    --local-files-only
else
  echo "Checkpoint/manifest is partial; refusing to guess or overwrite." >&2
  exit 3
fi

METHOD_ROOT="$OUTPUT_ROOT/sft"
ARTIFACT_MANIFEST="$METHOD_ROOT/artifact_manifest.json"
EVALUATION_DIR="$METHOD_ROOT/evaluation"
LATENCY_MANIFEST="$METHOD_ROOT/latency_manifest.json"
TABLE_DIR="$METHOD_ROOT/table4"
mkdir -p "$METHOD_ROOT"

download_flag=()
if [[ "$ALLOW_DATA_DOWNLOAD" == "1" ]]; then
  download_flag+=(--allow-data-download)
fi

if [[ -f "$ARTIFACT_MANIFEST" ]]; then
  "$PYTHON" - "$ARTIFACT_MANIFEST" "$CHECKPOINT_DIR" "$CHECKPOINT_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
from casafety.baselines.common import directory_hashes, verify_wanda_checkpoint
artifact_path, checkpoint_dir, checkpoint_manifest = map(Path, sys.argv[1:])
value = json.load(open(artifact_path, encoding="utf-8"))
if value.get("status") != "completed" or value.get("method") != "sft":
    raise SystemExit("Existing artifact manifest is not a completed SFT run")
wanda = verify_wanda_checkpoint(checkpoint_dir, checkpoint_manifest, verify_all_files=True)
base = value.get("base_checkpoint") or {}
if Path(base.get("path", "")).resolve() != checkpoint_dir.resolve():
    raise SystemExit("Existing SFT artifact uses another Wanda checkpoint")
if base.get("manifest_sha256") != wanda["manifest_sha256"]:
    raise SystemExit("Existing SFT artifact uses another Wanda manifest")
adapter = Path(value["adapter_dir"])
if directory_hashes(adapter) != value.get("adapter_files_sha256"):
    raise SystemExit("Existing SFT adapter hash mismatch")
PY
  echo "[safety-sft] skip completed fit"
else
  "$PYTHON" -m casafety.baselines.safety_sft \
    --model-id Qwen/Qwen2.5-3B-Instruct \
    --wanda-checkpoint-dir "$CHECKPOINT_DIR" \
    --wanda-manifest "$CHECKPOINT_MANIFEST" \
    --output-dir "$METHOD_ROOT" \
    --artifact-manifest "$ARTIFACT_MANIFEST" \
    --training-file "$TRAINING_FILE" \
    --safety-reference-file "$SAFETY_REFERENCE_FILE" \
    --benign-file "$BENIGN_FILE" \
    --seed 42 \
    "${download_flag[@]}"
fi

if [[ -f "$LATENCY_MANIFEST" ]]; then
  echo "[safety-sft] skip completed latency benchmark"
else
  "$PYTHON" -m casafety.baselines.latency \
    --model-id Qwen/Qwen2.5-3B-Instruct \
    --wanda-checkpoint-dir "$CHECKPOINT_DIR" \
    --wanda-manifest "$CHECKPOINT_MANIFEST" \
    --artifact-manifest "$ARTIFACT_MANIFEST" \
    --output "$LATENCY_MANIFEST" \
    --sequence-length 256 \
    --iterations 40 \
    --seed 0
fi

eval_targets=(
  "$EVALUATION_DIR/evaluation_manifest.json"
  "$EVALUATION_DIR/main_summary.csv"
  "$EVALUATION_DIR/safety_summary.csv"
  "$EVALUATION_DIR/benign_summary.csv"
  "$EVALUATION_DIR/ppl_summary.csv"
)
eval_complete=1
eval_present=0
for path in "${eval_targets[@]}"; do
  [[ -f "$path" ]] || eval_complete=0
  [[ -e "$path" ]] && eval_present=1
done
if [[ "$eval_complete" == "1" ]]; then
  echo "[safety-sft] skip completed evaluation"
elif [[ "$eval_present" == "1" ]]; then
  echo "Evaluation output is partial; move it aside before retrying." >&2
  exit 4
else
  "$PYTHON" -m casafety.baselines.evaluate \
    --method sft \
    --model-id Qwen/Qwen2.5-3B-Instruct \
    --wanda-checkpoint-dir "$CHECKPOINT_DIR" \
    --wanda-manifest "$CHECKPOINT_MANIFEST" \
    --artifact-manifest "$ARTIFACT_MANIFEST" \
    --output-dir "$EVALUATION_DIR" \
    --config configs/base.yaml \
    --judge-model meta-llama/Llama-Guard-3-8B \
    --benign-file "$BENIGN_FILE" \
    --window-index-file "$PPL_WINDOWS" \
    --max-new-tokens 128 \
    --response-ppl-threshold 100 \
    --judge-max-new-tokens 16 \
    --seed 0 \
    "${download_flag[@]}"
fi

table_record="$TABLE_DIR/table4_safety_sft_record.json"
table_csv="$TABLE_DIR/table4_safety_sft_record.csv"
table_tex="$TABLE_DIR/table4_safety_sft_row.tex"
if [[ -f "$table_record" && -f "$table_csv" && -f "$table_tex" ]]; then
  "$PYTHON" - "$table_record" "$ARTIFACT_MANIFEST" \
    "$EVALUATION_DIR/evaluation_manifest.json" \
    "$EVALUATION_DIR/main_summary.csv" "$LATENCY_MANIFEST" \
    configs/base.yaml "$PPL_WINDOWS" <<'PY'
import hashlib, json, sys
from pathlib import Path
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
record = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "artifact_manifest_sha256": sha(sys.argv[2]),
    "evaluation_manifest_sha256": sha(sys.argv[3]),
    "evaluation_summary_sha256": sha(sys.argv[4]),
    "latency_manifest_sha256": sha(sys.argv[5]),
    "config_sha256": sha(sys.argv[6]),
    "ppl_window_index_sha256": sha(sys.argv[7]),
}
for key, value in expected.items():
    if record.get(key) != value:
        raise SystemExit(f"Existing Table 4 record is stale: {key}")
PY
  echo "[safety-sft] verified existing Table 4 record: $TABLE_DIR"
elif [[ -e "$table_record" || -e "$table_csv" || -e "$table_tex" ]]; then
  echo "Table 4 output is partial; move it aside before retrying." >&2
  exit 5
else
  "$PYTHON" -m casafety.baselines.table4 \
    --artifact-manifest "$ARTIFACT_MANIFEST" \
    --evaluation-manifest "$EVALUATION_DIR/evaluation_manifest.json" \
    --evaluation-summary "$EVALUATION_DIR/main_summary.csv" \
    --latency-manifest "$LATENCY_MANIFEST" \
    --config configs/base.yaml \
    --window-index-file "$PPL_WINDOWS" \
    --output-dir "$TABLE_DIR"
fi

echo "[safety-sft] complete"
echo "[safety-sft] artifact=$ARTIFACT_MANIFEST"
echo "[safety-sft] evaluation=$EVALUATION_DIR/main_summary.csv"
echo "[safety-sft] table4=$TABLE_DIR/table4_safety_sft_row.tex"
