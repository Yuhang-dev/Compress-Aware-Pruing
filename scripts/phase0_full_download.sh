#!/usr/bin/env bash
set -euo pipefail

# Run this from /root/autodl-tmp/cap after entering the pbp environment.
# It installs lightweight dependencies with the preserved pip cache and
# downloads model assets into the preserved Hugging Face cache.
#
# Do not install vLLM / GPTQ / AWQ here by default. Those packages can pull
# large PyTorch CUDA component wheels such as nvidia-nccl-cu13 and may disturb
# the known-good torch 2.12.0+cu130 environment. Install them later only when
# the exact compatible wheel set is chosen.

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export CAP_ROOT="${CAP_ROOT:-$DATA_DISK/cap}"
export HF_HOME="${HF_HOME:-$DATA_DISK/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$DATA_DISK/torch_cache}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_DISK/pip_cache}"
export TOKENIZERS_PARALLELISM=false
# Disable Xet and use regular HTTP downloads. Xet can hang on this AutoDL setup.
export HF_HUB_DISABLE_XET=1
unset HF_XET_HIGH_PERFORMANCE
export PYTHONPATH="$CAP_ROOT/src:${PYTHONPATH:-}"

cd "$CAP_ROOT"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

python -m pip install \
  --index-url https://pypi.org/simple \
  --cache-dir "$PIP_CACHE_DIR" \
  transformers accelerate datasets peft pandas matplotlib pyyaml huggingface_hub

if [ "${INSTALL_HEAVY_EVAL_DEPS:-0}" = "1" ]; then
  python -m pip install \
    --index-url https://pypi.org/simple \
    --cache-dir "$PIP_CACHE_DIR" \
    vllm bitsandbytes auto-gptq autoawq
fi

python -m pip install \
  --index-url https://pypi.org/simple \
  --cache-dir "$PIP_CACHE_DIR" \
  lm-eval

python scripts/download_phase0_assets.py \
  --config configs/base.yaml \
  --include-generalization-model

python -m casafety.env_report --output env_report.txt

echo "Phase 0 lightweight package/model download completed."
echo "Dataset downloads are skipped until configs/base.yaml contains verified dataset paths."
