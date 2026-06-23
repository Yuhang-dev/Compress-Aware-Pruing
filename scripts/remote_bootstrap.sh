#!/usr/bin/env bash
set -euo pipefail

# Reuse the known-good AutoDL environment from the remote environment notes.
# This script deliberately does not reference or depend on the deprecated FFAP repo.

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export CAP_ROOT="${CAP_ROOT:-$DATA_DISK/cap}"
export CASAFETY_ROOT="${CASAFETY_ROOT:-$CAP_ROOT}"

export HF_HOME="$DATA_DISK/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$DATA_DISK/torch_cache"
export HF_XET_CACHE="$HF_HOME/xet"
export PIP_CACHE_DIR="$DATA_DISK/pip_cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
unset HF_XET_HIGH_PERFORMANCE

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate pbp
else
  echo "Missing conda bootstrap at /root/miniconda3/etc/profile.d/conda.sh" >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"

if [ -d "$CASAFETY_ROOT" ]; then
  cd "$CASAFETY_ROOT"
fi

export PYTHONPATH="$CASAFETY_ROOT/src:${PYTHONPATH:-}"
python -m pip install -e . --no-build-isolation --no-deps

echo "CAP_ROOT=$CAP_ROOT"
echo "CASAFETY_ROOT=$CASAFETY_ROOT"
echo "HF_HOME=$HF_HOME"
echo "TORCH_HOME=$TORCH_HOME"
echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
which python
