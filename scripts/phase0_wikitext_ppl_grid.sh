#!/usr/bin/env bash
set -euo pipefail

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
export HF_HUB_DISABLE_XET=1
unset HF_XET_HIGH_PERFORMANCE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONPATH="$CAP_ROOT/src:${PYTHONPATH:-}"

cd "$CAP_ROOT"

/root/miniconda3/envs/pbp/bin/python -m casafety.phase0_ppl_eval \
  --config configs/base.yaml \
  --model "${MODEL:-}" \
  --output results/phase0_wikitext_ppl_grid.csv \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --seq-len "${SEQ_LEN:-512}" \
  --limit "${LIMIT:-64}" \
  --local-files-only \
  --sparsities 0.5 0.6 0.7 \
  --pruners magnitude wanda
