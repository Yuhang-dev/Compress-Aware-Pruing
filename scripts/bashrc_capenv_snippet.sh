# Add this to ~/.bashrc on the AutoDL machine.
# Do not store API keys in this file or in the repository.

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export CAP_ROOT="$DATA_DISK/cap"

capenv() {
  cd "$CAP_ROOT"
  source /etc/network_turbo
  conda activate pbp

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
  export PYTHONPATH="$CAP_ROOT/src:${PYTHONPATH:-}"
}
