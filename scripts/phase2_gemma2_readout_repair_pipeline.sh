#!/usr/bin/env bash
set -euo pipefail

# Gemma-2-9B-it has a deeper stack than the 32-layer Llama family. Use a coarse
# full-depth sweep and keep repair layers data-selected through the manifest.
export MODEL="${MODEL:-google/gemma-2-9b-it}"
export MODEL_TAG="${MODEL_TAG:-gemma2_9b_it}"
export PROJECTION_LAYERS="${PROJECTION_LAYERS:-6,12,18,24,30,36,41}"
export PROJECTION_NEIGHBOR_RADIUS="${PROJECTION_NEIGHBOR_RADIUS:-3}"
export CONDITION="${CONDITION:-wanda_50}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/phase2_model_zoo}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/phase2_model_zoo}"
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
# Gemma-2-9B plus LlamaGuard can fit on a 96GB card, but 4 simultaneous shards
# may overlap at judge time. Default to 2; raise to 3/4 only after watching nvidia-smi.
export MAX_PARALLEL="${MAX_PARALLEL:-2}"

bash scripts/phase2_llama_readout_repair_pipeline.sh
