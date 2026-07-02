#!/usr/bin/env bash
set -euo pipefail

# Llama-3.1-8B-Instruct has 32 decoder layers. Sweep late and mid layers, then
# let phase15_vpref_projection choose the repair layer neighborhood from data.
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export MODEL_TAG="${MODEL_TAG:-llama31_8b_instruct}"
export PROJECTION_LAYERS="${PROJECTION_LAYERS:-4,8,12,16,20,24,28,31}"
export PROJECTION_NEIGHBOR_RADIUS="${PROJECTION_NEIGHBOR_RADIUS:-3}"
export CONDITION="${CONDITION:-wanda_50}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/phase2_model_zoo}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/phase2_model_zoo}"
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
export MAX_PARALLEL="${MAX_PARALLEL:-4}"

bash scripts/phase2_llama_readout_repair_pipeline.sh
