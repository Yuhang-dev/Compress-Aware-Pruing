#!/usr/bin/env bash
set -euo pipefail

args=(
  --model "${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
  --layers "${LAYERS:-24,28,32}"
  --direction-artifact-dir "${DIRECTION_ARTIFACT_DIR:-artifacts/vpref_projection}"
  --margin-dir "${MARGIN_DIR:-results/phase15_margin_calib}"
  --canonical-manifest "${CANONICAL_MANIFEST:-results/phase2_readout_repair_v2_w50_parallel/g_solve_manifest.json}"
  --output-artifact "${CANONICAL_ARTIFACT:-artifacts/phase2_oracle_hs_coverage_v2/canonical_remar.pt}"
  --output-dir "${OUTPUT_DIR:-results/phase2_oracle_hs_coverage_v2}"
  --target-margin "${TARGET_MARGIN:-20}"
  --lambda-benign "${LAMBDA_BENIGN:-20}"
  --fit-limit "${FIT_LIMIT:-128}"
  --benign-fit-limit "${BENIGN_FIT_LIMIT:-128}"
  --calib-max-length "${CALIB_MAX_LENGTH:-256}"
  --max-length "${MAX_LENGTH:-512}"
  --match-atol "${MATCH_ATOL:-1e-5}"
  --match-rtol "${MATCH_RTOL:-1e-6}"
)
if [[ -n "${BENIGN_FILE:-}" ]]; then args+=(--benign-file "$BENIGN_FILE"); fi
if [[ "${LOCAL_FILES_ONLY:-1}" == "1" ]]; then args+=(--local-files-only); fi

python -m casafety.canonical_remar "${args[@]}"
