#!/usr/bin/env bash
set -euo pipefail

SOURCE_PRUNED_MODEL_DIR="${SOURCE_PRUNED_MODEL_DIR-artifacts/phase2_oracle_policy_distill/pruned_model}"
SOURCE_PRUNED_MANIFEST="${SOURCE_PRUNED_MANIFEST-results/phase2_oracle_policy_distill/prepare_manifest.json}"

args=(
  --mode "${MODE:?Set MODE}"
  --condition "${CONDITION:-wanda_50}"
  --output-dir "${OUTPUT_DIR:-results/phase2_oracle_policy_distill_min_margin_sequential}"
  --artifact-dir "${ARTIFACT_DIR:-artifacts/phase2_oracle_policy_distill_min_margin_sequential}"
  --pruned-model-dir "${PRUNED_MODEL_DIR:-$SOURCE_PRUNED_MODEL_DIR}"
  --shard-dir "${SHARD_DIR:-results/phase2_oracle_policy_distill_min_margin_sequential/shards}"
  --canonical-artifact "${CANONICAL_ARTIFACT:-artifacts/phase2_oracle_hs_coverage_v2/canonical_remar.pt}"
  --prepare-variants "${PREPARE_VARIANTS:-multisource_prefill,adv_decode,multisource_decode,multisource_decode_aggressive,multisource_decode_r2}"
  --eval-dataset "${EVAL_DATASET:-advbench}"
  --calibration-positions "${CALIBRATION_POSITIONS:-0,1,4,8,16,32}"
  --adv-calibration-offset "${ADV_CALIBRATION_OFFSET:-0}"
  --adv-calibration-limit "${ADV_CALIBRATION_LIMIT:-64}"
  --ood-calibration-offset "${OOD_CALIBRATION_OFFSET:-0}"
  --ood-calibration-limit "${OOD_CALIBRATION_LIMIT:-32}"
  --benign-calibration-offset "${BENIGN_CALIBRATION_OFFSET:-0}"
  --benign-calibration-limit "${BENIGN_CALIBRATION_LIMIT:-64}"
  --adv-eval-offset "${ADV_EVAL_OFFSET:-128}"
  --ood-eval-offset "${OOD_EVAL_OFFSET:-64}"
  --eval-limit "${EVAL_LIMIT:-128}"
  --benign-eval-offset "${BENIGN_EVAL_OFFSET:-128}"
  --benign-eval-limit "${BENIGN_EVAL_LIMIT:-128}"
  --target-mode "${TARGET_MODE:-min_margin}"
  --solve-mode "${SOLVE_MODE:-sequential}"
  --dense-margin-condition "${DENSE_MARGIN_CONDITION:-dense}"
  --m-star-quantile "${M_STAR_QUANTILE:-0}"
  --m-star-tolerance "${M_STAR_TOLERANCE:-0.001}"
  --epsilon "${EPSILON:-0.5}"
  --lambda-benign "${LAMBDA_BENIGN:-20}"
  --aggressive-lambda-benign "${AGGRESSIVE_LAMBDA_BENIGN:-5}"
  --ridge-mu "${RIDGE_MU:-0.01}"
  --delta-max "${DELTA_MAX:-50}"
  --max-length "${MAX_LENGTH:-512}"
  --calib-max-length "${CALIB_MAX_LENGTH:-256}"
  --max-new-tokens "${MAX_NEW_TOKENS:-128}"
  --response-ppl-threshold "${RESPONSE_PPL_THRESHOLD:-100}"
  --judge-max-new-tokens "${JUDGE_MAX_NEW_TOKENS:-16}"
  --ppl-dataset "${PPL_DATASET:-Salesforce/wikitext}"
  --ppl-dataset-config "${PPL_DATASET_CONFIG:-wikitext-2-raw-v1}"
  --ppl-split "${PPL_SPLIT:-test}"
  --context-len "${CONTEXT_LEN:-1024}"
  --stride "${STRIDE:-512}"
  --sample-windows "${SAMPLE_WINDOWS:-128}"
  --seed "${SEED:-0}"
  --window-index-file "${WINDOW_INDEX_FILE:-results/phase1_v2/ppl_windows_wikitext2_seed0.json}"
  --harmbench-config "${HARMBENCH_CONFIG:-standard}"
)
if [[ -n "${DENSE_MARGIN_POINTS:-results/phase15_margin_calib/margin_points.csv}" ]]; then
  args+=(--dense-margin-points "${DENSE_MARGIN_POINTS:-results/phase15_margin_calib/margin_points.csv}")
fi
if [[ -n "${EXPECTED_M_STARS:-24:1.13720703125,28:4.049341201782227,32:7.153665542602539}" ]]; then
  args+=(--expected-m-stars "${EXPECTED_M_STARS:-24:1.13720703125,28:4.049341201782227,32:7.153665542602539}")
fi
if [[ -n "${SOURCE_PRUNED_MODEL_DIR:-}" ]]; then
  args+=(--source-pruned-model-dir "$SOURCE_PRUNED_MODEL_DIR")
fi
if [[ -n "${SOURCE_PRUNED_MANIFEST:-}" ]]; then
  args+=(--source-pruned-manifest "$SOURCE_PRUNED_MANIFEST")
fi
if [[ -n "${BENIGN_FILE:-}" ]]; then args+=(--benign-file "$BENIGN_FILE"); fi
if [[ -n "${JUDGE_MODEL:-}" ]]; then args+=(--judge-model "$JUDGE_MODEL"); fi
if [[ "${LOCAL_FILES_ONLY:-1}" == "1" ]]; then args+=(--local-files-only); fi
python -m casafety.oracle_policy_distill "${args[@]}"
