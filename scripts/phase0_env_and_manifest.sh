#!/usr/bin/env bash
set -euo pipefail

python -m casafety.env_report --output env_report.txt
python data/build_refusal_sft.py --config configs/base.yaml --write-manifest data/manifest.json

echo "Phase 0 initial files written:"
echo "  env_report.txt"
echo "  data/manifest.json"
echo
echo "Next: verify dataset HF paths/splits and implement/run compression evaluation before GATE-0."
