#!/usr/bin/env bash
set -euo pipefail

python -m casafety.vpref --config configs/base.yaml --output results/vpref_validation.csv
python -m casafety.mechanism_diagnosis --config configs/base.yaml --output results/mechanism_diagnosis.csv
