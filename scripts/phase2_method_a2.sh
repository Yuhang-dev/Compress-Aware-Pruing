#!/usr/bin/env bash
set -euo pipefail

python -m casafety.train --config configs/method_a2.yaml
python -m casafety.compress_and_eval --config configs/method_a2.yaml --output results/phase2_a2.csv
