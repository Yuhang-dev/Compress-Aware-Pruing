#!/usr/bin/env bash
set -euo pipefail

python -m casafety.train --config configs/method_b.yaml
python -m casafety.compress_and_eval --config configs/method_b.yaml --output results/phase3_b_grid.csv
