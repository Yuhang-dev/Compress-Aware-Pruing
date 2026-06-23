#!/usr/bin/env bash
set -euo pipefail

: "${PIP_CACHE_DIR:=/root/autodl-tmp/pip_cache}"

python -m pip install \
  --index-url https://pypi.org/simple \
  --cache-dir "$PIP_CACHE_DIR" \
  transformers accelerate datasets peft pandas matplotlib pyyaml

# Install heavy/optional packages explicitly when the phase needs them.
# python -m pip install --index-url https://pypi.org/simple --cache-dir "$PIP_CACHE_DIR" vllm lm-eval bitsandbytes auto-gptq autoawq
