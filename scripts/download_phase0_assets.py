"""Download Phase 0 model and dataset assets into the configured HF cache.

This script is intentionally conservative:
- model IDs come from the experiment plan;
- dataset IDs are read from config and skipped when unset, because the plan
  requires verifying current HF paths and splits before use;
- gated models rely on a private HF_TOKEN/HUGGINGFACE_HUB_TOKEN in the shell.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from huggingface_hub import snapshot_download

from casafety.config import load_config


DEFAULT_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-Guard-3-8B",
]


def _iter_dataset_entries(data_config: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key, value in data_config.items():
        if isinstance(value, dict):
            entries.append(value)
        elif isinstance(value, list):
            entries.extend(item for item in value if isinstance(item, dict))
    return entries


def download_models(model_ids: list[str], cache_dir: str | None) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    for model_id in model_ids:
        print(f"[model] downloading {model_id} with HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET')}")
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            cache_dir=cache_dir,
            token=token,
            resume_download=True,
        )


def download_configured_datasets(config: dict[str, Any], cache_dir: str | None) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    for entry in _iter_dataset_entries(config.get("data", {})):
        dataset_id = entry.get("path")
        name = entry.get("name", dataset_id or "unknown")
        if not dataset_id:
            print(f"[dataset] skip {name}: path is not verified in config")
            continue
        print(f"[dataset] downloading {dataset_id} with HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET')}")
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            cache_dir=cache_dir,
            token=token,
            resume_download=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--datasets-only", action="store_true")
    parser.add_argument("--include-generalization-model", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    model_ids = [config["model"]["name_or_path"], config["model"]["judge_name_or_path"]]
    if args.include_generalization_model:
        model_ids.append("meta-llama/Llama-3.1-8B-Instruct")

    if not args.datasets_only:
        download_models(model_ids or DEFAULT_MODELS, cache_dir)
    if not args.models_only:
        download_configured_datasets(config, cache_dir)


if __name__ == "__main__":
    main()
