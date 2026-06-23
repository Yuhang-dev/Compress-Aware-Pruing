from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
from pathlib import Path


PACKAGES = [
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "peft",
    "vllm",
    "lm-eval",
    "bitsandbytes",
    "auto-gptq",
    "autoawq",
    "pandas",
    "matplotlib",
    "pyyaml",
]


def collect_report() -> str:
    lines = [
        f"platform={platform.platform()}",
        f"python={platform.python_version()}",
        f"executable={os.sys.executable}",
        f"HF_HOME={os.environ.get('HF_HOME', '')}",
        f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE', '')}",
        f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE', '')}",
        f"TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE', '')}",
        f"TORCH_HOME={os.environ.get('TORCH_HOME', '')}",
        f"PIP_CACHE_DIR={os.environ.get('PIP_CACHE_DIR', '')}",
    ]

    try:
        import torch

        lines.extend(
            [
                f"torch={torch.__version__}",
                f"torch_cuda={torch.version.cuda}",
                f"cuda_available={torch.cuda.is_available()}",
                f"device_count={torch.cuda.device_count()}",
            ]
        )
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gib = props.total_memory / 1024**3
                lines.append(f"gpu_{idx}={props.name}, {gib:.1f} GiB")
    except Exception as exc:  # pragma: no cover - environment dependent
        lines.append(f"torch_error={exc!r}")

    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not_installed"
        lines.append(f"package.{package}={version}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = collect_report()
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
