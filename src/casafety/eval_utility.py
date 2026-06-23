from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_config(args.config)
    raise NotImplementedError("Utility evaluation should call lm-eval and a WikiText-2 PPL runner.")


if __name__ == "__main__":
    main()
