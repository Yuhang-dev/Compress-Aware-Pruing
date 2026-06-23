from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "status": "not_run",
                "reason": "Requires Method C safety-key identification and benign calibration activations.",
                "model": config["model"]["name_or_path"],
            }
        ]
    ).to_csv(args.output, index=False)
    raise NotImplementedError(f"mechanism diagnosis scaffold wrote placeholder to {args.output}")


if __name__ == "__main__":
    main()
