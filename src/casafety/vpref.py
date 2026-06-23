from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config


def run_vpref_validation(config: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "status": "not_run",
                "reason": "Requires verified harmful/benign train splits and model activation hooks.",
                "model": config["model"]["name_or_path"],
            }
        ]
    )
    df.to_csv(output, index=False)
    raise NotImplementedError(f"vpref extraction scaffold wrote placeholder to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_vpref_validation(load_config(args.config), args.output)


if __name__ == "__main__":
    main()
