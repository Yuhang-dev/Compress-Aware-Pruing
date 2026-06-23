from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def train(config: dict) -> None:
    method = config.get("method", {}).get("name", "unknown")
    if method == "method_a":
        from .methods.method_a import train_method_a

        train_method_a(config)
    elif method == "method_b":
        from .methods.method_b import train_method_b

        train_method_b(config)
    elif method == "method_c":
        from .methods.method_c import train_method_c

        train_method_c(config)
    elif method == "method_d":
        from .methods.method_d import train_method_d

        train_method_d(config)
    else:
        raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
