"""Build refusal SFT data and a dataset manifest.

This script intentionally does not hard-code unverified Hugging Face dataset IDs.
Fill or pass verified dataset paths before running Phase 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_placeholder_manifest() -> dict:
    return {
        "status": "requires_human_verification",
        "leakage_check": "not_run",
        "datasets": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--write-manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_placeholder_manifest()
    args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.write_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote placeholder manifest to {args.write_manifest}")


if __name__ == "__main__":
    main()
