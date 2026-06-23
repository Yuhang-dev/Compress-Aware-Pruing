"""Make the src-layout package importable from a plain repo checkout.

AutoDL shell helpers may put only the repository root on PYTHONPATH. Python
imports this file automatically when the repo root is on sys.path, so commands
like `python -m casafety.env_report` work before editable installation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if SRC.is_dir():
    src_text = str(SRC)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
