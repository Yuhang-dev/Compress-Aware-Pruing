from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - dependency bootstrap path
    raise ModuleNotFoundError(
        "Missing PyYAML. Install project dependencies or run "
        "`python -m pip install --index-url https://pypi.org/simple pyyaml`."
    ) from exc


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config.

    The lightweight scaffold supports single-file configs and a simple
    `defaults: [base.yaml]` convention used by this project.
    """

    path = Path(path)
    data = _read_yaml(path)
    defaults = data.pop("defaults", [])
    merged: dict[str, Any] = {}
    for item in defaults:
        if isinstance(item, str):
            default_path = path.parent / item
        else:
            raise TypeError(f"Unsupported defaults entry: {item!r}")
        merged = deep_merge(merged, _read_yaml(default_path))
    return deep_merge(merged, data)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return loaded
