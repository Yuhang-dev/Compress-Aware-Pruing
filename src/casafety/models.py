from __future__ import annotations

from typing import Any


def resolve_model_id(config: dict[str, Any], requested: str | None = None) -> str:
    model_config = config.get("model", {})
    if requested:
        return requested
    if "name_or_path" in model_config:
        return model_config["name_or_path"]
    if "default" in model_config:
        return model_config["default"]
    candidates = model_config.get("candidates") or []
    if candidates:
        first = candidates[0]
        return first["id"] if isinstance(first, dict) else str(first)
    raise KeyError("No model id found. Set model.default, model.name_or_path, or pass --model.")


def candidate_model_ids(config: dict[str, Any]) -> list[str]:
    model_config = config.get("model", {})
    candidates = model_config.get("candidates")
    if candidates:
        ids = []
        for candidate in candidates:
            ids.append(candidate["id"] if isinstance(candidate, dict) else str(candidate))
        return ids
    if "name_or_path" in model_config:
        return [model_config["name_or_path"]]
    if "default" in model_config:
        return [model_config["default"]]
    return []
