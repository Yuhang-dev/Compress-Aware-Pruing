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


def resolve_judge_model_id(config: dict[str, Any], requested: str | None = None) -> str:
    if requested:
        return requested
    model_config = config.get("model", {})
    if "judge_name_or_path" in model_config:
        return model_config["judge_name_or_path"]
    judge_config = config.get("judge", {})
    if isinstance(judge_config, dict) and "name_or_path" in judge_config:
        return judge_config["name_or_path"]
    return "meta-llama/Llama-Guard-3-8B"


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")
