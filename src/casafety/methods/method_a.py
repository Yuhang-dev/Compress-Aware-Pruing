from __future__ import annotations


def train_method_a(config: dict) -> None:
    variant = config.get("method", {}).get("variant", "a2")
    raise NotImplementedError(
        f"Method A/{variant} training requires verified refusal vectors, calibration activations, and LoRA setup."
    )
