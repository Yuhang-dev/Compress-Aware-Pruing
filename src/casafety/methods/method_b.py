from __future__ import annotations


def train_method_b(config: dict) -> None:
    if config.get("method", {}).get("mask_trainable", False):
        raise ValueError("Method B must not learn masks; masks come from deployment pruner criteria.")
    raise NotImplementedError(
        "Method B training requires MaskedLinearSTE replacement, sampled compression configs, and calibration caches."
    )
