from __future__ import annotations


def train_method_d(config: dict) -> None:
    if not config.get("method", {}).get("enabled", False):
        raise NotImplementedError("Method D is optional and disabled in the default config.")
    raise NotImplementedError("Method D low-rank structural constraint is not implemented yet.")
