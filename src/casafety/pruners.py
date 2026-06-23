from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

PrunerName = Literal["magnitude", "wanda", "sparsegpt"]


@dataclass(frozen=True)
class CompressionConfig:
    method: str
    sparsity: float | str | None = None


def compute_mask(
    layer_weight: torch.Tensor,
    calib_act_stats: dict[str, torch.Tensor] | None,
    sparsity: float | str,
    method: PrunerName,
) -> torch.Tensor:
    """Return a binary keep mask for one linear weight tensor.

    Wanda and magnitude are implemented for framework validation. SparseGPT is
    deliberately routed through an integration point because the experiment plan
    requires using a vetted external implementation for SparseGPT.
    """

    if method == "magnitude":
        scores = layer_weight.detach().abs()
    elif method == "wanda":
        if not calib_act_stats or "input_norm" not in calib_act_stats:
            raise ValueError("Wanda requires calib_act_stats['input_norm']")
        input_norm = calib_act_stats["input_norm"].to(layer_weight.device)
        scores = layer_weight.detach().abs() * input_norm.view(1, -1)
    elif method == "sparsegpt":
        raise NotImplementedError("Integrate locuslab/wanda or IST-DASLab/sparsegpt for SparseGPT.")
    else:
        raise ValueError(f"Unknown pruner: {method}")

    if sparsity == "2:4":
        return _mask_2_to_4(scores)
    if not isinstance(sparsity, float):
        raise TypeError(f"sparsity must be float or '2:4', got {sparsity!r}")
    return _mask_unstructured(scores, sparsity)


def _mask_unstructured(scores: torch.Tensor, sparsity: float) -> torch.Tensor:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    keep = max(1, int(scores.numel() * (1.0 - sparsity)))
    threshold = torch.topk(scores.flatten(), keep, largest=True).values.min()
    return (scores >= threshold).to(dtype=scores.dtype)


def _mask_2_to_4(scores: torch.Tensor) -> torch.Tensor:
    if scores.shape[-1] % 4 != 0:
        raise ValueError("2:4 mask requires input dimension divisible by 4")
    grouped = scores.reshape(*scores.shape[:-1], scores.shape[-1] // 4, 4)
    topk_idx = grouped.topk(k=2, dim=-1, largest=True).indices
    mask = torch.zeros_like(grouped)
    mask.scatter_(dim=-1, index=topk_idx, value=1.0)
    return mask.reshape_as(scores)


def quantize(model: torch.nn.Module, method: str) -> torch.nn.Module:
    raise NotImplementedError(f"Quantization integration is pending for method={method!r}.")
