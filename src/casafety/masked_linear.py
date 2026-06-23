from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MaskedLinearSTE(nn.Linear):
    """Linear layer whose forward uses a mask and backward is straight-through."""

    def __init__(self, source: nn.Linear):
        super().__init__(
            in_features=source.in_features,
            out_features=source.out_features,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self.weight = source.weight
        self.bias = source.bias
        self.register_buffer("mask", torch.ones_like(source.weight), persistent=False)

    def set_mask(self, mask: torch.Tensor) -> None:
        if mask.shape != self.weight.shape:
            raise ValueError(f"Mask shape {tuple(mask.shape)} != weight shape {tuple(self.weight.shape)}")
        self.mask = mask.detach().to(device=self.weight.device, dtype=self.weight.dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight + (self.weight * self.mask - self.weight).detach()
        return F.linear(input, masked_weight, self.bias)
