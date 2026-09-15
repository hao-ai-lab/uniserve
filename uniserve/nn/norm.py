"""Root-mean-square normalization as an ordinary numerical module."""

import torch
from torch import nn

from . import functional


class RMSNorm(nn.Module):
    """Normalize the final feature axis with FP32 variance accumulation."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or eps <= 0:
            raise ValueError(
                "RMS normalization requires positive width and epsilon"
            )
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.eps = eps

    def forward(
        self, x: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        return functional.rms_norm(x, self.weight, self.eps, out=out)
