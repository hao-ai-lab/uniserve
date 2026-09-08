"""Model-facing RMS normalization backed by worker operator dispatch."""

from __future__ import annotations

import torch
import torch.nn as nn

from uniserve_worker import ops
from uniserve_worker.ops.rms import eager_rms_norm

__all__ = [
    "RMSNorm",
]


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned per-feature weight.

    Providers accumulate variance in fp32 to keep composed low-precision model
    execution numerically stable. The module also exposes a fused residual-add
    entry point used by pre-normalization decoder blocks. ``affine_in_fp32``
    keeps normalization and weighting in FP32 until a single output cast.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        affine_in_fp32: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        """Create a unit scale vector for ``hidden_size`` features."""

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.variance_epsilon = eps
        self.affine_in_fp32 = affine_in_fp32

    @property
    def eps(self) -> float:
        """Return the variance stabilizer supplied at construction."""

        return self.variance_epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize ``hidden_states`` across their final dimension."""

        if self.affine_in_fp32:
            # This mode includes the tensor reduction's FP32 arithmetic and a
            # single rounding boundary after learned weighting.
            return eager_rms_norm(
                hidden_states, self.weight, self.variance_epsilon, affine_in_fp32=True
            )
        return ops.rms_norm(hidden_states, self.weight, self.variance_epsilon)

    def forward_with_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        *,
        in_place: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the normalized sum and the summed residual used to produce it.

        ``in_place`` permits an eligible provider to reuse caller-owned storage
        for the residual sum.
        """

        if self.affine_in_fp32:
            combined = hidden_states + residual
            return self(combined), combined
        return ops.add_rms_norm(
            hidden_states,
            residual,
            self.weight,
            self.variance_epsilon,
            in_place=in_place,
        )

    def extra_repr(self) -> str:
        """Render parameter geometry and epsilon in module representations."""

        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
