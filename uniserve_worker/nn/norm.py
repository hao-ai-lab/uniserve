"""RMSNorm module. Compute dispatch lives in ``uniserve_worker.ops``."""

from __future__ import annotations

import torch
import torch.nn as nn

from uniserve_worker import ops

__all__ = [
    "RMSNorm",
]


class RMSNorm(nn.Module):
    """RMSNorm with fp32 variance accumulation.

    Matches the local RMSNorm variants used by current model ports. Uses fp32
    for the variance reduction because low-precision norm drift is visible in
    long composed generations.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @property
    def eps(self) -> float:
        return self.variance_epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return ops.rms_norm(hidden_states, self.weight, self.variance_epsilon)

    def forward_with_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        *,
        in_place: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``RMSNorm(hidden_states + residual)`` and the summed residual."""

        return ops.add_rms_norm(
            hidden_states,
            residual,
            self.weight,
            self.variance_epsilon,
            in_place=in_place,
        )

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
