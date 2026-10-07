"""Root-mean-square normalization as an ordinary numerical module."""

import torch
from torch import nn

from . import functional


class RMSNorm(nn.Module):
    """Normalize the final feature axis with FP32 variance accumulation.

    With ``elementwise_affine`` (the default) the normalized rows multiply a
    learned ``weight`` Parameter. Without it the rows are only normalized;
    ``weight`` is then a non-persistent unit buffer, so both forms share one
    normalization primitive and multiplying by one leaves every value exact.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        elementwise_affine: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or eps <= 0:
            raise ValueError(
                "RMS normalization requires positive width and epsilon"
            )
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(hidden_size, device=device, dtype=dtype),
                requires_grad=False,
            )
        else:
            # The unit scale is a derived constant, not a checkpoint value:
            # it stays a real FP32 tensor under meta construction, and loading
            # moves it with its module.
            actual = torch.device("cpu" if device is None else device)
            if actual.type == "meta":
                actual = torch.device("cpu")
            self.register_buffer(
                "weight",
                torch.ones(hidden_size, device=actual, dtype=torch.float32),
                persistent=False,
            )
        self.eps = eps

    def forward(
        self, x: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        return functional.rms_norm(x, self.weight, self.eps, out=out)
