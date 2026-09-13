"""Numerical conversions between clean-sample predictions and flow velocity."""

import torch
from torch import nn


class ImageVelocity(nn.Module):
    """Compose a clean-sample predictor with the bounded flow-matching equation.

    ``timestep`` is the clean fraction. ``epsilon`` bounds the denominator at
    the clean endpoint. PyTorch promotion between the predictor, latent and
    timestep preserves their declared numerical dtypes; no cast precedes the
    subtraction or division. Subclasses provide their mathematical head layout.
    """

    def __init__(self, head: nn.Module, *, epsilon: float) -> None:
        super().__init__()
        self.head = head
        self.epsilon = epsilon

    def forward(
        self,
        latent: torch.Tensor,
        hidden: torch.Tensor,
        timestep: torch.Tensor,
        *,
        image_tokens: int,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        predicted = self.predict(
            latent,
            hidden,
            timestep,
            image_tokens=image_tokens,
            image_height=image_height,
            image_width=image_width,
        )
        return (predicted - latent) / (1 - timestep).clamp_min(self.epsilon)

    def predict(
        self,
        latent: torch.Tensor,
        hidden: torch.Tensor,
        timestep: torch.Tensor,
        *,
        image_tokens: int,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        """Return clean-sample values in the input latent's mathematical layout."""

        raise NotImplementedError
