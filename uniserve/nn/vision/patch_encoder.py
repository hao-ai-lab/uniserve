"""Image patch encoding with learned feature projection and spatial
positions.
"""  # noqa: D205

from __future__ import annotations

import torch
from torch import nn

from uniserve.nn.vision.patching import patchify_batch
from uniserve.nn.vision.position import get_flattened_position_ids_extrapolate


class PatchEncoder(nn.Module):
    """Encode uniform NCHW images into projected, position-aware patch
    sequences.

    The encoder accepts flattened patches and cumulative sequence boundaries;
    projection and position modules map its output into the consumer's width.
    Inputs must already reside with the modules. Dtype conversion is numerical;
    this module does not deliver tensors between devices.
    """  # noqa: D205

    def __init__(
        self,
        encoder: nn.Module,
        projection: nn.Module,
        position_embed: nn.Module,
        *,
        patch_size: int,
        position_grid: int,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.position_embed = position_embed
        self.patch_size = patch_size
        self.position_grid = position_grid
        self.hidden_size = hidden_size
        self.dtype = dtype

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return ``[images, patches, hidden]`` without attention across
        images.
        """  # noqa: D205
        if pixels.ndim != 4:
            raise ValueError("patch encoding requires NCHW pixels")

        batch, channels, height, width = pixels.shape
        patch = self.patch_size
        positions = get_flattened_position_ids_extrapolate(
            height, width, patch, self.position_grid, device=pixels.device
        ).repeat(batch)
        patches = patchify_batch(pixels, patch).reshape(
            -1, patch * patch * channels
        )
        tokens = (height // patch) * (width // patch)

        # Cumulative bounds isolate attention while all images share one grid.
        boundaries = torch.arange(
            0,
            (batch + 1) * tokens,
            tokens,
            dtype=torch.int32,
            device=pixels.device,
        )
        features = self.encoder(
            patches.to(self.dtype),
            {
                "position_ids": positions,
                "cu_seqlens": boundaries,
                "seq_lens": (tokens,) * batch,
            },
            None,
        )

        projected = self.projection(features) + self.position_embed(positions)
        return projected.reshape(batch, tokens, -1).to(self.dtype)
