"""Shared SigLIP-NaViT encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ...execution.forward_batch import ForwardBatch
from ..layer import LayerConfig
from ..linear import LinearBase
from .encoder import VisionEncoder, VisionEncoderConfig
from .patching import patchify_batch
from .position import get_flattened_position_ids_extrapolate

__all__ = [
    "SiglipNavitConfig",
    "SIGLIP_SO400M",
    "SiglipNavitEncoder",
]


@dataclass(frozen=True)
class SiglipNavitConfig:
    patch_size: int = 14
    hidden_size: int = 1152
    image_size: int = 980
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    num_hidden_layers: int = 26
    layer_norm_eps: float = 1e-6
    num_channels: int = 3


# Named SigLIP-so400m geometry preset. The dataclass defaults equal this preset;
# build a config from the checkpoint where available and fall back to it.
SIGLIP_SO400M = SiglipNavitConfig()


class SiglipNavitEncoder(nn.Module):
    """Linear-patch, learned-absolute-position NaViT encoder."""

    def __init__(self, cfg: SiglipNavitConfig, *, layer_config: LayerConfig) -> None:
        super().__init__()
        self.patch_size = int(cfg.patch_size)
        hidden = int(cfg.hidden_size)
        image_size = int(cfg.image_size)
        num_heads = int(cfg.num_attention_heads)
        intermediate = int(cfg.intermediate_size)
        layers = int(cfg.num_hidden_layers)
        eps = float(cfg.layer_norm_eps)
        self.num_channels = int(cfg.num_channels)
        self.max_num_patch_per_side = image_size // self.patch_size
        self.patch_embedding = LinearBase(
            self.num_channels * self.patch_size * self.patch_size,
            hidden,
            layer_config=layer_config,
            bias=True,
        )
        self.position_embedding = nn.Embedding(self.max_num_patch_per_side**2, hidden)
        self.encoder = VisionEncoder(
            VisionEncoderConfig(
                hidden_size=hidden,
                num_attention_heads=num_heads,
                intermediate_size=intermediate,
                num_hidden_layers=layers,
                layer_norm_eps=eps,
            ),
            layer_config=layer_config,
        )

    def forward(
        self,
        pixels: torch.Tensor,
        grid: Any,
        context: ForwardBatch,
    ) -> torch.Tensor:
        packed, pos_ids, cu_seqlens, seq_lens = self._pack_inputs(pixels, grid)
        x = self.patch_embedding(packed) + self.position_embedding(pos_ids)
        return self.encoder(x, context, cu_seqlens, seq_lens=seq_lens)

    def _pack_inputs(
        self, pixels: torch.Tensor, grid: Any | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
        if isinstance(grid, dict):
            pos = grid.get("position_ids")
            if pos is None:
                pos = grid.get("pos_ids")
            cu = grid.get("cu_seqlens")
            seq_lens = grid.get("seq_lens")
            if pos is not None and cu is not None:
                if seq_lens is None:
                    raise ValueError(
                        "SiglipNavitEncoder grid dict must carry host-known seq_lens "
                        "alongside cu_seqlens"
                    )
                return (
                    pixels,
                    pos.to(device=pixels.device, dtype=torch.long),
                    cu.to(device=pixels.device, dtype=torch.int32),
                    tuple(int(length) for length in seq_lens),
                )
        if pixels.ndim == 2:
            n = int(pixels.shape[0])
            pos = torch.arange(n, device=pixels.device, dtype=torch.long)
            cu = torch.tensor([0, n], device=pixels.device, dtype=torch.int32)
            return pixels, pos, cu, (n,)
        if pixels.ndim != 4:
            raise ValueError("SiglipNavitEncoder expects packed patches or NCHW pixels")
        patches = patchify_batch(pixels, self.patch_size).reshape(
            -1, self.num_channels * self.patch_size * self.patch_size
        )
        batch, _, height, width = pixels.shape
        per_image = (height // self.patch_size) * (width // self.patch_size)
        pos = get_flattened_position_ids_extrapolate(
            height,
            width,
            self.patch_size,
            self.max_num_patch_per_side,
            device=pixels.device,
        ).repeat(batch)
        cu = torch.arange(
            0, (batch + 1) * per_image, per_image, device=pixels.device, dtype=torch.int32
        )
        return patches, pos, cu, (per_image,) * int(batch)
