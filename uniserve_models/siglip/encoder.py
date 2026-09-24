"""SigLIP-NaViT image transformer composition and patch coordinates.

``Encoder`` packs the patches of several images of different grid sizes into
one ``[total_patches, patch_features]`` sequence. Each image attends only to
its own patches through non-causal varlen attention, and each patch adds the
learned position of its ``(row, column)`` in a fixed square table; grids
smaller than the table select a subset of positions without interpolation.
Patch rows use the order ``uniserve.nn.functional.patchify`` produces: pixel
row, pixel column, then channel within each patch.
"""

from __future__ import annotations

import torch
from torch import nn

from uniserve.model import TransformerEncoder
from uniserve.nn.attention import Attention as ScaledAttention
from uniserve.nn.attention import SequenceLengths, VarlenInput
from uniserve.nn.functional import patchify
from uniserve.nn.linear import (
    ColumnParallelLinear,
    Linear,
    QKVParallelLinear,
    RowParallelLinear,
)
from uniserve.nn.vision.position import build_abs_positions_from_grid_hw

from .config import Config, TransformerConfig


class Attention(nn.Module):
    """Multi-head self-attention over a variable-length patch sequence."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        heads = config.num_attention_heads
        dim = config.hidden_size // heads
        self.qkv = QKVParallelLinear(
            config.hidden_size, heads, heads, dim, bias=True
        )
        self.attention = ScaledAttention(heads, heads, dim)
        self.output = RowParallelLinear(
            config.hidden_size, config.hidden_size, bias=True
        )

    def forward(
        self, patches: torch.Tensor, attention: VarlenInput
    ) -> torch.Tensor:
        projections = self.qkv(patches)

        # [total_patches, heads, head_dim], with the heads this rank holds
        # under tensor-parallel head partitioning.
        query, key, value = (
            projections[name].reshape(
                patches.shape[0], -1, self.attention.head_dim
            )
            for name in ("q", "k", "v")
        )
        hidden = self.attention(query, key, value, attention)
        return self.output(hidden.flatten(1))


class TransformerLayer(nn.Module):
    """Pre-norm attention and GELU MLP with residual connections."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.input_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = Attention(config)
        self.output_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                config.hidden_size, config.intermediate_size, bias=True
            ),
            nn.GELU(approximate="tanh"),
            RowParallelLinear(
                config.intermediate_size, config.hidden_size, bias=True
            ),
        )

    def forward(
        self, patches: torch.Tensor, attention: VarlenInput
    ) -> torch.Tensor:
        patches = patches + self.attention(self.input_norm(patches), attention)
        return patches + self.mlp(self.output_norm(patches))


class Encoder(nn.Module):
    """Encode canonical HWC patch rows with separate attention per image."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.patch_embedding = Linear(
            config.num_channels * config.patch_size**2,
            config.encoder.hidden_size,
            bias=True,
        )
        self.position_embedding = nn.Embedding(
            (config.image_size // config.patch_size) ** 2,
            config.encoder.hidden_size,
        )
        self.encoder = TransformerEncoder(
            nn.ModuleList(
                TransformerLayer(config.encoder)
                for _ in range(config.encoder.num_hidden_layers)
            ),
            nn.LayerNorm(
                config.encoder.hidden_size, eps=config.encoder.layer_norm_eps
            ),
        )

    def forward(
        self,
        pixels: torch.Tensor,
        grids: torch.Tensor,
        grid_shapes: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        """Encode the packed patches of one or more images.

        Args:
            pixels: ``[total_patches, num_channels * patch_size**2]`` patch
                rows with images concatenated in order, or
                ``[images, channels, height, width]`` pixels of same-sized
                images, which are patchified here.
            grids: ``[images, 2]`` int32 or int64 tensor of each image's
                ``(height, width)`` in patches, on the encoder's device.
            grid_shapes: The same ``(height, width)`` pairs as host integers,
                so patch counts and kernel shapes are known without reading
                device values.

        Returns:
            ``[total_patches, hidden_size]`` features after the final
            layernorm, in input patch order.

        Raises:
            ValueError: If the patch rows are not ``[sum of grid areas,
                num_channels * patch_size**2]``, ``grids`` is not an
                ``[images, 2]`` int32 or int64 tensor, or a grid side lies
                outside ``1..image_size // patch_size``. ``patchify`` also
                raises for NCHW pixels whose sides the patch size does not
                divide.
        """
        if pixels.ndim == 4:
            pixels = patchify(
                pixels, patch_size=self.config.patch_size
            ).flatten(0, 1)

        # pixels: [total_patches, channels * patch_size**2] after flattening;
        # every image grid must fit the fixed learned position table. The
        # checks compare shapes and host values only; ``grids`` values are
        # trusted to agree with ``grid_shapes``.
        counts = tuple(height * width for height, width in grid_shapes)
        side = self.config.image_size // self.config.patch_size
        if (
            pixels.ndim != 2
            or pixels.shape != (sum(counts), self.patch_embedding.in_features)
            or grids.shape != (len(counts), 2)
            or grids.dtype not in (torch.int32, torch.int64)
            or any(min(shape) < 1 or max(shape) > side for shape in grid_shapes)
        ):
            raise ValueError(
                "SigLIP patches and grid coordinates must cover "
                "the declared image grids"
            )

        # Row-major absolute position index within the side x side patch grid.
        columns, rows = build_abs_positions_from_grid_hw(
            grids, total=sum(counts)
        )
        positions = rows * side + columns

        # Callers may stage pixels in another dtype; embed in the weight dtype.
        features = self.patch_embedding(
            pixels.to(self.patch_embedding.weight.dtype)
        )
        features = features + self.position_embedding(positions)

        # Varlen attention needs each image's patch count on device and their
        # prefix-sum offsets, while the host copy drives kernel launch shapes.
        # Queries and keys share the same lengths, and no image is causal.
        values = grids.prod(dim=1).to(torch.int32)
        offsets = torch.cat(
            (values.new_zeros(1), values.cumsum(0, dtype=torch.int32))
        )
        lengths = SequenceLengths(host=counts, values=values, offsets=offsets)
        return self.encoder(
            features, VarlenInput(lengths, lengths, (False,) * len(counts))
        )
