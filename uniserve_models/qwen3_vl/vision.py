"""Qwen3-VL vision encoder: tubelet patches to language-width tokens.

``VisionTower`` packs the patches of several images and video blocks into one
``[patches, tubelet_pixels]`` sequence in the processor's merge-block order
(``uniserve.nn.vision.merged_grid_coordinates``). Each patch receives a
learned position bilinearly resampled to its grid and a 2D rotary position
(row, column). Attention never crosses a time step: every temporal patch of
every grid is its own sequence. After the blocks, ``PatchMerger`` merges each
``merge x merge`` patch block into one language-width token, and DeepStack
mergers do the same with the outputs of selected intermediate blocks.
"""

from __future__ import annotations

from math import prod

import torch
from torch import nn

from uniserve.nn.activation import get_act_fn
from uniserve.nn.attention import Attention as ScaledAttention
from uniserve.nn.attention import AttentionBatch, SequenceLengths, VarlenInput
from uniserve.nn.functional import apply_rotary
from uniserve.nn.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from uniserve.nn.rope import RotaryEmbedding
from uniserve.nn.vision import (
    PatchEmbed,
    PositionEmbedding,
    merged_grid_coordinates,
)

from .config import VisionConfig

# Fixed by the Qwen3-VL architecture rather than its checkpoint metadata:
# every vision LayerNorm uses this epsilon, and the 2D rotary embedding this
# frequency base.
LAYER_NORM_EPS = 1e-6
ROPE_THETA = 10_000.0


class Attention(nn.Module):
    """Bidirectional rotary attention within each patch sequence."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        heads, width = config.num_heads, config.head_dim
        self.qkv = QKVParallelLinear(
            config.hidden_size, heads, heads, width, bias=True
        )
        self.attention = ScaledAttention(heads, heads, width)
        self.output = RowParallelLinear(
            config.hidden_size, config.hidden_size, bias=True
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention: AttentionBatch,
    ) -> torch.Tensor:
        # hidden: [patches, hidden]; q/k/v: [patches, local heads, head_dim].
        # cos/sin: compact [patches, head_dim / 2] factors rotating each head
        # split-half over its full width.
        projections = self.qkv(hidden)
        query, key, value = (
            projections[name].reshape(
                hidden.shape[0], -1, self.attention.head_dim
            )
            for name in ("q", "k", "v")
        )
        query = apply_rotary(query, cos, sin, rotation="split")
        key = apply_rotary(key, cos, sin, rotation="split")
        attended = self.attention(query, key, value, attention)
        return self.output(attended.flatten(1))


class TransformerLayer(nn.Module):
    """Pre-norm attention and MLP, each added to the residual stream."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.input_norm = nn.LayerNorm(config.hidden_size, eps=LAYER_NORM_EPS)
        self.attention = Attention(config)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=LAYER_NORM_EPS)
        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                config.hidden_size, config.intermediate_size, bias=True
            ),
            get_act_fn(config.hidden_act),
            RowParallelLinear(
                config.intermediate_size, config.hidden_size, bias=True
            ),
        )

    def forward(self, hidden, cos, sin, attention):
        hidden = hidden + self.attention(
            self.input_norm(hidden), cos, sin, attention
        )
        return hidden + self.mlp(self.output_norm(hidden))


class PatchMerger(nn.Module):
    """Merge each block of ``merge**2`` patch features into one token.

    The block's consecutive patch rows concatenate into one row of
    ``merge**2 * hidden`` values, which an exact-GELU MLP projects to the
    language width. The final merger normalizes each patch before the
    concatenation; DeepStack mergers normalize the concatenated row.
    """

    def __init__(self, config: VisionConfig, *, normalize_merged: bool):
        super().__init__()
        self.width = config.hidden_size * config.spatial_merge_size**2
        self.normalize_merged = normalize_merged
        self.norm = nn.LayerNorm(
            self.width if normalize_merged else config.hidden_size,
            eps=LAYER_NORM_EPS,
        )
        self.mlp = nn.Sequential(
            ColumnParallelLinear(self.width, self.width, bias=True),
            nn.GELU(),
            RowParallelLinear(self.width, config.out_hidden_size, bias=True),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Return ``[patches / merge**2, out_hidden]`` merged tokens."""
        if self.normalize_merged:
            merged = self.norm(patches.reshape(-1, self.width))
        else:
            merged = self.norm(patches).reshape(-1, self.width)
        return self.mlp(merged)


class VisionTower(nn.Module):
    """Encode packed image and video patches into language-width tokens."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        # Tubelets span temporal_patch_size frames of one spatial patch.
        self.patch_embedding = PatchEmbed(
            config.in_channels,
            config.hidden_size,
            (
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            ),
        )
        self.position_embedding = PositionEmbedding(
            (config.position_grid, config.position_grid), config.hidden_size
        )
        # Half of each head rotates by the patch row and half by its column,
        # so each axis has head_dim / 4 frequencies.
        self.rotary = RotaryEmbedding(config.head_dim // 2, theta=ROPE_THETA)
        self.layers = nn.ModuleList(
            TransformerLayer(config) for _ in range(config.depth)
        )
        self.merger = PatchMerger(config, normalize_merged=False)
        self.deepstack = nn.ModuleList(
            PatchMerger(config, normalize_merged=True)
            for _ in config.deepstack_visual_indexes
        )

    def forward(
        self,
        pixels: torch.Tensor,
        grids: torch.Tensor,
        grid_shapes: tuple[tuple[int, int, int], ...],
    ) -> torch.Tensor:
        """Encode the packed tubelets of one or more images or video blocks.

        Args:
            pixels: ``[patches, tubelet_pixels]`` tubelet rows, grids
                concatenated in order, each in merge-block order over its
                time steps.
            grids: ``[grids, 3]`` device copy of ``grid_shapes``. Every
                coordinate here derives from the host shapes, so only its
                shape is checked.
            grid_shapes: Each grid's ``(time, height, width)`` in patches;
                an image has one time step.

        Returns:
            ``[patches / merge**2, feature_size]`` merged tokens, each its
            language embedding followed by its DeepStack features
            (``VisionConfig.feature_size``), in input order.

        Raises:
            ValueError: The patch rows do not cover the declared grids, or a
                grid side is not a whole number of merge blocks.
        """
        config = self.config
        merge = config.spatial_merge_size
        if (
            grids.shape != (len(grid_shapes), 3)
            or any(len(shape) != 3 or min(shape) < 1 for shape in grid_shapes)
            or any(
                height % merge or width % merge
                for _, height, width in grid_shapes
            )
            or pixels.shape
            != (
                sum(prod(shape) for shape in grid_shapes),
                self.patch_embedding.in_features,
            )
        ):
            raise ValueError(
                "Qwen3-VL patches must cover whole merge blocks of their "
                "declared grids"
            )

        hidden = self.patch_embedding(pixels)
        device = hidden.device

        # Learned positions resampled to each grid (FP32), shared by every
        # time step, then rounded to the activation dtype.
        positions = torch.cat(
            tuple(
                self.position_embedding.interpolate(
                    height, width, merge=merge
                ).repeat(time, 1)
                for time, height, width in grid_shapes
            )
        )
        hidden = hidden + positions.to(hidden.dtype)

        # [patches, 2] (row, column) coordinates -> compact [patches,
        # head_dim / 2] factors: row frequencies, then column frequencies.
        coordinates = torch.cat(
            tuple(
                torch.stack(
                    merged_grid_coordinates(
                        height, width, merge, device=device
                    ),
                    dim=-1,
                ).repeat(time, 1)
                for time, height, width in grid_shapes
            )
        )
        cos, sin = self.rotary(
            coordinates, dtype=torch.float32, sequence_length=0
        )
        cos, sin = cos.flatten(1), sin.flatten(1)

        # One bidirectional sequence per time step of every grid.
        lengths = SequenceLengths.from_lengths(
            tuple(
                height * width
                for time, height, width in grid_shapes
                for _ in range(time)
            ),
            device=device,
        )
        attention = AttentionBatch.single(
            VarlenInput(lengths, lengths, (False,) * lengths.batch_size)
        )

        deepstack = []
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, attention)
            if index in config.deepstack_visual_indexes:
                merger = self.deepstack[
                    config.deepstack_visual_indexes.index(index)
                ]
                deepstack.append(merger(hidden))
        return torch.cat((self.merger(hidden), *deepstack), dim=-1)
