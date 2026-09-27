"""Gemma-4 vision tower: patch embedding, 2D-rotary encoder, pooling, output.

``Encoder`` is the ``PatchEncoder`` network. It embeds packed patch rows of
one or more images (pixel row, pixel column, then channel within each patch,
with values in ``[0, 1]``), runs non-causal attention within each image,
averages each image's ``k x k`` squares of patches into soft tokens and
standardizes them. ``Embedder`` is the connector that projects soft tokens
to the text width. Patch positions are ``(x, y)`` = (column, row) in the
image's patch grid; patches follow row-major order.
"""

from __future__ import annotations

import torch
from torch import nn

from uniserve.nn.attention import Attention as ScaledAttention
from uniserve.nn.attention import (
    AttentionBatch,
    SequenceLengths,
    VarlenInput,
)
from uniserve.nn.functional import patchify, qk_norm_rope
from uniserve.nn.linear import Linear, QKVParallelLinear, RowParallelLinear
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.norm import RMSNorm
from uniserve.nn.rope import RotaryEmbedding
from uniserve.nn.vision.position import build_abs_positions_from_grid_hw

from .config import VisionConfig

# One rotary factor pair per spatial axis: ``(cos, sin)`` tuples ordered
# (x, y), each ``[patches, head_dim / 4]``.
RotaryFactors = tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]


class PatchEmbedder(nn.Module):
    """Project patch pixels and add learned column and row positions.

    ``position_table`` is ``[2, position_embedding_size, hidden]``: row 0
    embeds a patch's column index, row 1 its row index.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.projection = Linear(
            3 * config.patch_size**2, config.hidden_size, bias=False
        )
        self.position_table = nn.Parameter(
            torch.ones(2, config.position_embedding_size, config.hidden_size),
            requires_grad=False,
        )

    def forward(
        self, patches: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # Pixels in [0, 1] map to [-1, 1] in their staged dtype, then round to
        # the weight dtype for the projection.
        pixels = 2 * (patches - 0.5)
        features = self.projection(pixels.to(self.projection.weight.dtype))
        positions = self.position_table[0].index_select(
            0, x
        ) + self.position_table[1].index_select(0, y)
        return features + positions


class Attention(nn.Module):
    """Non-causal multi-head attention over one image's patches.

    Scores are unscaled; queries and keys carry weighted RMS norms and values
    an unweighted one. The first half of every head rotates by the patch
    column and the second half by its row.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        heads, head_dim = config.num_attention_heads, config.head_dim
        self.head_dim, self.eps = head_dim, config.rms_norm_eps
        self.qkv = QKVParallelLinear(
            config.hidden_size, heads, heads, head_dim, bias=False
        )
        self.query_norm = RMSNorm(head_dim, config.rms_norm_eps)
        self.key_norm = RMSNorm(head_dim, config.rms_norm_eps)
        self.value_norm = RMSNorm(
            head_dim, config.rms_norm_eps, elementwise_affine=False
        )
        self.attention = ScaledAttention(heads, heads, head_dim, scale=1.0)
        self.output = RowParallelLinear(
            heads * head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden: torch.Tensor,
        rotary: RotaryFactors,
        attention: AttentionBatch,
    ) -> torch.Tensor:
        tokens = hidden.shape[0]
        projections = self.qkv(hidden)
        query, key, value = (
            projections[name].reshape(tokens, -1, self.head_dim)
            for name in ("q", "k", "v")
        )
        cos, sin = rotary
        half = self.head_dim // 2
        # One normalization domain spans the whole head; the two rotary axes
        # each rotate split halves of their own half of the head.
        query, key = qk_norm_rope(
            query,
            key,
            (self.query_norm.weight,),
            (self.key_norm.weight,),
            cos,
            sin,
            eps=self.eps,
            axis_dims=(half, half),
        )
        attended = self.attention(query, key, self.value_norm(value), attention)
        return self.output(attended.flatten(1))


class EncoderLayer(nn.Module):
    """Gemma sandwich-normalized attention and tanh-GELU gated MLP."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        size, eps = config.hidden_size, config.rms_norm_eps
        self.input_norm = RMSNorm(size, eps)
        self.attention = Attention(config)
        self.post_attention_norm = RMSNorm(size, eps)
        self.pre_feedforward_norm = RMSNorm(size, eps)
        self.mlp = GatedMLP(
            size, config.intermediate_size, activation="gelu_pytorch_tanh"
        )
        self.post_feedforward_norm = RMSNorm(size, eps)

    def forward(
        self,
        hidden: torch.Tensor,
        rotary: RotaryFactors,
        attention: AttentionBatch,
    ) -> torch.Tensor:
        attended = self.attention(self.input_norm(hidden), rotary, attention)
        hidden = hidden + self.post_attention_norm(attended)
        update = self.mlp(self.pre_feedforward_norm(hidden))
        return hidden + self.post_feedforward_norm(update)


class VisionTransformer(nn.Module):
    """The encoder layers with shared two-axis rotary factors.

    Each spatial axis rotates ``head_dim / 2`` channels with the default
    frequencies of that width. The stack has no final norm.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.rotary = RotaryEmbedding(
            config.head_dim // 2, theta=config.rope_theta
        )
        self.layers = nn.ModuleList(
            EncoderLayer(config) for _ in range(config.num_hidden_layers)
        )

    def forward(
        self,
        features: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        attention: AttentionBatch,
    ) -> torch.Tensor:
        factors = tuple(
            self.rotary(axis, dtype=torch.float32, sequence_length=axis.numel())
            for axis in (x, y)
        )
        rotary = (
            tuple(cos for cos, _ in factors),
            tuple(sin for _, sin in factors),
        )
        for layer in self.layers:
            features = layer(features, rotary, attention)
        return features


class Pooler(nn.Module):
    """Average ``k x k`` patch squares, rescale and standardize them.

    Each image's patches average in FP32 and round to the feature dtype;
    the averages then multiply by ``sqrt(hidden_size)`` and, when the
    checkpoint standardizes, become ``(value - std_bias) * std_scale``, all
    in FP32 before a final rounding. Soft tokens follow row-major order of
    the pooled grid, images in packing order.

    The squares are found from the device grids alone, so the computation
    has one shape for every packing of the same number of patch rows.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.kernel_size = config.pooling_kernel_size
        self.root_size = config.hidden_size**0.5
        self.std_bias: nn.Parameter | None = None
        self.std_scale: nn.Parameter | None = None
        if config.standardize:
            self.std_bias = nn.Parameter(
                torch.zeros(config.hidden_size), requires_grad=False
            )
            self.std_scale = nn.Parameter(
                torch.ones(config.hidden_size), requires_grad=False
            )

    def forward(
        self, features: torch.Tensor, grids: torch.Tensor
    ) -> torch.Tensor:
        """Pool ``[patches, hidden]`` features of the ``[images, 2]`` grids.

        Every grid side is a multiple of ``k``, and the grids cover the
        patch rows in order; a ``(0, 0)`` grid is an empty segment.
        """
        kernel = self.kernel_size
        device = features.device
        tokens = features.shape[0] // kernel**2

        # Soft token t of the packing belongs to image image[t] and is its
        # local[t]-th token in row-major order of that image's pooled grid.
        heights, widths = grids[:, 0], grids[:, 1]
        patch_counts = heights * widths
        token_counts = patch_counts // kernel**2
        image = torch.repeat_interleave(
            torch.arange(grids.shape[0], device=device),
            token_counts,
            output_size=tokens,
        )
        first_token = torch.cumsum(token_counts, 0) - token_counts
        first_patch = torch.cumsum(patch_counts, 0) - patch_counts
        local = torch.arange(tokens, device=device) - first_token[image]
        width = widths[image]
        row, column = local // (width // kernel), local % (width // kernel)

        # The patch rows of each token's k x k square in row-major order:
        # [tokens, k * k] indices into the packed rows.
        corner = first_patch[image] + (row * width + column) * kernel
        offsets = torch.arange(kernel, device=device)
        square = (
            corner[:, None, None]
            + offsets[None, :, None] * width[:, None, None]
            + offsets[None, None, :]
        ).flatten(1)
        pooled = features.float()[square].mean(dim=1).to(features.dtype)

        values = pooled.float() * self.root_size
        if self.std_bias is not None and self.std_scale is not None:
            values = (values - self.std_bias.float()) * self.std_scale.float()
        return values.to(features.dtype)


class Encoder(nn.Module):
    """Encode the patches of one or more images into pooled soft tokens."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedder = PatchEmbedder(config)
        self.transformer = VisionTransformer(config)
        self.pooler = Pooler(config)

    def forward(
        self,
        pixels: torch.Tensor,
        grids: torch.Tensor,
        grid_shapes: tuple[tuple[int, int], ...] | None,
    ) -> torch.Tensor:
        """Encode packed patch rows or same-sized CHW images.

        The computation reads the grids only as device values, so its shapes
        depend only on the number of patch rows and grids, and a padded
        packing (``PatchEncoder.encode_packed``) replays one captured call.

        Args:
            pixels: ``[total_patches, 3 * patch_size**2]`` patch rows with
                images concatenated in order, or ``[images, 3, height,
                width]`` pixels, which are patchified here.
            grids: ``[images, 2]`` integer ``(height, width)`` of each
                image in patches, on the encoder's device. Every side is a
                multiple of the pooling kernel, image sides lie within the
                position table (a padding segment of a packed batch may run
                past it), and a ``(0, 0)`` grid is an empty segment.
            grid_shapes: The same pairs as host integers, checked against
                the patch rows, or None when the caller has checked the
                packing.

        Returns:
            ``[total_patches / k**2, hidden_size]`` standardized soft tokens,
            images in input order.

        Raises:
            ValueError: If the patch rows do not cover the declared grids, a
                grid side is not a multiple of the pooling kernel, or a side
                exceeds the learned position table.
        """
        config = self.config
        if pixels.ndim == 4:
            pixels = patchify(pixels, patch_size=config.patch_size).flatten(
                0, 1
            )

        counts = (
            None
            if grid_shapes is None
            else tuple(height * width for height, width in grid_shapes)
        )
        if (
            pixels.ndim != 2
            or pixels.shape[1] != 3 * config.patch_size**2
            or pixels.shape[0] % config.pooling_kernel_size**2
            or grids.ndim != 2
            or grids.shape[1] != 2
        ) or (
            grid_shapes is not None
            and (
                pixels.shape[0] != sum(counts)
                or grids.shape[0] != len(counts)
                or any(
                    side % config.pooling_kernel_size
                    or not 0 < side <= config.position_embedding_size
                    for shape in grid_shapes
                    for side in shape
                )
            )
        ):
            raise ValueError(
                "vision patches must cover pooled grids within the position "
                "table"
            )

        # Image rows lie within the position table (checked above, or by the
        # caller of a packed batch). A padding segment of a packed batch may
        # run past it, and its rows read the last entry instead.
        x, y = (
            axis.clamp(max=config.position_embedding_size - 1)
            for axis in build_abs_positions_from_grid_hw(
                grids, total=pixels.shape[0]
            )
        )
        features = self.patch_embedder(pixels, x, y)

        # Each image attends only to its own patches, in both directions.
        values = grids.prod(dim=1).to(torch.int32)
        offsets = torch.cat(
            (values.new_zeros(1), values.cumsum(0, dtype=torch.int32))
        )
        lengths = SequenceLengths(host=counts, values=values, offsets=offsets)
        attention = AttentionBatch.single(
            VarlenInput(lengths, lengths, (False,) * grids.shape[0])
        )
        features = self.transformer(features, x, y, attention)
        return self.pooler(features, grids)


class Embedder(nn.Module):
    """Project soft tokens to the text width after an unweighted RMS norm."""

    def __init__(self, config: VisionConfig, text_hidden_size: int):
        super().__init__()
        self.norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, elementwise_affine=False
        )
        self.projection = Linear(
            config.hidden_size, text_hidden_size, bias=False
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(features))
