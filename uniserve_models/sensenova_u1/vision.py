"""SenseNova's NEO patch projection, axial RoPE and dense spatial reduction."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from uniserve.nn.functional import apply_rotary
from uniserve.nn.rope import RotaryEmbedding
from uniserve.nn.vision.position import build_abs_positions_from_grid_hw


@dataclass(frozen=True)
class Config:
    hidden_size: int
    output_size: int
    downsample_ratio: float
    patch_size: int
    num_channels: int
    rope_theta: float

    def __post_init__(self):
        if (
            any(
                type(value) is not int or value < 1
                for value in (
                    self.hidden_size,
                    self.output_size,
                    self.patch_size,
                    self.num_channels,
                )
            )
            or self.hidden_size % 4
        ):
            raise ValueError(
                "NEO patch dimensions must be positive and its rotary "
                "width divisible by four"
            )
        if (
            not math.isfinite(self.downsample_ratio)
            or not 0 < self.downsample_ratio <= 1
            or round(1 / self.downsample_ratio) * self.downsample_ratio != 1
        ):
            raise ValueError(
                "NEO downsampling must be the reciprocal of a positive integer"
            )
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("NEO rotary theta must be finite and positive")


class Encoder(nn.Module):
    """Encode NCHW pixels or flattened CHW patches with explicit image grids.

    Columns rotate the first channel half and rows rotate the second, using
    interleaved pairs in FP32 before returning to the patch convolution dtype.
    Spatial reduction never crosses an image boundary.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        factor = round(1 / config.downsample_ratio)
        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
        )
        self.dense_embedding = nn.Conv2d(
            config.hidden_size, config.output_size, factor, stride=factor
        )
        self.rotary = RotaryEmbedding(
            config.hidden_size // 2, theta=config.rope_theta
        )
        self.activation = nn.GELU()

    def forward(
        self,
        pixels: torch.Tensor,
        grids: torch.Tensor,
        grid_shapes: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        counts = tuple(height * width for height, width in grid_shapes)
        if grids.shape != (len(counts), 2) or grids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "NEO grids require one integer height/width pair per image"
            )
        factor = round(1 / self.config.downsample_ratio)
        if any(
            min(shape) < 1 or any(axis % factor for axis in shape)
            for shape in grid_shapes
        ):
            raise ValueError(
                "NEO image grids must align with dense spatial downsampling"
            )

        if pixels.ndim == 2:
            # Packed patches arrive as [total_patches, channels*patch*patch].
            if pixels.shape != (
                sum(counts),
                self.config.num_channels * self.config.patch_size**2,
            ):
                raise ValueError(
                    "NEO packed CHW pixels must cover the declared image grids"
                )
            pixels = pixels.reshape(
                -1,
                self.config.num_channels,
                self.config.patch_size,
                self.config.patch_size,
            )
        if pixels.ndim != 4:
            raise ValueError(
                "NEO pixels must be NCHW images or flattened CHW patches"
            )

        features = self.activation(
            self.patch_embedding(pixels.to(self.patch_embedding.weight.dtype))
        )
        # [patches, hidden, h, w] -> [total_patches, hidden] across all images.
        features = features.permute(0, 2, 3, 1).reshape(
            -1, self.config.hidden_size
        )
        if features.shape[0] != sum(counts):
            raise ValueError("NEO image grids must cover all projected patches")

        columns, rows = build_abs_positions_from_grid_hw(
            grids, total=sum(counts)
        )
        parts = []
        for hidden, coordinates, axis in zip(
            features.float().chunk(2, dim=-1),
            (columns, rows),
            (1, 0),
            strict=True,
        ):
            cosine, sine = self.rotary(
                coordinates,
                dtype=torch.float32,
                sequence_length=max(
                    (shape[axis] for shape in grid_shapes), default=0
                ),
            )
            parts.append(
                apply_rotary(
                    hidden.unsqueeze(1), cosine, sine, rotation="interleaved"
                ).squeeze(1)
            )
        features = torch.cat(parts, dim=-1).to(features.dtype)

        if not grid_shapes:
            return features.new_empty((0, self.config.output_size))

        # Equal grids reduce in one batched convolution; mixed grids loop.
        if all(shape == grid_shapes[0] for shape in grid_shapes):
            height, width = grid_shapes[0]
            spatial = features.reshape(
                len(grid_shapes), height, width, -1
            ).permute(0, 3, 1, 2)
            return (
                self.dense_embedding(spatial)
                .permute(0, 2, 3, 1)
                .reshape(-1, self.config.output_size)
            )
        outputs = []
        for values, (height, width) in zip(
            features.split(counts), grid_shapes, strict=True
        ):
            spatial = values.reshape(1, height, width, -1).permute(0, 3, 1, 2)
            outputs.append(
                self.dense_embedding(spatial)
                .permute(0, 2, 3, 1)
                .reshape(-1, self.config.output_size)
            )
        return torch.cat(outputs)
