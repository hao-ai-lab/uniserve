"""Shared NEO-ViT embedding tower."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from uniserve.nn.rope import RotaryEmbedding, apply_rotary_emb
from uniserve.nn.vision.patching import build_abs_positions_from_grid_hw

__all__ = [
    "NeoVitConfig",
    "NeoVitEncoder",
]


@dataclass(frozen=True)
class NeoVitConfig:
    """Defines NEO-ViT patch, width, head, layer, projection, and position
    settings.
    """  # noqa: D205

    hidden_size: int = 1024
    llm_hidden_size: int = 1024
    downsample_ratio: float = 0.5
    patch_size: int = 16
    num_channels: int = 3
    rope_theta_vision: float = 10000.0

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "llm_hidden_size",
            "patch_size",
            "num_channels",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"NEO-ViT {name} must be a positive integer")
        if self.hidden_size % 4:
            raise ValueError("NEO-ViT hidden_size must be divisible by 4")
        for name in ("downsample_ratio", "rope_theta_vision"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"NEO-ViT {name} must be finite and positive")
        if (
            self.downsample_ratio > 1
            or round(1 / self.downsample_ratio) * self.downsample_ratio != 1
        ):
            raise ValueError(
                "NEO-ViT downsample_ratio must be the reciprocal of an integer"
            )


class NeoVitEncoder(nn.Module):
    """Conv patch + dense downsample NEO vision encoder with 2D RoPE."""

    def __init__(self, cfg: NeoVitConfig) -> None:
        """Build patch projection, positional encoding, downsampling, and
        language-width output.
        """  # noqa: D205
        super().__init__()
        hidden = int(cfg.hidden_size)
        llm_hidden = int(cfg.llm_hidden_size)
        self.downsample_factor = round(1 / cfg.downsample_ratio)
        self.patch_size = int(cfg.patch_size)
        channels = int(cfg.num_channels)
        self.num_channels = channels
        theta = float(cfg.rope_theta_vision)

        self.patch_embedding = nn.Conv2d(
            channels,
            hidden,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.dense_embedding = nn.Conv2d(
            hidden,
            llm_hidden,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
        )
        self.gelu = nn.GELU()

        # Per-axis rotary frequencies from the shared helper, so the inv_freq /
        # cos-sin construction has a single owner. dim is hidden // 2 because
        # the feature width is split into x/y halves.
        self.rope = RotaryEmbedding(dim=hidden // 2, theta=theta)

    def forward(
        self,
        pixels: torch.Tensor,
        grid_hw: torch.Tensor,
        *,
        grid_shapes: tuple[tuple[int, int], ...] | None = None,
    ) -> torch.Tensor:
        """Encode flattened or image-shaped patches with grid-aware rotary
        positions.
        """  # noqa: D205
        if pixels.ndim == 2:
            pixels = pixels.view(
                -1, self.num_channels, self.patch_size, self.patch_size
            )
        if pixels.ndim != 4:
            raise ValueError(
                "NeoVitEncoder expects flattened patches or NCHW patch pixels"
            )

        # ``grid_hw`` carries the (h, w) of each image and drives the 2D RoPE
        # positions and dense downsample; require the (B, 2) shape so the
        # downstream ``[0][0]`` / ``[:, 0]`` indexing is well-defined.
        if grid_hw.ndim != 2 or grid_hw.shape[1] != 2:
            raise ValueError(
                "NeoVitEncoder expects grid_hw of shape (B, 2), got "
                f"{tuple(grid_hw.shape)}"
            )

        patch_embeds = self.gelu(self.patch_embedding(pixels)).view(
            -1, self.patch_embedding.out_channels
        )
        patch_embeds = self._apply_2d_rope(patch_embeds.float(), grid_hw).to(
            dtype=patch_embeds.dtype
        )
        return self._dense_downsample(patch_embeds, grid_shapes=grid_shapes)

    def _apply_2d_rope(
        self, patch_embeds: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor:
        """Apply independent height and width rotary coordinates to patch
        embeddings.
        """  # noqa: D205
        # ``patch_embeds.shape[0]`` is the total patch count as a static tensor
        # shape, so passing it avoids a host sync and keeps this capturable.
        abs_x, abs_y = build_abs_positions_from_grid_hw(
            grid_hw,
            device=patch_embeds.device,
            total=int(patch_embeds.shape[0]),
        )
        half = patch_embeds.shape[-1] // 2
        x_cos, x_sin = self.rope.cos_sin_1d(abs_x)
        y_cos, y_sin = self.rope.cos_sin_1d(abs_y)
        x_part = apply_rotary_emb(
            patch_embeds[..., :half],
            x_cos,
            x_sin,
            rotation="interleaved",
        )
        y_part = apply_rotary_emb(
            patch_embeds[..., half:],
            y_cos,
            y_sin,
            rotation="interleaved",
        )
        return torch.cat([x_part, y_part], dim=-1)

    def _dense_downsample(
        self,
        patch_embeds: torch.Tensor,
        *,
        grid_shapes: tuple[tuple[int, int], ...] | None,
    ) -> torch.Tensor:
        """Merge fixed patch neighborhoods and project them to language
        width.
        """  # noqa: D205
        # Host-known per-image (h, w) grids drive convolution dimensions, so the
        # downsample never reads the device ``grid_hw`` tensor back to the host;
        # ``grid_hw`` remains the device source of truth for 2D RoPE positions.
        if grid_shapes is None:
            raise ValueError(
                "NeoVitEncoder requires host-known grid shapes for dense "
                "downsample"
            )
        shapes = [(int(h), int(w)) for h, w in grid_shapes]
        if not shapes:
            return patch_embeds.new_empty(
                (0, self.dense_embedding.out_channels)
            )

        # Conv2d acts independently per batch element, so when every image
        # shares the same (h, w) grid (the common batched-serving case) the
        # per-image Python loop is equivalent to a single batched conv over
        # (N, C, h, w): it combines their independent image projections in
        # one launch.
        h0, w0 = shapes[0]
        if all(h == h0 and w == w0 for h, w in shapes):
            n = len(shapes)
            if n * h0 * w0 != patch_embeds.shape[0]:
                raise ValueError(
                    "grid shapes do not cover all NEO-ViT patch embeddings"
                )
            image = patch_embeds.view(n, h0, w0, -1).permute(0, 3, 1, 2)
            dense = self.dense_embedding(image).permute(0, 2, 3, 1)
            return dense.reshape(-1, self.dense_embedding.out_channels)

        out = []
        cursor = 0
        for h, w in shapes:
            image = (
                patch_embeds[cursor : cursor + h * w]
                .view(1, h, w, -1)
                .permute(0, 3, 1, 2)
            )
            dense = (
                self.dense_embedding(image)
                .permute(0, 2, 3, 1)
                .reshape(-1, self.dense_embedding.out_channels)
            )
            out.append(dense)
            cursor += h * w
        if cursor != patch_embeds.shape[0]:
            raise ValueError(
                "grid shapes do not cover all NEO-ViT patch embeddings"
            )
        return (
            torch.cat(out, dim=0)
            if out
            else patch_embeds.new_empty((0, self.dense_embedding.out_channels))
        )
