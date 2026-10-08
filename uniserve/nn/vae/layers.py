"""Reusable spatial, causal video, attention and posterior layers."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from uniserve.nn.attention import Attention, AttentionBatch, DenseInput
from uniserve.nn.functional import frame_norm_pad, frame_pad


class ResidualBlock(nn.Module):
    """Apply two normalized convolutions and an optional channel projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norms = nn.ModuleList(
            (
                nn.GroupNorm(groups, in_channels, eps=eps),
                nn.GroupNorm(groups, out_channels, eps=eps),
            )
        )
        self.convolutions = nn.ModuleList(
            (
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
            )
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for norm, convolution in zip(
            self.norms, self.convolutions, strict=True
        ):
            hidden = convolution(F.silu(norm(hidden)))
        return self.shortcut(x) + hidden


class AttentionBlock(nn.Module):
    """Add full spatial attention, with channels forming a single head."""

    def __init__(self, channels: int, *, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, eps=eps)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.attention = Attention(1, 1, channels)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape

        # [batch, 1 head, H*W tokens, channels] per projection.
        query, key, value = (
            tensor.flatten(2).transpose(1, 2).unsqueeze(1).contiguous()
            for tensor in self.qkv(self.norm(x)).chunk(3, dim=1)
        )

        # Every query chunk attends to the full spatial context. This bounds
        # portable score workspace without changing the attention equation.
        result = torch.cat(
            tuple(
                self.attention(
                    chunk,
                    key,
                    value,
                    AttentionBatch.single(DenseInput(False, None)),
                )
                for chunk in query.split(4096, dim=2)
            ),
            dim=2,
        )
        hidden = (
            result.squeeze(1)
            .transpose(1, 2)
            .reshape(batch, channels, height, width)
        )
        return x + self.output(hidden)


class Downsample(nn.Module):
    """Halve the spatial extent with a strided convolution and asymmetric
    pad.
    """  # noqa: D205

    def __init__(self, channels: int):
        super().__init__()
        self.convolution = nn.Conv2d(channels, channels, 3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad bottom/right only so the output extent follows the input parity.
        return self.convolution(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    """Double the spatial extent with nearest interpolation and a
    convolution.
    """  # noqa: D205

    def __init__(self, channels: int):
        super().__init__()
        self.convolution = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convolution(
            F.interpolate(x, scale_factor=2.0, mode="nearest")
        )


class CausalConv3d(nn.Conv3d):
    """Convolve NCTHW values with symmetric spatial, causal temporal padding.

    ``spatial_padding`` pixels pad both sides of height and width in
    ``spatial_padding_mode``. ``temporal_padding`` zero frames precede the
    input and none follow it, so no output frame reads a later input frame.
    ``forward`` pads (``uniserve.nn.functional.frame_pad``) into
    ``memory_format`` storage and convolves; ``convolve`` takes input already
    padded by ``padding_extents``. The storage order changes only how the
    convolution reads its input: a channels-last input reaches convolution
    kernels that compute channels-last without a transposition, and the
    result follows that order.

    ``forward(values, norm=norm)`` convolves the pre-activation
    ``silu(norm(values))`` of a ``FrameGroupNorm``, which
    ``uniserve.nn.functional.frame_norm_pad`` normalizes, activates and pads
    together, bit for bit as the three operations compute it.

    ``add_bias=False`` returns the convolution without its bias, for a
    caller that adds it with ``uniserve.nn.functional.bias_add``. PyTorch
    adds the bias of a cuDNN convolution in a separate pass after it, so the
    sum equals the biased convolution bit for bit, and ``bias_add`` also
    stores a channels-last result channel-first and adds a residual in that
    pass.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
        memory_format: torch.memory_format = torch.contiguous_format,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=0
        )
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode
        self.memory_format = memory_format
        # The weight is stored in the input's order, which the convolution
        # would otherwise convert it to on every call.
        self.weight = nn.Parameter(
            self.weight.detach().contiguous(memory_format=memory_format)
        )

    @property
    def padding_extents(self) -> tuple[int, int, int, int, int]:
        """``(left, right, top, bottom, front)`` padding of the input."""
        extent = self.spatial_padding
        return (extent, extent, extent, extent, self.temporal_padding)

    def convolve(
        self, padded: torch.Tensor, *, add_bias: bool = True
    ) -> torch.Tensor:
        """Convolve input that already carries this convolution's padding."""
        bias = self.bias if add_bias else None
        return F.conv3d(padded, self.weight, bias, stride=self.stride)

    def forward(
        self,
        values: torch.Tensor,
        norm: "FrameGroupNorm | None" = None,
        *,
        add_bias: bool = True,
    ) -> torch.Tensor:
        if norm is not None:
            return self.convolve(
                frame_norm_pad(
                    values,
                    self.padding_extents,
                    groups=norm.num_groups,
                    weight=norm.weight,
                    bias=norm.bias,
                    eps=norm.eps,
                    mode=self.spatial_padding_mode,
                    memory_format=self.memory_format,
                ),
                add_bias=add_bias,
            )
        if self.spatial_padding or self.temporal_padding:
            values = frame_pad(
                values,
                self.padding_extents,
                mode=self.spatial_padding_mode,
                memory_format=self.memory_format,
            )
        return self.convolve(values, add_bias=add_bias)


class FrameGroupNorm(nn.GroupNorm):
    """Group-normalize every NCTHW frame on its own; frames never mix.

    Returns the normalized values as a permuted view of the frame-folded
    result; the causal convolution's padding (``frame_pad``) reads that view
    directly instead of a contiguous copy.
    """

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = values.shape
        # Fold time into the batch: [batch * frames, channels, height, width].
        folded = values.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        normalized = super().forward(folded)
        return normalized.view(batch, frames, channels, height, width).permute(
            0, 2, 1, 3, 4
        )


class DiagonalGaussian(nn.Module):
    """Split mean/log-variance moments and optionally sample the posterior.

    ``log_variance_range`` clamps the log-variance before it scales the
    noise, as latent diffusion codecs bound it against overflow; None leaves
    it unbounded. A sample draws standard normal noise from ``generator`` on
    the moments' device unless the caller supplies ``noise``, a draw with the
    mean's shape, dtype and device. A caller that reproduces a reference
    draw, such as one fixed-seed host draw over a longer timeline than these
    moments cover, supplies its share of that draw.
    """

    def __init__(
        self,
        sample: bool = True,
        chunk_dim: int = 1,
        log_variance_range: tuple[float, float] | None = None,
    ):
        super().__init__()
        if log_variance_range is not None and (
            len(log_variance_range) != 2
            or not all(math.isfinite(bound) for bound in log_variance_range)
            or log_variance_range[0] >= log_variance_range[1]
        ):
            raise ValueError(
                "posterior log-variance range must be a finite increasing "
                "interval"
            )
        self.sample, self.chunk_dim = sample, chunk_dim
        self.log_variance_range = log_variance_range

    def forward(
        self,
        moments: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if moments.shape[self.chunk_dim] % 2:
            raise ValueError(
                "posterior moments require equally sized mean and "
                "log-variance fields"
            )

        mean, log_variance = moments.chunk(2, dim=self.chunk_dim)
        if not self.sample:
            if noise is not None:
                raise ValueError("the posterior mean takes no noise")
            return mean

        if self.log_variance_range is not None:
            log_variance = log_variance.clamp(*self.log_variance_range)
        if noise is None:
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
        elif (
            noise.shape != mean.shape
            or noise.dtype != mean.dtype
            or noise.device != mean.device
        ):
            raise ValueError(
                "posterior noise must match the mean's shape, dtype and device"
            )
        return mean + torch.exp(0.5 * log_variance) * noise
