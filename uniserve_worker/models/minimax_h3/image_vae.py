# SPDX-License-Identifier: Apache-2.0
"""Checkpoint causal CNN for H3 single-frame reference posteriors.

Parameter paths match the released video encoder. Temporal left padding remains
zero even for T=1; repeating the image would produce a different posterior.
"""

from torch import nn
from torch.nn import functional as F


class CausalConv3d(nn.Conv3d):
    """Reflect spatial boundaries and zero-pad only the temporal past."""

    def __init__(self, inputs, outputs, kernel_size=3, *, spatial_padding=1, stride=1):
        super().__init__(inputs, outputs, kernel_size, stride=stride)
        self.spatial_padding = spatial_padding
        self.temporal_padding = kernel_size - 1

    def forward(self, value):
        p = self.spatial_padding
        if p:
            value = F.pad(value, (p, p, p, p, 0, 0), mode="reflect")
        if self.temporal_padding:
            value = F.pad(value, (0, 0, 0, 0, self.temporal_padding, 0))
        return F.conv3d(value, self.weight, self.bias, stride=self.stride)


class FrameGroupNorm(nn.GroupNorm):
    """Normalize each frame independently, without temporal leakage."""

    def forward(self, value):
        batch, channels, frames, height, width = value.shape
        flat = value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, 1, height, width)
        flat = super().forward(flat)
        return (
            flat.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        )


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.norm1 = FrameGroupNorm(32, inputs, eps=1e-6)
        self.conv1 = CausalConv3d(inputs, outputs)
        self.norm2 = FrameGroupNorm(32, outputs, eps=1e-6)
        self.conv2 = CausalConv3d(outputs, outputs)
        self.conv_shortcut = (
            CausalConv3d(inputs, outputs, 1, spatial_padding=0) if inputs != outputs else None
        )

    def forward(self, value):
        residual = value if self.conv_shortcut is None else self.conv_shortcut(value)
        value = self.conv1(F.silu(self.norm1(value)))
        return residual + self.conv2(F.silu(self.norm2(value)))


class Downsample(nn.Module):
    def __init__(self, channels, temporal_stride, spatial_stride):
        super().__init__()
        self.spatial_stride = spatial_stride
        self.conv = CausalConv3d(
            channels,
            channels,
            spatial_padding=0,
            stride=(temporal_stride, spatial_stride, spatial_stride),
        )

    def forward(self, value):
        if self.spatial_stride == 2:
            value = F.pad(value, (0, 1, 0, 1, 0, 0), mode="reflect")
        return self.conv(value)


class DownBlock(nn.Module):
    def __init__(self, inputs, outputs, temporal_stride, spatial_stride):
        super().__init__()
        self.resnets = nn.ModuleList(
            (ResidualBlock(inputs, outputs), ResidualBlock(outputs, outputs))
        )
        self.downsamplers = (
            nn.ModuleList((Downsample(outputs, temporal_stride, spatial_stride),))
            if temporal_stride * spatial_stride > 1
            else None
        )

    def forward(self, value):
        for block in self.resnets:
            value = block(value)
        if self.downsamplers is not None:
            for block in self.downsamplers:
                value = block(value)
        return value


class H3ImageEncoder(nn.Module):
    """Released six-stage causal encoder producing 48 posterior channels."""

    def __init__(self):
        super().__init__()
        widths = (128, 256, 256, 512, 512, 1024)
        spatial = (2, 2, 2, 2, 1, 1)
        temporal = (1, 2, 2, 1, 1, 1)
        self.conv_in = CausalConv3d(3, widths[0])
        self.down_blocks = nn.ModuleList(
            DownBlock(inputs, outputs, t, s)
            for inputs, outputs, t, s in zip(
                (widths[0], *widths[:-1]), widths, temporal, spatial, strict=True
            )
        )
        self.norm_out = FrameGroupNorm(32, widths[-1], eps=1e-6)
        self.conv_out = CausalConv3d(widths[-1], 48)

    def forward(self, pixels):
        if pixels.ndim != 5 or pixels.shape[:3] != (1, 3, 1):
            raise ValueError("H3 image encoder requires [1, 3, 1, H, W]")
        value = self.conv_in(pixels)
        for block in self.down_blocks:
            value = block(value)
        return self.conv_out(F.silu(self.norm_out(value)))
