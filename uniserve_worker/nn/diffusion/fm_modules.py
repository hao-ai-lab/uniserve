"""Flow-matching heads and pixel decoders used by served image models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from ..layer import LayerConfig
from ..linear import LinearBase
from .timestep import TimestepEmbedder

__all__ = [
    'FlowHeadConfig',
    'modulate',
    'ResBlock',
    'FlowMatchingHead',
    'FinalLayer',
    'ConvDecoder',
]


@dataclass(frozen=True)
class FlowHeadConfig:
    """Geometry of the time-conditioned flow-matching MLP head.

    The single source for the head defaults shared by ``_TimeConditionedMLPAdaLN``
    and its public ``FlowMatchingHead`` wrapper.
    """

    dim: int = 1536
    layers: int = 12
    mlp_ratio: float = 1.0


_DEFAULT_FLOW_HEAD = FlowHeadConfig()

# PixelShuffle(2) used twice before the final upscale; each folds 2x2=4 channels
# into spatial resolution, so the preceding conv's input channel count must be
# divisible by this.
_INTERMEDIATE_UPSCALE = 2
_INTERMEDIATE_CHANNEL_DIVISOR = _INTERMEDIATE_UPSCALE**2


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor | None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply optional adaptive scale and shift terms to normalized activations."""

    scaled = x if scale is None else x * (1 + scale)
    return scaled if shift is None else scaled + shift


class ResBlock(nn.Module):
    """Applies time-conditioned residual convolutions with optional spatial resampling."""

    def __init__(self, channels: int, *, layer_config: LayerConfig, mlp_ratio: float = 1.0):
        """Build a time-modulated residual MLP at a fixed channel width."""

        super().__init__()
        self.channels = int(channels)
        self.intermediate_size = int(channels * mlp_ratio)
        self.in_ln = nn.LayerNorm(self.channels, eps=1e-6)
        self.mlp = nn.Sequential(
            LinearBase(self.channels, self.intermediate_size, layer_config=layer_config),
            nn.SiLU(),
            LinearBase(self.intermediate_size, self.channels, layer_config=layer_config),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            LinearBase(self.channels, 3 * self.channels, layer_config=layer_config, bias=True),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply a timestep-modulated MLP update while preserving the residual stream."""

        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = self.mlp(modulate(self.in_ln(x), shift_mlp, scale_mlp))
        return x + gate_mlp * h


class _TimeAdaptiveFinalLayer(nn.Module):
    """Applies timestep-conditioned normalization and the final flow projection."""

    def __init__(self, model_channels: int, out_channels: int, *, layer_config: LayerConfig):
        """Build adaptive normalization and the terminal flow projection."""

        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = LinearBase(model_channels, out_channels, layer_config=layer_config, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            LinearBase(model_channels, 2 * model_channels, layer_config=layer_config, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Normalize hidden rows, apply timestep affine parameters, and project flow values."""

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class _TimeConditionedMLPAdaLN(nn.Module):
    """Applies adaptive layer normalization and a gated MLP conditioned on timestep embeddings."""

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        *,
        layer_config: LayerConfig,
        dim: int = _DEFAULT_FLOW_HEAD.dim,
        layers: int = _DEFAULT_FLOW_HEAD.layers,
        mlp_ratio: float = _DEFAULT_FLOW_HEAD.mlp_ratio,
        init_weights: bool = False,
    ):
        """Build the input projection, adaptive residual stack, and flow output layer."""

        super().__init__()
        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        self.dim = int(dim)
        self.layers = int(layers)
        self.mlp_ratio = float(mlp_ratio)

        self.time_embed = TimestepEmbedder(self.dim)
        self.input_proj = LinearBase(self.input_dim, self.dim, layer_config=layer_config)
        self.res_blocks = nn.ModuleList(
            [
                ResBlock(self.dim, layer_config=layer_config, mlp_ratio=self.mlp_ratio)
                for _ in range(self.layers)
            ]
        )
        self.final_layer = _TimeAdaptiveFinalLayer(self.dim, self.out_dim, layer_config=layer_config)
        # Random init is wasted on the serving path (the checkpoint overwrites it);
        # gate it off by default so construction is cheap and never-loaded weights
        # surface as garbage rather than a plausible random init.
        if init_weights:
            self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize dense layers and zero residual gates for stable flow-head startup."""

        def _basic_init(module: nn.Module) -> None:
            """Initialize linear weights with Xavier scaling and clear their biases."""

            if isinstance(module, (nn.Linear, LinearBase)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        time_input = cast(nn.Linear, self.time_embed.mlp[0])
        time_output = cast(nn.Linear, self.time_embed.mlp[2])
        nn.init.normal_(time_input.weight, std=0.02)
        nn.init.normal_(time_output.weight, std=0.02)

        for block_module in self.res_blocks:
            block = cast(ResBlock, block_module)
            modulation = cast(LinearBase, block.adaLN_modulation[-1])
            nn.init.constant_(modulation.weight, 0)
            if modulation.bias is not None:
                nn.init.constant_(modulation.bias, 0)
        final_modulation = cast(LinearBase, self.final_layer.adaLN_modulation[-1])
        nn.init.constant_(final_modulation.weight, 0)
        if final_modulation.bias is not None:
            nn.init.constant_(final_modulation.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        if self.final_layer.linear.bias is not None:
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Project input rows, apply time-conditioned residual blocks, and emit patch flow."""

        h = self.input_proj(x)
        c = self.time_embed(t.to(device=x.device))
        for block in self.res_blocks:
            h = block(h, c)
        return self.final_layer(h, c)


class FlowMatchingHead(nn.Module):
    """Predicts patch-space flow values from image tokens and timestep conditioning."""

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        *,
        layer_config: LayerConfig,
        dim: int = _DEFAULT_FLOW_HEAD.dim,
        layers: int = _DEFAULT_FLOW_HEAD.layers,
        mlp_ratio: float = _DEFAULT_FLOW_HEAD.mlp_ratio,
        init_weights: bool = False,
    ):
        """Build the timestep-conditioned patch-flow prediction network."""

        super().__init__()
        self.net = _TimeConditionedMLPAdaLN(
            input_dim=input_dim,
            out_dim=out_dim,
            layer_config=layer_config,
            dim=dim,
            layers=layers,
            mlp_ratio=mlp_ratio,
            init_weights=init_weights,
        )

    @property
    def dtype(self):
        """Expose the activation dtype expected by the flow head's input projection."""

        return self.net.input_proj.weight.dtype

    @property
    def device(self):
        """Expose the device that owns the flow head's input projection."""

        return self.net.input_proj.weight.device

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict patch-space flow values for rows paired with diffusion timesteps."""

        return self.net(x, t)


class FinalLayer(nn.Module):
    """Untimed DiT-style final projection used by patch-space decoders."""

    def __init__(self, model_channels: int, out_channels: int, *, layer_config: LayerConfig):
        """Build untimed normalization and the final patch-space projection."""

        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = LinearBase(model_channels, out_channels, layer_config=layer_config, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize decoder features and project them to output channels."""

        return self.linear(self.norm_final(x))


class ConvDecoder(nn.Module):
    """Reconstructs RGB images from spatial latent feature maps."""

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 1024,
        *,
        out_channels: int = 3,
        final_upscale: int = 8,
    ):
        """Build the convolutional upsampling stack for RGB reconstruction."""

        super().__init__()
        self.out_channels = int(out_channels)
        self.final_upscale = int(final_upscale)
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        # Each intermediate PixelShuffle(_INTERMEDIATE_UPSCALE) divides the channel
        # count by _INTERMEDIATE_CHANNEL_DIVISOR, so the preceding conv input dims
        # must be exactly divisible.
        if input_dim % _INTERMEDIATE_CHANNEL_DIVISOR != 0:
            raise ValueError(
                f"ConvDecoder input_dim={input_dim} must be divisible by "
                f"{_INTERMEDIATE_CHANNEL_DIVISOR} (PixelShuffle({_INTERMEDIATE_UPSCALE}))"
            )
        if hidden_dim % _INTERMEDIATE_CHANNEL_DIVISOR != 0:
            raise ValueError(
                f"ConvDecoder hidden_dim={hidden_dim} must be divisible by "
                f"{_INTERMEDIATE_CHANNEL_DIVISOR} (PixelShuffle({_INTERMEDIATE_UPSCALE}))"
            )
        # conv2 emits out_channels * final_upscale**2 planes so the final
        # PixelShuffle(final_upscale) folds them back into out_channels.
        conv2_out = self.out_channels * self.final_upscale**2
        self.ps1 = nn.PixelShuffle(_INTERMEDIATE_UPSCALE)
        self.conv1 = nn.Conv2d(input_dim // _INTERMEDIATE_CHANNEL_DIVISOR, hidden_dim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.ps2 = nn.PixelShuffle(_INTERMEDIATE_UPSCALE)
        self.conv2 = nn.Conv2d(hidden_dim // _INTERMEDIATE_CHANNEL_DIVISOR, conv2_out, kernel_size=3, padding=1)
        self.ps3 = nn.PixelShuffle(self.final_upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upscale latent feature maps through pixel shuffles and reconstruct RGB channels."""

        x = self.act1(self.conv1(self.ps1(x)))
        return self.ps3(self.conv2(self.ps2(x)))
