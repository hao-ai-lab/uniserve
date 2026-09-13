"""Shared image diffusion geometry, numerical transforms, and prompt framing."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Literal

import torch

from uniserve_worker.modeling.tensors import FlowPatches, PositionLayout

from ..foundation.errors import invalid_descriptor, unsupported_setup
from ..nn.diffusion.cfg import Branch, CfgRecipe
from ..nn.diffusion.schedule import (
    ScheduleDirection,
    ScheduleShiftDomain,
    flow_match_coordinate,
)
from ..nn.vision.patching import patchify_batch, unpatchify_batch


class LatentLayout(StrEnum):
    """Selects patch-token or image-tensor representation at the generation boundary."""

    PATCH_TOKENS = "patch_tokens"
    IMAGE_NCHW = "image_nchw"


class BranchSource(StrEnum):
    """Selects conditioning, negative conditioning, or start-state input for a guidance branch."""

    CONDITIONING = "conditioning"
    NEGATIVE_OR_START = "negative_or_start"
    START = "start"


class NoiseScaleMode(StrEnum):
    """Selects constant, resolution-aware, or dynamic latent-noise scaling."""

    CONSTANT = "constant"
    RESOLUTION = "resolution"
    DYNAMIC = "dynamic"
    DYNAMIC_SQRT = "dynamic_sqrt"


class FlowPrompt:
    """Model-owned framing that produces one classifier-free-guidance prefix."""

    def __init__(
        self,
        *,
        user_prefix: str,
        user_suffix: str,
        assistant_suffix: str,
        conditioned_append: str,
        unconditional_append: str,
        system_prefix: str = "",
        system_message: str = "",
        system_suffix: str = "",
        add_special_tokens: bool = True,
    ) -> None:
        """Store model-specific chat framing for conditioned and unconditional flow branches."""

        self.user_prefix = user_prefix
        self.user_suffix = user_suffix
        self.assistant_suffix = assistant_suffix
        self.conditioned_append = conditioned_append
        self.unconditional_append = unconditional_append
        self.system_prefix = system_prefix
        self.system_message = system_message
        self.system_suffix = system_suffix
        self.add_special_tokens = bool(add_special_tokens)

    def encode(self, tokenizer: Any, *, text: str, conditioned: bool) -> tuple[int, ...]:
        """Frame and tokenize either the conditioned or unconditional diffusion prompt."""

        if tokenizer is None:
            raise unsupported_setup("the configured generation prompt requires a tokenizer")
        append = self.conditioned_append if conditioned else self.unconditional_append
        framed = (
            self.system_prefix
            + self.system_message
            + self.system_suffix
            + self.user_prefix
            + text
            + self.user_suffix
            + self.assistant_suffix
            + append
        )
        return tuple(
            int(value)
            for value in tokenizer.encode(framed, add_special_tokens=self.add_special_tokens)
        )


class ImageDiffusion:
    """Image latent geometry, noise transforms, and analytical diffusion mathematics."""

    def __init__(
        self,
        *,
        latent_downsample: int,
        prediction: Literal["velocity", "sample"],
        prediction_dtype: str,
        schedule_direction: ScheduleDirection,
        schedule_shift_domain: ScheduleShiftDomain,
        max_latent_tokens: int,
        max_vae_grid_tokens: int,
        marker_tokens: int,
        rope_advance: int,
        max_cfg_branches: int,
        latent_layout: LatentLayout,
        latent_channels: int,
        latent_patch_size: int,
        positions: PositionLayout,
        text_unconditional: BranchSource,
        image_unconditional: BranchSource,
        cfg_recipe: CfgRecipe,
        noise_scale: float = 1.0,
        noise_scale_mode: NoiseScaleMode = NoiseScaleMode.CONSTANT,
        noise_scale_base_tokens: float = 1.0,
        noise_scale_maximum: float = 1.0,
        timestep_shift: float | None = None,
        prompt: FlowPrompt | None = None,
    ) -> None:
        """Validate and freeze latent geometry, flow math, CFG, and prompt semantics."""

        # These bounds describe one legal numerical image and its framing,
        # independently of how many trajectories the caller admits or stores.
        self.latent_downsample = int(latent_downsample)
        self.prediction = prediction
        self.prediction_dtype = str(prediction_dtype)
        self.schedule_direction = schedule_direction
        self.schedule_shift_domain = schedule_shift_domain
        self.max_latent_tokens = int(max_latent_tokens)
        self.max_vae_grid_tokens = int(max_vae_grid_tokens)
        self.marker_tokens = int(marker_tokens)
        self.rope_advance = int(rope_advance)
        self.max_cfg_branches = int(max_cfg_branches)
        self.latent_layout = latent_layout
        self.latent_channels = int(latent_channels)
        self.latent_patch_size = int(latent_patch_size)
        self.positions = positions
        self.text_unconditional = text_unconditional
        self.image_unconditional = image_unconditional
        self.cfg_recipe = cfg_recipe
        self.noise_scale_value = float(noise_scale)
        self.noise_scale_mode = noise_scale_mode
        self.noise_scale_base_tokens = float(noise_scale_base_tokens)
        self.noise_scale_maximum = float(noise_scale_maximum)
        self.timestep_shift = None if timestep_shift is None else float(timestep_shift)
        self.prompt = prompt

        # Reject geometry that cannot describe a latent, framing marker, or
        # mathematical guidance branch before it is consumed by public layers.
        if self.prediction not in {"velocity", "sample"}:
            raise ValueError("image diffusion prediction must be velocity or sample")
        if min(
            self.latent_downsample,
            self.max_latent_tokens,
            self.max_vae_grid_tokens,
            self.marker_tokens,
            self.rope_advance,
            self.max_cfg_branches,
            self.latent_channels,
            self.latent_patch_size,
        ) < 1 or self.max_cfg_branches > len(Branch):
            raise invalid_descriptor("generation geometry exceeds the concrete model bounds")

    def schedule_pair(
        self,
        steps: int,
        requested_shift: float,
        index: int,
    ) -> tuple[float, float]:
        """Return one immutable analytical schedule pair for fixed tensor staging."""

        shift = float(requested_shift if requested_shift > 0 else self.timestep_shift or 1.0)
        current = flow_match_coordinate(
            int(steps),
            shift,
            self.schedule_direction,
            self.schedule_shift_domain,
            int(index),
        )
        following = flow_match_coordinate(
            int(steps),
            shift,
            self.schedule_direction,
            self.schedule_shift_domain,
            int(index) + 1,
        )
        return current, following

    def branch_source(self, branch: Branch) -> BranchSource:
        """Map a classifier-free-guidance branch to its configured conditioning source."""

        if branch is Branch.COND:
            return BranchSource.CONDITIONING
        if branch is Branch.TEXT_UNCOND:
            return self.text_unconditional
        return self.image_unconditional

    def image_tokens(self, height: int, width: int) -> int:
        """Count latent-grid tokens for an output image at the model downsample ratio."""

        return (int(height) // self.latent_downsample) * (int(width) // self.latent_downsample)

    def sequence_length(self, height: int, width: int) -> int:
        """Count one numerical sequence, including its model framing markers."""

        count = self.image_tokens(height, width)
        return (
            count + self.marker_tokens if self.latent_layout is LatentLayout.PATCH_TOKENS else count
        )

    def latent_shape(self, height: int, width: int) -> tuple[int, ...]:
        """Describe the stored latent tensor for the configured image or patch layout."""

        if self.latent_layout is LatentLayout.PATCH_TOKENS:
            width_per_token = self.latent_patch_size**2 * self.latent_channels
            return (self.image_tokens(height, width), width_per_token)
        return (1, self.latent_channels, int(height), int(width))

    def noise_scale(self, height: int, width: int) -> float:
        """Scale initial noise by latent resolution and clamp it to the model limit."""

        value = self.noise_scale_value
        if self.noise_scale_mode in {
            NoiseScaleMode.RESOLUTION,
            NoiseScaleMode.DYNAMIC,
            NoiseScaleMode.DYNAMIC_SQRT,
        }:
            value *= math.sqrt(self.image_tokens(height, width) / self.noise_scale_base_tokens)
        if self.noise_scale_mode is NoiseScaleMode.DYNAMIC_SQRT:
            value = math.sqrt(value)
        return min(value, self.noise_scale_maximum)

    def patchify(self, latent: torch.Tensor) -> torch.Tensor:
        """Convert image-layout latent storage into patch rows consumed by the denoiser."""

        if self.latent_layout is LatentLayout.PATCH_TOKENS:
            return latent
        return patchify_batch(latent, self.latent_patch_size)

    def unpatchify(self, latent: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Restore one image's canonical patch rows to its numerical latent layout.

        Patch-token models retain their row representation. Image-space models
        return [1, channels, height, width], preserving raster patch order.
        """

        if self.latent_layout is LatentLayout.PATCH_TOKENS:
            return latent
        return unpatchify_batch(
            latent.reshape(1, self.image_tokens(height, width), -1),
            self.latent_patch_size,
            height=int(height),
            width=int(width),
            channels=self.latent_channels,
        )

    def conditioning(
        self,
        latent: torch.Tensor,
        height: int,
        width: int,
        *,
        patch_size: int | None,
    ) -> FlowPatches | None:
        """Build image-conditioning patches and their grid/noise metadata when required."""

        if self.latent_layout is LatentLayout.PATCH_TOKENS:
            return None
        if patch_size is None:
            raise invalid_descriptor("image-space generation requires an image patch processor")
        pixels = patchify_batch(latent, patch_size, channel_first=True).reshape(
            -1,
            patch_size * patch_size * int(latent.shape[1]),
        )
        grid = torch.tensor(
            [[int(height) // patch_size, int(width) // patch_size]],
            dtype=torch.long,
            device=latent.device,
        )
        return FlowPatches(
            pixels=pixels,
            grid=grid,
            noise_scale=latent.new_tensor([self.noise_scale(height, width)]),
        )


__all__ = [
    "BranchSource",
    "FlowPrompt",
    "ImageDiffusion",
    "LatentLayout",
    "NoiseScaleMode",
]
