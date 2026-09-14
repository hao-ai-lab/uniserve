"""Shared image diffusion sizes, numerical transforms, and solver coordinates."""

from __future__ import annotations

import math
from enum import StrEnum

import torch

from uniserve.model.tensors import FlowPatches, PositionLayout
from uniserve.nn.diffusion.cfg import Branch, CfgPlan, CfgRecipe, build_flow_cfg_plan
from uniserve.nn.diffusion.config import DiffusionConfig
from uniserve.nn.diffusion.schedule import (
    DiffusionSchedule,
    ScheduleDirection,
    ScheduleShiftDomain,
    flow_match_coordinate,
)
from uniserve.nn.vision.patching import patchify_batch, unpatchify_batch


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


class ImageDiffusion:
    """Image latent geometry, noise transforms, and analytical diffusion mathematics."""

    def __init__(
        self,
        *,
        latent_downsample: int,
        prediction_dtype: torch.dtype,
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
    ) -> None:
        """Validate and freeze latent sizes, flow math, and CFG semantics."""

        # These bounds describe one legal numerical image and its framing,
        # independently of how many trajectories the caller admits or stores.
        self.latent_downsample = int(latent_downsample)
        self.prediction_dtype = prediction_dtype
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

        # Reject geometry that cannot describe a latent, framing marker, or
        # mathematical guidance branch before it is consumed by public layers.
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
            raise ValueError("generation geometry exceeds the concrete model bounds")

    def timestep(self, config: DiffusionConfig, index: int) -> float:
        """Return an analytical coordinate before its numerical FP32 cast."""

        shift = config.timestep_shift
        if shift is None:
            shift = self.timestep_shift or 1.0
        return flow_match_coordinate(
            config.steps, shift, self.schedule_direction, self.schedule_shift_domain, index
        )

    def create_schedule(
        self, config: DiffusionConfig, *, device: torch.device | str
    ) -> DiffusionSchedule:
        """Materialize all image endpoints once in their numerical representation."""

        times = torch.tensor(
            tuple(self.timestep(config, index) for index in range(config.steps + 1)),
            dtype=torch.float32,
            device=device,
        )
        sigmas = times if self.schedule_direction is ScheduleDirection.DESCENDING else 1.0 - times
        return DiffusionSchedule((sigmas,), (times,))

    def guidance(self, config: DiffusionConfig, index: int) -> CfgPlan:
        """Select CFG branches using the analytical, unrounded timestep.

        Rounding the interval comparison to FP32 can change whether a boundary
        evaluation uses guidance, even though its network time is staged in FP32.
        """

        if not 0 <= index < config.steps:
            raise IndexError(index)
        time = self.timestep(config, index)
        plan = build_flow_cfg_plan(
            cfg_text_scale=config.cfg_text_scale,
            cfg_img_scale=config.cfg_img_scale,
            recipe=self.cfg_recipe,
            renorm=config.cfg_renorm,
            renorm_min=config.cfg_renorm_min,
            use_cfg=config.cfg_interval[0] <= time <= config.cfg_interval[1],
        )
        if len(plan.branches) > self.max_cfg_branches:
            raise ValueError("guidance exceeds the model branch bound")
        return plan

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
            raise ValueError("image-space generation requires an image patch processor")
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
    "ImageDiffusion",
    "LatentLayout",
    "NoiseScaleMode",
]
