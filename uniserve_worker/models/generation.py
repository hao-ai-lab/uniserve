"""Imperative cross-modal generation behavior owned by concrete models."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

import torch

from ..execution.forward_batch import FlowPatches
from ..foundation.errors import unsupported_setup, invalid_descriptor
from ..nn.diffusion.cfg import Branch, CfgRecipe
from ..nn.diffusion.schedule import (
    ScheduleDirection,
    ScheduleShiftDomain,
    flow_match_coordinate,
)
from ..nn.vision.patching import patchify_batch, unpatchify_batch
from .runtime import PositionLayout


class LatentLayout(StrEnum):
    PATCH_TOKENS = "patch_tokens"
    IMAGE_NCHW = "image_nchw"


class BranchSource(StrEnum):
    CONDITIONING = "conditioning"
    NEGATIVE_OR_START = "negative_or_start"
    START = "start"


class Materialization(StrEnum):
    DECODE_ROUTE = "decode_route"
    RGB_LATENT = "rgb_latent"


class NoiseScaleMode(StrEnum):
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


class GenerationPipeline:
    """Shared mechanics parameterized by one concrete model's generation math."""

    def __init__(
        self,
        *,
        latent_downsample: int,
        prediction: str,
        prediction_dtype: str,
        schedule_direction: ScheduleDirection,
        schedule_shift_domain: ScheduleShiftDomain,
        max_latent_tokens: int,
        max_vae_grid_tokens: int,
        commit_marker_tokens: int,
        rope_advance: int,
        max_cfg_branches: int,
        latent_layout: LatentLayout,
        latent_channels: int,
        latent_patch_size: int,
        positions: PositionLayout,
        materialization: Materialization,
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
        self.latent_downsample = int(latent_downsample)
        self.prediction = str(prediction)
        self.prediction_dtype = str(prediction_dtype)
        self.schedule_direction = schedule_direction
        self.schedule_shift_domain = schedule_shift_domain
        self.max_latent_tokens = int(max_latent_tokens)
        self.max_vae_grid_tokens = int(max_vae_grid_tokens)
        self.commit_marker_tokens = int(commit_marker_tokens)
        self.rope_advance = int(rope_advance)
        self.max_cfg_branches = int(max_cfg_branches)
        self.latent_layout = latent_layout
        self.latent_channels = int(latent_channels)
        self.latent_patch_size = int(latent_patch_size)
        self.positions = positions
        self.materialization = materialization
        self.text_unconditional = text_unconditional
        self.image_unconditional = image_unconditional
        self.cfg_recipe = cfg_recipe
        self.noise_scale_value = float(noise_scale)
        self.noise_scale_mode = noise_scale_mode
        self.noise_scale_base_tokens = float(noise_scale_base_tokens)
        self.noise_scale_maximum = float(noise_scale_maximum)
        self.timestep_shift = None if timestep_shift is None else float(timestep_shift)
        self.prompt = prompt
        if min(
            self.latent_downsample,
            self.max_latent_tokens,
            self.max_vae_grid_tokens,
            self.commit_marker_tokens,
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
        if branch is Branch.COND:
            return BranchSource.CONDITIONING
        if branch is Branch.TEXT_UNCOND:
            return self.text_unconditional
        return self.image_unconditional

    def prefix(
        self,
        source: BranchSource,
        *,
        image_prompt: str,
        negative_prompt: str,
        negative_token_ids: tuple[int, ...],
        tokenizer: Any | None,
    ) -> tuple[tuple[int, ...], bool]:
        if source is BranchSource.CONDITIONING and not image_prompt.strip():
            return (), True
        if source is BranchSource.NEGATIVE_OR_START and negative_token_ids:
            return negative_token_ids, False
        if self.prompt is None:
            if source is BranchSource.CONDITIONING:
                raise invalid_descriptor("this model does not accept a generation prompt override")
            return (), False
        if source is BranchSource.CONDITIONING:
            text = image_prompt.strip()
            conditioned = True
        elif source is BranchSource.NEGATIVE_OR_START:
            text = negative_prompt.strip()
            conditioned = False
        else:
            text = ""
            conditioned = False
        return self.prompt.encode(tokenizer, text=text, conditioned=conditioned), False

    def image_tokens(self, height: int, width: int) -> int:
        return (int(height) // self.latent_downsample) * (int(width) // self.latent_downsample)

    def physical_tokens(self, height: int, width: int) -> int:
        count = self.image_tokens(height, width)
        return (
            count + self.commit_marker_tokens
            if self.latent_layout is LatentLayout.PATCH_TOKENS
            else count
        )

    def latent_shape(self, height: int, width: int) -> tuple[int, ...]:
        if self.latent_layout is LatentLayout.PATCH_TOKENS:
            width_per_token = self.latent_patch_size**2 * self.latent_channels
            return (self.image_tokens(height, width), width_per_token)
        return (1, self.latent_channels, int(height), int(width))

    def noise_scale(self, height: int, width: int) -> float:
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

    def neural_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.latent_layout is LatentLayout.PATCH_TOKENS:
            return latent
        return patchify_batch(latent, self.latent_patch_size)

    def stored_latent(self, latent: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if self.latent_layout is LatentLayout.PATCH_TOKENS:
            return latent
        return unpatchify_batch(
            latent,
            self.latent_patch_size,
            height=int(height),
            width=int(width),
            channels=self.latent_channels,
        )

    def materialization_latent(
        self,
        latent: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Project canonical page rows into the model's materialization layout."""

        if self.latent_layout is LatentLayout.PATCH_TOKENS:
            return latent
        return self.stored_latent(
            latent.reshape(1, self.image_tokens(height, width), -1),
            height,
            width,
        )

    def conditioning(
        self,
        latent: torch.Tensor,
        height: int,
        width: int,
        *,
        patch_size: int | None,
    ) -> FlowPatches | None:
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
    "GenerationPipeline",
    "LatentLayout",
    "Materialization",
    "NoiseScaleMode",
]
