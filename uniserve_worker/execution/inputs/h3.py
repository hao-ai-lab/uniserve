"""Construct H3 numerical inputs from the worker's admitted media requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch

from uniserve.diffusion import Schedule, normal_noise
from uniserve.model import LatentInput
from uniserve.tensors import BufferConfig
from uniserve_models.minimax_h3 import Denoiser, DenoiserInput, DenoiserSize


class Inputs:
    """Own input bounds while borrowing the denoiser's numerical capability.

    Native CPU noise, pinned transfer sources and retained conditioning belong
    to serving. The model sees only the exact sample, constants and workspace
    views required for one invocation.
    """

    def __init__(self, denoiser: Denoiser, *, max_frames: int, max_text_tokens: int) -> None:
        # Admission advertises complete native windows, including the final
        # overlap. Cover the configured duration with the next legal input.
        frames = max(22, max_frames + (5 - max_frames) % 17)
        self.maximum = DenoiserSize(frames, max_text_tokens)
        self.denoiser = denoiser
        self.num_steps = len(denoiser.diffusion.ladder)

    def size(self, num_frames: int, num_text_tokens: int) -> DenoiserSize:
        size = DenoiserSize(num_frames, num_text_tokens)
        if (
            size.num_frames > self.maximum.num_frames
            or size.num_text_tokens > self.maximum.num_text_tokens
        ):
            raise ValueError("media input exceeds the worker frame or conditioning capacity")
        return size

    def buffers(self, size: DenoiserSize) -> Mapping[str, BufferConfig]:
        """Describe request state, complete CPU draws and transfer source views."""

        result = dict(self.denoiser.state_buffers(size))
        for name in self.denoiser.modalities:
            result[f"{name}_noise"] = BufferConfig(
                self.denoiser.noise_shape(name, size), torch.float32, host=True
            )
            result[f"{name}_source"] = replace(result[name], host=True)
        result["text_condition"] = BufferConfig(
            (size.num_text_tokens, self.denoiser.config.hidden_size), torch.bfloat16
        )
        return result

    def capacity_buffers(self) -> Mapping[str, BufferConfig]:
        """Cover every shard produced by an admitted duration and prompt length.

        A shorter prompt can move additional media tokens onto a particular
        sequence rank. Global modality extents provide a conservative bound
        without assuming that the maximum-size request has the largest shard.
        """

        result = dict(self.buffers(self.maximum))
        for name in self.denoiser.modalities:
            capacity = self.denoiser.latent_shape(name, self.maximum)
            for key in (name, f"{name}_source"):
                result[key] = replace(result[key], capacity_shape=capacity)
        return result

    def schedules(self, *, device: torch.device | str) -> Mapping[str, Schedule]:
        return self.denoiser.make_schedules(self.num_steps, shift=None, device=device)

    @torch.inference_mode()
    def initialize(
        self,
        size: DenoiserSize,
        tensors: Mapping[str, torch.Tensor],
        *,
        seed: int,
        constants: Mapping[str, torch.Tensor],
        workspace: Mapping[str, torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Fill CPU transfer sources and return destination/source pairs to stage."""

        names = self.denoiser.modalities
        noise = {name: tensors[f"{name}_noise"].unsqueeze(0) for name in names}
        source = {name: tensors[f"{name}_source"].unsqueeze(0) for name in names}
        normal_noise((seed,), out=tuple(noise.values()))
        self.denoiser.prepare_latents(
            (size,), noise=noise, state=source, constants=constants, workspace=workspace
        )
        return tuple((tensors[name], tensors[f"{name}_source"]) for name in names)

    def bind(
        self,
        size: DenoiserSize,
        tensors: Mapping[str, torch.Tensor],
        schedules: Mapping[str, Schedule],
        index: int,
    ) -> DenoiserInput:
        if not 0 <= index < self.num_steps:
            raise ValueError("denoising index is outside the fixed schedule")
        return DenoiserInput(
            latents={
                name: (LatentInput(tensors[name], schedules[name].timesteps[index]),)
                for name in self.denoiser.modalities
            },
            sizes=(size,),
            step_index=index,
            text_features=(tensors["text_condition"],),
        )
