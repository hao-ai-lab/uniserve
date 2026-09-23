"""Stage numerical media inputs for the worker's admitted requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch

from uniserve.diffusion import Schedule, normal_noise
from uniserve.model import Denoiser, LatentInput
from uniserve.tensors import BufferConfig


class MediaBuilder:
    """Own input bounds while borrowing the denoiser's numerical capability.

    Native CPU noise, pinned transfer sources and retained conditioning belong
    to serving. The model sees only the exact sample, constants and workspace
    views required for one invocation, and states its own native window and
    size descriptor.

    A request's storage, constants and denoising calls follow the layout its
    exact size occupies, so every request of one layout binds the same shapes
    and replays the same captured ladder. What distinguishes the request
    within its layout is state the builder stages with its samples.
    """

    def __init__(
        self, denoiser: Denoiser, *, max_frames: int, max_text_tokens: int
    ) -> None:
        # Admission advertises complete native windows, including the final
        # overlap. Cover the configured duration with the next legal input.
        frames = denoiser.legal_frame_count(max_frames)
        self.maximum = denoiser.make_size(frames, max_text_tokens)
        self.denoiser = denoiser
        self.num_steps = len(denoiser.diffusion.ladder)
        # State descriptions per layout. Describing one builds the layout's
        # packing, which costs milliseconds of host time; layouts are a small
        # finite set bounded by the admitted frame and prompt capacity.
        self._states: dict[object, Mapping[str, BufferConfig]] = {}

    def size(self, num_frames: int, num_text_tokens: int):
        size = self.denoiser.make_size(num_frames, num_text_tokens)
        if (
            size.num_frames > self.maximum.num_frames
            or size.num_text_tokens > self.maximum.num_text_tokens
        ):
            raise ValueError(
                "media input exceeds the worker frame or conditioning capacity"
            )
        return size

    def layout(self, size):
        """Return the size whose numerical layout ``size`` occupies.

        Requests of one layout share its prepared context and captured
        graphs; see ``Denoiser.layout_size``.
        """
        return self.denoiser.layout_size(size)

    def _state(self, size) -> Mapping[str, BufferConfig]:
        layout = self.layout(size)
        state = self._states.get(layout)
        if state is None:
            state = self._states[layout] = self.denoiser.state_buffers(layout)
        return state

    def tables(self, size) -> tuple[str, ...]:
        """Name the state fields other than the samples, filled per request."""
        return tuple(
            name
            for name in self._state(size)
            if name not in self.denoiser.modalities
        )

    def buffers(self, size) -> Mapping[str, BufferConfig]:
        """Describe one request's storage, shaped by its layout.

        The denoiser's device state, the complete CPU draws, a CPU source for
        every state field to stage from, and the retained conditioning over
        the layout's text rows, zero past the prompt.
        """
        layout = self.layout(size)
        state = self._state(size)
        result = dict(state)
        for name in self.denoiser.modalities:
            result[f"{name}_noise"] = BufferConfig(
                self.denoiser.noise_shape(name, layout),
                torch.float32,
                host=True,
            )
        for name, config in state.items():
            result[f"{name}_source"] = replace(config, host=True)
        result["text_condition"] = BufferConfig(
            (layout.num_text_tokens, self.denoiser.text_condition_width),
            torch.bfloat16,
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

    @torch.inference_mode()
    def stage_request(
        self, size, tensors: Mapping[str, torch.Tensor], *, seed: int
    ) -> None:
        """Fill a request's host inputs that its admission determines.

        The seeded native draw and the request's own state tables depend only
        on the seed and the exact size, so they can be prepared on another
        thread before the request's latents are.
        """
        # The denoiser's numerical calls batch over leading size 1.
        normal_noise(
            (seed,),
            out=tuple(
                tensors[f"{name}_noise"].unsqueeze(0)
                for name in self.denoiser.modalities
            ),
        )
        self.denoiser.prepare_state(
            (size,),
            out={name: tensors[f"{name}_source"] for name in self.tables(size)},
        )

    @torch.inference_mode()
    def initialize(
        self,
        size,
        tensors: Mapping[str, torch.Tensor],
        *,
        constants: Mapping[str, torch.Tensor],
        workspace: Mapping[str, torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Fill sample sources from the drawn noise.

        Returns destination/source pairs of every state field for staging.
        """
        names = self.denoiser.modalities
        noise = {name: tensors[f"{name}_noise"].unsqueeze(0) for name in names}
        source = {
            name: tensors[f"{name}_source"].unsqueeze(0) for name in names
        }
        self.denoiser.prepare_latents(
            (self.layout(size),),
            noise=noise,
            state=source,
            constants=constants,
            workspace=workspace,
        )
        return tuple(
            (tensors[name], tensors[f"{name}_source"])
            for name in (*names, *self.tables(size))
        )

    def store_conditioning(
        self, size, tensors: Mapping[str, torch.Tensor], features
    ) -> None:
        """Retain a prompt's conditioning in the leading rows of its layout.

        The rows past the prompt are zero, as the layout's denoising calls
        require. Both writes are ordered on the caller's current stream.
        """
        target = tensors["text_condition"]
        rows = size.num_text_tokens
        target[:rows].copy_(features.reshape(rows, -1), non_blocking=True)
        target[rows:].zero_()

    def bind(
        self,
        size,
        tensors: Mapping[str, torch.Tensor],
        schedules: Mapping[str, Schedule],
        index: int,
    ):
        """Assemble one denoising step's typed input from resident tensors."""
        if not 0 <= index < self.num_steps:
            raise ValueError("denoising index is outside the fixed schedule")

        return self.denoiser.bind_inputs(
            latents={
                name: (
                    LatentInput(
                        tensors[name], schedules[name].timesteps[index]
                    ),
                )
                for name in self.denoiser.modalities
            },
            sizes=(self.layout(size),),
            step_index=index,
            text_features=(tensors["text_condition"],),
        )
