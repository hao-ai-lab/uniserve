"""Numerical preparation for the native media input owner.

Rust selects admitted layouts, describes request storage and places samples in
latent pages. These methods draw noise, prepare model state and bind tensor
views through the public denoiser interface.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from uniserve.diffusion import Schedule, normal_noise
from uniserve.model import LatentInput
from uniserve_worker._uniserve_ipc import MediaBuilder as _MediaBuilder


class MediaBuilder(_MediaBuilder):
    """Prepare numerical state within the native owner's capacity layouts."""

    def condition_noise(
        self, size, tensors: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        """View a request's condition draws in its ``condition_noise`` field.

        The draws of ``VideoDenoiser.condition_noise_shapes``, each leading
        with the request's batch of one, occupy the field's leading elements
        one after another.
        """
        flat, views, offset = tensors["condition_noise"], [], 0
        for shape in self.denoiser.condition_noise_shapes(size):
            count = math.prod(shape)
            views.append(flat[offset : offset + count].view(shape))
            offset += count
        return tuple(views)

    @torch.inference_mode()
    def prepare_request(
        self,
        size,
        tensors: Mapping[str, torch.Tensor],
        *,
        seed: int,
        layout=None,
    ) -> None:
        """Fill a request's host inputs that its admission determines.

        The seeded native draw and the request's own state tables depend only
        on the seed and the exact size, so they can be prepared on another
        thread before the request's latents are. The seed's stream draws the
        conditions' noise first, then each generated modality's. ``layout``
        is the layout the request evaluates in, ``layout(size)`` by default.
        """
        layout = self.layout(size) if layout is None else layout
        # The denoiser's numerical calls batch over leading size 1, which a
        # condition's native draw already leads with.
        normal_noise(
            (seed,),
            out=(
                *self.condition_noise(size, tensors),
                *(
                    tensors[f"{name}_noise"].unsqueeze(0)
                    for name in self.denoiser.modalities
                ),
            ),
        )
        self.denoiser.prepare_state(
            (size,),
            layouts=(layout,),
            out={
                name: tensors[f"{name}_source"]
                for name in self._layout_tables(layout)
            },
        )

    @torch.inference_mode()
    def initialize(
        self,
        size,
        tensors: Mapping[str, torch.Tensor],
        samples: Mapping[str, torch.Tensor],
        *,
        constants: Mapping[str, torch.Tensor],
        workspace: Mapping[str, torch.Tensor],
        layout=None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Fill sample sources from the drawn noise.

        Returns destination/source pairs for copying: each modality's source
        into its device view in ``samples``, and every table's source into
        the slot. ``layout`` is the layout the request evaluates in,
        ``layout(size)`` by default.
        """
        layout = self.layout(size) if layout is None else layout
        names = self.denoiser.modalities
        noise = {name: tensors[f"{name}_noise"].unsqueeze(0) for name in names}
        source = {
            name: tensors[f"{name}_source"].unsqueeze(0) for name in names
        }
        self.denoiser.prepare_latents(
            (layout,),
            noise=noise,
            state=source,
            constants=constants,
            workspace=workspace,
        )
        return (
            *((samples[name], tensors[f"{name}_source"]) for name in names),
            *(
                (tensors[name], tensors[f"{name}_source"])
                for name in self._layout_tables(layout)
            ),
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

    def encode_conditions(
        self,
        size,
        tensors: Mapping[str, torch.Tensor],
        latents: tuple[torch.Tensor, ...],
    ) -> None:
        """Write a request's encoded conditions into its retained rows.

        ``latents`` are the condition latents in request order as the latent
        encoders produced them (``VideoDenoiser.encode_conditions``). The
        rows follow the prompt, so this runs after ``store_conditioning``, on
        the same stream.
        """
        self.denoiser.encode_conditions(
            size,
            self.layout(size),
            latents=latents,
            noise=self.condition_noise(size, tensors),
            out=tensors["text_condition"],
        )

    def bind(
        self,
        size,
        tensors: Mapping[str, torch.Tensor],
        samples: Mapping[str, torch.Tensor],
        schedules: Mapping[str, Schedule],
        index: int,
        *,
        layout=None,
    ):
        """Assemble one denoising step's typed input.

        The latents are the ``samples`` views a denoising step advances and
        the text features are the slot's retained conditioning. ``layout``
        is the layout the request evaluates in, ``layout(size)`` by default.
        """
        if not 0 <= index < self.num_steps:
            raise ValueError("denoising index is outside the fixed schedule")

        # Every modality's schedule enumerates the same evaluations; the
        # first names the step as device data (``Schedule.step``).
        names = self.denoiser.modalities
        return self.denoiser.bind_inputs(
            latents={
                name: (
                    LatentInput(
                        samples[name], schedules[name].timesteps[index]
                    ),
                )
                for name in names
            },
            sizes=(self.layout(size) if layout is None else layout,),
            step=schedules[names[0]].step(index),
            text_features=(tensors["text_condition"],),
        )
