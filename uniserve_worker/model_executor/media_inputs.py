"""Stage numerical media inputs for the worker's admitted requests."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property

import torch

from uniserve.diffusion import Schedule, normal_noise
from uniserve.model import LatentInput, VideoDenoiser
from uniserve.tensors import BufferConfig

#: Pages one request's samples span in the latent pool. Every denoising step
#: validates and addresses the request's pages on the host, so a small fixed
#: count keeps that work constant while padding stays under one page.
REQUEST_PAGES = 8

#: Element alignment of each modality's samples within a request's pages.
SAMPLE_ALIGNMENT = 256


@dataclass(frozen=True, slots=True)
class SamplePages:
    """How a standalone denoiser's samples occupy latent pool pages.

    A pool unit is one sample element and the pool's rows are one element
    wide. Each request owns ``pages`` pages of ``page_units`` elements, which
    hold every modality's local samples of the admitted maximum. Within one
    layout the modalities follow each other in modality order, each starting
    on a ``SAMPLE_ALIGNMENT`` boundary, so gathering the layout's leading
    pages yields every sample as a contiguous tensor.
    """

    page_units: int
    pages: int
    dtype: torch.dtype

    @property
    def units(self) -> int:
        """Pool units one request's pages hold."""
        return self.pages * self.page_units


def _aligned(elements: int) -> int:
    return -(-int(elements) // SAMPLE_ALIGNMENT) * SAMPLE_ALIGNMENT


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

    The samples a solver step rewrites live in the worker's latent pool (see
    ``sample_pages``); the request's slot holds only state written once: the
    denoiser's tables, the retained conditioning and the host staging.
    """

    def __init__(
        self, denoiser: VideoDenoiser, *, max_frames: int, max_text_tokens: int
    ) -> None:
        # Admission advertises complete native windows, including the final
        # overlap. Cover the configured duration with the next legal input.
        frames = denoiser.legal_frame_count(max_frames)
        self.maximum = denoiser.make_size(frames, max_text_tokens)
        self.denoiser = denoiser
        self.num_steps = denoiser.num_steps
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

    @cached_property
    def sample_pages(self) -> SamplePages:
        """Size one request's pages from the admitted maximum.

        Each modality's global extent at the maximum bounds its local shard
        on every rank for every admitted size, as in ``capacity_buffers``.
        """
        state = self._state(self.maximum)
        names = self.denoiser.modalities
        dtypes = {state[name].dtype for name in names}
        if len(dtypes) != 1:
            raise ValueError("pooled sample modalities must share one dtype")
        capacity = sum(
            _aligned(math.prod(self.denoiser.latent_shape(name, self.maximum)))
            for name in names
        )
        page_units = _aligned(-(-capacity // REQUEST_PAGES))
        return SamplePages(page_units, -(-capacity // page_units), dtypes.pop())

    def slot_pages(self, slot: int) -> tuple[int, ...]:
        """Return the consecutive pool pages request ``slot`` owns.

        Page zero is the pool's sentinel; slot ``s`` owns the ``s``-th run
        of ``sample_pages.pages`` pages after it.
        """
        count = self.sample_pages.pages
        return tuple(range(1 + (int(slot) - 1) * count, 1 + int(slot) * count))

    def _sample_offsets(self, size) -> tuple[dict[str, int], int]:
        state = self._state(size)
        offsets, cursor = {}, 0
        for name in self.denoiser.modalities:
            offsets[name] = cursor
            cursor += _aligned(math.prod(state[name].shape))
        return offsets, cursor

    def layout_pages(self, size) -> int:
        """Count the leading request pages that hold the samples of ``size``.

        A denoising step of the layout gathers and writes back only these.
        """
        _, elements = self._sample_offsets(size)
        page_units = self.sample_pages.page_units
        return -(-elements // page_units)

    def sample_views(
        self, size, flat: torch.Tensor
    ) -> Mapping[str, torch.Tensor]:
        """View each modality's local sample of ``size`` in gathered pages.

        ``flat`` holds a request's leading pages contiguously, in any shape.
        """
        flat = flat.view(-1)
        state = self._state(size)
        offsets, _ = self._sample_offsets(size)
        return {
            name: flat[
                offsets[name] : offsets[name] + math.prod(state[name].shape)
            ].view(state[name].shape)
            for name in self.denoiser.modalities
        }

    def tables(self, size) -> tuple[str, ...]:
        """Name the state fields other than the samples, filled per request."""
        return tuple(
            name
            for name in self._state(size)
            if name not in self.denoiser.modalities
        )

    def buffers(self, size) -> Mapping[str, BufferConfig]:
        """Describe one request's slot storage, shaped by its layout.

        The denoiser's device tables, the complete CPU draws, a CPU source for
        every state field to stage from, samples included, and the retained
        conditioning over the layout's text rows, zero past the prompt. The
        device samples live in the latent pool, not here.
        """
        layout = self.layout(size)
        state = self._state(size)
        result = {name: state[name] for name in self.tables(size)}
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
            key = f"{name}_source"
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
        samples: Mapping[str, torch.Tensor],
        *,
        constants: Mapping[str, torch.Tensor],
        workspace: Mapping[str, torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Fill sample sources from the drawn noise.

        Returns destination/source pairs for staging: each modality's source
        into its device view in ``samples``, and every table's source into
        the slot.
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
        return (
            *((samples[name], tensors[f"{name}_source"]) for name in names),
            *(
                (tensors[name], tensors[f"{name}_source"])
                for name in self.tables(size)
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

    def bind(
        self,
        size,
        tensors: Mapping[str, torch.Tensor],
        samples: Mapping[str, torch.Tensor],
        schedules: Mapping[str, Schedule],
        index: int,
    ):
        """Assemble one denoising step's typed input.

        The latents are the ``samples`` views a denoising step advances and
        the text features are the slot's retained conditioning.
        """
        if not 0 <= index < self.num_steps:
            raise ValueError("denoising index is outside the fixed schedule")

        return self.denoiser.bind_inputs(
            latents={
                name: (
                    LatentInput(
                        samples[name], schedules[name].timesteps[index]
                    ),
                )
                for name in self.denoiser.modalities
            },
            sizes=(self.layout(size),),
            step_index=index,
            text_features=(tensors["text_condition"],),
        )
