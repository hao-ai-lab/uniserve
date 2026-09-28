"""Stage numerical media inputs for the worker's admitted requests.

``MediaBuilder`` wraps a standalone ``VideoDenoiser`` for serving: it bounds
admitted sizes, maps each size to the capacity layout whose prepared
constants and captured ladder it shares, describes a request slot's storage,
and places a request's solver samples in latent pool pages (``SamplePages``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property

import torch

from uniserve.diffusion import Schedule, normal_noise
from uniserve.media import image
from uniserve.model import LatentInput, VideoDenoiser
from uniserve.tensors import BufferConfig

#: Pages one request's samples span in the latent pool. Every denoising step
#: validates and addresses the request's pages on the host, so a small fixed
#: count keeps that work constant while padding stays under one page.
REQUEST_PAGES = 8

#: Element alignment of each modality's samples within a request's pages.
SAMPLE_ALIGNMENT = 256

#: Default text capacities, in prompt tokens: a first rung, then steps of
#: ``TEXT_CAPACITY_STEP`` up to the prompt capacity. A request's layout holds
#: at most one step of padding rows beyond its prompt's tiles. Padding rows
#: cost GEMM work, about 2% of a 4-5 second step per 1024 rows on 4 x GB200
#: under Ulysses-4, so the first rung spares prompts of up to 1024 tokens
#: that padding.
TEXT_CAPACITY_FIRST = 1024
TEXT_CAPACITY_STEP = 2048


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


def bound(first: BufferConfig, second: BufferConfig) -> BufferConfig:
    """Return the field whose extents hold both, dimension by dimension.

    Raises:
        ValueError: The fields differ in rank, dtype or placement.
    """
    if (
        len(first.shape) != len(second.shape)
        or first.dtype != second.dtype
        or first.host != second.host
    ):
        raise ValueError("bounded fields must share rank, dtype and placement")

    def largest(a, b):
        return tuple(max(x, y) for x, y in zip(a, b, strict=True))

    capacity = (
        None
        if first.capacity_shape is None and second.capacity_shape is None
        else largest(
            first.capacity_shape or first.shape,
            second.capacity_shape or second.shape,
        )
    )
    return replace(
        first,
        shape=largest(first.shape, second.shape),
        capacity_shape=capacity,
    )


class MediaBuilder:
    """Own input bounds while borrowing the denoiser's numerical capability.

    Native CPU noise, pinned transfer sources and retained conditioning belong
    to serving. The model sees only the exact sample, constants and workspace
    views required for one invocation, and states its own native window and
    size descriptor.

    Serving evaluates a bounded set of capacity layouts (``layouts``): every
    frame count the worker admits at every served frame raster, each with
    every text capacity. A request
    evaluates in the smallest capacity layout that holds it (``layout``), so
    every request of one layout binds the same shapes and replays the same
    captured ladder, and no admitted size needs a layout of its own. What
    distinguishes the request within its layout is state the builder stages
    with its samples.

    The samples a solver step rewrites live in the worker's latent pool (see
    ``sample_pages``); the request's slot holds only state written once: the
    denoiser's tables, the retained conditioning and the host staging.
    """

    def __init__(
        self,
        denoiser: VideoDenoiser,
        *,
        frame_sizes: tuple[image.Config, ...],
        max_frames: int,
        max_text_tokens: int,
        min_frames: int = 1,
        text_capacities: tuple[int, ...] = (),
    ) -> None:
        """Bound admitted sizes and fix the capacity layouts.

        Admitted frame counts are the legal counts from ``min_frames`` up to
        ``max_frames`` rounded up to complete native windows, each at every
        raster of ``frame_sizes``. Text
        capacities are prompt token counts; the largest must hold
        ``max_text_tokens``, and without any they are ``TEXT_CAPACITY_FIRST``
        and then steps of ``TEXT_CAPACITY_STEP`` up to it.

        Raises:
            ValueError: No frame count or raster is admitted, the denoiser does
                not generate a raster, or a text capacity is not positive or
                none holds ``max_text_tokens``.
        """
        if not frame_sizes or len(set(frame_sizes)) != len(frame_sizes):
            raise ValueError("media input requires distinct frame rasters")
        # Admission advertises complete native windows, including the final
        # overlap. Cover the configured duration with the next legal input.
        frames = denoiser.legal_frame_count(max_frames)
        self.frame_sizes = tuple(frame_sizes)
        self.max_frames = frames
        self.max_text_tokens = max_text_tokens
        # The largest admitted size at each raster; together they bound every
        # admitted size's shapes dimension by dimension.
        self.maxima = tuple(
            denoiser.make_size(frames, frame, max_text_tokens)
            for frame in self.frame_sizes
        )
        self.denoiser = denoiser
        self.num_steps = denoiser.num_steps

        counts = [denoiser.legal_frame_count(max(1, int(min_frames)))]
        while counts[-1] < frames:
            counts.append(denoiser.legal_frame_count(counts[-1] + 1))
        if counts[-1] != frames:
            raise ValueError("media input admits no frame count")
        self.frame_counts = tuple(counts)

        requested = tuple(int(value) for value in text_capacities) or (
            TEXT_CAPACITY_FIRST,
            *range(TEXT_CAPACITY_STEP, max_text_tokens, TEXT_CAPACITY_STEP),
            max_text_tokens,
        )
        if min(requested) < 1 or max(requested) < max_text_tokens:
            raise ValueError(
                "text capacities must be positive and hold the prompt capacity"
            )
        # Each capacity is the text region of the layout that holds it,
        # capped at the one that holds the prompt capacity.
        largest = denoiser.layout_size(self.maxima[0]).num_text_tokens
        self.text_capacities = tuple(
            sorted(
                {
                    min(
                        largest,
                        denoiser.layout_size(
                            denoiser.make_size(
                                frames, self.frame_sizes[0], value
                            )
                        ).num_text_tokens,
                    )
                    for value in requested
                }
            )
        )
        # State descriptions per layout. Describing one builds the layout's
        # packing on the host, so each is cached; layouts are the bounded set
        # ``layouts`` lists.
        self._states: dict[object, Mapping[str, BufferConfig]] = {}

    def size(self, num_frames: int, frame: image.Config, num_text_tokens: int):
        """Return the denoiser's exact size for an admitted request.

        Raises:
            ValueError: The frame count or raster is not one the worker
                admits, or the prompt exceeds its conditioning capacity.
        """
        size = self.denoiser.make_size(num_frames, frame, num_text_tokens)
        if (
            size.num_frames not in self.frame_counts
            or frame not in self.frame_sizes
            or size.num_text_tokens > self.max_text_tokens
        ):
            raise ValueError(
                "media input exceeds the worker frame, raster or conditioning "
                "capacity"
            )
        return size

    def layout(self, size):
        """Return the capacity layout a request of ``size`` evaluates in.

        It is the smallest text capacity at the request's frame count and raster
        that holds the request (``Denoiser.holds``). Requests of one layout
        share its prepared constants and captured graphs.

        Raises:
            ValueError: ``size`` is not an admitted frame count and raster,
                or no capacity holds it.
        """
        if (
            size.num_frames not in self.frame_counts
            or size.frame not in self.frame_sizes
        ):
            raise ValueError("media input has no admitted capacity layout")
        for capacity in self.text_capacities:
            layout = self.denoiser.layout_size(
                self.denoiser.make_size(size.num_frames, size.frame, capacity)
            )
            if self.denoiser.holds(layout, size):
                return layout
        raise ValueError("media input has no admitted capacity layout")

    def layouts(self) -> tuple:
        """List every capacity layout, the longest and largest first.

        The leading layout of each raster holds every admitted size at that
        raster and bounds the shapes of that raster's other layouts dimension
        by dimension.
        """
        return tuple(
            self.denoiser.layout_size(
                self.denoiser.make_size(frames, frame, capacity)
            )
            for frames in reversed(self.frame_counts)
            for frame in self.frame_sizes
            for capacity in reversed(self.text_capacities)
        )

    def _state(self, size) -> Mapping[str, BufferConfig]:
        layout = self.layout(size)
        state = self._states.get(layout)
        if state is None:
            state = self._states[layout] = self.denoiser.state_buffers(layout)
        return state

    @cached_property
    def sample_pages(self) -> SamplePages:
        """Size one request's pages from the admitted maxima.

        Each modality's global extent at a raster's maximum bounds its local
        shard on every rank for every admitted size at that raster, as in
        ``capacity_buffers``.
        """
        names = self.denoiser.modalities
        dtypes = {
            self._state(maximum)[name].dtype
            for maximum in self.maxima
            for name in names
        }
        if len(dtypes) != 1:
            raise ValueError("pooled sample modalities must share one dtype")
        capacity = max(
            sum(
                _aligned(math.prod(self.denoiser.latent_shape(name, maximum)))
                for name in names
            )
            for maximum in self.maxima
        )
        # Round the per-page share up to the alignment, then count the pages
        # that share actually needs (at most ``REQUEST_PAGES``).
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
        """Cover every shard produced by an admitted size.

        A shorter prompt can move additional media tokens onto a particular
        sequence rank. Global modality extents provide a conservative bound
        without assuming that the maximum-size request has the largest shard.
        Each field's extents are the largest any raster's maximum needs,
        dimension by dimension.

        Raises:
            ValueError: The rasters' fields disagree in rank, dtype or
                placement.
        """
        result: dict[str, BufferConfig] = {}
        for maximum in self.maxima:
            fields = dict(self.buffers(maximum))
            for name in self.denoiser.modalities:
                capacity = self.denoiser.latent_shape(name, maximum)
                key = f"{name}_source"
                fields[key] = replace(fields[key], capacity_shape=capacity)
            for name, field in fields.items():
                result[name] = (
                    field if name not in result else bound(result[name], field)
                )
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
            layouts=(self.layout(size),),
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
            sizes=(self.layout(size),),
            step=schedules[names[0]].step(index),
            text_features=(tensors["text_condition"],),
        )
