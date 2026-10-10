"""Numerical denoiser capabilities and shared image latent transforms."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, Protocol, TypeVar, cast

import torch
from torch import nn

from uniserve.diffusion import FixedGrid, Grid, NoiseScale, Schedule, Solver
from uniserve.distributed import DeviceMesh
from uniserve.media import image, video
from uniserve.nn.functional import patchify
from uniserve.processing import BranchSource
from uniserve.tensors import BufferConfig, OutputLayout, TensorOutput

from .inputs import DenoiserInput, LatentInput

InputT = TypeVar("InputT", bound=DenoiserInput)
SizeT = TypeVar("SizeT")


@dataclass(frozen=True, slots=True)
class ConditionTiles:
    """Whole-tile condition packing of a multi-region video denoiser.

    Each condition occupies whole tiles of ``rows`` rows: its audio rows
    fill tiles of their own, then an image's rows fill tiles, while a
    video's ``(latent frames, height, width)`` token grid is cut into tiles
    of ``video`` tokens along those axes, each taking ``rows`` rows however
    few tokens it holds. A keyframe has no place in such a packing.

    Attributes:
        rows: Rows of one tile.
        video: Tokens of one video tile along latent frames, height and
            width; their product is ``rows``.
    """

    rows: int
    video: tuple[int, int, int]

    def __post_init__(self) -> None:
        if (
            type(self.rows) is not int
            or self.rows < 1
            or len(self.video) != 3
            or any(type(size) is not int or size < 1 for size in self.video)
            or self.video[0] * self.video[1] * self.video[2] != self.rows
        ):
            raise ValueError(
                "condition tiles need positive video tile sides whose "
                "product is the tile's rows"
            )


class ConditionRole(StrEnum):
    """How a conditioning input relates to the generated video."""

    # Anchors the first generated frame.
    FIRST_FRAME = "first_frame"
    # Anchors the last generated frame.
    LAST_FRAME = "last_frame"
    # Conditions the generation as a whole.
    REFERENCE = "reference"


@dataclass(frozen=True, slots=True)
class Condition:
    """One conditioning input of a video request, as its encoders see it.

    Attributes:
        role: Keyframe anchor or reference.
        video: The encoded pixels' frame count and raster, as the video
            encoder receives them; an image is one frame. None for audio
            alone.
        audio_samples: Samples of the encoded track at the model's audio
            rate: an audio reference, or a video's soundtrack; 0 without one.
    """

    role: ConditionRole
    video: video.Config | None
    audio_samples: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.role, ConditionRole):
            raise TypeError("a condition's role must be a ConditionRole")
        if self.video is not None and not isinstance(self.video, video.Config):
            raise TypeError("a condition's pixels must be a video.Config")
        if type(self.audio_samples) is not int or self.audio_samples < 0:
            raise ValueError("a condition's audio samples must be nonnegative")
        if self.video is None and self.audio_samples == 0:
            raise ValueError("a condition must carry pixels or audio")
        if self.role != ConditionRole.REFERENCE and (
            self.video is None
            or self.video.num_frames != 1
            or self.audio_samples
        ):
            raise ValueError("a keyframe is one encoded frame without audio")


class VideoSize(Protocol):
    """A video request's output timeline, raster and conditioning lengths.

    ``conditions`` lists the request's conditioning inputs in request order
    and ``condition_rows`` the rows the network gives them;
    ``vision_spans`` are the ``(start, stop)`` token ranges of the presented
    prompt that hold vision tokens.
    """

    @property
    def num_frames(self) -> int: ...

    @property
    def canvas(self) -> image.Config: ...

    @property
    def num_text_tokens(self) -> int: ...

    @property
    def condition_rows(self) -> int: ...

    @property
    def conditions(self) -> tuple[Condition, ...]: ...

    @property
    def vision_spans(self) -> tuple[tuple[int, int], ...]: ...


VideoSizeT = TypeVar("VideoSizeT", bound=VideoSize)


class Denoiser(nn.Module, Generic[InputT, SizeT], ABC):
    """A network prediction.

    With explicit solver, evaluation grids and mathematical partitioning.
    ``grids`` declares each modality's ``Grid``; every modality's trajectory
    evaluates the network equally often, so the grids either all fix the
    same step count or all leave it to requests.
    """

    def __init__(
        self,
        *,
        modalities: tuple[str, ...],
        prediction_dtype: torch.dtype,
        solver: Solver,
        grids: Mapping[str, Grid],
    ):
        super().__init__()
        if (
            not modalities
            or len(set(modalities)) != len(modalities)
            or any(not name for name in modalities)
        ):
            raise ValueError(
                "denoiser modalities must be distinct nonempty names"
            )
        if set(grids) != set(modalities) or any(
            not isinstance(grid, Grid) for grid in grids.values()
        ):
            raise ValueError("a denoiser declares one grid per modality")
        if len({grid.num_steps for grid in grids.values()}) != 1:
            raise ValueError(
                "every modality's grid must evaluate the network equally often"
            )
        self.modalities, self.prediction_dtype, self.solver = (
            modalities,
            prediction_dtype,
            solver,
        )
        self.grids = dict(grids)
        self.mesh = DeviceMesh(ranks=(0,), shape=(1,), axes=("tp",), rank=0)

    @property
    def prediction_type(self):
        return self.solver.prediction_type

    @abstractmethod
    def latent_shape(self, modality: str, size: SizeT) -> tuple[int, ...]:
        """Describe the canonical sample tensor consumed by this network."""
        raise NotImplementedError

    @abstractmethod
    def noise_shape(self, modality: str, size: SizeT) -> tuple[int, ...]:
        """Describe a complete native random draw.

        Before numerical conversion.
        """
        raise NotImplementedError

    @property
    def num_steps(self) -> int | None:
        """Network evaluations of a trajectory, None when requests choose."""
        return next(iter(self.grids.values())).num_steps

    def make_schedules(
        self,
        steps: int | None,
        *,
        shift: float | None,
        device: torch.device | str,
    ) -> Mapping[str, Schedule]:
        """Build every modality's schedule for a request.

        ``steps`` and ``shift`` are the request's choices, None leaving a
        parameter to the grids (``Grid.schedule``).

        Raises:
            ValueError: A grid does not admit the requested parameters.
        """
        return {
            name: self.grids[name].schedule(
                steps=steps, shift=shift, device=device
            )
            for name in self.modalities
        }

    def layout_size(self, size: SizeT) -> SizeT:
        """Return the smallest layout that holds ``size``.

        A layout is the size whose numerical shapes a call evaluates. Every
        size a layout holds (see ``holds``) shares its constants, workspace
        and captured graphs, so callers key those by the layout; whatever
        distinguishes the sizes within a layout is request state, which
        ``prepare_state`` fills. By default every size is its own layout.
        """
        return size

    def holds(self, layout: SizeT, size: SizeT) -> bool:
        """Whether a request of ``size`` can be evaluated in ``layout``.

        A caller may evaluate a request in any layout that holds it, such as
        a capacity many sizes share, rather than in its smallest layout. By
        default a layout holds only the sizes whose smallest layout it is.
        """
        return self.layout_size(size) == layout

    def prepare_state(
        self,
        sizes: tuple[SizeT, ...],
        *,
        layouts: tuple[SizeT, ...],
        out: Mapping[str, torch.Tensor],
    ) -> None:
        """Fill the request state that does not derive from a native draw.

        Each size is evaluated in the aligned layout, which must hold it.
        ``out`` holds CPU views of the state fields other than the sample
        modalities, shaped by ``state_buffers`` of the layouts, which the
        caller stages next to the samples. A network whose state is its
        samples alone receives no views.
        """
        if out:
            raise NotImplementedError(
                "denoiser declares request state it does not prepare"
            )

    @abstractmethod
    def prepare_latents(
        self, sizes: tuple[SizeT, ...], *, noise, state, constants, workspace
    ) -> None:
        """Convert caller-supplied native draws.

        Into canonical sample storage.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, inputs: InputT, *, state, constants, workspace
    ) -> Mapping[str, tuple[TensorOutput | None, ...]]:
        """Predict each sample.

        Without committing a solver step or request progress. ``inputs.sizes``
        are layouts (see ``layout_size``) and ``inputs.latents`` carry the
        samples. ``state`` names the device fields ``prepare_state`` fills for
        the request being advanced, which a captured step reads through fixed
        staging rather than at the request's own addresses; storage a request
        draws on the host, as ``prepare_latents`` receives, is the
        preparation's and is absent here.
        """
        raise NotImplementedError


class VideoDenoiser(Denoiser[InputT, VideoSizeT]):
    """A standalone denoiser that generates a video timeline from conditions.

    A request is sized by its output frame count, canvas, prompt length and
    condition rows. The network's fixed grids set its step count
    (``num_steps``) and trained shifts; it rounds a requested duration to its
    native windows and describes the request state and outputs that the
    caller's storage holds; the caller advances the fixed schedule and
    supplies each step's borrowed tensors to ``bind_inputs``.
    """

    def __init__(self, **options):
        super().__init__(**options)
        if any(not isinstance(grid, FixedGrid) for grid in self.grids.values()):
            raise ValueError("a video denoiser evaluates its fixed grids")

    @property
    def canvases(self) -> tuple[image.Config, ...]:
        """Canvases a deployment may prepare layouts for before serving.

        A deployment prepares a selection of them, or all of them without
        one, and admits only the canvases it prepares.
        """
        raise NotImplementedError

    @property
    def tasks(self) -> tuple[str, ...]:
        """Task names this network serves, in canonical order."""
        raise NotImplementedError

    @property
    def schedule_shifts(self) -> Mapping[str, float]:
        """Each modality's trained schedule shift."""
        # ``__init__`` admits only fixed grids, which carry a trained shift.
        return {
            name: cast(FixedGrid, grid).shift
            for name, grid in self.grids.items()
        }

    @property
    def fixed_canvases(self) -> tuple[image.Config, ...] | None:
        """The only canvases the network generates, or None for any canvas.

        ``None`` generates every canvas of the model's canvas rule; a
        deployment still prepares and admits only its selection of
        ``canvases``.
        """
        raise NotImplementedError

    @property
    def max_sequence_rows(self) -> int | None:
        """Rows the denoiser packs at most, or None without such a bound."""
        raise NotImplementedError

    @property
    def condition_tiles(self) -> ConditionTiles | None:
        """How ``make_size`` counts condition rows.

        ``None`` packs conditions densely, one row per condition token; a
        multi-region network returns the whole tiles it packs them in. A
        serving owner admits requests by the same count.
        """
        return None

    def max_conditions(
        self, num_frames: int, canvas: image.Config
    ) -> tuple[Condition, ...]:
        """The largest condition set a request of this network may carry.

        Returns the conditions, among those a request generating
        ``num_frames`` frames on ``canvas`` may bring under the network's
        tasks, whose packed rows (``make_size``) are the most. A serving
        owner that states no smaller condition capacity provisions for them,
        so it refuses no request for its conditions. Empty for a network
        without a conditioned task.
        """
        return ()

    def text_condition_rows(self, layout: VideoSizeT) -> int:
        """Rows of a request's retained conditioning in ``layout``.

        The caller retains the refined prompt in the leading rows, zero past
        the prompt; a network may reserve further rows it fills itself.
        """
        return layout.num_text_tokens

    @property
    def num_steps(self) -> int:
        """Network evaluations of the fixed schedule."""
        # The constructor accepts only fixed grids, which fix the count.
        return cast(int, super().num_steps)

    @property
    def text_condition_width(self) -> int:
        """Feature width of the retained text conditioning."""
        raise NotImplementedError

    def legal_frame_count(self, requested: int) -> int:
        """Round a requested frame count up to one the network generates."""
        raise NotImplementedError

    def make_size(
        self,
        num_frames: int,
        num_text_tokens: int,
        *,
        canvas: image.Config,
        conditions: tuple[Condition, ...] = (),
        vision_spans: tuple[tuple[int, int], ...] = (),
    ) -> VideoSizeT:
        """Build the size descriptor of one request.

        ``conditions`` are the request's conditioning inputs in request
        order; ``vision_spans`` the presented prompt's vision-token ranges.
        """
        raise NotImplementedError

    def condition_noise_shapes(
        self, size: VideoSizeT
    ) -> tuple[tuple[int, ...], ...]:
        """Native FP32 noise draws of a request's conditions, in order.

        They precede the generated modalities' draws in the request's seeded
        stream, each drawn as its own tensor. A network without noised
        conditions draws none.
        """
        return ()

    def condition_noise_capacity(self, layout: VideoSizeT) -> int:
        """Bound the condition draws of every size ``layout`` holds.

        Returns the FP32 elements that hold all of one request's
        ``condition_noise_shapes`` draws.
        """
        return 0

    def condition_layout(
        self, layout: VideoSizeT, condition_rows: int
    ) -> VideoSizeT:
        """Widen ``layout`` so its condition region holds ``condition_rows``.

        Returns ``layout`` with a condition region that holds
        ``condition_rows`` packed condition rows, and is otherwise equal; a
        layout that already holds them is returned as it is. A serving owner
        sizes the layout that bounds its condition capacity this way.

        Raises:
            ValueError: The network takes no conditions, or the widened
                layout exceeds ``max_sequence_rows``.
        """
        raise ValueError("this network takes no conditioning inputs")

    def encode_conditions(
        self,
        size: VideoSizeT,
        layout: VideoSizeT,
        *,
        latents: tuple[torch.Tensor, ...],
        noise: tuple[torch.Tensor, ...],
        out: torch.Tensor,
    ) -> None:
        """Write a request's encoded conditions into its retained rows.

        ``latents`` are the condition latents in request order as the latent
        encoders produced them; ``noise`` the draws of
        ``condition_noise_shapes``; ``out`` the request's retained
        conditioning (``text_condition_rows(layout)`` rows, the refined
        prompt leading), into whose rows past the prompt capacity the network
        writes.

        Raises:
            ValueError: The request has conditions this network does not
                take, or inputs that do not match ``size``.
        """
        if size.conditions:
            raise ValueError("this network takes no conditioning inputs")

    def state_buffers(self, size: VideoSizeT) -> Mapping[str, BufferConfig]:
        """Describe one request's state for the layout ``size`` occupies."""
        raise NotImplementedError

    def output_layout(self, size: VideoSizeT) -> Mapping[str, OutputLayout]:
        """Describe each modality's predicted samples for ``size``."""
        raise NotImplementedError

    def bind_inputs(
        self,
        *,
        latents: Mapping[str, tuple[LatentInput, ...]],
        sizes: tuple[VideoSizeT, ...],
        step: torch.Tensor,
        text_features: tuple[torch.Tensor, ...],
    ) -> InputT:
        """Assemble one denoising step's typed input from borrowed tensors.

        ``step`` is the evaluation's [1] int64 device index
        (``Schedule.step``); see ``DenoiserInput``.
        """
        raise NotImplementedError


class ImageDenoiser(Denoiser[InputT, image.Config]):
    """Shared image patch dimensions and scale-before-patch noise conversion.

    downsample measures output pixels per canonical patch token. Native spatial
    latents therefore have patch_size pixels per token on each spatial axis.
    A network with a different native draw layout overrides noise_shape.

    ``framing_tokens`` and ``image_unconditional`` describe how this network
    frames a generated image and which prefix its image-unconditional guidance
    branch reads. Execution owns the buffers and the stepping; the network
    states these layout facts and assembles its own typed input.
    """

    framing_tokens: int = 0
    image_unconditional: BranchSource = BranchSource.START

    @property
    def max_sequence_tokens(self) -> int:
        """Longest image sequence this network accepts, in latent patch rows.

        ``framing_tokens`` are not counted; the framed sequence holds up to
        ``max_sequence_tokens + framing_tokens`` tokens.
        """
        raise NotImplementedError

    def bind_inputs(
        self,
        *,
        latents: Mapping[str, tuple[LatentInput, ...]],
        sizes: tuple[image.Config, ...],
        step: torch.Tensor,
        positions: torch.Tensor,
        sequence_lengths: tuple[int, ...],
        attention: Any,
    ) -> InputT:
        """Assemble this network's typed input for one denoising step.

        The caller supplies borrowed numerical views; the network selects the
        fields its own forward consumes and derives any per-step conditioning
        from them.
        """
        raise NotImplementedError

    def __init__(
        self,
        *,
        patch_size: int,
        latent_channels: int,
        downsample: int,
        noise_scale: NoiseScale,
        prediction_dtype: torch.dtype,
        solver: Solver,
        grid: Grid,
    ):
        super().__init__(
            modalities=("image",),
            prediction_dtype=prediction_dtype,
            solver=solver,
            grids={"image": grid},
        )
        if any(
            type(value) is not int or value < 1
            for value in (patch_size, latent_channels, downsample)
        ):
            raise ValueError(
                "image patch, channel and downsample dimensions must be "
                "positive"
            )
        self.patch_size, self.latent_channels, self.downsample = (
            patch_size,
            latent_channels,
            downsample,
        )
        self.noise_scale = noise_scale

    def latent_shape(
        self, modality: str, size: image.Config
    ) -> tuple[int, ...]:
        if modality != "image":
            raise ValueError("image denoisers accept the image modality")
        rows = size.height // self.downsample * (size.width // self.downsample)
        if not rows:
            raise ValueError(
                "image dimensions must contain at least one latent patch"
            )
        return rows, self.patch_size**2 * self.latent_channels

    def noise_shape(self, modality: str, size: image.Config) -> tuple[int, ...]:
        self.latent_shape(modality, size)
        # [1, channels, height, width] in output pixels, before patchify
        return (
            1,
            self.latent_channels,
            size.height // self.downsample * self.patch_size,
            size.width // self.downsample * self.patch_size,
        )

    def prepare_latents(
        self, sizes, *, noise, state, constants, workspace
    ) -> None:
        source, destination = noise["image"], state["image"]
        if source.shape[0] != len(sizes) or destination.shape[0] != len(sizes):
            raise ValueError(
                "noise and sample storage must align with image sizes"
            )
        if (
            source.device != destination.device
            or source.dtype != destination.dtype
        ):
            raise ValueError("image noise must use the sample device and dtype")

        for index, size in enumerate(sizes):
            native = self.noise_shape("image", size)
            canonical = self.latent_shape("image", size)
            if (
                tuple(source[index].shape) != native
                or tuple(destination[index].shape) != canonical
            ):
                raise ValueError(
                    "image storage must match the native draw and canonical "
                    "patch shapes"
                )

            # Scaling precedes the layout permutation and rounds in the sample
            # dtype. A separate value protects permitted source/state aliases.
            scaled = source[index] * self.noise_scale.scale(canonical[0])
            patches = (
                scaled
                if native == canonical
                else patchify(scaled, patch_size=self.patch_size)
            )
            destination[index].copy_(patches.reshape(canonical))
