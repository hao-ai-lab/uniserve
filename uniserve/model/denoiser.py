"""Numerical denoiser capabilities and shared image latent transforms."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, TypeVar

import torch
from torch import nn

from uniserve.diffusion import NoiseScale, Schedule, Solver
from uniserve.distributed import DeviceMesh
from uniserve.media import image
from uniserve.nn.functional import patchify
from uniserve.processing import BranchSource
from uniserve.tensors import TensorOutput

from .inputs import DenoiserInput, LatentInput

InputT = TypeVar("InputT", bound=DenoiserInput)
SizeT = TypeVar("SizeT")


class Denoiser(nn.Module, Generic[InputT, SizeT], ABC):
    """A network prediction.

    With explicit solver and mathematical partitioning.
    """

    def __init__(
        self,
        *,
        modalities: tuple[str, ...],
        prediction_dtype: torch.dtype,
        solver: Solver,
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
        self.modalities, self.prediction_dtype, self.solver = (
            modalities,
            prediction_dtype,
            solver,
        )
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

    @abstractmethod
    def make_schedules(
        self, steps: int, *, shift: float | None, device: torch.device | str
    ) -> Mapping[str, Schedule]:
        """Construct complete schedules.

        Using this network's time parameterization.
        """
        raise NotImplementedError

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

        Without committing a solver step or request progress.
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
        """Longest image sequence this network accepts, framing included."""
        raise NotImplementedError

    def bind_inputs(
        self,
        *,
        latents: Mapping[str, tuple[LatentInput, ...]],
        sizes: tuple[image.Config, ...],
        step_index: int,
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
    ):
        super().__init__(
            modalities=("image",),
            prediction_dtype=prediction_dtype,
            solver=solver,
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
