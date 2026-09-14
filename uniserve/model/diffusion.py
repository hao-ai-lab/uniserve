"""Shared image latent preparation and explicit diffusion capability."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from uniserve.model.batch import DiffusionBatch, TensorOutput
from uniserve.model.image_diffusion import ImageDiffusion
from uniserve.model.media import ImageSize
from uniserve.model.tensors import TensorViews
from uniserve.nn.diffusion.integrator import CleanSampleEulerSolver, EulerSolver
from uniserve.nn.diffusion.schedule import DiffusionSchedule

if TYPE_CHECKING:
    from uniserve.nn.parallel_pipeline import LayerPipeline


Size = TypeVar("Size")


class DiffusionMixin(Generic[Size]):
    """Prepare latent rows while leaving RNG ownership and solver advancement outside.

    Image models share scaling and conversion from their declared normal-draw
    representation to canonical patch rows. Multimodal models can implement the
    same numerical entry with their own packing and conditioning mathematics.
    """

    generation: ImageDiffusion | None = None

    @property
    def diffusion_pipeline(self) -> LayerPipeline | None:
        """Expose numerical pipeline feedback when the denoiser is partitioned."""

        return None

    if TYPE_CHECKING:
        # Compositions may expose nested modules through properties or register
        # them directly. A read-only typing contract must not install a runtime
        # descriptor that would mask PyTorch's registered-module lookup.
        @property
        def modalities(self) -> tuple[str, ...]: ...

        @property
        def solver(self) -> EulerSolver | CleanSampleEulerSolver: ...
    else:
        modalities = ("image",)

    @property
    def prediction_type(self) -> str:
        return self.solver.prediction_type

    @property
    def prediction_dtype(self) -> torch.dtype:
        if self.generation is None:
            raise ValueError("image prediction requires an image diffusion module")
        return self.generation.prediction_dtype

    def validate_schedule(self, schedule: DiffusionSchedule) -> None:
        """Check that numerical endpoints cover every ordered modality."""

        if len(schedule.sigmas) != len(self.modalities):
            raise ValueError("diffusion schedule must follow the model modality order")

    def latent_shape(self, name: str, size: Size) -> tuple[int, ...]:
        """Return canonical image patch rows used by the numerical prediction."""

        generation = self.generation
        if name != "image" or generation is None or not isinstance(size, ImageSize):
            raise ValueError("image latent shape requires its image modality and raster")
        return (
            generation.image_tokens(size.height, size.width),
            generation.latent_patch_size**2 * generation.latent_channels,
        )

    def noise_shape(self, name: str, size: Size) -> tuple[int, ...]:
        """Return the native draw representation before patch conversion."""

        if name != "image" or self.generation is None or not isinstance(size, ImageSize):
            raise ValueError("image noise shape requires its image modality and raster")
        return self.generation.latent_shape(size.height, size.width)

    @torch.inference_mode()
    def prepare_latents(
        self,
        batch: DiffusionBatch[Size],
        *,
        noise: TensorViews,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> None:
        """Scale native noise before patch conversion into caller-owned state.

        Views have a leading logical-row dimension. Noise may alias state's
        backing; all writes are to the supplied state view. This preserves the
        native normal-draw order without staging copies inside the model.
        """

        generation = self.generation
        if generation is None or tuple(batch.latents) != ("image",):
            raise ValueError("shared image latent preparation requires one image modality")
        target, source = state["image"], noise["image"]
        if target.shape[0] != batch.row_count or source.shape[0] != batch.row_count:
            raise ValueError("image noise and state must align with their logical rows")
        for index, shape in enumerate(batch.sizes):
            if not isinstance(shape, ImageSize):
                raise ValueError("image latent preparation requires image sizes")
            expected = generation.latent_shape(shape.height, shape.width)
            if tuple(source[index].shape) != expected:
                raise ValueError("image noise disagrees with its native draw geometry")
            if target.device != source.device or target.dtype != source.dtype:
                raise ValueError("noise must be delivered in the state device and dtype")
            raw = target[index].reshape(expected)
            raw.copy_(source[index])
            raw.mul_(generation.noise_scale(shape.height, shape.width))
            neural = generation.patchify(raw)
            target[index].copy_(neural.reshape_as(target[index]))

    @abstractmethod
    def forward_diffusion(
        self,
        batch: DiffusionBatch[Size],
        *,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        """Predict each named modality without advancing its supplied latent state."""
