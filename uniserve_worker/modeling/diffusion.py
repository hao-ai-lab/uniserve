"""Shared image latent preparation and explicit diffusion capability."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, cast

import torch

from ..nn.diffusion.spec import DiffusionSpec, ModalitySpec, ScheduleRule
from .batch import DiffusionBatch, TensorOutput
from .components import Call
from .geometry import MediaShape, Shape
from .image_diffusion import ImageDiffusion
from .resources import TensorNeeds, TensorSchema
from .tensors import TensorViews

if TYPE_CHECKING:
    from ..nn.parallel_pipeline import LayerPipeline
    from .model import Model


class DiffusionMixin:
    """Prepare latent rows while leaving RNG ownership and solver advancement outside.

    Image models share scaling and conversion from their declared normal-draw
    representation to canonical patch rows. Multimodal models can implement the
    same numerical entry with their own packing and conditioning mathematics.
    """

    generation: ImageDiffusion | None

    def tensor_specs(self, call: Call, shape: Shape) -> TensorNeeds:
        """Declare canonical image prediction rows independently of solver storage."""

        if call is not Call.DIFFUSION:
            # Capability mixins cooperate before the single Model base in MRO.
            return cast("Model", super()).tensor_specs(call, shape)
        generation = self.generation
        if generation is None or not isinstance(shape, MediaShape) or shape.frames != 1:
            raise ValueError("image diffusion requires a single-frame media shape")
        rows = generation.image_tokens(shape.height, shape.width)
        width = generation.latent_patch_size**2 * generation.latent_channels
        return TensorNeeds(
            outputs={
                "image": TensorSchema((rows, width), getattr(torch, generation.prediction_dtype))
            }
        )

    @property
    def diffusion_pipeline(self) -> LayerPipeline | None:
        """Expose numerical pipeline feedback when the denoiser is partitioned."""

        return None

    def diffusion_spec(self, shape: MediaShape, steps: int) -> DiffusionSpec:
        """Declare image noise, prediction, CFG, and schedule mathematics."""

        generation = self.generation
        if generation is None or shape.frames != 1:
            raise ValueError("image diffusion requires image generation geometry")
        rows = generation.image_tokens(shape.height, shape.width)
        width = generation.latent_patch_size**2 * generation.latent_channels
        return DiffusionSpec(
            modalities=(
                ModalitySpec(
                    name="image",
                    latent_shape=(rows, width),
                    noise_shape=generation.latent_shape(shape.height, shape.width),
                    schedule=ScheduleRule(
                        generation.schedule_direction,
                        generation.schedule_shift_domain,
                        generation.timestep_shift or 1.0,
                    ),
                    prediction=generation.prediction,
                    prediction_dtype=getattr(torch, generation.prediction_dtype),
                    state_dtype="input",
                    noise_dtype="state",
                    noise_scale=generation.noise_scale(shape.height, shape.width),
                ),
            ),
            steps=steps,
            cfg=generation.cfg_recipe,
            max_cfg_branches=generation.max_cfg_branches,
            solver="euler",
            noise_device="input",
            seed_transform="splitmix_coordinate",
        )

    @torch.inference_mode()
    def prepare_latents(
        self,
        batch: DiffusionBatch,
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
        for index, shape in enumerate(batch.shapes):
            if shape.frames != 1:
                raise ValueError("image latent preparation requires single-frame geometry")
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
        batch: DiffusionBatch,
        *,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        """Predict each named modality without advancing its supplied latent state."""
