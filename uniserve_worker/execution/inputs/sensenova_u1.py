"""Construct SenseNova U1's image conditioning at the worker boundary."""

import torch

from uniserve.model import LatentInput
from uniserve.nn.functional import unpatchify
from uniserve_models.sensenova_u1 import DenoiserInput, ImageConditioning

from .image import ImageInputs


class Inputs(ImageInputs):
    @property
    def max_tokens(self):
        return self.denoiser.config.max_image_seq_len

    def bind(self, *, samples, sizes, timesteps, positions, attention, step_index):
        images = []
        for sample, size in zip(samples, sizes, strict=True):
            # This tower consumes the current image, which must be rebuilt
            # after every solver update and must remain distinct from a
            # trajectory's conditioning prefix in the K/V cache.
            pixels = unpatchify(
                sample.unsqueeze(0),
                size,
                patch_size=self.denoiser.patch_size,
                channels=self.denoiser.latent_channels,
            )
            patch = self.denoiser.config.vision.patch_size
            grid = torch.tensor(
                [[size.height // patch, size.width // patch]],
                device=sample.device,
                dtype=torch.int64,
            )
            scale = sample.new_tensor([self.denoiser.noise_scale.scale(sample.shape[0])])
            images.append(ImageConditioning(pixels, grid, scale))
        return DenoiserInput(
            latents={
                "image": tuple(
                    LatentInput(value, time) for value, time in zip(samples, timesteps, strict=True)
                )
            },
            sizes=sizes,
            step_index=step_index,
            positions=positions,
            sequence_lengths=tuple(self.sequence_length(size) for size in sizes),
            attention=attention,
            images=tuple(images),
        )
