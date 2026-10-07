"""Encoder numerical calls and reusable text input storage."""

from __future__ import annotations

from typing import cast

import torch

from uniserve.model import PatchEncoder, TextSize
from uniserve.nn.vae import PatchAutoencoder
from uniserve_worker._uniserve_ipc import EncoderRunner as _EncoderRunner
from uniserve_worker.errors import InputError

from .output import ExecutionOutput


def encoder_inputs(kind, values):
    """Flatten text sequences and describe the numerical preparation size."""
    if kind == "text":
        values = tuple(value.reshape(-1) for value in values)
        return values, TextSize(
            sum(value.numel() for value in values), len(values)
        )
    return values, tuple(value.shape for value in values)


def text_positions(encoder, tokens, device, image_grids, video_grids):
    """Place the encoder's rotary coordinates beside its input tokens."""
    return encoder.positions(
        tokens, image_grids=image_grids, video_grids=video_grids
    ).to(device, non_blocking=True)


def conditioning_inputs(features, capacity):
    """Pad text features and retain the valid length for attention masking."""
    count = features.shape[0]
    padded = features.new_zeros((capacity, *features.shape[1:]))
    padded[:count].copy_(features)
    lengths = torch.full((1,), count, dtype=torch.int32, device=features.device)
    return padded, lengths


def text_features(encoder, capacity, dtype, device):
    """Construct startup features in the text encoder's output layout."""
    layout = encoder.output_layout(capacity, dtype)["conditioning"]
    return torch.zeros(layout.shape, dtype=layout.dtype, device=device)


class EncoderRunner(_EncoderRunner):
    """Evaluate encoder tensors and retain reusable text input storage.

    Execution owns packed vision graph selection and replay. Numerical helpers
    allocate fixed image slots and preserve each image's unpadded features.
    """

    def batch_forward(self, batch, *, padded=False):
        """Encode a homogeneous image batch through its declared capability."""
        if isinstance(self.model, PatchEncoder):
            if self.packs_images:
                return self.execution.encode_images(self, batch.inputs)
            return ExecutionOutput(self.model.encode(batch.inputs))
        if isinstance(self.model, PatchAutoencoder):
            return ExecutionOutput(
                tuple(
                    self.model.encode(torch.stack(batch.inputs.images)).unbind(
                        0
                    )
                )
            )
        raise InputError("encoder has no batched image capability")

    def allocate_packed(self, slots, dtype):
        """Allocate shared patch rows; graph capacities borrow leading slots."""
        encoder = cast(PatchEncoder, self.model)
        return torch.zeros(
            (slots * cast(int, encoder.max_patches), 3 * encoder.patch_size**2),
            dtype=dtype,
            device=self.device,
        )

    def packed_inputs(self, buffer, slots):
        """Bind pixel rows and empty grid metadata for one captured shape."""
        encoder = cast(PatchEncoder, self.model)
        pixels = buffer[: slots * cast(int, encoder.max_patches)]
        grids = torch.tensor(
            encoder.packed_grids((), slots),
            dtype=torch.long,
            device=self.device,
        )
        return pixels, grids

    def prepare_packed(self, inputs, images, shapes):
        """Copy patch rows and grid metadata into a captured numerical input."""
        encoder = cast(PatchEncoder, self.model)
        capacity = cast(int, encoder.max_patches)
        if any(
            shape is None
            or image.ndim != 2
            or image.shape[0] != shape[0] * shape[1]
            for image, shape in zip(images, shapes, strict=True)
        ):
            raise InputError(
                "packed vision images require patch rows covering their grids"
            )

        # The graph reads its static input in place: image i's rows fill the
        # front of slot i, and the grids describe the packing. Rows past an
        # image keep earlier, finite values, which only padding reads.
        pixels, grids = inputs
        for index, image in enumerate(images):
            pixels[index * capacity : index * capacity + image.shape[0]].copy_(
                image
            )
        layout = torch.tensor(
            encoder.packed_grids(shapes, len(images)),
            dtype=torch.long,
            device="cpu",
            pin_memory=grids.is_cuda,
        )
        grids.copy_(layout, non_blocking=True)

    def unpack_packed(self, output, shapes):
        """Retain unpadded feature views across subsequent graph replays."""
        encoder = cast(PatchEncoder, self.model)
        capacity = cast(int, encoder.max_patches)
        # One copy keeps each image's features past reuse of captured outputs.
        features = output.clone()
        tokens = capacity // encoder.downsample**2
        return tuple(
            features[
                index * tokens : index * tokens
                + height * width // encoder.downsample**2
            ]
            for index, (height, width) in enumerate(shapes)
        )
