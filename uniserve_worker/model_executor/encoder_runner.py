"""Encoder numerical calls and reusable text input storage."""

from __future__ import annotations

import torch

from uniserve.model import PatchEncoder
from uniserve.nn.vae import PatchAutoencoder
from uniserve.runtime.device import fill_cpu_ints
from uniserve.runtime.resources import close_resources
from uniserve_worker.errors import InputError
from uniserve_worker.storage.host_buffers import HostBuffers

from .cuda_graph import CUDAGraphRunner
from .model_runner import ModelRunner
from .output import ExecutionOutput


class EncoderRunner(ModelRunner):
    """Evaluate encoder tensors and retain reusable text input storage.

    Execution owns packed vision graph selection and replay. Numerical helpers
    allocate fixed image slots and preserve each image's unpadded features.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokens: torch.Tensor | None = None
        self._host: HostBuffers | None = None

    @property
    def packs_images(self) -> bool:
        """Whether vision calls run through packed image graphs."""
        return (
            bool(self.execution.pools)
            and isinstance(self.model, PatchEncoder)
            and self.model.max_patches is not None
        )

    @torch.inference_mode()
    def prepare_tokens(self, tokens, *, capacity):
        """Copy a flat token sequence into the runner's device token buffer.

        The first call allocates a ``capacity``-long int64 device buffer and a
        two-deep ring of host sources; later calls reuse both, so every call
        must pass the same ``capacity``. The copy is issued on the
        current stream and, on CUDA, fenced so its host source is not
        refilled before the copy completes.

        Returns:
            A [1, len(tokens)] view of the device buffer, which the next call
            overwrites.

        Raises:
            InputError: If ``tokens`` is empty or longer than ``capacity``.
        """
        if not 1 <= len(tokens) <= capacity:
            raise InputError("text encoder input exceeds its token capacity")
        if self._tokens is None or self._host is None:
            self._tokens = torch.empty(
                capacity, dtype=torch.int64, device=self.device
            )
            self._host = HostBuffers(
                (capacity,), dtype=torch.int64, depth=2, device=self.device
            )
        index, host = self._host.acquire()
        fill_cpu_ints(host, tokens)
        target = self._tokens[: len(tokens)]
        target.copy_(host[: len(tokens)], non_blocking=self._tokens.is_cuda)
        self._host.record_copy(index)
        return target.view(1, -1)

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
        return torch.zeros(
            (slots * self.model.max_patches, 3 * self.model.patch_size**2),
            dtype=dtype,
            device=self.device,
        )

    def packed_inputs(self, buffer, slots):
        """Bind pixel rows and empty grid metadata for one captured shape."""
        pixels = buffer[: slots * self.model.max_patches]
        grids = torch.tensor(
            self.model.packed_grids((), slots),
            dtype=torch.long,
            device=self.device,
        )
        return pixels, grids

    def capture_packed(self, inputs):
        """Capture encoding over fixed pixel and grid tensors."""
        return CUDAGraphRunner.capture(
            self.execution.context,
            inputs,
            lambda values: self.model.encode_packed(*values),
            pools=self.execution.pools,
        )

    def prepare_packed(self, inputs, images, shapes):
        """Copy patch rows and grid metadata into a captured numerical input."""
        encoder = self.model
        capacity = encoder.max_patches
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
            encoder.packed_grids(shapes, len(images)), dtype=torch.long
        )
        grids.copy_(
            layout.pin_memory() if grids.is_cuda else layout,
            non_blocking=True,
        )

    def unpack_packed(self, output, shapes):
        """Retain unpadded feature views across subsequent graph replays."""
        encoder = self.model
        capacity = encoder.max_patches
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

    def close(self):
        close_resources(
            super().close,
            *(() if self._host is None else (self._host.close,)),
        )
        self._tokens = self._host = None
