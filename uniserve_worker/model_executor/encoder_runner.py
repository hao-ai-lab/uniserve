"""Encoder numerical calls and reusable text input storage."""

from __future__ import annotations

import torch

from uniserve.model import PatchEncoder, VisionInput
from uniserve.nn.vae import PatchAutoencoder
from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve.runtime.device import fill_cpu_ints
from uniserve.runtime.resources import close_resources
from uniserve_worker.errors import InputError
from uniserve_worker.protocol.output import ForwardStats
from uniserve_worker.storage.host_buffers import HostBuffers

from .cuda_graph import CUDAGraphRunner
from .model_runner import ModelRunner
from .output import ExecutionOutput

# Most images one packed vision graph encodes. The largest graph's
# activations bound the graph pool the vision graphs share (for
# DiffusionGemma, 16 slots of 2520 patch rows), and a call with more images
# replays it once per such group.
MAX_PACKED_IMAGES = 16


def packed_capacities(max_images: int) -> tuple[int, ...]:
    """Slot counts of the packed vision graphs for calls of ``max_images``.

    Every count from one to ``max_images``, capped at ``MAX_PACKED_IMAGES``,
    so a call pads no empty slots below the cap. The graphs share one pixel
    buffer and one pool, so each further count adds only its output.
    """
    return tuple(range(1, min(max_images, MAX_PACKED_IMAGES) + 1))


class EncoderRunner(ModelRunner):
    """Evaluate encoder inputs and retain their transfer backing.

    A ``PatchEncoder`` that packs fixed image slots (``max_patches``) runs
    on a runner with graph pools only through packed graphs, one per slot
    count of ``packed_capacities``, captured at startup (``capture_packed``).
    A call takes the graph of its image count, or the largest once per full
    group, and every packing of a capacity replays its graph.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokens: torch.Tensor | None = None
        self._host: HostBuffers | None = None
        # Captured packed vision graphs by slot count; each retains its
        # static ``(pixels, grids)`` input, the pixels a leading view of one
        # buffer every count shares.
        self._packed: dict[int, CUDAGraphRunner] = {}

    @property
    def packs_images(self) -> bool:
        """Whether vision calls run through packed image graphs."""
        return (
            bool(self.pools)
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

    @torch.inference_mode()
    def capture_packed(self, *, max_images: int, dtype: torch.dtype) -> None:
        """Capture every packed vision slot count at startup, largest first.

        ``max_images`` is the most vision calls one batch carries and
        ``dtype`` the dtype images are staged in. Each graph's static input
        holds ``dtype`` patch rows of empty slots; the graphs share the
        runner's pools and are charged to its graph storage.

        Raises:
            CUDAGraphError: After startup is sealed, or when graph residency
                exceeds its byte budget.
        """
        if not self.packs_images or self._packed:
            return
        if self._startup_complete:
            raise CUDAGraphError("packed capture is outside startup")
        encoder = self.model
        assert isinstance(encoder, PatchEncoder)
        assert encoder.max_patches is not None
        row = 3 * encoder.patch_size**2
        capacities = packed_capacities(max_images)

        # Only one packed call runs at a time on the runner's stream, so
        # every capacity reads the leading slots of one pixel buffer.
        with self.graph_storage.allocate(self):
            buffer = torch.zeros(
                (max(capacities) * encoder.max_patches, row),
                dtype=dtype,
                device=self.device,
            )
        # Prepare the largest call first, as for token and canvas graphs.
        # Smaller calls borrow its GEMM workspaces and capture-pool blocks;
        # growing through the catalog would retain every earlier workspace
        # because the captured kernels still address it.
        for slots in reversed(capacities):
            pixels = buffer[: slots * encoder.max_patches]
            with self.graph_storage.allocate(self):
                grids = torch.tensor(
                    encoder.packed_grids((), slots),
                    dtype=torch.long,
                    device=self.device,
                )
            graph = CUDAGraphRunner.capture(
                self.context,
                (pixels, grids),
                lambda inputs: encoder.encode_packed(*inputs),
                pools=self.pools,
            )
            try:
                self.graph_storage.check()
            except BaseException:
                graph.close()
                raise
            self._packed[slots] = graph

    @torch.inference_mode()
    def run_batch(self, batch, forward, *, eligible, borrow_output=False):
        """Replay packed vision graphs, or run the batch as ``ModelRunner``.

        With packed graphs every vision call replays them: its images are
        copied into the static slots of the smallest capacity that holds
        them, one full group of the largest capacity at a time, and each
        image's features are sliced from a copy of the graph's output.

        Raises:
            CUDAGraphError: When packed graphs are enabled but a capacity
                was not captured before serving.
        """
        if not self.packs_images or not isinstance(batch.inputs, VisionInput):
            return super().run_batch(
                batch, forward, eligible=eligible, borrow_output=borrow_output
            )

        if not self._packed:
            raise CUDAGraphError("packed vision graphs are not resident")
        inputs = batch.inputs
        largest = max(self._packed)
        values: list[torch.Tensor] = []
        with self.context.activate():
            for start in range(0, inputs.batch_size, largest):
                stop = min(start + largest, inputs.batch_size)
                values.extend(
                    self._replay_packed(
                        inputs.images[start:stop],
                        inputs.grid_shapes[start:stop],
                    )
                )
        replays = -(-inputs.batch_size // largest)
        return ExecutionOutput(
            tuple(values),
            stats=ForwardStats(
                cuda_graph_runtime_mode_counts={"graph_replay": replays},
                cuda_graph_replays=replays,
            ),
        )

    def _replay_packed(self, images, shapes):
        """Encode up to the largest capacity of images with one replay."""
        encoder = self.model
        assert isinstance(encoder, PatchEncoder)
        capacity = encoder.max_patches
        assert capacity is not None

        # Every slot count through the cap is captured, so selecting the
        # smallest sufficient count adds no empty image slots.
        slots = min(count for count in self._packed if count >= len(images))
        graph = self._packed[slots]
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
        pixels, grids = graph.inputs.value
        for index, image in enumerate(images):
            pixels[index * capacity : index * capacity + image.shape[0]].copy_(
                image
            )
        layout = torch.tensor(
            encoder.packed_grids(shapes, slots), dtype=torch.long
        )
        grids.copy_(
            layout.pin_memory() if grids.is_cuda else layout,
            non_blocking=True,
        )

        # One copy of the output keeps every image's features past the next
        # replay; image i's soft tokens lead slot i's.
        features = graph.replay().clone()
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
            *(graph.close for graph in self._packed.values()),
            *(() if self._host is None else (self._host.close,)),
        )
        self._packed.clear()
        self._tokens = self._host = None
