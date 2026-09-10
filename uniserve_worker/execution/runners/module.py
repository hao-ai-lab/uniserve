"""Fixed and exact-shape tensor module execution."""

from collections.abc import Callable
from functools import partial
from typing import TypeAlias

import torch

from ...nn.mesh import Communicator
from ..graph.backend import CudaGraphBackend, GraphExecutionError

TensorOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]
TensorSignature: TypeAlias = tuple[tuple[tuple[int, ...], torch.dtype, tuple[int, ...]], ...]


def tensor_signature(inputs: tuple[torch.Tensor, ...]) -> TensorSignature:
    """Include layout as well as shape in a tensor-only computation's identity."""

    return tuple((tuple(value.shape), value.dtype, tuple(value.stride())) for value in inputs)


def capture_required(missing: bool, groups: tuple[Communicator, ...], device: torch.device) -> bool:
    """Coordinate first-use work over the actual numerical communication groups."""

    if not groups:
        return missing
    decision = torch.tensor(int(missing), dtype=torch.int32, device=device)
    for group in groups:
        group.all_reduce_max(decision)
    return bool(decision.item())


class ModuleRunner:
    """Bind a callable to fixed inputs or one resident exact tensor signature.

    Outputs borrow this module's storage until its next execution. Different
    module instances retain independent outputs unless explicitly given a shared
    graph pool. The caller orders consumers before execution or retirement.
    """

    def __init__(
        self,
        forward: Callable[..., TensorOutput],
        *,
        device: torch.device,
        backend: CudaGraphBackend[TensorOutput] | None,
        inputs: tuple[torch.Tensor, ...] | None = None,
        groups: tuple[Communicator, ...] = (),
    ) -> None:
        self.forward = forward
        self.device = device
        self.backend = backend
        self.groups = groups
        self._fixed_inputs = inputs
        self._inputs: dict[TensorSignature, tuple[torch.Tensor, ...]] = {}
        self._closed = False

    @property
    def fixed(self) -> bool:
        return self._fixed_inputs is not None

    @torch.inference_mode()
    def warmup(self, *inputs: torch.Tensor) -> None:
        """Prepare eager/provider work with the same numerical callable."""

        values = inputs or self._fixed_inputs
        if values is None:
            raise ValueError("module warmup requires representative inputs")
        self.forward(*values)

    @torch.inference_mode()
    def capture(self) -> None:
        """Capture the fixed configuration after its numerical resources exist."""

        if self._fixed_inputs is None or self.backend is None:
            return
        key = tensor_signature(self._fixed_inputs)
        self.backend.capture_one(
            key, partial(self.forward, *self._fixed_inputs), keepalive=self._fixed_inputs
        )
        self._inputs[key] = self._fixed_inputs

    @torch.inference_mode()
    def run(self, *inputs: torch.Tensor) -> tuple[TensorOutput, str]:
        if self._closed:
            raise GraphExecutionError("module runner is closed")
        if any(not isinstance(value, torch.Tensor) for value in inputs):
            raise GraphExecutionError("module arguments must be tensors")
        if any(value.device != self.device for value in inputs):
            raise GraphExecutionError("module input device changed")
        key = tensor_signature(inputs)
        if self._fixed_inputs is not None:
            fixed_key = tensor_signature(self._fixed_inputs)
            if tuple(value[:2] for value in key) != tuple(value[:2] for value in fixed_key):
                raise GraphExecutionError("CUDA graph input geometry changed")
            # Copying preserves logical values from strided source views. The
            # executable's layout belongs to its fixed destination buffers.
            key = fixed_key
        if self.backend is None:
            return self.forward(*inputs), "eager"
        if self._fixed_inputs is not None:
            missing = False
        else:
            missing = capture_required(not self.backend.contains(key), self.groups, self.device)
        if missing:
            torch.cuda.current_stream(self.device).synchronize()
            self._discard_inputs()
            stable = tuple(
                torch.empty_strided(
                    value.shape, value.stride(), dtype=value.dtype, device=value.device
                )
                for value in inputs
            )
            for source, target in zip(inputs, stable, strict=True):
                target.copy_(source)
            self.backend.capture_one(key, partial(self.forward, *stable), keepalive=stable)
            self._inputs[key] = stable
        resident = self._inputs.get(key)
        if resident is not None:
            for source, target in zip(inputs, resident, strict=True):
                target.copy_(source)
        return self.backend.replay(key), "graph_capture" if missing else "graph_replay"

    def _discard_inputs(self) -> None:
        if self.backend is not None:
            for key in self._inputs:
                self.backend.discard(key)
        self._inputs.clear()

    def close(self) -> None:
        if self._closed:
            return
        if self.backend is not None:
            self.backend.close()
        self._inputs.clear()
        self._fixed_inputs = None
        self._closed = True
