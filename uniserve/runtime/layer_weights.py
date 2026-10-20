"""Host-resident layer weights streamed to a few device buffers ahead of use.

A stack of identical layers keeps the dense weights of its projections in
page-locked host memory. The device holds ``slots`` layer-sized buffers, and
layer ``i``'s projection parameters are fixed views of buffer ``i % slots``,
so bound numerical operators read whichever layer was last copied there. A
layer's first projection call waits for its copy and enqueues the copies of
the layers after it; every projection call records that its buffer was read.
Copies run on their own stream, so the device's copy engines move the next
layers while the current ones compute.

Evaluation order makes two layers live at once: a layer's projections start
while the previous layer's feed-forward still produces rows, and that layer
is complete before the next one attends. With ``slots`` buffers, ``slots -
2`` layers can therefore be copied ahead. ``slots`` divides the layer count,
so a repeated pass (the next denoising step) finds every layer in its own
buffer; a copy never targets a buffer whose layer is live, and a layer whose
copy was not enqueued in time (the first pass, or one after an interrupted
pass) is copied when first used. A pass acquires its layers in order, so a
first use that does not follow the newest live layer starts a new pass.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from uniserve.quantization import QuantizedTensor

# Byte alignment of every storage placed in a layer buffer.
_ALIGNMENT = 256


def _slots(layers: int, minimum: int) -> int:
    """Return the fewest buffers, at least ``minimum``, dividing ``layers``."""
    for count in range(minimum, layers + 1):
        if layers % count == 0:
            return count
    raise ValueError(
        f"no buffer count of at least {minimum} divides {layers} layers"
    )


def layer_stack(module: nn.Module) -> tuple[nn.Module, ...]:
    """Return the repeated layer stack of ``module``.

    The stack is the ``ModuleList`` or ``ModuleDict`` under ``module`` whose
    children, at least four, share one class and hold the most parameter
    bytes among such containers.

    Raises:
        ValueError: No container holds four children of one class.
    """
    best, most = None, -1
    for container in module.modules():
        if not isinstance(container, (nn.ModuleList, nn.ModuleDict)):
            continue
        children = tuple(container.children())
        if len(children) < 4 or len({type(child) for child in children}) != 1:
            continue
        size = sum(
            parameter.numel() * parameter.element_size()
            for parameter in container.parameters()
        )
        if size > most:
            best, most = children, size
    if best is None:
        raise ValueError(
            f"{type(module).__name__} has no stack of four layers of one class"
        )
    return best


def _projections(layer: nn.Module) -> list[nn.Module]:
    """Return a layer's outermost projection modules in module order.

    A projection is a ``Linear`` or a merged column projection; a merged
    projection evaluates its branches itself, so its branches are not
    projections of their own.
    """
    from uniserve.nn.linear import Linear, MergedColumnParallelLinear

    found: list[nn.Module] = []

    def visit(module: nn.Module) -> None:
        if isinstance(module, (Linear, MergedColumnParallelLinear)):
            found.append(module)
            return
        for child in module.children():
            visit(child)

    visit(layer)
    return found


def _place(modules: Sequence[nn.Module]):
    """Lay out the storages of the modules' projection weight matrices.

    Returns ``(parameters, storages, size)``: every weight matrix with its
    byte offset in a layer buffer, every distinct backing storage with its
    offset, and the buffer bytes. Matrices sharing one storage keep their
    relative placement, so merged projection views stay merged. Vectors
    (biases) stay resident.

    Raises:
        ValueError: A projection weight is an encoded tensor.
    """
    parameters, storages, offsets = [], [], {}
    size = 0
    for module in modules:
        for parameter in module.parameters():
            if isinstance(parameter.data, QuantizedTensor):
                raise ValueError(
                    "streamed projection weights must be dense tensors"
                )
            if parameter.ndim < 2:
                continue
            storage = parameter.untyped_storage()
            if storage.data_ptr() not in offsets:
                offsets[storage.data_ptr()] = size
                storages.append((storage, size))
                size += -(-storage.nbytes() // _ALIGNMENT) * _ALIGNMENT
            offset = (
                offsets[storage.data_ptr()]
                + parameter.storage_offset() * parameter.element_size()
            )
            parameters.append((parameter, offset))
    return parameters, storages, size


class LayerWeights:
    """Stream the projection weights of identical layers from host memory.

    ``layers`` are evaluated in order, repeatedly, on one CUDA device. Build
    this after loading them and before binding any numerical operator to
    their parameters: operators prepared earlier keep views of the device
    storage this releases. Projection weight matrices move to pinned host
    memory; every other parameter stays on the device. The layers' calls
    must not be captured into CUDA graphs.

    Raises:
        ValueError: Fewer than four layers, no dense projection weights,
            projections of differing layout, an encoded projection weight,
            or weights not on one CUDA device.
    """

    @torch.no_grad()
    def __init__(self, layers: Sequence[nn.Module]):
        layers = tuple(layers)
        if len(layers) < 4:
            raise ValueError("streaming needs at least four layers")
        projections = [_projections(layer) for layer in layers]
        placements = [_place(modules) for modules in projections]
        shapes = {
            (
                size,
                tuple((storage.nbytes(), offset) for storage, offset in stores),
                tuple(
                    (tuple(p.shape), tuple(p.stride()), p.dtype, offset)
                    for p, offset in parameters
                ),
            )
            for parameters, stores, size in placements
        }
        if len(shapes) != 1 or not placements[0][0]:
            raise ValueError(
                "streamed layers need one common projection weight layout"
            )
        devices = {
            parameter.device
            for parameters, _, _ in placements
            for parameter, _ in parameters
        }
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("streamed weights must share one CUDA device")

        self.device = next(iter(devices))
        self.slots = _slots(len(layers), 4)
        self.ahead = self.slots - 2
        self.layer_bytes = placements[0][2]
        self._count = len(layers)
        self._buffers = [
            torch.empty(self.layer_bytes, dtype=torch.uint8, device=self.device)
            for _ in range(self.slots)
        ]
        # One page-locked host region of exactly every layer's bytes; the
        # caching pinned allocator would round each layer to a power of two.
        self._region = torch.empty(
            len(layers) * self.layer_bytes, dtype=torch.uint8
        )
        torch.cuda.check_error(
            torch.cuda.cudart().cudaHostRegister(
                self._region.data_ptr(), self._region.numel(), 0
            )
        )
        self._host = list(self._region.split(self.layer_bytes))
        self._layers: dict[int, int] = {}
        self._hooks = []
        for index, (modules, (parameters, stores, _)) in enumerate(
            zip(projections, placements, strict=True)
        ):
            host = self._host[index]
            for storage, offset in stores:
                source = torch.empty(
                    0, dtype=torch.uint8, device=self.device
                ).set_(storage)
                host[offset : offset + source.numel()].copy_(source)
            buffer = self._buffers[index % self.slots].untyped_storage()
            for parameter, offset in parameters:
                # Rebind in place: the parameter keeps its identity, which
                # operator bindings key on, and reads its layer's buffer.
                parameter.data = torch.empty(
                    0, dtype=parameter.dtype, device=self.device
                ).set_(
                    buffer,
                    offset // parameter.element_size(),
                    parameter.shape,
                    parameter.stride(),
                )
            for module in modules:
                self._layers[id(module)] = index
                self._hooks.append(
                    module.register_forward_pre_hook(self._before)
                )
                self._hooks.append(module.register_forward_hook(self._after))
        # The released device storages return to the device.
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        self._stream = torch.cuda.Stream(device=self.device)
        self._ready = [torch.cuda.Event() for _ in range(self.slots)]
        self._read = [torch.cuda.Event() for _ in range(self.slots)]
        # The layer whose copy each buffer last received, and the layers
        # whose projections may still run (at most two, newest last).
        self._loaded: list[int | None] = [None] * self.slots
        self._live: list[int] = []
        self.copied_bytes = 0

    def _occupied(self, slot: int) -> bool:
        return any(layer % self.slots == slot for layer in self._live)

    def _copy(self, layer: int) -> None:
        """Copy ``layer`` into its buffer after the buffer's last reads."""
        slot = layer % self.slots
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(self._read[slot])
            self._buffers[slot].copy_(self._host[layer], non_blocking=True)
            self._ready[slot].record(self._stream)
        self._loaded[slot] = layer
        self.copied_bytes += self.layer_bytes

    def _before(self, module, args):
        layer = self._layers[id(module)]
        if layer in self._live:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("streamed layer weights cannot be captured")
        if self._live and layer != (self._live[-1] + 1) % self._count:
            # A pass acquires its layers in order, so another layer starts
            # a new pass: the previous one, complete or interrupted, has
            # enqueued every read of its layers.
            self._live = []
        slot = layer % self.slots
        if self._occupied(slot):
            raise RuntimeError(
                "a streamed layer shares its buffer with a live layer"
            )
        if self._loaded[slot] != layer:
            self._copy(layer)
        torch.cuda.current_stream(self.device).wait_event(self._ready[slot])
        self._live = [*self._live[-1:], layer]
        # Copy the following layers into buffers no live layer reads; every
        # read of their previous layers is already enqueued.
        for step in range(1, self.ahead + 1):
            following = (layer + step) % self._count
            target = following % self.slots
            if self._loaded[target] != following and not self._occupied(target):
                self._copy(following)

    def _after(self, module, args, output):
        slot = self._layers[id(module)] % self.slots
        self._read[slot].record(torch.cuda.current_stream(self.device))

    def close(self) -> None:
        """Remove the hooks and release the host copies.

        The parameters keep their last buffer views.
        """
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._stream.synchronize()
        if self._host:
            self._host.clear()
            torch.cuda.check_error(
                torch.cuda.cudart().cudaHostUnregister(self._region.data_ptr())
            )
            del self._region
