"""Distributed immutable expert weights with independent local prefetch.

The DWDP mechanism follows TensorRT-LLM's composite-VA weight buffer:
resident expert pages alias their published allocation, while remote pages
alias two reusable local slots. Peer reads use CUDA copy engines. Only setup
is collective; inference waits solely on local copy/compute events.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from uniserve.model.inputs import TextSize
from uniserve.nn.moe import FusedMoE
from uniserve.quantization import QuantizedTensor
from uniserve.runtime._peer_storage import allocate_peer_tensor
from uniserve.runtime.backends import moe
from uniserve_kernels import peer_storage


@dataclass
class _Call:
    started: bool = False
    queued: int | None = None


class WeightPrefetch:
    """Own DWDP expert storage and one serialized execution domain.

    Construct collectively after loading expert shards, before preparing
    numerical operators. The modules receive full, contiguous borrowed
    weight views; their routing and numerical methods remain unchanged.
    The owner must outlive every context, graph, and module using those views.
    Calls sharing this owner must be serialized, including graph replays.
    Peers may execute unrelated batches, different numbers of invocations,
    or no invocations at all, while retaining their immutable weight storage.

    ``activate`` brackets a numerical invocation. Bindings call ``before``
    and ``after`` around each expert computation. The first call forks a
    copy stream; scope exit joins it, also when a forward skips its final
    expert layer. Thus the complete dependency chain is capturable.
    """

    @torch.inference_mode()
    def __init__(self, model: nn.Module, *, backend="auto"):
        layers = [
            layer
            for layer in model.modules()
            if isinstance(layer, FusedMoE)
            and layer.expert_group.size > 1
            and not layer.up_gate.weight.is_meta
        ]
        if not layers:
            raise ValueError("DWDP requires loaded expert shards")
        self.group = layers[0].expert_group
        self.device = layers[0].up_gate.weight.device
        if any(layer.expert_group != self.group for layer in layers):
            raise ValueError("DWDP weights must share one expert group")
        if self.device.type != "cuda":
            raise ValueError("DWDP requires CUDA peer memory")
        self._layers = {id(layer): index for index, layer in enumerate(layers)}
        self._copies: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        self._storage: list[torch.Tensor] = []
        self._slots: dict[tuple, torch.Tensor] = {}
        self._calls: list[_Call] = []
        self._stream = torch.cuda.Stream(device=self.device)
        self._ready = [torch.cuda.Event() for _ in range(2)]
        self._consumed = [torch.cuda.Event() for _ in range(2)]
        self._closed = False
        self.resident_bytes = self.buffer_bytes = self.transfer_bytes = 0

        try:
            self._prepare(layers, backend)
        except BaseException:
            # A peer can still read already-published pages when this rank
            # fails during setup. Do not unmap them or wait collectively in
            # the exception path; the worker terminates this failed process.
            from .resources import retain_until_exit

            retain_until_exit(self)
            raise

    def _prepare(self, layers, backend):
        for index, layer in enumerate(layers):
            provider = moe.resolve(backend, module=layer, device=self.device)
            if provider.name != "cutedsl":
                raise ValueError("DWDP requires CuTeDSL expert kernels")
            # Place the checkpoint encoding before publishing immutable pages.
            # Prepared operators only borrow views of that representation.
            provider.prepare(
                module=layer, size=TextSize(1, 1), workspace={}
            ).close()
            copies: list[tuple[torch.Tensor, torch.Tensor]] = []
            self._copies.append(copies)
            for name in ("up_gate", "down"):
                linear = getattr(layer, name)
                weight = linear.weight
                if isinstance(weight, QuantizedTensor):
                    fields = {}
                    for field, value in weight.buffers().items():
                        if field == "tensor_scale":
                            # Kernels derive alpha at preparation, so these
                            # few immutable scalars remain fully resident.
                            fields[field] = (
                                value
                                if value.ndim == 0
                                else self.group.all_gather(value, dim=0)
                            )
                        else:
                            fields[field] = self._field(
                                value, (index % 2, name, field), copies
                            )
                    full = weight.quantizer.from_tensors(
                        fields,
                        shape=(layer.num_experts, *weight.shape[1:]),
                        dtype=weight.dtype,
                        scale_layout=weight.scale_layout,
                        row_order=weight.row_order,
                    )
                else:
                    full = self._field(weight, (index % 2, name), copies)
                linear.weight = nn.Parameter(full, requires_grad=False)
                linear.num_experts = layer.num_experts
            layer.expert_slice = slice(0, layer.num_experts)
            # Keep the distributed group: an unbound standalone call must
            # still reject access to weights requiring runtime prefetch.

        # Publish every local page, including the page-edge bytes filled at
        # setup. After this fence no rank writes its resident shard again.
        self._publish()

    def _publish(self) -> None:
        marker = torch.ones(1, device=self.device, dtype=torch.int32)
        self.group.all_gather(marker)

    def _field(self, value, role, copies):
        """Create a contiguous full field with page-aligned resident edges."""
        value = value.contiguous()
        size, rank = self.group.size, self.group.rank
        page = peer_storage.allocation_granularity(self.device)
        shard_bytes = value.numel() * value.element_size()
        total = shard_bytes * size
        begin, end = rank * shard_bytes, (rank + 1) * shard_bytes
        first = begin // page * page
        # Every rank allocates the same extent, even when its local expert
        # interval starts at a different offset within a CUDA page.
        capacity = max(
            ((peer * shard_bytes % page + shard_bytes + page - 1) // page)
            * page
            for peer in range(size)
        )
        # Fabric handles must be mapped in full on GB200. All ranks publish
        # the same capacity, which can exceed one rank's minimally rounded
        # shard. Those extra edge pages remain resident and are filled once;
        # the remote reusable region starts after the complete allocation.
        last = first + capacity
        backing = allocate_peer_tensor(
            self.group, (capacity,), dtype=torch.uint8
        )
        self._storage.append(backing)
        resident = backing[rank * capacity : (rank + 1) * capacity]
        resident[begin - first : end - first].copy_(
            value.view(torch.uint8).flatten()
        )
        self.resident_bytes += capacity
        self._publish()

        pieces = []
        for side, length in (
            ("pre", first),
            ("post", max(0, (total + page - 1) // page * page - last)),
        ):
            if side == "post":
                pieces.append(resident)
            if not length:
                continue
            key = (*role, side, length)
            slot = self._slots.get(key)
            if slot is None:
                slot = peer_storage.empty(
                    (length,), dtype=torch.uint8, device=self.device
                )
                self._slots[key] = slot
                self.buffer_bytes += length
            pieces.append(slot)
        whole = peer_storage.map_segments(pieces)[:total]
        self._storage.append(whole)

        # Edge bytes share immutable local pages. Fetch them once at setup;
        # steady-state copies target only remote pages. Source ranges always
        # stay inside the peer's owned expert interval, never its edge bytes.
        for peer in range(size):
            if peer == rank:
                continue
            peer_begin, peer_end = peer * shard_bytes, (peer + 1) * shard_bytes
            source_begin = peer * capacity + peer_begin % page
            for low, high, persistent in (
                (0, first, False),
                (first, begin, True),
                (end, min(last, total), True),
                (last, total, False),
            ):
                start, stop = max(low, peer_begin), min(high, peer_end)
                if start >= stop:
                    continue
                offset = source_begin + start - peer_begin
                destination = whole[start:stop]
                source = backing[offset : offset + stop - start]
                if persistent:
                    destination.copy_(source)
                else:
                    copies.append((destination, source))
                    self.transfer_bytes += stop - start
        shape = (value.shape[0] * size, *value.shape[1:])
        return whole.view(value.dtype).view(shape)

    @contextmanager
    def activate(self):
        """Bracket one eager or captured invocation, lazily for graph replay."""
        if self._closed:
            raise RuntimeError("DWDP weight owner is closed")
        call = _Call()
        self._calls.append(call)
        try:
            yield
        finally:
            if call.started:
                torch.cuda.current_stream(self.device).wait_stream(self._stream)
            self._calls.pop()

    def _prefetch(self, index: int) -> None:
        slot = index % 2
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(self._consumed[slot])
            for destination, source in self._copies[index]:
                # Both views have this process's local CUDA device identity;
                # the source mapping resolves to immutable peer pages. A
                # contiguous same-dtype copy uses DMA rather than an SM kernel.
                destination.copy_(source, non_blocking=True)
            self._ready[slot].record(self._stream)

    def before(self, module: nn.Module) -> None:
        """Wait for this layer and overlap the next layer's peer reads."""
        index = self._layers[id(module)]
        if not self._calls:
            raise RuntimeError("DWDP expert call requires an active context")
        call = self._calls[-1]
        compute = torch.cuda.current_stream(self.device)
        if not call.started:
            for event in self._consumed:
                event.record(compute)
            self._stream.wait_stream(compute)
            call.started = True
        if call.queued != index:
            self._prefetch(index)
        compute.wait_event(self._ready[index % 2])
        following = index + 1
        call.queued = following if following < len(self._copies) else None
        if call.queued is not None:
            self._prefetch(call.queued)

    def after(self, module: nn.Module) -> None:
        """Release the layer's slot only after its last weight reader."""
        slot = self._layers[id(module)] % 2
        self._consumed[slot].record(torch.cuda.current_stream(self.device))

    def contains(self, module: nn.Module) -> bool:
        return id(module) in self._layers

    def close(self) -> None:
        """Release owned references after all contexts and graphs retire."""
        if not self._closed:
            self._stream.synchronize()
            self._copies.clear()
            self._storage.clear()
            self._slots.clear()
            self._closed = True
