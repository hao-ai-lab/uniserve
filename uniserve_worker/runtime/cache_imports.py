"""Bounded physical KV imports and explicit page-representation conversion."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from threading import Condition, Event
from typing import TYPE_CHECKING

import torch

from uniserve.quantization import QuantizedTensor

from ..foundation.errors import invalid_descriptor, resource_error
from ..protocol.identity import BufferId, RequestKey
from ..protocol.transfer import KvTransfer, TensorTransfer
from ..transfer.layout import fetch_tensor
from ..transfer.tickets import TransferTicket, Transport
from .block_tables import page_spans
from .cpu import CpuPool

if TYPE_CHECKING:
    from .cache_manager import CacheManager


def cache_transfer_workspace_bytes(
    *,
    num_layers: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    capacity: int,
) -> int:
    """Return the startup bytes for every bounded import workspace.

    Each worker has two raw pages large enough for float64, one FP32 conversion
    page, and at most one source scale per destination token and head. The raw
    page becomes rounding/absolute-value scratch after its input has been
    consumed. No prefix-sized conversion allocation is needed.
    """
    workers = min(4, int(capacity))
    elements = (
        int(page_size) * int(num_layers) * int(num_kv_heads) * int(head_dim)
    )
    scales = 2 * int(page_size) * int(num_layers) * int(num_kv_heads)
    # 20 bytes per element: two raw float64 pages (2 x 8) plus one FP32
    # conversion page (4). Scale entries are FP32 (4 bytes each).
    return workers * (20 * elements + 4 * scales)


@dataclass(slots=True)
class CacheImport:
    """A scheduler-owned KV destination retained through physical input.

    access.
    """

    buffer: BufferId
    request_pool_idx: int
    group_id: int
    pages: tuple[int, ...]
    initialized_pages: tuple[int, ...]
    publication: KvTransfer
    ranges: dict[int, tuple[int, int]]
    completion: Future[None] = field(default_factory=Future)
    retirement: Future[None] = field(default_factory=Future)
    cancelled: bool = False
    released: bool = False
    _tickets: set[TransferTicket] = field(default_factory=set, repr=False)
    _workspace: TransferBuffer | None = field(default=None, repr=False)
    _work_finished: bool = field(default=False, repr=False)
    _stream_finished: bool = field(default=False, repr=False)


@dataclass(slots=True)
class TransferBuffer:
    """One physical import worker's fixed page buffers and device stream."""

    raw: torch.Tensor
    values: torch.Tensor
    scales: torch.Tensor
    stream: torch.cuda.Stream | None


class CacheImports:
    """Own bounded import execution, destination leases and conversion.

    storage.
    """

    def __init__(self, pool: CacheManager, *, capacity: int) -> None:
        self.pool = pool
        workers = min(4, int(capacity))
        self._tasks = CpuPool(capacity=capacity, workers=workers)

        # One conversion page holds a full KV page:
        # [tokens, layers, heads, dim].
        shape = (
            pool.info.block_size,
            pool.info.num_layers,
            pool.info.num_kv_heads,
            pool.info.head_dim,
        )
        elements = (
            pool.info.block_size
            * pool.info.num_layers
            * pool.info.num_kv_heads
            * pool.info.head_dim
        )

        device = pool.cache.device
        self._available = deque(
            TransferBuffer(
                raw=torch.empty(
                    (2, elements * 8), dtype=torch.uint8, device=device
                ),
                values=torch.empty(shape, dtype=torch.float32, device=device),
                scales=torch.empty(
                    (
                        pool.info.block_size,
                        2,
                        pool.info.num_layers,
                        pool.info.num_kv_heads,
                    ),
                    dtype=torch.float32,
                    device=device,
                ),
                stream=torch.cuda.Stream(device=device)
                if device.type == "cuda"
                else None,
            )
            for _ in range(workers)
        )

        self._writes: dict[BufferId, CacheImport] = {}
        self._condition = Condition()
        self._closed = False
        self._wake: Callable[[], None] | None = None

    def set_completion_wake(self, wake: Callable[[], None] | None) -> None:
        """Wake the owner after import jobs or destination retirements.

        finish.
        """
        self._wake = wake
        self._tasks.set_completion_wake(wake)

    def __bool__(self) -> bool:
        with self._condition:
            return bool(self._writes)

    def dependencies(
        self, ranges: tuple[tuple[int, int, int], ...]
    ) -> tuple[Future[None], ...]:
        """Return writer retirements that precede reuse of overlapping.

        storage.
        """
        with self._condition:
            return tuple(
                write.retirement
                for write in self._writes.values()
                if self.pool._ranges_overlap(write.ranges, ranges)
            )

    def reserve(
        self,
        buffer: BufferId,
        publication: KvTransfer,
        *,
        request_pool_idx: int,
        group: int,
        pages: tuple[int, ...],
        initialized_pages: tuple[int, ...],
        transports: Mapping[str, Transport],
    ) -> CacheImport:
        """Reserve exact destination ranges before any host or device read.

        starts.
        """
        suffix = publication.published_extent - publication.base_extent
        ranges = {
            page: (offset, count)
            for page, offset, count in page_spans(
                pages,
                publication.base_extent,
                suffix,
                page_size=self.pool.info.block_size,
            )
        }
        ranges.update(
            (page, (0, self.pool.info.block_size)) for page in initialized_pages
        )
        for page, (offset, count) in ranges.items():
            self.pool.require_reusable(
                (page,), group=group, start=offset, length=count
            )

        reservation = (
            self._tasks.reserve()
            if publication.tensors or initialized_pages
            else None
        )
        write = CacheImport(
            buffer,
            request_pool_idx,
            group,
            pages,
            initialized_pages,
            publication,
            ranges,
        )

        try:
            with self._condition:
                if self._closed or buffer in self._writes:
                    raise invalid_descriptor(
                        "KV import destination is closed or already reserved"
                    )
                self._writes[buffer] = write
            if reservation is None:
                write._work_finished = True
                write._stream_finished = True
                write.completion.set_result(None)
            else:
                write.completion = reservation.submit(
                    self._copy, write, transports
                )
        except BaseException:
            if reservation is not None:
                reservation.abandon()
            with self._condition:
                self._writes.pop(buffer, None)
            raise
        return write

    def owns(self, write: CacheImport) -> bool:
        with self._condition:
            return self._writes.get(write.buffer) is write

    def adopt(self, write: CacheImport) -> None:
        """Hand a completed import to resident cache ownership."""
        if not write.completion.done():
            raise RuntimeError("KV import was observed before input readiness")
        write.completion.result()
        with self._condition:
            if not self.owns(write) or write.cancelled:
                raise invalid_descriptor(
                    "KV import destination is no longer active"
                )
            write.released = True
            self._reclaim(write)

    def abandon(self, write: CacheImport) -> None:
        """Revoke consumption while preserving every started physical access."""
        with self._condition:
            if not self.owns(write):
                return
            write.cancelled = True
            write.released = True
            for ticket in tuple(write._tickets):
                ticket.cancel()
            self._reclaim(write)
            self._condition.notify_all()

    def release(self, buffers: tuple[BufferId, ...]) -> None:
        with self._condition:
            for buffer in buffers:
                write = self._writes.get(buffer)
                if write is not None:
                    self.abandon(write)

    def cancel_requests(
        self,
        requests: frozenset[RequestKey],
        *,
        retained: frozenset[BufferId] = frozenset(),
    ) -> None:
        with self._condition:
            for write in tuple(self._writes.values()):
                if (
                    write.buffer.owner in requests
                    and write.buffer not in retained
                ):
                    self.abandon(write)

    def retirement_ready(
        self,
        buffers: set[BufferId],
        requests: set[RequestKey],
        retained: frozenset[BufferId],
    ) -> bool:
        with self._condition:
            return not any(
                buffer in buffers
                or (buffer.owner in requests and buffer not in retained)
                for buffer in self._writes
            )

    def stop(self) -> None:
        """Cancel queued imports and stop submission before transport.

        shutdown.
        """
        with self._condition:
            self._closed = True
            for write in tuple(self._writes.values()):
                self.abandon(write)
            self._condition.notify_all()
        self._tasks.close()

    def require_retired(self) -> None:
        with self._condition:
            if self._writes:
                raise resource_error("KV imports still own physical storage")

    def _acquire(self, write: CacheImport) -> TransferBuffer:
        with self._condition:
            while not self._available:
                self._require_active(write)
                self._condition.wait()
            self._require_active(write)
            workspace = self._available.popleft()
            write._workspace = workspace
            return workspace

    def _require_active(self, write: CacheImport) -> None:
        if self._closed or write.cancelled:
            raise resource_error("KV import was cancelled")

    def _retain(self, write: CacheImport, ticket: TransferTicket) -> None:
        with self._condition:
            write._tickets.add(ticket)
            ticket.add_retirement_callback(
                partial(self._read_retired, write, ticket)
            )
            if write.cancelled:
                ticket.cancel()

    def _read_retired(self, write: CacheImport, ticket: TransferTicket) -> None:
        with self._condition:
            write._tickets.discard(ticket)
            self._reclaim(write)

    def _reclaim(self, write: CacheImport) -> None:
        # A failed or cancelled task can finish before its transport access.
        # Its workspace and cache pages remain unavailable until both retire.
        if (
            not write._work_finished
            or not write._stream_finished
            or write._tickets
        ):
            return
        if write._workspace is not None:
            self._available.append(write._workspace)
            write._workspace = None
            self._condition.notify_all()
        if write.released and not write.retirement.done():
            self._writes.pop(write.buffer, None)
            write.retirement.set_result(None)
            if self._wake is not None:
                self._wake()

    def _fetch(
        self,
        write: CacheImport,
        tensor: TensorTransfer,
        destination: torch.Tensor | tuple[torch.Tensor, ...],
        transports: Mapping[str, Transport],
        *,
        region: tuple[slice, ...] | None = None,
    ) -> tuple[TransferTicket, ...]:
        self._require_active(write)
        return fetch_tensor(
            tensor,
            destination,
            bindings={
                (location.source, location.backend): transports[
                    location.backend
                ]
                for location in tensor.locations
                if location.backend in transports
            },
            region=region,
            retain=partial(self._retain, write),
        )

    @staticmethod
    def _consume(
        tickets: tuple[TransferTicket, ...], workspace: TransferBuffer
    ) -> None:
        for ticket in tickets:
            ready = Event()
            ticket.add_done_callback(ready.set)
            ready.wait()
            ticket.result(workspace.stream)

    def _copy(
        self, write: CacheImport, transports: Mapping[str, Transport]
    ) -> None:
        workspace = None
        stream_finished = True
        try:
            workspace = self._acquire(write)
            context = (
                torch.cuda.stream(workspace.stream)
                if workspace.stream is not None
                else nullcontext()
            )
            with context:
                self.pool.initialize_import(write)
                if workspace.stream is not None:
                    # Backend copy streams must not race initialization of the
                    # reserved pages. This host wait belongs to the bounded
                    # import worker, never to the request execution loop.
                    workspace.stream.synchronize()

                publication = write.publication
                if not publication.tensors:
                    return

                same_format = (
                    publication.tensors[0].dtype == self.pool.info.dtype
                )
                direct = same_format and (
                    self.pool.info.dtype != "float8_e4m3fn"
                    or (
                        publication.page_size == self.pool.info.block_size
                        # A partial installed page owns its destination scale.
                        # Its base may have arrived through another TP layout.
                        and publication.base_extent % self.pool.info.block_size
                        == 0
                        and publication.compute_dtype
                        == str(self.pool.compute_dtype).removeprefix("torch.")
                        and self.pool.info.kv_head_offset
                        // publication.scale_head_size
                        == (
                            self.pool.info.kv_head_offset
                            + self.pool.info.num_kv_heads
                            - 1
                        )
                        // publication.scale_head_size
                    )
                )
                if direct:
                    self._copy_direct(write, transports, workspace)
                else:
                    self._copy_converted(write, transports, workspace)
        finally:
            try:
                if workspace is not None and workspace.stream is not None:
                    workspace.stream.synchronize()
            except BaseException:
                stream_finished = False
                raise
            finally:
                with self._condition:
                    write._work_finished = True
                    write._stream_finished = stream_finished
                    self._reclaim(write)

    def _copy_direct(
        self,
        write: CacheImport,
        transports: Mapping[str, Transport],
        workspace: TransferBuffer,
    ) -> None:
        publication = write.publication
        suffix = publication.published_extent - publication.base_extent
        spans = page_spans(
            write.pages,
            publication.base_extent,
            suffix,
            self.pool.info.block_size,
        )
        info = self.pool.info

        for layer, name in enumerate(self.pool.layers, info.layer_offset):
            # Limit physical reads to one layer, independent of model depth.
            tickets = []
            state = self.pool.cache.state(name)
            for index, tensor_name in enumerate(("key", "value")):
                tensor = state.tensors[tensor_name]
                values = (
                    tensor.buffers()["values"]
                    if isinstance(tensor, QuantizedTensor)
                    else tensor
                )
                # Each span view is [tokens, 1, kv heads, head dim]: the
                # unsqueezed axis matches the single-layer fetch region.
                destination = tuple(
                    values[page, offset : offset + count].unsqueeze(1)
                    for page, offset, count in spans
                )
                tickets.extend(
                    self._fetch(
                        write,
                        publication.tensors[index],
                        destination,
                        transports,
                        region=(
                            slice(0, suffix),
                            slice(layer, layer + 1),
                            slice(
                                info.kv_head_offset,
                                info.kv_head_offset + info.num_kv_heads,
                            ),
                            slice(0, info.head_dim),
                        ),
                    )
                )
                if isinstance(tensor, QuantizedTensor):
                    scales = tensor.buffers()["scale"]
                    destination = tuple(
                        scales[page : page + 1] for page, _, _ in spans
                    )
                    head = info.kv_head_offset // publication.scale_head_size
                    tickets.extend(
                        self._fetch(
                            write,
                            publication.tensors[2],
                            destination,
                            transports,
                            region=(
                                slice(0, len(spans)),
                                slice(index, index + 1),
                                slice(layer, layer + 1),
                                slice(head, head + 1),
                            ),
                        )
                    )
            self._consume(tuple(tickets), workspace)
            for ticket in tickets:
                ticket.close()
        self.pool.mark_import_scales(write)

    def _copy_converted(
        self,
        write: CacheImport,
        transports: Mapping[str, Transport],
        workspace: TransferBuffer,
    ) -> None:
        publication = write.publication
        start = publication.base_extent
        suffix = publication.published_extent - start
        dtype = getattr(torch, publication.tensors[0].dtype)
        quantized = dtype is torch.float8_e4m3fn
        trailing = (
            self.pool.info.num_layers,
            self.pool.info.num_kv_heads,
            self.pool.info.head_dim,
        )
        itemsize = workspace.raw.view(dtype).element_size()
        logical = 0

        for page, offset, count in page_spans(
            write.pages, start, suffix, page_size=self.pool.info.block_size
        ):
            elements = (
                count
                * self.pool.info.num_layers
                * self.pool.info.num_kv_heads
                * self.pool.info.head_dim
            )
            # Raw staging views per K/V field: [tokens, layers, kv heads, dim].
            raw = tuple(
                workspace.raw[field, : elements * itemsize]
                .view(dtype)
                .reshape(count, *trailing)
                for field in range(2)
            )
            # Fetch this span's tokens and this worker's layer/head shard.
            region = tuple(
                slice(start, start + extent)
                for start, extent in zip(
                    (
                        logical,
                        self.pool.info.layer_offset,
                        self.pool.info.kv_head_offset,
                        0,
                    ),
                    (count, *trailing),
                    strict=True,
                )
            )
            tickets = tuple(
                ticket
                for tensor, destination in zip(
                    publication.tensors[:2], raw, strict=True
                )
                for ticket in self._fetch(
                    write, tensor, destination, transports, region=region
                )
            )

            source_offset = (start + logical) % publication.page_size
            if quantized:
                head_start = (
                    self.pool.info.kv_head_offset // publication.scale_head_size
                )
                head_end = (
                    self.pool.info.kv_head_offset
                    + self.pool.info.num_kv_heads
                    - 1
                ) // publication.scale_head_size + 1
                # Source scale rows cover the publication pages this
                # span touches.
                scale_start = (
                    start + logical
                ) // publication.page_size - start // publication.page_size
                scale_count = (
                    source_offset + count + publication.page_size - 1
                ) // publication.page_size
                tickets += self._fetch(
                    write,
                    publication.tensors[2],
                    workspace.scales[
                        :scale_count, :, :, : head_end - head_start
                    ],
                    transports,
                    region=(
                        slice(scale_start, scale_start + scale_count),
                        slice(0, 2),
                        slice(
                            self.pool.info.layer_offset,
                            self.pool.info.layer_offset
                            + self.pool.info.num_layers,
                        ),
                        slice(head_start, head_start + head_end - head_start),
                    ),
                )

            self._consume(tickets, workspace)

            for field_index, source in enumerate(raw):
                values = source
                if quantized:
                    values = workspace.values[:count]
                    values.copy_(source)
                    # Dequantize in FP32. Source scale pages and head groups do
                    # not align with the span boundaries, so walk both.
                    position, scale_index, token_offset = 0, 0, source_offset
                    while position < count:
                        length = min(
                            publication.page_size - token_offset,
                            count - position,
                        )
                        head, group = 0, 0
                        while head < self.pool.info.num_kv_heads:
                            heads = min(
                                publication.scale_head_size
                                - (self.pool.info.kv_head_offset + head)
                                % publication.scale_head_size,
                                self.pool.info.num_kv_heads - head,
                            )
                            scale = workspace.scales[
                                scale_index, field_index, :, group
                            ].reshape(1, self.pool.info.num_layers, 1, 1)
                            values[
                                position : position + length,
                                :,
                                head : head + heads,
                            ].mul_(scale)
                            head += heads
                            group += 1
                        position += length
                        scale_index += 1
                        token_offset = 0

                    compute_dtype = getattr(torch, publication.compute_dtype)
                    if compute_dtype != torch.float32:
                        # An encoded source first rounds through its original
                        # compute representation, before destination encoding.
                        rounded = (
                            workspace.raw[
                                field_index,
                                : elements
                                * torch.empty(
                                    (), dtype=compute_dtype
                                ).element_size(),
                            ]
                            .view(compute_dtype)
                            .reshape_as(values)
                        )
                        rounded.copy_(values)
                        values.copy_(rounded)

                # Store the converted tokens into the destination page by layer.
                for layer, name in enumerate(self.pool.layers):
                    trailing_slice = (
                        slice(0, self.pool.info.num_kv_heads),
                        slice(0, self.pool.info.head_dim),
                    )
                    self.pool.cache.state(name).copy_region(
                        values[:, layer],
                        field=("key", "value")[field_index],
                        block=page,
                        source_slice=(slice(0, count), *trailing_slice),
                        target_slice=(
                            slice(offset, offset + count),
                            *trailing_slice,
                        ),
                        workspace={},
                    )

            if workspace.stream is not None:
                workspace.stream.synchronize()
            for ticket in tickets:
                ticket.close()
            logical += count
