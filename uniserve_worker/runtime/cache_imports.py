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

from uniserve.runtime.kv_cache import page_spans
from uniserve.tensors import TensorRegion

from ..foundation.errors import invalid_descriptor, resource_error
from ..protocol.batch import BufferId, KvTransfer, RequestKey, TensorTransfer
from ..transfer.layout import fetch_tensor
from ..transfer.tickets import TransferTicket, Transport
from .cpu import CpuPool

if TYPE_CHECKING:
    from .cache_manager import CacheManager


def cache_transfer_workspace_bytes(
    *, num_layers: int, page_size: int, num_kv_heads: int, head_dim: int, capacity: int
) -> int:
    """Return the startup bytes for every bounded import workspace.

    Each worker has two raw pages large enough for float64, one FP32 conversion
    page, at most one source scale per destination token and head, and a per-layer scale
    reduction. The raw page becomes rounding/absolute-value scratch after its
    input has been consumed. No prefix-sized conversion allocation is needed.
    """

    workers = min(4, int(capacity))
    elements = int(page_size) * int(num_layers) * int(num_kv_heads) * int(head_dim)
    scales = 2 * int(page_size) * int(num_layers) * int(num_kv_heads)
    return workers * (20 * elements + 4 * (scales + int(num_layers)))


@dataclass(slots=True)
class CacheImport:
    """A scheduler-owned KV destination retained through physical input access."""

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
    scale: torch.Tensor
    stream: torch.cuda.Stream | None


class CacheImports:
    """Own bounded import execution, destination leases and conversion storage."""

    def __init__(self, pool: CacheManager, *, capacity: int) -> None:
        self.pool = pool
        workers = min(4, int(capacity))
        self._tasks = CpuPool(capacity=capacity, workers=workers)
        shape = (
            pool.cache.page_size,
            pool.cache.config.num_layers,
            pool.cache.config.num_kv_heads,
            pool.cache.config.head_dim,
        )
        elements = (
            pool.cache.page_size
            * pool.cache.config.num_layers
            * pool.cache.config.num_kv_heads
            * pool.cache.config.head_dim
        )
        device = pool.cache.k.device
        self._available = deque(
            TransferBuffer(
                raw=torch.empty((2, elements * 8), dtype=torch.uint8, device=device),
                values=torch.empty(shape, dtype=torch.float32, device=device),
                scales=torch.empty(
                    (
                        pool.cache.page_size,
                        2,
                        pool.cache.config.num_layers,
                        pool.cache.config.num_kv_heads,
                    ),
                    dtype=torch.float32,
                    device=device,
                ),
                scale=torch.empty(
                    (pool.cache.config.num_layers,), dtype=torch.float32, device=device
                ),
                stream=torch.cuda.Stream(device=device) if device.type == "cuda" else None,
            )
            for _ in range(workers)
        )
        self._writes: dict[BufferId, CacheImport] = {}
        self._condition = Condition()
        self._closed = False
        self._wake: Callable[[], None] | None = None

    def set_completion_wake(self, wake: Callable[[], None] | None) -> None:
        """Wake the owner after import jobs or destination retirements finish."""

        self._wake = wake
        self._tasks.set_completion_wake(wake)

    def __bool__(self) -> bool:
        with self._condition:
            return bool(self._writes)

    def dependencies(self, ranges: tuple[tuple[int, int, int], ...]) -> tuple[Future[None], ...]:
        """Return writer retirements that precede reuse of overlapping storage."""

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
        """Reserve exact destination ranges before any host or device read starts."""

        suffix = publication.published_extent - publication.base_extent
        ranges = {
            page: (offset, count)
            for page, offset, count in page_spans(
                pages, publication.base_extent, suffix, page_size=self.pool.cache.page_size
            )
        }
        ranges.update((page, (0, self.pool.cache.page_size)) for page in initialized_pages)
        for page, (offset, count) in ranges.items():
            self.pool.require_reusable((page,), group=group, start=offset, length=count)
        reservation = self._tasks.reserve() if publication.tensors or initialized_pages else None
        write = CacheImport(
            buffer, request_pool_idx, group, pages, initialized_pages, publication, ranges
        )
        try:
            with self._condition:
                if self._closed or buffer in self._writes:
                    raise invalid_descriptor("KV import destination is closed or already reserved")
                self._writes[buffer] = write
            if reservation is None:
                write._work_finished = True
                write._stream_finished = True
                write.completion.set_result(None)
            else:
                write.completion = reservation.submit(self._copy, write, transports)
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
                raise invalid_descriptor("KV import destination is no longer active")
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
        self, requests: frozenset[RequestKey], *, retained: frozenset[BufferId] = frozenset()
    ) -> None:
        with self._condition:
            for write in tuple(self._writes.values()):
                if write.buffer.owner in requests and write.buffer not in retained:
                    self.abandon(write)

    def retirement_ready(
        self, buffers: set[BufferId], requests: set[RequestKey], retained: frozenset[BufferId]
    ) -> bool:
        with self._condition:
            return not any(
                buffer in buffers or (buffer.owner in requests and buffer not in retained)
                for buffer in self._writes
            )

    def stop(self) -> None:
        """Cancel queued imports and stop submission before transport shutdown."""

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
            ticket.add_retirement_callback(partial(self._read_retired, write, ticket))
            if write.cancelled:
                ticket.cancel()

    def _read_retired(self, write: CacheImport, ticket: TransferTicket) -> None:
        with self._condition:
            write._tickets.discard(ticket)
            self._reclaim(write)

    def _reclaim(self, write: CacheImport) -> None:
        # A failed or cancelled task can finish before its transport access.
        # Its workspace and cache pages remain unavailable until both retire.
        if not write._work_finished or not write._stream_finished or write._tickets:
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
        region: TensorRegion | None = None,
    ) -> tuple[TransferTicket, ...]:
        self._require_active(write)
        return fetch_tensor(
            tensor,
            destination,
            bindings={
                (location.source, location.backend): transports[location.backend]
                for location in tensor.locations
                if location.backend in transports
            },
            region=region,
            retain=partial(self._retain, write),
        )

    @staticmethod
    def _consume(tickets: tuple[TransferTicket, ...], workspace: TransferBuffer) -> None:
        for ticket in tickets:
            ready = Event()
            ticket.add_done_callback(ready.set)
            ready.wait()
            ticket.result(workspace.stream)

    def _copy(self, write: CacheImport, transports: Mapping[str, Transport]) -> None:
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
                same_format = publication.tensors[0].dtype == str(
                    self.pool.cache.k.dtype
                ).removeprefix("torch.")
                direct = same_format and (
                    not self.pool.cache.is_quantized
                    or (
                        publication.page_size == self.pool.cache.page_size
                        # A partial installed page owns its destination scale.
                        # Its base may have arrived through another TP layout.
                        and publication.base_extent % self.pool.cache.page_size == 0
                        and publication.compute_dtype
                        == str(self.pool.cache.config.dtype).removeprefix("torch.")
                        and self.pool.cache.config.kv_head_offset // publication.scale_head_size
                        == (
                            self.pool.cache.config.kv_head_offset
                            + self.pool.cache.config.num_kv_heads
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
        fields = self.pool.cache.transfer_views(
            write.pages, start=publication.base_extent, length=suffix
        )
        tickets: list[TransferTicket] = []
        for index, (tensor, destination) in enumerate(
            zip(publication.tensors, fields, strict=True)
        ):
            if index < 2:
                region = TensorRegion(
                    (
                        0,
                        self.pool.cache.config.layer_offset,
                        self.pool.cache.config.kv_head_offset,
                        0,
                    ),
                    (
                        suffix,
                        self.pool.cache.config.num_layers,
                        self.pool.cache.config.num_kv_heads,
                        self.pool.cache.config.head_dim,
                    ),
                )
            else:
                region = TensorRegion(
                    (
                        0,
                        0,
                        self.pool.cache.config.layer_offset,
                        self.pool.cache.config.kv_head_offset // publication.scale_head_size,
                    ),
                    (len(destination), 2, self.pool.cache.config.num_layers, 1),
                )
            tickets.extend(self._fetch(write, tensor, destination, transports, region=region))
        self._consume(tuple(tickets), workspace)
        self.pool.mark_import_scales(write)
        for ticket in tickets:
            ticket.close()

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
            self.pool.cache.config.num_layers,
            self.pool.cache.config.num_kv_heads,
            self.pool.cache.config.head_dim,
        )
        itemsize = workspace.raw.view(dtype).element_size()
        logical = 0
        for page, offset, count in page_spans(
            write.pages, start, suffix, page_size=self.pool.cache.page_size
        ):
            elements = (
                count
                * self.pool.cache.config.num_layers
                * self.pool.cache.config.num_kv_heads
                * self.pool.cache.config.head_dim
            )
            raw = tuple(
                workspace.raw[field, : elements * itemsize].view(dtype).reshape(count, *trailing)
                for field in range(2)
            )
            region = TensorRegion(
                (
                    logical,
                    self.pool.cache.config.layer_offset,
                    self.pool.cache.config.kv_head_offset,
                    0,
                ),
                (count, *trailing),
            )
            tickets = tuple(
                ticket
                for tensor, destination in zip(publication.tensors[:2], raw, strict=True)
                for ticket in self._fetch(write, tensor, destination, transports, region=region)
            )
            source_offset = (start + logical) % publication.page_size
            if quantized:
                head_start = self.pool.cache.config.kv_head_offset // publication.scale_head_size
                head_end = (
                    self.pool.cache.config.kv_head_offset + self.pool.cache.config.num_kv_heads - 1
                ) // publication.scale_head_size + 1
                scale_start = (
                    start + logical
                ) // publication.page_size - start // publication.page_size
                scale_count = (
                    source_offset + count + publication.page_size - 1
                ) // publication.page_size
                tickets += self._fetch(
                    write,
                    publication.tensors[2],
                    workspace.scales[:scale_count, :, :, : head_end - head_start],
                    transports,
                    region=TensorRegion(
                        (scale_start, 0, self.pool.cache.config.layer_offset, head_start),
                        (scale_count, 2, self.pool.cache.config.num_layers, head_end - head_start),
                    ),
                )
            self._consume(tickets, workspace)
            for field_index, values in enumerate(raw):
                self.pool.cache.copy_page(
                    values,
                    source_page_size=publication.page_size,
                    source_head_size=publication.scale_head_size,
                    source_compute_dtype=getattr(torch, publication.compute_dtype),
                    source_scales=workspace.scales,
                    values_buffer=workspace.values,
                    raw_buffer=workspace.raw,
                    scale_buffer=workspace.scale,
                    field=field_index,
                    page=page,
                    offset=offset,
                    source_offset=source_offset,
                )
            if workspace.stream is not None:
                workspace.stream.synchronize()
            for ticket in tickets:
                ticket.close()
            logical += count
