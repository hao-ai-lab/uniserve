"""Bounded physical KV imports and explicit page-representation conversion.

A KV import installs a published KV extent (`KvTransfer`) into pages of this
worker's cache that the scheduler has assigned.
`KVCacheManager.prepare_install` validates the destination and calls
`CacheImports.reserve`, which claims the destination token ranges and runs
the copy on a dedicated `HostLane`. `KVCacheManager.install` later hands the
completed import to resident ownership through `CacheImports.adopt`. An
import that is not adopted is abandoned instead, for example when its batch
closes its inputs, its request is cancelled, its buffer is released, or the
imports stop.

Two copy paths exist. The direct path fetches the source bytes straight into
the destination pages when both use the same representation. The converted
path fetches each destination span into a fixed workspace, one of a pool
sized to the import threads, decodes FP8 sources through the numerical cache
library, and writes the result through the destination state's own encoding.

An import's workspace returns to the idle set only after the copy task has
finished, its workspace stream has drained, and every transport read has
physically retired. The destination stays reserved until the same holds and
the import has also been adopted or abandoned. Cancellation revokes
consumption but never releases storage a started read may still touch.
"""

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

from uniserve.cache import block_spans, mha
from uniserve.cache.state import decode_region
from uniserve.quantization import QuantizedTensor, Quantizer
from uniserve_worker.errors import invalid_descriptor, resource_error
from uniserve_worker.execution.host import HostLane
from uniserve_worker.protocol.identity import BufferId, RequestKey
from uniserve_worker.protocol.transfer import KvTransfer, TensorTransfer
from uniserve_worker.transport.fetch import fetch_tensor
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.ticket import TransferTicket

if TYPE_CHECKING:
    from uniserve_worker.storage.kv_cache import KVCacheManager


def cache_transfer_workspace_bytes(
    *,
    num_layers: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    capacity: int,
) -> int:
    """Return the startup bytes for every bounded import workspace.

    Startup memory accounting (`bootstrap.report`) reserves this amount, so it
    must match the `TransferBuffer` allocation in `CacheImports.__init__`.
    Each import thread has two raw pages (one per K/V field) large enough for
    float64 elements, one FP32 conversion page, and an FP32 scale buffer of
    ``[page_size, 2, num_layers, num_kv_heads]``: a destination span touches
    at most one source page per token and one source scale group per head.
    The tail of a raw page serves as rounding scratch while it holds an FP8
    input. No prefix-sized conversion allocation is needed.
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
    """A scheduler-owned KV destination retained through physical input access.

    ``ranges`` maps each destination page to the ``(offset, count)`` token
    range the import writes: the published suffix span, or the whole page
    for ``initialized_pages``, which the import zeroes before copying.
    ``completion`` is the copy task's future, resolved at reservation when
    there is nothing to copy or zero. ``retirement`` resolves once the import
    is adopted or abandoned and all of its physical access has finished;
    until then the import stays registered and
    `KVCacheManager.require_writable` rejects writes overlapping its
    ``ranges``. ``cancelled`` marks abandonment; ``released`` marks adoption
    or abandonment.

    The underscored fields track the copy task, its leased workspace and its
    started reads; `CacheImports._reclaim` uses them to decide when the
    workspace and destination are reclaimed.
    """

    request_pool_idx: int
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
    """One physical import worker's fixed page buffers and device stream.

    With ``elements`` the element count of one KV page
    (``block_size * layers * kv_heads * head_dim``), ``raw`` is uint8
    ``[2, elements * 8]``: one staging page per K/V field, sized for float64
    source elements. ``values`` is FP32
    ``[block_size, layers, kv_heads, head_dim]``, the conversion page.
    ``scales`` is FP32 ``[block_size, 2, layers, kv_heads]`` and holds the
    source scale rows of one span. ``stream`` is None off CUDA.
    """

    raw: torch.Tensor
    values: torch.Tensor
    scales: torch.Tensor
    stream: torch.cuda.Stream | None


class CacheImports:
    """Own bounded import execution, destination leases and conversion storage.

    At most ``capacity`` copy tasks are admitted at once, executed by up to
    four threads that each lease one `TransferBuffer`. Registered imports are
    keyed by their publication's source `BufferId`.

    The import registry, idle workspaces and closed flag are guarded by
    ``_condition``, whose default lock is reentrant: public methods such as
    `release` and `stop` call `abandon` while holding it.
    """

    def __init__(self, pool: KVCacheManager, *, capacity: int) -> None:
        self.pool = pool
        # `cache_transfer_workspace_bytes` accounts for one workspace per
        # thread with the same bound.
        workers = min(4, int(capacity))
        self._tasks = HostLane(max_inflight=capacity, workers=workers)

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

        # Idle workspaces. A task leases one in `_acquire`; `_reclaim` returns
        # it only after the task's physical access has retired.
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
        """Register the owner's wake for import completions and retirements.

        The host lane calls it after each copy task; `_reclaim` calls it
        after resolving a destination's retirement.
        """
        self._wake = wake
        self._tasks.set_completion_wake(wake)

    def __bool__(self) -> bool:
        # True while any import still holds a destination.
        with self._condition:
            return bool(self._writes)

    def dependencies(
        self, ranges: tuple[tuple[int, int, int], ...]
    ) -> tuple[Future[None], ...]:
        """Return retirements of imports whose destinations overlap ``ranges``.

        ``ranges`` holds ``(page, offset, count)`` token spans. Each returned
        future is a registered import's `CacheImport.retirement`.
        """
        with self._condition:
            return tuple(
                write.retirement
                for write in self._writes.values()
                if self.pool._ranges_overlap(write.ranges, ranges)
            )

    def reserve(
        self,
        publication: KvTransfer,
        *,
        request_pool_idx: int,
        pages: tuple[int, ...],
        initialized_pages: tuple[int, ...],
        transports: Mapping[str, Transport],
    ) -> CacheImport:
        """Reserve exact destination ranges before any host or device read.

        Checks with `KVCacheManager.require_reusable` that each destination
        range is free, registers the import, and submits its copy to the
        import lane. An import with no tensors and no pages to initialize
        completes immediately without using lane capacity. The caller
        (`KVCacheManager.prepare_install`) has already validated the pages
        against the pool and, when the import extends an installed base,
        against the request's installed base pages.

        Raises:
            ResourceError: When a destination range is still in use, or the
                import lane is closed or has no capacity.
            WorkerError: ``invalid_descriptor`` when the pages are invalid,
                this is closed, or the publication source already has a
                registered import.
        """
        # Destination token ranges: the published suffix within its pages,
        # plus every page this import initializes, which it zeroes whole.
        suffix = publication.published_extent - publication.base_extent
        ranges = {
            page: (offset, count)
            for page, offset, count in block_spans(
                pages,
                publication.base_extent,
                suffix,
                block_size=self.pool.info.block_size,
            )
        }
        ranges.update(
            (page, (0, self.pool.info.block_size)) for page in initialized_pages
        )
        for page, (offset, count) in ranges.items():
            self.pool.require_reusable(
                (page,), group=publication.group_id, start=offset, length=count
            )

        reservation = (
            self._tasks.reserve()
            if publication.tensors or initialized_pages
            else None
        )
        write = CacheImport(
            request_pool_idx,
            pages,
            initialized_pages,
            publication,
            ranges,
        )

        buffer = publication.source
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
        # A released import is removed from ``_writes`` once it retires.
        with self._condition:
            return self._writes.get(write.publication.source) is write

    def adopt(self, write: CacheImport) -> None:
        """Hand a completed import to resident cache ownership.

        The destination stays registered until its physical reads retire;
        its `CacheImport.retirement` resolves then.

        Raises:
            RuntimeError: When the copy task has not finished.
            WorkerError: ``invalid_descriptor`` when the import is no longer
                registered or was cancelled. A failed or cancelled copy task
                re-raises through its future.
        """
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
        """Revoke consumption while preserving every started physical access.

        Cancels the import's transport reads and makes its running task stop
        at its next `_require_active` check or cancelled read. The
        destination and workspace stay reserved until that task and every
        started read retire. A no-op for an import that is not registered
        here.
        """
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
        # Abandon the imports whose publication source is among ``buffers``.
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
        # Abandon every import owned by ``requests`` except retained sources.
        with self._condition:
            for write in tuple(self._writes.values()):
                if (
                    write.publication.source.owner in requests
                    and write.publication.source not in retained
                ):
                    self.abandon(write)

    def retirement_ready(
        self,
        buffers: set[BufferId],
        requests: set[RequestKey],
        retained: frozenset[BufferId],
    ) -> bool:
        # True when no registered import belongs to the selected buffers or,
        # outside ``retained``, to the selected requests.
        with self._condition:
            return not any(
                buffer in buffers
                or (buffer.owner in requests and buffer not in retained)
                for buffer in self._writes
            )

    def stop(self) -> None:
        """Abandon every import and stop the import lane.

        Must run before transport shutdown. Rejects later reservations,
        cancels unsubmitted tasks, and returns after the lane threads exit.
        Destinations whose reads have not retired stay registered, which
        `require_retired` reports.
        """
        with self._condition:
            self._closed = True
            for write in tuple(self._writes.values()):
                self.abandon(write)
            self._condition.notify_all()
        self._tasks.close()

    def require_retired(self) -> None:
        """Require that no import still holds a destination.

        Raises:
            ResourceError: When any import is still registered.
        """
        with self._condition:
            if self._writes:
                raise resource_error("KV imports still own physical storage")

    def _acquire(self, write: CacheImport) -> TransferBuffer:
        # Runs on an import thread. A thread can wait here because a finished
        # import keeps its workspace until its reads retire; cancellation
        # and `stop` wake the wait so it can fail.
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
        # Track a started read until the transport reports physical
        # retirement. A read started after cancellation is cancelled at once.
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
        # Caller holds ``_condition``. A failed or cancelled task can finish
        # before its transport access. Its workspace and cache pages remain
        # unavailable until both retire. An import whose stream failed to
        # drain (`_stream_finished` False) is never reclaimed.
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
            self._writes.pop(write.publication.source, None)
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
        # Start transport reads of ``region`` of the source tensor into
        # ``destination``, using only locations whose backend this worker has
        # a transport for. Each started read is retained on ``write``.
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
        # Block this import thread until each read is stream-ready, then
        # order the workspace stream after its fence, if it has one. A failed
        # or cancelled read raises from `result`.
        for ticket in tickets:
            ready = Event()
            ticket.add_done_callback(ready.set)
            ready.wait()
            ticket.result(workspace.stream)

    def _copy(
        self, write: CacheImport, transports: Mapping[str, Transport]
    ) -> None:
        """Run one import on an import thread.

        Leases a workspace, zeroes ``initialized_pages``, then copies the
        published suffix through the direct or converted path. On every exit
        it attempts to drain the workspace stream and then records the task
        as finished for `_reclaim`.
        """
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
                self._require_active(write)
                if write.initialized_pages:
                    for name in self.pool.layers:
                        self.pool.cache.zero_blocks(
                            name, write.initialized_pages
                        )
                if workspace.stream is not None:
                    # Backend copy streams must not race initialization of the
                    # reserved pages. This host wait belongs to the bounded
                    # import worker, never to the request execution loop.
                    workspace.stream.synchronize()

                publication = write.publication
                if not publication.tensors:
                    return

                # The direct path needs an identical page representation. For
                # FP8 the page scales must also transfer unchanged: equal page
                # sizes, a page-aligned base, the same compute dtype, and all
                # of this rank's heads in one source scale group, because each
                # destination page holds one scale per layer and K/V field.
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
                # Publish completion even when the drain failed, so
                # `_reclaim` sees a finished task with an undrained stream.
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
        """Fetch source values, and FP8 scales, straight into cache pages.

        The fetch region selects this worker's layers and KV heads by their
        global offsets (``layer_offset``, ``kv_head_offset``) on the
        publication's layer and head axes.
        """
        publication = write.publication
        suffix = publication.published_extent - publication.base_extent
        spans = block_spans(
            write.pages,
            publication.base_extent,
            suffix,
            self.pool.info.block_size,
        )
        info = self.pool.info

        # ``layer`` indexes the publication's global layer axis.
        for layer, name in enumerate(self.pool.layers, info.layer_offset):
            # Limit physical reads to one layer, independent of model depth.
            tickets: list[TransferTicket] = []
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
                    # Page-aligned bases and equal page sizes make source page
                    # ``i`` of the suffix destination span ``i``.
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

        # Mark the pages initialized only while the import is still active.
        self._require_active(write)
        pages = tuple(page for page, _, _ in spans)
        for name in self.pool.layers:
            self.pool.cache.mark_initialized(
                name, pages, fields=("key", "value")
            )

    def _copy_converted(
        self,
        write: CacheImport,
        transports: Mapping[str, Transport],
        workspace: TransferBuffer,
    ) -> None:
        """Copy the suffix span by span through the conversion workspace.

        For each destination span, fetches the source K/V tokens of this
        worker's shard into ``workspace.raw``. FP8 sources are decoded per
        source page and scale head group into ``workspace.values``; other
        sources are copied as fetched. Each layer is then written through
        `mha.State.copy_region`, which applies the destination encoding.
        """
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
        # Tokens of the suffix already copied.
        logical = 0

        for page, offset, count in block_spans(
            write.pages, start, suffix, block_size=self.pool.info.block_size
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

            # Token offset of this span within its first source page.
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
                # Source scale rows cover the publication pages this span
                # touches, counted from the source page holding the first
                # suffix token.
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
                    # Intersect source scale pages and head groups here; the
                    # numerical cache library owns decoding and rounding.
                    compute_dtype = getattr(torch, publication.compute_dtype)
                    # Rounding scratch lies in the raw page past the FP8
                    # input, which occupies at most its first eighth. FP32 and
                    # FP64 compute dtypes need no rounding scratch.
                    rounded = (
                        workspace.raw[field_index, elements * 4 :].view(
                            compute_dtype
                        )
                        if compute_dtype not in {torch.float32, torch.float64}
                        else workspace.raw[field_index, :0].view(compute_dtype)
                    )
                    # Walk source pages (one scale row each) and, within each,
                    # this worker's heads grouped by source scale group.
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
                            for layer in range(self.pool.info.num_layers):
                                page_region = (
                                    slice(position, position + length),
                                    layer,
                                    slice(head, head + heads),
                                )
                                encoded = source[page_region]
                                numerical = Quantizer("fp8").from_tensors(
                                    {
                                        "values": encoded,
                                        "scale": workspace.scales[
                                            scale_index,
                                            field_index,
                                            layer,
                                            group,
                                        ].reshape(()),
                                    },
                                    shape=tuple(encoded.shape),
                                    dtype=compute_dtype,
                                )
                                decode_region(
                                    numerical,
                                    tuple(
                                        slice(0, size) for size in encoded.shape
                                    ),
                                    device=values.device,
                                    workspace={
                                        "values": values[page_region],
                                        "rounded": rounded,
                                    },
                                )
                            head += heads
                            group += 1
                        position += length
                        scale_index += 1
                        token_offset = 0

                # Store the converted tokens into the destination page by layer.
                for layer, name in enumerate(self.pool.layers):
                    trailing_slice = (
                        slice(0, self.pool.info.num_kv_heads),
                        slice(0, self.pool.info.head_dim),
                    )
                    # The cache manager admits only MHA state layers.
                    state = self.pool.cache.state(name)
                    assert isinstance(state, mha.State)
                    state.copy_region(
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

            # The next span fetches into the same workspace buffers, so this
            # span's conversion and copies must finish first.
            if workspace.stream is not None:
                workspace.stream.synchronize()
            for ticket in tickets:
                ticket.close()
            logical += count
