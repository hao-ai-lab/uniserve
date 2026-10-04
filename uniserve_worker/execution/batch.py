"""Numerical resources owned by one in-flight scheduler submission.

`Executor.submit` creates one `BatchState` per `Batch` and owns it until
`Executor.poll` consumes its result, `submit` itself fails, or the worker
closes. The execution modules fill it in stage order: `prepare_batch` and
`prepare_inputs` record storage dependencies, input reservations and
predicate captures, and `image.reserve_images` the host preparation of
inline input images; `reserve_outputs` binds one `PendingOutput` per call;
`commit_batch` (or `execute_batch` on failure) records the final outputs.
The native executor owns admission, launch order, failure and delivery.
The batch runner materializes outputs and retires their resources. Its callbacks
registered by `on_dependencies_ready` may run on a thread that completes a
dependency rather than the worker thread.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from threading import Lock
from typing import cast

import torch

from uniserve.runtime.resources import close_resources
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution.host import HostTask
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.protocol.batch import Batch, TensorPublication
from uniserve_worker.protocol.call import Call, CallStatus
from uniserve_worker.protocol.identity import (
    BufferId,
    CallId,
    CallIdentity,
    RequestKey,
)
from uniserve_worker.protocol.output import (
    BatchOutput,
    ForwardStats,
    RequestOutput,
)
from uniserve_worker.protocol.transfer import KvTransfer
from uniserve_worker.storage.cache_imports import CacheImport
from uniserve_worker.storage.kv_cache import KVCacheManager
from uniserve_worker.storage.latent_pool import LatentImport, LatentPool
from uniserve_worker.storage.output import OutputBuffer
from uniserve_worker.storage.tensor_store import TensorRead, TensorStore
from uniserve_worker.transport.ticket import TransferTicket


@dataclass(slots=True)
class BatchState:
    """Retain numerical inputs, physical dependencies, and outputs.

    The `Executor` submits inputs, launches computation, and materializes
    results. This object has no callback that can execute its batch or
    advance the worker.

    Every call has the same kind and component, so execution, publication,
    and failure belong to the batch as a whole. Physical retirement remains
    distinct from making its outputs visible.
    """

    batch: Batch
    propagate_errors: bool = False
    # The call each of this batch's calls follows in its request, derived by
    # `RequestPool.predecessors` from this rank's request state; None for
    # independent work.
    predecessors: dict[CallId, CallId | None] = field(default_factory=dict)

    # Physical input reservations keyed by input buffer id, held until
    # `close_inputs` completes the reads and abandons unadopted imports.
    tensor_reads: dict[BufferId, TensorRead] = field(default_factory=dict)
    latent_imports: dict[BufferId, LatentImport] = field(default_factory=dict)
    cache_imports: dict[BufferId, CacheImport] = field(default_factory=dict)
    # Media inputs a host call reads in place from their producers' segments
    # at execution; they are neither imported nor staged here.
    borrowed_inputs: set[BufferId] = field(default_factory=set)

    # Completion-valued (U8) predicates, one row each in a dedicated output
    # buffer. An entry is (call identity, captured (offset, count) span,
    # row). Transferred sources wait in ``predicate_transfers`` as
    # (identity, source buffer, row) until `capture_predicates` captures them
    # and seals the buffer; no capture is possible after sealing.
    predicate_buffer: OutputBuffer | None = None
    predicate_entries: list[tuple[CallIdentity, tuple[int, int], int]] = field(
        default_factory=list
    )
    predicate_transfers: tuple[tuple[CallIdentity, BufferId, int], ...] = ()
    predicates_sealed: bool = False
    # Cache filled by the first successful `predicate_values` read.
    _predicate_values: dict[CallIdentity, bool] | None = None

    # Set by `prepare_batch`: futures that must complete before this batch
    # writes its target latent, cache-page and KV storage (they gate
    # execution even without imports), and the products and KV transfers it
    # imports.
    storage_dependencies: tuple[Future[None], ...] = ()
    input_products: tuple[TensorPublication, ...] = ()
    kv_inputs: tuple[KvTransfer, ...] = ()
    # Entries of ``input_products`` whose reads `prepare_inputs` has started;
    # preparation refused for read tickets resumes at the next one.
    inputs_started: int = 0
    # Host-lane tasks preparing the inline input images of this batch's
    # encoder calls, by call id (`image.reserve_images`). Execution waits
    # for them like the storage dependencies, and `close_inputs` cancels the
    # ones still queued and drops every result.
    image_tasks: dict[CallId, HostTask] = field(default_factory=dict)

    # Preparation waits for read tickets while ``awaiting_reads`` is set.
    # Closing the numerical inputs suppresses their readiness callbacks.
    awaiting_reads: bool = False
    inputs_submitted: bool = False
    inputs_closed: bool = False

    # Final values addressed by original call index: None until
    # `bind_outputs`, then a `PendingOutput`, then the materialized
    # `RequestOutput`.
    outputs: list[PendingOutput | RequestOutput | None] = field(
        default_factory=list
    )
    # Whether every pending output has become its wire value. Retirement may
    # still be outstanding after this, so it is distinct from ``complete``.
    materialized: bool = False

    # Completion storage leased by `reserve_outputs`; `record_outputs` drops
    # this reference while the pending outputs keep their rows.
    buffer: OutputBuffer | None = None
    # The first active call's capability stream (`ModelExecutor.call_stream`),
    # or None to run on the current stream; see `scope`.
    stream: torch.cuda.Stream | None = None
    # ``time.perf_counter_ns`` when `reserve_outputs` began binding outputs;
    # `commit_batch` and the post-registration failure paths of
    # `execute_batch` derive ``execution_us`` from it.
    started_ns: int = 0
    # Per-execution scratch that `record_outputs` clears; `commit_batch`
    # folds the forward stats and component timings into ``stats``.
    forward_stats: list[ForwardStats] = field(default_factory=list)
    component_us: dict[str, int] = field(default_factory=dict)
    forward_indices: dict[CallIdentity, tuple[int, ...]] = field(
        default_factory=dict
    )
    # Set once `reserve_outputs` succeeds.
    registered: bool = False
    # Set by `commit_batch` once resource commits begin; from then on the
    # batch cannot be discarded and a failure is fatal.
    published: bool = False
    # Output index of each request's call. Looking up one request must not
    # scan the other calls of the batch.
    request_indexes: dict[int, int] = field(default_factory=dict)
    products: tuple[TensorPublication, ...] = ()
    stats: ForwardStats | None = None
    execution_us: int | None = None

    # Retirement of the batch's ``Finish`` and ``Free`` commands, recorded by
    # `BatchRunner.begin_retirement` and advanced by
    # `BatchRunner.poll`. ``retirement_events`` fence the device
    # work issued up to the batch's launch, including its request-state
    # writes, and ``retirement_cleaned`` is set once the stores have retired
    # the closed requests and freed buffers (at once for a batch without such
    # commands).
    retirement_requests: frozenset[RequestKey] = frozenset()
    retirement_local_requests: frozenset[RequestKey] = frozenset()
    retirement_buffers: frozenset[BufferId] = frozenset()
    retained_buffers: frozenset[BufferId] = frozenset()
    retirement_exports: tuple[BufferId, ...] = ()
    retirement_events: tuple[torch.cuda.Event, ...] = ()
    retirement_cleaned: bool = False

    def __post_init__(self) -> None:
        self.outputs = [None] * len(self.batch.calls)
        self.request_indexes = {
            call.request_key.request_id: index
            for index, call in enumerate(self.batch.calls)
        }

    def predecessor(self, call: Call) -> CallId | None:
        """Return the call this call follows, or None for independent work."""
        return self.predecessors.get(call.call_id)

    def scope(self):
        """Return a context that makes the batch stream current.

        `reserve_outputs`, `dispatch_batch`, `commit_batch` and
        `discard_batch` enter it, so the device work and fences they enqueue
        land on the batch stream; with no batch stream the context changes
        nothing.
        """
        stream = self.stream
        return nullcontext() if stream is None else torch.cuda.stream(stream)

    @property
    def output_buffer(self) -> OutputBuffer:
        """Borrow completion storage during execution, before publication.

        Raises:
            RuntimeError: Before `bind_outputs` or after `record_outputs`.
        """
        if self.buffer is None:
            raise RuntimeError("batch has no reserved output buffer")
        return self.buffer

    def bind_outputs(
        self,
        outputs: tuple[PendingOutput, ...],
        buffer: OutputBuffer,
        started_ns: int,
    ) -> None:
        """Bind reserved outputs to call indexes before preparing resources.

        ``outputs`` is aligned with ``batch.calls``. Indexes bound before a
        failing one stay bound, and ``buffer`` is retained only on success.

        Raises:
            RuntimeError: An index already holds an output.
            WorkerError: ``invalid_descriptor`` when an output's request key
                and call id differ from its call's.
            ValueError: ``outputs`` and the calls differ in length.
        """
        for index, (call, output) in enumerate(
            zip(self.batch.calls, outputs, strict=True)
        ):
            if self.outputs[index] is not None:
                raise RuntimeError("call output is already reserved")
            if (output.request_key, output.call_id) != (
                call.request_key,
                call.call_id,
            ):
                raise invalid_descriptor(
                    "reserved output does not match its call"
                )
            self.outputs[index] = output

        self.buffer = buffer
        self.started_ns = started_ns

    def pending_outputs(self) -> tuple[PendingOutput, ...]:
        """Borrow every call's `PendingOutput`, in call order.

        Raises:
            RuntimeError: Any output is unbound or already materialized.
        """
        values = tuple(self.outputs)
        if any(not isinstance(value, PendingOutput) for value in values):
            raise RuntimeError("batch has no reserved pending outputs")
        return cast(tuple[PendingOutput, ...], values)

    def pending_output(self, request_id: int) -> PendingOutput:
        """Borrow the reserved pending output of one request of the batch.

        Raises:
            WorkerError: ``invalid_descriptor`` when no call of the batch
                belongs to ``request_id``.
            RuntimeError: That call's output is unbound or materialized.
        """
        index = self.request_indexes.get(int(request_id))
        if index is None:
            raise invalid_descriptor(f"batch has no request {request_id}")
        value = self.outputs[index]
        if not isinstance(value, PendingOutput):
            raise RuntimeError("request has no reserved pending output")
        return value

    @property
    def batch_id(self) -> int:
        return int(self.batch.batch_id)

    @property
    def route(self) -> str | None:
        """The call kind every call executes, which failures report as route.

        None for a lifecycle-only batch, which has no calls and so no
        execution route.
        """
        calls = self.batch.calls
        return calls[0].kind.value if calls else None

    def inputs_ready(self) -> bool:
        """Check readiness without submitting inputs or running the model.

        True once inputs were submitted, every storage dependency, input
        image preparation, input transfer and cache import is done, and
        completion predicates are absent, already read, or sealed with their
        copies complete.
        """
        return (
            self.inputs_submitted
            and all(
                dependency.done() for dependency in self.storage_dependencies
            )
            and all(task.ready() for task in self.image_tasks.values())
            and all(ticket.ready() for ticket in self.input_tickets())
            and all(
                write.completion.done() for write in self.cache_imports.values()
            )
            and (
                self.predicate_buffer is None
                or self._predicate_values is not None
                or (self.predicates_sealed and self.predicate_buffer.ready())
            )
        )

    def predicate_values(self) -> dict[CallIdentity, bool]:
        """Read each completion predicate as a boolean, keyed by call.

        The first read validates every captured value, marks each row
        observed, and caches the result; later calls return the cache. An
        empty mapping means the batch has no completion-valued (U8)
        predicates; I64 predicates are not read here.

        Raises:
            RuntimeError: The predicate buffer is not yet ready.
            WorkerError: ``invalid_descriptor`` when a captured predicate is
                not a single 0 or 1, or an `OutputBuffer` invariant
                violation. The buffer is abandoned on any failure while
                reading.
        """
        buffer = self.predicate_buffer
        if buffer is None:
            return {}
        if self._predicate_values is not None:
            return self._predicate_values
        if not buffer.ready():
            raise RuntimeError(
                "prepared predicates were observed before readiness"
            )

        values: dict[CallIdentity, bool] = {}
        generation = buffer.generation
        try:
            for identity, capture, row in sorted(
                self.predicate_entries, key=lambda entry: entry[2]
            ):
                captured = buffer.read_tokens(*capture)
                if len(captured) != 1 or captured[0] not in {0, 1}:
                    raise invalid_descriptor(
                        "call predicate is not a canonical boolean"
                    )
                values[identity] = bool(captured[0])
                buffer.observe(row, generation)
        except BaseException:
            buffer.abandon()
            raise

        self._predicate_values = values
        return values

    def on_dependencies_ready(self, callback: Callable[[], None]) -> None:
        """Wake the owner once physical dependencies permit its next step.

        Waits on the input transfer tickets, the storage dependencies, the
        input image preparations, the cache import completions and, once
        sealed, the predicate buffer's completion. With no such dependency,
        ``callback`` runs synchronously and unconditionally. Otherwise it
        runs at most once, when all of them are complete, and not after
        `close_inputs`: synchronously when they already are at registration,
        else on a thread that completes one. The dependency set is captured
        now, so the owner re-checks `inputs_ready` when woken and registers
        again if the batch is still not ready.
        """
        tickets = tuple(self.input_tickets())
        dependencies = (
            self.storage_dependencies
            + tuple(task.promise for task in self.image_tasks.values())
            + tuple(write.completion for write in self.cache_imports.values())
        )
        if self.predicate_buffer is not None and self.predicates_sealed:
            dependencies += (self.predicate_buffer.completion_future(),)
        if not tickets and not dependencies:
            callback()
            return

        lock = Lock()
        fired = False

        # Done callbacks may run concurrently on different completing
        # threads; the lock lets at most one of them invoke ``callback``.
        def notify_if_ready() -> None:
            nonlocal fired
            if self.inputs_closed:
                return
            if not all(ticket.ready() for ticket in tickets) or not all(
                dependency.done() for dependency in dependencies
            ):
                return
            with lock:
                if fired:
                    return
                fired = True
            callback()

        for ticket in tickets:
            ticket.add_done_callback(notify_if_ready)

        for dependency in dependencies:
            dependency.add_done_callback(lambda _future: notify_if_ready())

        notify_if_ready()

    def input_tickets(self) -> Iterator[TransferTicket]:
        """Borrow physical transfers from their actual storage reservations."""
        for read in self.tensor_reads.values():
            # `TensorStore.complete_reads` clears ``imported`` when a read
            # completes and drops its shared import. A callback registered
            # after synchronous execution must not revive that retired
            # dependency.
            if read.imported is not None:
                yield from read.imported.tickets
        for write in self.latent_imports.values():
            yield from write.transfers

    def input_ready(self, buffer: BufferId) -> bool:
        """Query one reserved input without publishing or consuming it.

        Inputs read in place and tensor reads without a pending import are
        always ready. A buffer this batch has not reserved reports False.
        """
        if buffer in self.borrowed_inputs:
            return True
        if (read := self.tensor_reads.get(buffer)) is not None:
            return read.imported is None or all(
                ticket.ready() for ticket in read.imported.tickets
            )
        if (latent := self.latent_imports.get(buffer)) is not None:
            return all(ticket.ready() for ticket in latent.transfers)
        if (cache := self.cache_imports.get(buffer)) is not None:
            return cache.completion.done()
        return False

    def close_inputs(
        self,
        tensor_store: TensorStore,
        latent_pool: LatentPool | None,
        kv_cache: KVCacheManager | None,
    ) -> None:
        """Release this submission's readers and unadopted destinations.

        Runs once, whether after execution, on a preparation failure, or when
        the batch is closed; later calls do nothing, even when an earlier
        release raised. Every release is attempted, and the first failure is
        raised with later ones noted. Shared tensor fills outlive cancellation
        while another read retains them. Latent and cache owners retain
        cancelled writes until retirement.
        """
        if self.inputs_closed:
            return

        self.inputs_closed = True
        actions: list[Callable[[], object]] = []

        # A preparation still queued is withdrawn and returns its lane
        # capacity; a running one finishes on its lane thread. Dropping the
        # tasks releases their page-locked results, whose device copies the
        # caching host allocator fences.
        image_tasks = tuple(self.image_tasks.values())
        self.image_tasks = {}
        actions.extend(task.cancel for task in image_tasks)

        if self.latent_imports:
            assert latent_pool is not None
            actions.extend(
                partial(latent_pool.abandon_import, write)
                for write in self.latent_imports.values()
                if not write.adopted
            )

        if self.cache_imports:
            assert kv_cache is not None
            actions.extend(
                partial(kv_cache.imports.abandon, write)
                for write in self.cache_imports.values()
                if not write.released
            )

        # `predicate_values` observes every row it reads, so only an unread
        # predicate buffer is abandoned here.
        if self.predicate_buffer is not None and self._predicate_values is None:
            actions.append(self.predicate_buffer.abandon)

        if self.tensor_reads:
            actions.append(
                partial(
                    tensor_store.complete_reads,
                    tuple(self.tensor_reads.values()),
                )
            )

        actions.extend(
            ticket.close
            for write in self.latent_imports.values()
            for ticket in write.transfers
        )
        close_resources(*actions)

    def record_outputs(
        self,
        outputs: tuple[PendingOutput | RequestOutput, ...],
        *,
        products: tuple[TensorPublication, ...] = (),
        execution_us: int,
        stats: ForwardStats,
    ) -> None:
        """Record the batch's final call outputs, products and statistics.

        Called once per batch, by `commit_batch` for a published batch or by
        `execute_batch` with error completions. ``outputs`` is aligned with
        ``batch.calls``, and a `PendingOutput` must be the one reserved at its
        index. Clears the output buffer reference and the per-execution
        scratch. Outputs stored before a failure stay stored.

        Raises:
            RuntimeError: Outputs were already recorded, or a pending output
                replaces a different reserved one.
            WorkerError: ``invalid_descriptor`` when an output or product does
                not belong to a call of this batch.
            ValueError: ``outputs`` and the calls differ in length.
        """
        if self.stats is not None:
            raise RuntimeError("batch was published more than once")

        for index, (call, output) in enumerate(
            zip(self.batch.calls, outputs, strict=True)
        ):
            previous = self.outputs[index]
            if (
                isinstance(output, PendingOutput)
                and previous is not None
                and previous is not output
            ):
                raise RuntimeError("result replaced another reserved output")
            if (output.request_key, output.call_id) != (
                call.request_key,
                call.call_id,
            ):
                raise invalid_descriptor(
                    "result does not match its submitted call"
                )
            self.outputs[index] = output

        identities = {
            (output.request_key, output.call_id) for output in outputs
        }
        if any(
            (value.product.request_key, value.product.producer_call_id)
            not in identities
            for value in products
        ):
            raise invalid_descriptor("product does not belong to its batch")

        self.products = products
        self.stats = stats
        self.execution_us = execution_us
        self.buffer = None
        self.forward_indices.clear()
        self.forward_stats.clear()
        self.component_us.clear()

    def result(self) -> BatchOutput:
        """Build the materialized batch result for native delivery.

        A batch is one numerical call on one component, so every call it
        carries completes together and its retirement is already applied.
        Only products of calls that completed with `CallStatus.OK` are
        reported.

        Raises:
            RuntimeError: An output is not yet materialized.
        """
        values: list[RequestOutput] = []
        for value in self.outputs:
            if not isinstance(value, RequestOutput):
                raise RuntimeError(
                    "batch delivery encountered an unmaterialized output"
                )
            values.append(value)

        successful = {
            (value.request_key, value.call_id)
            for value in values
            if value.status is CallStatus.OK
        }

        # A batch without calls carries only lifecycle commands and records
        # no completion, so it reports no execution statistics.
        return BatchOutput(
            batch_id=self.batch_id,
            completions=tuple(values),
            products=tuple(
                value
                for value in self.products
                if (value.product.request_key, value.product.producer_call_id)
                in successful
            ),
            worker_exec_us=self.execution_us,
            forward_stats=self.stats,
        )

    def close(
        self,
        tensor_store: TensorStore,
        latent_pool: LatentPool | None,
        kv_cache: KVCacheManager | None,
    ) -> None:
        """Abandon delivery while physical readers retain their own leases.

        Closes the inputs and abandons every output still pending. Every
        release is attempted, and the first failure is raised with later ones
        noted.
        """
        actions: list[Callable[[], object]] = [
            partial(self.close_inputs, tensor_store, latent_pool, kv_cache)
        ]
        actions.extend(
            output.abandon
            for output in self.outputs
            if isinstance(output, PendingOutput)
        )
        close_resources(*actions)
