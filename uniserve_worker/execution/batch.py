"""Numerical resources owned by one in-flight scheduler submission.

`Executor.submit` creates one `BatchState` per `Batch` and owns it until
`Executor.poll` consumes its result, `submit` itself fails, or the worker
closes. The execution modules fill it in stage order: `prepare_batch` and
`prepare_inputs` record storage dependencies, input reservations and
predicate captures, and `image.reserve_images` the host preparation of
inline input images; `reserve_outputs` binds one `PendingOutput` per call;
`commit_batch` (or `execute_batch` on failure) records the final outputs.
The native executor owns admission, launch order, failure and delivery.
The batch runner materializes outputs; Rust retires batch resources. Native
`BatchInputs` retains input leases and notifies the executor when their
physical dependencies become consumable.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import cast

import torch

from uniserve.runtime.resources import close_resources
from uniserve_worker._uniserve_ipc import BatchInputs
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.protocol.batch import Batch, TensorPublication
from uniserve_worker.protocol.call import Call, CallStatus
from uniserve_worker.protocol.identity import (
    BufferId,
    CallId,
    CallIdentity,
)
from uniserve_worker.protocol.output import (
    BatchOutput,
    ForwardStats,
    RequestOutput,
)
from uniserve_worker.protocol.transfer import KvTransfer
from uniserve_worker.storage.kv_cache import KVCacheManager
from uniserve_worker.storage.latent_pool import LatentPool
from uniserve_worker.storage.output import OutputBuffer
from uniserve_worker.storage.tensor_store import TensorStore


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

    # Rust retains physical inputs, host preparations and their readiness.
    inputs: BatchInputs = field(default_factory=BatchInputs)

    # Completion-valued (U8) predicates, one row each in a dedicated output
    # buffer. An entry is (call identity, captured (offset, count) span,
    # row). Transferred sources wait in ``predicate_transfers`` as
    # (identity, source buffer, row) until `capture_predicates` captures them
    # and seals the buffer; no capture is possible after sealing.
    predicate_entries: list[tuple[CallIdentity, tuple[int, int], int]] = field(
        default_factory=list
    )
    predicate_transfers: tuple[tuple[CallIdentity, BufferId, int], ...] = ()
    # Cache filled by the first successful `predicate_values` read.
    _predicate_values: dict[CallIdentity, bool] | None = None

    # Numerical transfer descriptions consumed by input preparation.
    input_products: tuple[TensorPublication, ...] = ()
    kv_inputs: tuple[KvTransfer, ...] = ()

    # Final values addressed by original call index: None until
    # `bind_outputs`, then a `PendingOutput`, then the materialized
    # `RequestOutput`.
    outputs: list[PendingOutput | RequestOutput | None] = field(
        default_factory=list
    )

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
        if self._predicate_values is not None:
            return self._predicate_values
        buffer = self.inputs.predicate
        if buffer is None:
            return {}
        if not buffer.ready():
            raise RuntimeError(
                "prepared predicates were observed before readiness"
            )

        values: dict[CallIdentity, bool] = {}
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
                buffer.observe(row)
        except BaseException:
            buffer.abandon()
            raise

        self._predicate_values = values
        self.inputs.predicate = None
        return values

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
            partial(
                self.inputs.close,
                tensor_store,
                latent_pool,
                None if kv_cache is None else kv_cache.imports,
            )
        ]
        actions.extend(
            output.abandon
            for output in self.outputs
            if isinstance(output, PendingOutput)
        )
        close_resources(*actions)
