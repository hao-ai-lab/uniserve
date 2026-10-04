"""Numerical resources owned by one in-flight scheduler submission.

`Executor.submit` creates one `BatchState` per `Batch` and owns it until
`Executor.poll` consumes its result, `submit` itself fails, or the worker
closes. The execution modules fill it in stage order: `prepare_batch` and
`prepare_inputs` record storage dependencies, input reservations and
predicate captures, and `image.reserve_images` the host preparation of
inline input images; `reserve_outputs` binds one `PendingOutput` per call;
`commit_batch` records published products and execution measurements.
The native executor owns admission, launch order, failure, result assembly
and delivery. Rust resolves outputs and retires batch resources. Native
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
from uniserve_worker.protocol.call import Call
from uniserve_worker.protocol.identity import (
    BufferId,
    CallId,
    CallIdentity,
)
from uniserve_worker.protocol.output import ForwardStats
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
    # Native request ordering exposed to numerical staging as a snapshot;
    # None denotes independent work without a state predecessor.
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

    # Output owners addressed by original call index, bound by reserve_outputs.
    outputs: list[PendingOutput | None] = field(default_factory=list)

    # Completion storage leased by `reserve_outputs`; `record_execution` drops
    # this reference while the pending outputs keep their rows.
    buffer: OutputBuffer | None = None
    # The first active call's capability stream (`ModelExecutor.call_stream`),
    # or None to run on the current stream; see `scope`.
    stream: torch.cuda.Stream | None = None
    # ``time.perf_counter_ns`` when `reserve_outputs` began binding outputs;
    # `commit_batch` and the post-registration failure paths of
    # `execute_batch` derive ``execution_us`` from it.
    started_ns: int = 0
    # Per-execution scratch that `record_execution` clears; `commit_batch`
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
            RuntimeError: Before `bind_outputs` or after `record_execution`.
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
            RuntimeError: Any output is unbound.
        """
        values = tuple(self.outputs)
        if any(value is None for value in values):
            raise RuntimeError("batch has no reserved pending outputs")
        return cast(tuple[PendingOutput, ...], values)

    def pending_output(self, request_id: int) -> PendingOutput:
        """Borrow the reserved pending output of one request of the batch.

        Raises:
            WorkerError: ``invalid_descriptor`` when no call of the batch
                belongs to ``request_id``.
            RuntimeError: That call's output is unbound.
        """
        index = self.request_indexes.get(int(request_id))
        if index is None:
            raise invalid_descriptor(f"batch has no request {request_id}")
        value = self.outputs[index]
        if value is None:
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

    def record_execution(
        self, *, execution_us: int, stats: ForwardStats
    ) -> None:
        """Keep execution measurements and release numerical scratch."""
        self.stats = stats
        self.execution_us = execution_us
        self.buffer = None
        self.forward_indices.clear()
        self.forward_stats.clear()
        self.component_us.clear()

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
            output.abandon for output in self.outputs if output is not None
        )
        close_resources(*actions)
