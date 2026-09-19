"""Resources and delivery facts owned by one in-flight scheduler submission."""

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
from uniserve_worker.execution.output import OutputBuffer, PendingOutput
from uniserve_worker.execution.rows import (
    CallIdentity,
)
from uniserve_worker.foundation.errors import WorkerError, invalid_descriptor
from uniserve_worker.protocol.batch import (
    Batch,
    RegistrationAck,
    TensorPublication,
)
from uniserve_worker.protocol.call import Call, CallStatus
from uniserve_worker.protocol.identity import BufferId, CallId, RequestKey
from uniserve_worker.protocol.output import (
    BatchOutput,
    ForwardStats,
    RequestOutput,
)
from uniserve_worker.protocol.transfer import KvTransfer
from uniserve_worker.runtime.cache_imports import CacheImport
from uniserve_worker.runtime.cache_manager import CacheManager
from uniserve_worker.runtime.latent_pool import LatentImport, LatentPool
from uniserve_worker.runtime.tensor_store import TensorRead, TensorStore
from uniserve_worker.transfer.tickets import TransferTicket


@dataclass(slots=True)
class BatchState:
    """Retain original input, physical dependencies, outputs.

    and delivery position.

    Worker submits inputs, launches computation, and materializes results. This
    object has no callback that can execute its batch or advance the worker.
    """

    batch: Batch
    propagate_errors: bool = False
    # The call each of this batch's calls follows in its request, derived by
    # the rank from its own request state; None for independent work.
    predecessors: dict[CallId, CallId | None] = field(default_factory=dict)

    # Physical input reservations held until execution observes readiness.
    tensor_reads: dict[BufferId, TensorRead] = field(default_factory=dict)
    latent_imports: dict[BufferId, LatentImport] = field(default_factory=dict)
    cache_imports: dict[BufferId, CacheImport] = field(default_factory=dict)

    # Completion predicates staged in a sealed buffer and read as booleans.
    predicate_buffer: OutputBuffer | None = None
    predicate_entries: list[tuple[CallIdentity, tuple[int, int], int]] = field(
        default_factory=list
    )
    predicate_transfers: tuple[tuple[CallIdentity, BufferId, int], ...] = ()
    predicates_sealed: bool = False
    _predicate_values: dict[CallIdentity, bool] | None = None

    # Declared physical inputs and their outstanding transfer dependencies.
    storage_dependencies: tuple[Future[None], ...] = ()
    input_products: tuple[TensorPublication, ...] = ()
    kv_inputs: tuple[KvTransfer, ...] = ()

    # Lifecycle flags from input submission through terminal delivery.
    inputs_submitted: bool = False
    inputs_closed: bool = False
    launched: bool = False
    complete: bool = False
    error: WorkerError | None = None

    # Final values addressed by original call index and completion group.
    outputs: list[PendingOutput | RequestOutput | None] = field(
        default_factory=list
    )
    output_groups: dict[int, tuple[int, ...]] = field(default_factory=dict)
    completed_groups: set[int] = field(default_factory=set)
    accepted_groups: set[int] = field(default_factory=set)

    # Per-group execution and publication bookkeeping.
    group_buffers: dict[int, OutputBuffer] = field(default_factory=dict)
    group_streams: dict[int, torch.cuda.Stream] = field(default_factory=dict)
    group_started_ns: dict[int, int] = field(default_factory=dict)
    group_forward_stats: dict[int, list[ForwardStats]] = field(
        default_factory=dict
    )
    group_component_us: dict[int, dict[str, int]] = field(default_factory=dict)
    group_forward_indices: dict[int, dict[CallIdentity, tuple[int, ...]]] = (
        field(default_factory=dict)
    )
    group_registered: dict[int, bool] = field(default_factory=dict)
    group_published: dict[int, bool] = field(default_factory=dict)
    request_locations: dict[int, tuple[int, int]] = field(default_factory=dict)
    group_products: dict[int, tuple[TensorPublication, ...]] = field(
        default_factory=dict
    )
    group_stats: dict[int, ForwardStats] = field(default_factory=dict)
    group_execution_us: dict[int, int] = field(default_factory=dict)
    visible_groups: set[int] = field(default_factory=set)

    # Resource retirement decided during execution, applied by the owner.
    retirement_requests: frozenset[RequestKey] = frozenset()
    retirement_local_requests: frozenset[RequestKey] = frozenset()
    retirement_buffers: frozenset[BufferId] = frozenset()
    retained_buffers: frozenset[BufferId] = frozenset()
    retirement_exports: tuple[BufferId, ...] = ()
    retirement_events: tuple[torch.cuda.Event, ...] = ()
    retirement_cleaned: bool = False

    # Whether the batch's single result has been delivered.
    result_sent: bool = False

    def __post_init__(self) -> None:
        self.outputs = [None] * len(self.batch.calls)

        groups: dict[tuple[object, str], list[int]] = {}
        for index, call in enumerate(self.batch.calls):
            groups.setdefault((call.kind, call.entry), []).append(index)

        self.output_groups = {
            group: tuple(indexes)
            for group, indexes in enumerate(groups.values(), start=1)
        }
        # Resolve group ownership alongside the output index. Looking up one
        # request must not scan the other requests in its completion group.
        self.request_locations = {
            self.batch.calls[index].request_key.request_id: (group, index)
            for group, indexes in self.output_groups.items()
            for index in indexes
        }

    def predecessor(self, call: Call) -> CallId | None:
        """Return the call this call follows, or None for independent work."""
        return self.predecessors.get(call.call_id)

    def group_calls(self, group: int) -> tuple[Call, ...]:
        """Borrow original call values belonging to one completion.

        group.
        """
        return tuple(
            self.batch.calls[index] for index in self.output_groups[group]
        )

    def group_scope(self, group: int):
        """Keep numerical access and its retirement fences on the selected.

        stream.
        """
        stream = self.group_streams.get(group)
        return nullcontext() if stream is None else torch.cuda.stream(stream)

    def bind_outputs(
        self,
        group: int,
        outputs: tuple[PendingOutput, ...],
        buffer: OutputBuffer,
        started_ns: int,
    ) -> None:
        """Bind reserved outputs to original call indexes before resource.

        preparation.
        """
        for index, output in zip(
            self.output_groups[group], outputs, strict=True
        ):
            call = self.batch.calls[index]
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

        self.group_buffers[group] = buffer
        self.group_started_ns[group] = started_ns
        self.group_forward_stats[group] = []
        self.group_component_us[group] = {}
        self.group_forward_indices[group] = {}
        self.group_registered[group] = False
        self.group_published[group] = False

    def pending_outputs(self, group: int) -> tuple[PendingOutput, ...]:
        """Borrow the currently executing outputs of one completion group."""
        values = tuple(
            self.outputs[index] for index in self.output_groups[group]
        )
        if any(not isinstance(value, PendingOutput) for value in values):
            raise RuntimeError(
                "completion group has no reserved pending outputs"
            )
        return cast(tuple[PendingOutput, ...], values)

    def pending_output(self, group: int, request_id: int) -> PendingOutput:
        """Borrow the reserved pending output of one request in a completion.

        group.
        """
        location = self.request_locations.get(int(request_id))
        if location is None or location[0] != group:
            raise invalid_descriptor(
                f"completion group has no request {request_id}"
            )
        value = self.outputs[location[1]]
        if not isinstance(value, PendingOutput):
            raise RuntimeError("request has no reserved pending output")
        return value

    @property
    def batch_id(self) -> int:
        return int(self.batch.batch_id)

    @property
    def request_ids(self) -> frozenset[int]:
        return frozenset(
            key.request_id
            for key in (
                *(admission.request_key for admission in self.batch.admissions),
                *(call.request_key for call in self.batch.calls),
                *(command.request_key for command in self.batch.commands),
            )
        )

    def inputs_ready(self) -> bool:
        """Query physical readiness without submitting inputs or executing a.

        model.
        """
        return (
            self.inputs_submitted
            and all(
                dependency.done() for dependency in self.storage_dependencies
            )
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
        """Read validated predicate scalars and index them by semantic product.

        reference.
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
        """Wake the owner once physical dependencies permit its next.

        preparation step.
        """
        tickets = tuple(self.input_tickets())
        dependencies = self.storage_dependencies + tuple(
            write.completion for write in self.cache_imports.values()
        )
        if self.predicate_buffer is not None and self.predicates_sealed:
            dependencies += (self.predicate_buffer.completion_future(),)
        if not tickets and not dependencies:
            callback()
            return

        lock = Lock()
        fired = False

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
            # Completed reads release their shared import. A callback registered
            # after synchronous execution must not revive that retired
            # dependency.
            if read.imported is not None:
                yield from read.imported.tickets
        for write in self.latent_imports.values():
            yield from write.transfers

    def input_ready(self, buffer: BufferId) -> bool:
        """Query one reserved input without publishing or consuming it."""
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
        kv_cache: CacheManager | None,
    ) -> None:
        """Release this submission's readers and unadopted physical.

        destinations.

        Shared tensor fills outlive cancellation while another read retains
        them. Latent and cache owners retain cancelled writes until retirement.
        """
        if self.inputs_closed:
            return

        self.inputs_closed = True
        actions: list[Callable[[], object]] = []

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
        group: int,
        outputs: tuple[PendingOutput | RequestOutput, ...],
        *,
        products: tuple[TensorPublication, ...] = (),
        visible: bool,
        execution_us: int,
        stats: ForwardStats,
    ) -> None:
        """Retain original call outputs and statistics at their completion.

        boundary.
        """
        if group in self.group_stats:
            raise RuntimeError("completion group was published more than once")

        indexes = self.output_groups[group]
        for index, output in zip(indexes, outputs, strict=True):
            call = self.batch.calls[index]
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
            raise invalid_descriptor(
                "product does not belong to its completion group"
            )

        self.group_products[group] = products
        self.group_stats[group] = stats
        self.group_execution_us[group] = execution_us
        self.group_buffers.pop(group, None)
        self.group_forward_indices.pop(group, None)
        self.group_forward_stats.pop(group, None)
        self.group_component_us.pop(group, None)

        if visible:
            self.visible_groups.add(group)

    def ready(self) -> bool:
        """Query whether the batch's single result can be delivered."""
        if self.result_sent:
            return False
        return self.error is not None or self.complete

    def take_output(self) -> BatchOutput:
        """Consume the batch's one result.

        A batch is one numerical call on one component, so every call it
        carries completes together and its retirement is already applied.
        """
        if self.error is not None:
            raise RuntimeError(
                "terminal error must be consumed through take_error"
            )
        if not self.ready():
            raise RuntimeError("batch has no ready output")
        self.result_sent = True

        groups = tuple(self.output_groups)
        values: list[RequestOutput] = []
        for group in groups:
            for index in self.output_groups[group]:
                value = self.outputs[index]
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

        return BatchOutput(
            batch_id=self.batch_id,
            completions=tuple(values),
            products=tuple(
                value
                for group in groups
                for value in self.group_products[group]
                if (value.product.request_key, value.product.producer_call_id)
                in successful
            ),
            registration=RegistrationAck(
                visible=all(group in self.visible_groups for group in groups)
            ),
            worker_exec_us=(
                max(self.group_execution_us[group] for group in groups)
                if groups
                else None
            ),
            forward_stats=(
                ForwardStats.combine(
                    tuple(self.group_stats[group] for group in groups)
                )
                if groups
                else None
            ),
        )

    def take_error(self) -> WorkerError:
        """Consume the terminal error exactly once."""
        error = self.error
        if error is None or self.result_sent:
            raise RuntimeError("batch has no unread terminal error")
        self.result_sent = True
        return error

    def close(
        self,
        tensor_store: TensorStore,
        latent_pool: LatentPool | None,
        kv_cache: CacheManager | None,
    ) -> None:
        """Abandon delivery while physical readers retain their own resource.

        leases.
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
