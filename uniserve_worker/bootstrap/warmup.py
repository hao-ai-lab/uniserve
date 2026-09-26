"""Bounded request, sampling, and product startup scenarios.

``Worker.warmup`` calls `warmup_requests` after the model executor's
``warmup`` and ``capture`` and before serving. It drives synthetic requests
through the same ``Worker.submit``/``advance``/``poll`` path that serving
batches take, on CUDA devices only:

- a token scenario: one PREFILL call followed, when supported, by a DECODE
  call that consumes the prefill's device token;
- a flow scenario per configured guidance-branch count: a KV_PUBLISH of the
  conditioning, a LATENT_PREPARATION, and two chained DENOISING calls.

No engine scheduler is present at startup, so this module takes over the
engine storage pools' role: the scenarios choose request slots,
`_build_warmup_batch` and `_warmup_flow_tables` lease KV units and latent
pages,
and `_WarmupRequests` tracks those leases and allocates persistent buffer
spans. Each ``Batch`` carries the resulting assignments. Batch ids and
collective sequences are private to this sequence; ``Worker.warmup`` resets
the executor's last batch id and collective sequence afterwards and raises
if any request is still resident, so every successful scenario drops its
requests before returning.
"""

from __future__ import annotations

import logging
import math
import time
from itertools import islice
from typing import TYPE_CHECKING

import torch

from uniserve.math import ceil_div
from uniserve.media.image import Config as ImageConfig
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution.diffusion import image_state
from uniserve_worker.model_executor.diffusion_inputs import resolve_prefix
from uniserve_worker.model_executor.graph_inputs import DiffusionShape
from uniserve_worker.model_executor.startup import capture_image_parameters
from uniserve_worker.protocol.batch import (
    Batch,
    BatchCommand,
    BlockTable,
    BufferAllocation,
    CacheUnitAllocation,
    Finish,
    Free,
    LatentParams,
    NewRequest,
    Start,
    TensorPublication,
)
from uniserve_worker.protocol.call import (
    Call,
    CallStatus,
    ForwardMode,
    MediaCall,
    TransferMode,
)
from uniserve_worker.protocol.identity import BufferId, CallId, RequestKey
from uniserve_worker.protocol.output import BatchOutput, RequestOutput
from uniserve_worker.protocol.tensor import TensorRef

if TYPE_CHECKING:
    from uniserve_worker.worker import Worker

logger = logging.getLogger(__name__)


class _WarmupRequests:
    """Track synthetic request allocations.

    Holds the leases that stand in for the engine's storage pools while no
    scheduler is attached. KV units and latent pages have no free list:
    occupancy is recomputed from the lease maps whenever units are leased,
    so popping a lease (`drop_request`) returns its units. Persistent buffer
    spans use a first-fit extent list. Request slot zero, KV unit zero, and
    latent page zero are sentinels and are never assigned.
    """

    def __init__(self, worker: Worker) -> None:
        """Borrow the worker whose storage the synthetic requests occupy."""
        self.worker = worker
        # Main KV units of each (request, KV group), page-major in logical
        # order.
        self._kv_units: dict[tuple[RequestKey, int], list[int]] = {}
        # KV units of every group and the request slot of each request's
        # alternative guidance prefix (see `_warmup_flow_tables`).
        self._prefix_units: dict[tuple[RequestKey, int], list[int]] = {}
        self._prefix_slots: dict[RequestKey, int] = {}
        # Latent page table of each image request, in logical order.
        self._latent_pages: dict[RequestKey, list[int]] = {}
        self._buffers: dict[BufferId, BufferAllocation] = {}
        # Free extents of the synthetic persistent buffer pool, as
        # (offset, bytes).
        self._free_buffer_ranges: list[tuple[int, int]] = [
            (0, int(self.worker.info.buffer_pool_bytes))
        ]
        # Last batch id assigned. `Executor.submit` requires strictly
        # increasing ids, so control batches and call batches share this one
        # counter.
        self._batch_id = 0

    def occupied_units(self) -> set[int]:
        """Return every KV unit a main or prefix lease holds."""
        return {
            unit
            for leases in (self._kv_units, self._prefix_units)
            for units in leases.values()
            for unit in units
        }

    def grow_tables(
        self,
        leases: dict[tuple[RequestKey, int], list[int]],
        request_key: RequestKey,
        slot: int,
        tokens: int,
    ) -> tuple[tuple[BlockTable, ...], tuple[CacheUnitAllocation, ...]]:
        """Grow a request's leased tables of every cache group to ``tokens``.

        Each group's lease keeps its units until `drop_request` and gains
        the units of the whole pages ``tokens`` newly needs, taken from the
        lowest units no lease holds. Returns every group's resulting table
        and the units newly allocated to each.

        Raises:
            WorkerError: From `invalid_descriptor` when the worker has no KV
                cache, a lease would shrink, or the pool has too few free
                units.
        """
        cache = self.worker.kv_cache
        if cache is None:
            raise invalid_descriptor("warmup KV call requires cache storage")
        occupied = self.occupied_units()
        free = (
            unit
            for unit in range(1, int(cache.info.num_units))
            if unit not in occupied
        )

        tables: list[BlockTable] = []
        allocations: list[CacheUnitAllocation] = []
        for group, shape in enumerate(cache.shapes):
            lease = leases.setdefault((request_key, group), [])
            pages = ceil_div(tokens, shape.page_tokens)
            missing = pages * shape.units_per_page - len(lease)
            if missing < 0:
                raise invalid_descriptor(
                    "warmup call regresses its KV capacity"
                )
            allocated = tuple(islice(free, missing))
            if len(allocated) != missing:
                raise invalid_descriptor(
                    "warmup KV allocation exceeds resident capacity: "
                    f"request={request_key.request_id}, group={group}, "
                    f"required_units={missing}, "
                    f"available_units={len(allocated)}, "
                    f"leased_units={len(occupied)}"
                )
            lease.extend(allocated)
            occupied.update(allocated)
            tables.append(
                BlockTable(
                    request_pool_idx=slot,
                    group_id=group,
                    start_page=0,
                    unit_ids=tuple(lease),
                    allocated_tokens=pages * shape.page_tokens,
                )
            )
            if allocated:
                allocations.append(
                    CacheUnitAllocation(
                        request_pool_idx=slot,
                        group_id=group,
                        unit_ids=allocated,
                    )
                )
        return tuple(tables), tuple(allocations)

    def drop_request(self, request_id: int) -> None:
        """Retire one synthetic request and release everything it holds.

        Submits a ``Finish`` command and waits for it, removes the worker's
        request row, forgets the request's KV, prefix, and latent leases, and
        returns the buffer spans of products it owns to the synthetic pool.
        An unknown id is a no-op.
        """
        request = self.worker.requests.peek(int(request_id))
        if request is None:
            return

        self._execute_controls((Finish(request.request_key),))
        # Serving keeps a terminal row until its slot is reassigned. Synthetic
        # requests have no further scheduler messages and can leave the table.
        self.worker.requests.drop(request_id)

        for leases in (self._kv_units, self._prefix_units):
            for key in tuple(leases):
                if key[0] == request.request_key:
                    del leases[key]
        self._prefix_slots.pop(request.request_key, None)
        self._latent_pages.pop(request.request_key, None)

        released = tuple(
            buffer
            for buffer, allocation in self._buffers.items()
            if int(allocation.buffer.owner.request_id) == int(request_id)
        )
        self._release_buffer_allocations(released)

    def free_products(self, buffers: tuple[BufferId, ...]) -> None:
        """Free products on the worker and recycle their buffer spans.

        Submits one ``Free`` command per buffer and waits for the batch.
        """
        if not buffers:
            return
        self._execute_controls(tuple(Free(buffer) for buffer in buffers))
        self._release_buffer_allocations(buffers)

    def _execute_controls(self, commands: tuple[BatchCommand, ...]) -> None:
        """Submit a command-only batch and wait for it to finalize.

        The batch carries no calls, so it keeps the default collective
        sequence; the executor checks ordering only for batches with calls
        on a multi-rank worker.
        """
        self._batch_id += 1
        _execute_warmup(
            self,
            Batch(
                batch_id=self._batch_id,
                commands=commands,
            ),
            retain_device_outputs=True,
        )

    def _release_buffer_allocations(
        self, buffers: tuple[BufferId, ...]
    ) -> None:
        """Release persistent warmup allocations by exact buffer identity."""
        for buffer in buffers:
            allocation = self._buffers.pop(buffer, None)
            if allocation is not None:
                self._free_buffer_ranges.append(
                    (allocation.offset, allocation.bytes)
                )
        if not self._free_buffer_ranges:
            return

        # Coalesce adjacent free extents so later large products still fit.
        merged: list[tuple[int, int]] = []
        for offset, extent in sorted(self._free_buffer_ranges):
            if merged and merged[-1][0] + merged[-1][1] == offset:
                previous, size = merged[-1]
                merged[-1] = (previous, size + extent)
            else:
                merged.append((offset, extent))
        self._free_buffer_ranges = merged

    def buffer_allocation(self, product: TensorRef) -> BufferAllocation:
        """Return the product's buffer span, allocating it on first use.

        Repeated calls for one buffer id return the same span until it is
        released.

        Raises:
            WorkerError: From `invalid_descriptor` when no free extent fits
                ``product.max_bytes``.
        """
        existing = self._buffers.get(product.buffer_id)
        if existing is not None:
            return existing

        # First-fit over the free extents with the 256-byte alignment the
        # engine scheduler also uses for buffer spans.
        alignment = 256
        required = int(product.max_bytes)
        for index, (offset, extent) in enumerate(self._free_buffer_ranges):
            aligned = (offset + alignment - 1) & ~(alignment - 1)
            end = aligned + required
            if end > offset + extent:
                continue

            replacement: list[tuple[int, int]] = []
            if aligned > offset:
                replacement.append((offset, aligned - offset))
            if end < offset + extent:
                replacement.append((end, offset + extent - end))

            self._free_buffer_ranges[index : index + 1] = replacement
            allocation = BufferAllocation(product.buffer_id, aligned, required)
            self._buffers[product.buffer_id] = allocation
            return allocation

        raise invalid_descriptor(
            "warmup persistent buffer allocation exceeds resident capacity"
        )


def _warmup_batch(
    *,
    batch_id: int,
    admissions: tuple[NewRequest, ...],
    calls: tuple[Call, ...],
    block_tables: dict[tuple[RequestKey, CallId], tuple[BlockTable, ...]],
    new_cache_units: dict[
        tuple[RequestKey, CallId], tuple[CacheUnitAllocation, ...]
    ],
    forward_inputs: dict[
        tuple[RequestKey, CallId],
        tuple[
            tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[bool, ...]
        ],
    ],
    latent_params: dict[tuple[RequestKey, CallId], LatentParams],
    buffer_allocations: tuple[BufferAllocation, ...],
    input_products: tuple[TensorPublication, ...] = (),
) -> Batch:
    """Assemble warmup calls and their physical input columns.

    forward_inputs maps each call to its forward-row columns:
    (request_pool_indices, seq_lens, query_lens, write_kv). The batch
    flattens them into the Batch's parallel row arrays in call order. Every
    LATENT_PREPARATION, DENOISING, or IMAGE_DECODING call must have an entry
    in latent_params. Only the buffer allocations that some call references
    are attached, and each admission becomes a ``Start`` command.
    """
    return Batch(
        batch_id=batch_id,
        # Strictly increases with batch_id, which `Executor` requires of
        # batches with calls when the world size exceeds one.
        collective_seq=max(1, int(batch_id) * 16 + 1),
        calls=calls,
        block_tables=tuple(
            table
            for call in calls
            for table in block_tables.get(
                (call.request_key, call.call_id),
                (),
            )
        ),
        new_cache_units=tuple(
            allocation
            for call in calls
            for allocation in new_cache_units.get(
                (call.request_key, call.call_id),
                (),
            )
        ),
        forward_call_indices=tuple(
            call_index
            for call_index, call in enumerate(calls)
            for _ in forward_inputs.get(
                (call.request_key, call.call_id), ((), (), (), ())
            )[0]
        ),
        request_pool_indices=tuple(
            value
            for call in calls
            for value in forward_inputs.get(
                (call.request_key, call.call_id), ((), (), (), ())
            )[0]
        ),
        seq_lens=tuple(
            value
            for call in calls
            for value in forward_inputs.get(
                (call.request_key, call.call_id), ((), (), (), ())
            )[1]
        ),
        query_lens=tuple(
            value
            for call in calls
            for value in forward_inputs.get(
                (call.request_key, call.call_id), ((), (), (), ())
            )[2]
        ),
        write_kv=tuple(
            value
            for call in calls
            for value in forward_inputs.get(
                (call.request_key, call.call_id), ((), (), (), ())
            )[3]
        ),
        latent_params=tuple(
            latent_params[(call.request_key, call.call_id)]
            for call in calls
            if call.kind
            in {
                MediaCall.LATENT_PREPARATION,
                MediaCall.DENOISING,
                MediaCall.IMAGE_DECODING,
            }
        ),
        buffer_allocations=tuple(
            allocation
            for allocation in buffer_allocations
            if any(
                product.buffer_id == allocation.buffer
                for call in calls
                for product in (
                    *call.tensor_inputs(),
                    *call.tensor_outputs(),
                    *((call.predicate,) if call.predicate is not None else ()),
                )
            )
        ),
        input_products=input_products,
        commands=tuple(Start(request) for request in admissions),
    )


def _warmup_token_output(
    request_key: RequestKey, call_id: CallId, generation: int
) -> TensorRef:
    """Declare a call's ``token_output``: one packed int64 device scalar.

    The scalar packs the sampled token with its continuation bit, which lets
    a following DECODE call use it as its ``predicate``.
    """
    from uniserve_worker.protocol.tensor import DType, ShapeBound

    return TensorRef(
        request_key=request_key,
        producer_call_id=call_id,
        output_index=0,
        generation=generation,
        dtype=DType.I64,
        shape_bound=ShapeBound(),
    )


def _execute_warmup(
    requests: _WarmupRequests,
    batch: Batch,
    *,
    retain_device_outputs: bool = False,
) -> BatchOutput:
    """Submit one batch and block until `Worker.poll` consumes its result.

    Unless ``retain_device_outputs`` is set, a successful batch then frees
    each call's ``outputs`` and its token, completion, transition, and image
    outputs. Latent and KV outputs are never freed here: the flow scenario
    frees consumed latents itself, and the ``Finish`` that
    `_WarmupRequests.drop_request` submits retires the request's remaining
    storage. Errors recorded during submission propagate from
    ``Worker.submit``. On any failure the outputs stay allocated, and
    release is left to the enclosing Worker scope (see `warmup_requests`).

    Raises:
        RuntimeError: A call completed with ``CallStatus.ERROR``, or a
            completion is not a ``RequestOutput``.
    """
    worker = requests.worker
    state = worker.submit(batch, propagate_errors=True)

    # No service loop runs during startup, so warmup advances the executor
    # itself until the submission finalizes.
    while True:
        worker.advance()
        finalized = worker.poll(state)
        if finalized is not None:
            break
        time.sleep(0.00005)

    device_buffers = tuple(
        output.buffer_id
        for call in batch.calls
        for output in (
            *call.outputs,
            *(
                value
                for value in (
                    call.token_output,
                    call.completion_output,
                    call.transition_output,
                    call.image_output,
                )
                if value is not None
            ),
        )
    )
    failures: list[RequestOutput] = []
    for completion in finalized.completions:
        if not isinstance(completion, RequestOutput):
            raise RuntimeError(
                "finalized warmup result retains unresolved device output"
            )
        if completion.status is CallStatus.ERROR:
            failures.append(completion)

    if failures:
        parts = []
        for completion in failures:
            code = (
                completion.error_code.value
                if completion.error_code is not None
                else "internal"
            )
            parts.append(
                f"request={completion.request_key.request_id} "
                f"call={completion.call_id} code={code}"
            )
        details = ", ".join(parts)
        raise RuntimeError(f"startup warmup execution failed: {details}")

    if not retain_device_outputs:
        requests.free_products(device_buffers)

    return finalized


def _build_warmup_batch(
    requests: _WarmupRequests,
    *,
    admissions: tuple[NewRequest, ...],
    calls: tuple[Call, ...],
    input_products: tuple[TensorPublication, ...] = (),
    image_size: tuple[int, int] | None = None,
) -> Batch:
    """Derive allocations for a warmup submission and assemble its batch.

    Takes the next batch id, so each call's ``CallId`` must already name
    ``requests._batch_id + 1`` (``Batch.validate`` rejects calls of another
    batch). Reserves buffer spans for the calls' persistent products, binds
    request slots from the worker's rows or ``admissions``, grows the KV
    leases of KV-using calls to cover ``kv_visible_len`` plus, for token
    forwards, ``bounds.max_tokens``, leases latent pages for latent-holding
    calls at ``image_size`` (default `_warmup_image_size`), and adds the
    guidance-branch rows of DENOISING calls (`_warmup_flow_tables`). Leases
    persist in ``requests`` across batches until `drop_request`.

    Raises:
        WorkerError: From `invalid_descriptor` when a call has no request
            slot, a KV admission has a nonzero initial position, the worker
            has no KV cache, a lease would shrink, or KV, latent, or buffer
            capacity is exhausted, and from `_warmup_flow_tables` and
            ``Batch.validate``.
    """
    requests._batch_id += 1
    admissions_by_key = {
        admission.request_key: admission for admission in admissions
    }
    request_pool_indices: dict[RequestKey, int] = {}
    block_tables: dict[tuple[RequestKey, CallId], tuple[BlockTable, ...]] = {}
    new_cache_units: dict[
        tuple[RequestKey, CallId], tuple[CacheUnitAllocation, ...]
    ] = {}
    forward_inputs: dict[
        tuple[RequestKey, CallId],
        tuple[
            tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[bool, ...]
        ],
    ] = {}
    latent_params: dict[tuple[RequestKey, CallId], LatentParams] = {}
    buffer_allocations: dict[BufferId, BufferAllocation] = {}

    # Persistent products keep one span for their lifetime: an input
    # resolves to the span its producer was given.
    for call in calls:
        for product in (
            *call.buffer_inputs(),
            *call.buffer_outputs(),
        ):
            allocation = requests.buffer_allocation(product)
            buffer_allocations[allocation.buffer] = allocation

    # Bind request slots and grow the KV leases of each call that uses the
    # KV cache to the call's maximum shape.
    for call in calls:
        request = requests.worker.requests.peek(
            int(call.request_key.request_id)
        )
        admission = admissions_by_key.get(call.request_key)
        if request is None and admission is None:
            raise invalid_descriptor("warmup call has no request-pool binding")
        if request is None:
            assert admission is not None
            request_pool_indices[call.request_key] = admission.request_pool_idx
        else:
            request_pool_indices[call.request_key] = request.request_pool_idx

        if call.kind not in {
            ForwardMode.PREFILL,
            ForwardMode.DECODE,
            ForwardMode.VERIFY,
            ForwardMode.TOKEN_DENOISING,
            TransferMode.KV_PUBLISH,
            TransferMode.KV_INSTALL,
            MediaCall.LATENT_PREPARATION,
            MediaCall.DENOISING,
        }:
            continue

        if (
            request is None
            and admission is not None
            and admission.generation is not None
            and admission.generation.initial_position != 0
        ):
            raise invalid_descriptor(
                "warmup KV admission requires an empty prefix"
            )
        visible = 0
        if request is not None:
            runtime = request.accepted_progress
            visible = int(runtime.kv_visible_len)

        input_length = (
            int(call.bounds.max_tokens)
            if call.kind
            in {
                ForwardMode.PREFILL,
                ForwardMode.DECODE,
                ForwardMode.VERIFY,
            }
            else 0
        )

        # A lease keeps every unit granted to the request's earlier calls
        # until `drop_request`; a call whose extent needs fewer pages than
        # the lease already holds is rejected. Warmup retires no window page,
        # so every group's table covers the whole extent.
        tables, allocations = requests.grow_tables(
            requests._kv_units,
            call.request_key,
            request_pool_indices[call.request_key],
            visible + input_length,
        )
        identity = (call.request_key, call.call_id)
        block_tables[identity] = tables
        new_cache_units[identity] = allocations
        if input_length > 0:
            forward_inputs[identity] = (
                (request_pool_indices[call.request_key],),
                (visible + input_length,),
                (input_length,),
                (True,),
            )
        elif call.kind is ForwardMode.TOKEN_DENOISING:
            # A canvas step is one read-only row of its canvas over the
            # visible prefix.
            canvas = int(call.bounds.max_tokens)
            forward_inputs[identity] = (
                (request_pool_indices[call.request_key],),
                (visible + canvas,),
                (canvas,),
                (False,),
            )

    # Size each image request's latent page table: its latent units are the
    # leading dimension of the denoiser's image ``latent_shape`` (one per
    # patch), or a single unit on a worker without an image builder.
    height, width = image_size or _warmup_image_size(requests)
    builder = requests.worker.runner.image_builder
    latent_units = (
        builder.denoiser.latent_shape("image", ImageConfig(height, width))[0]
        if builder is not None
        else 1
    )
    page_units = int(requests.worker.info.latent_page_units)
    latent_page_count = (
        (latent_units + page_units - 1) // page_units if page_units > 0 else 0
    )
    occupied_latent_pages = {
        page for pages in requests._latent_pages.values() for page in pages
    }

    for call in calls:
        if (
            call.kind
            not in {
                MediaCall.LATENT_PREPARATION,
                MediaCall.DENOISING,
            }
            and call.latent_input is None
        ):
            continue
        page_table = requests._latent_pages.setdefault(call.request_key, [])
        missing = latent_page_count - len(page_table)
        if missing < 0:
            raise invalid_descriptor(
                "warmup latent allocation regresses its physical extent"
            )

        # Page zero is the latent pool's sentinel, outside its capacity.
        allocated = tuple(
            page
            for page in range(1, int(requests.worker.info.latent_pages))
            if page not in occupied_latent_pages
        )[:missing]
        if len(allocated) != missing:
            raise invalid_descriptor(
                "warmup latent allocation exceeds resident capacity"
            )

        page_table.extend(allocated)
        occupied_latent_pages.update(allocated)

        # The call runs solver steps from the request's accepted flow step;
        # warmup takes a DENOISING call's step count from its
        # ``bounds.max_tokens``.
        request = requests.worker.requests.peek(
            int(call.request_key.request_id)
        )
        start_step = (
            0 if request is None else int(request.accepted_progress.flow_step)
        )
        latent_params[(call.request_key, call.call_id)] = LatentParams(
            request_key=call.request_key,
            call_id=call.call_id,
            page_table=tuple(page_table),
            latent_units=latent_units,
            height=height,
            width=width,
            start_step=start_step,
            step_count=(
                int(call.bounds.max_tokens)
                if call.kind is MediaCall.DENOISING
                else 0
            ),
        )

        # A DENOISING call adds the alternative prefix's block table, if any,
        # and its forward rows are the guidance-branch rows.
        if call.kind is MediaCall.DENOISING:
            extra_tables, extra_allocations, flow_rows = _warmup_flow_tables(
                requests,
                call,
                request_pool_indices[call.request_key],
                height,
                width,
            )
            identity = (call.request_key, call.call_id)
            block_tables[identity] = (
                *block_tables.get(identity, ()),
                *extra_tables,
            )
            new_cache_units[identity] = (
                *new_cache_units.get(identity, ()),
                *extra_allocations,
            )
            forward_inputs[identity] = flow_rows

    return _warmup_batch(
        batch_id=requests._batch_id,
        admissions=admissions,
        calls=calls,
        block_tables=block_tables,
        new_cache_units=new_cache_units,
        forward_inputs=forward_inputs,
        latent_params=latent_params,
        buffer_allocations=tuple(buffer_allocations.values()),
        input_products=input_products,
    )


def _warmup_flow_tables(
    requests: _WarmupRequests,
    call: Call,
    main_slot: int,
    height: int,
    width: int,
) -> tuple[
    tuple[BlockTable, ...],
    tuple[CacheUnitAllocation, ...],
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[bool, ...]],
]:
    """Build KV tables and forward rows for a DENOISING call's CFG branches.

    The branches are those the request's guidance evaluates at its accepted
    flow step, with prefixes from `resolve_prefix`. A branch that reuses the
    request's conditioning reads the main slot; all other branches must
    share one alternative prefix. A nonempty alternative prefix is leased KV
    units in every group and a request slot of its own, and the rows start
    with one
    row that writes it into KV. Then follows one read-only row per branch
    whose query is the image sequence (patches plus framing tokens). The
    request must already be admitted.

    Returns:
        The alternative prefix's block tables, its newly allocated units,
        and the forward-row columns (request_pool_indices, seq_lens,
        query_lens, write_kv).

    Raises:
        WorkerError: From `invalid_descriptor` when the request is unknown
            or has no image parameters, the worker has no image builder or
            KV cache, the image has no guidance, `resolve_prefix` rejects a
            branch, branches need distinct alternative prefixes, or KV units
            or a request slot for the prefix are unavailable.
    """
    request = requests.worker.requests.get(call.request_key.request_id)
    image = request.image
    generation = requests.worker.runner.image_builder
    if image is None or generation is None:
        raise invalid_descriptor(
            "generation warmup has no admitted image runtime"
        )
    trajectory = image_state(generation, ImageConfig(height, width), image)
    guidance = trajectory.guidance
    if guidance is None:
        raise invalid_descriptor("generation warmup requires image guidance")
    branches = guidance.branches(
        trajectory.schedules["image"], request.accepted_progress.flow_step
    )

    runtime = request.accepted_progress
    query = generation.sequence_length(ImageConfig(height, width))
    image_prompt = image.image_prompts[0] if image.image_prompts else ""

    # Each branch either reuses the request's conditioning KV (the flag is
    # True) or conditions on a prefix of its own.
    branch_prefixes: list[tuple[tuple[int, ...], bool]] = []
    for branch in branches:
        prefix, copy_conditioning = resolve_prefix(
            requests.worker.runner.flow_prompt,
            generation.branch_source(branch),
            image_prompt=image_prompt,
            negative_prompt=image.negative_prompt,
            negative_token_ids=request.negative_token_ids,
            tokenizer=requests.worker.tokenizer,
        )
        branch_prefixes.append((prefix, copy_conditioning))

    alternatives = {
        prefix
        for prefix, copy_conditioning in branch_prefixes
        if not copy_conditioning
    }
    if len(alternatives) > 1:
        raise invalid_descriptor(
            "warmup flow has multiple distinct alternative prefixes"
        )
    alternative = next(iter(alternatives), ())

    tables: tuple[BlockTable, ...] = ()
    allocations: tuple[CacheUnitAllocation, ...] = ()
    alternative_slot = main_slot
    request_pool_indices: list[int] = []
    seq_lens: list[int] = []
    query_lens: list[int] = []
    write_kv: list[bool] = []

    if alternative:
        # Prefix slots are taken downward from the highest slot while the
        # flow scenario admits its requests upward from slot one; the
        # scenario's batch bound keeps the two apart.
        alternative_slot = requests._prefix_slots.setdefault(
            call.request_key,
            int(requests.worker.info.request_slots)
            - len(requests._prefix_slots),
        )
        admitted_slots = {
            requests.worker.requests.get(request_id).request_pool_idx
            for request_id in requests.worker.requests.request_ids()
        }
        if alternative_slot < 1 or alternative_slot in admitted_slots:
            raise invalid_descriptor(
                "warmup has no request slot for an alternative prefix"
            )
        # The prefix lease persists until `drop_request`, so every DENOISING
        # call of the request sees the same prefix tables and allocates only
        # new units, apart from the units of other requests' leases.
        tables, allocations = requests.grow_tables(
            requests._prefix_units,
            call.request_key,
            alternative_slot,
            len(alternative),
        )
        request_pool_indices.append(alternative_slot)
        seq_lens.append(len(alternative))
        query_lens.append(len(alternative))
        write_kv.append(True)

    # One row per branch, in the order ``guidance.branches`` returns them.
    for prefix, copy_conditioning in branch_prefixes:
        request_pool_indices.append(
            main_slot if copy_conditioning else alternative_slot
        )
        seq_lens.append(
            (int(runtime.kv_visible_len) if copy_conditioning else len(prefix))
            + query
        )
        query_lens.append(query)
        write_kv.append(False)

    return (
        tables,
        allocations,
        (
            tuple(request_pool_indices),
            tuple(seq_lens),
            tuple(query_lens),
            tuple(write_kv),
        ),
    )


def warmup_requests(worker: Worker) -> None:
    """Exercise synthetic requests through the configured execution paths.

    Runs only on a CUDA worker device: the token scenario when the worker
    supports PREFILL, and the flow scenario when it has an image builder.
    Successful scenarios retire their requests before returning. If execution
    fails, leave resource release to the enclosing Worker scope instead of
    issuing more execution commands that could replace the startup error.
    """
    requests = _WarmupRequests(worker)

    if torch.device(worker.worker_config.device).type == "cuda":
        if ForwardMode.PREFILL in worker.info.supported_calls:
            _warmup_tokens(requests)
            logger.info("completed token runtime warmup")

        if worker.canvas_slots is not None:
            _warmup_canvas(requests)
            logger.info("completed canvas generation warmup")

        if worker.runner.image_builder is not None:
            _warmup_flow(requests)
            logger.info("completed flow runtime warmup")


def _warmup_image_size(requests: _WarmupRequests) -> tuple[int, int]:
    """Derive a square warmup image size, in pixels.

    The side, in latent patches, is the integer square root of the worker's
    ``latent_capacity_units``, capped first at the image builder's
    ``max_tokens + framing`` when one exists, and is at least one patch.
    Without an image builder a patch is one pixel.
    """
    builder = requests.worker.runner.image_builder
    downsample = 1 if builder is None else builder.denoiser.downsample
    capacity = int(requests.worker.info.latent_capacity_units)
    if builder is not None:
        capacity = min(capacity, builder.max_tokens + builder.framing)

    side = max(1, math.isqrt(max(1, capacity)))
    return side * downsample, side * downsample


def _warmup_tokens(requests: _WarmupRequests) -> None:
    """Exercise the prefill-to-decode token handoff.

    Admits one greedy request, runs a one-token PREFILL, then, when DECODE is
    supported, a DECODE call gated on the prefill's device token. Returns
    without running when the worker lacks PREFILL or already holds requests;
    a PREFILL worker without KV cache storage raises ``WorkerError`` even if
    it holds requests. Drops its request afterwards.
    """
    from uniserve.sampling import SamplingParams
    from uniserve_worker.protocol.batch import GenerationParams, NewRequest
    from uniserve_worker.protocol.call import Bounds, Call, CallCoordinates

    variants = requests.worker.info.supported_calls
    if ForwardMode.PREFILL not in variants:
        return

    pool = requests.worker.kv_cache
    if pool is None:
        raise invalid_descriptor(
            "autoregressive warmup requires KV cache storage"
        )

    # Synthetic scenarios require an empty request pool.
    if requests.worker.requests.request_ids():
        return

    # Request ids double as one-based request slots; slot zero is the
    # padding sentinel.
    batch_sizes = (1,)
    request_ids = tuple(range(1, max(batch_sizes) + 1))
    keys = {sid: RequestKey(0, sid, 1) for sid in request_ids}
    admissions = {
        sid: NewRequest(
            keys[sid],
            request_pool_idx=sid,
            generation=GenerationParams(
                sampling=SamplingParams(temperature=0.0, ignore_eos=True),
                initial_position=0,
            ),
        )
        for sid in request_ids
    }

    next_product_generation = 1
    # Warmup owns this synthetic sequence, so it states the position each call
    # enters at and advances it by the tokens that call adds. Every warmup
    # request starts at the origin and holds no cached prefix, so its visible
    # and computed KV extents both track the logical position.
    positions = dict.fromkeys(request_ids, 0)

    def coordinates_for(sid: int, tokens: int) -> CallCoordinates:
        """State the entry coordinates of one call and advance the sequence."""
        position = positions[sid]
        positions[sid] = position + tokens
        return CallCoordinates(
            logical_position=position,
            kv_visible_len=position,
            kv_computed_len=position,
        )

    def prompt_op(
        sid: int,
        call_id: CallId,
        tokens: tuple[int, ...],
    ) -> Call:
        """Build one prompt computation with direct token inputs."""
        nonlocal next_product_generation
        outputs = _warmup_token_output(
            keys[sid], call_id, next_product_generation
        )
        next_product_generation += 1
        call = Call(
            request_key=keys[sid],
            call_id=call_id,
            coordinates=coordinates_for(sid, len(tokens)),
            kind=ForwardMode.PREFILL,
            bounds=Bounds(max_tokens=max(1, len(tokens))),
            input_token_ids=tokens,
            token_output=outputs,
        )
        return call

    def decode_op(sid: int, call_id: CallId, predecessor: Call) -> Call:
        """Build one decode call.

        The call consumes the predecessor's token product.
        """
        nonlocal next_product_generation
        token_output = predecessor.token_output
        assert token_output is not None
        outputs = _warmup_token_output(
            keys[sid], call_id, next_product_generation
        )
        next_product_generation += 1
        return Call(
            request_key=keys[sid],
            call_id=call_id,
            coordinates=coordinates_for(sid, 1),
            kind=ForwardMode.DECODE,
            bounds=Bounds(max_tokens=1),
            token_output=outputs,
            predicate=token_output,
        )

    # Call ids name the batch id `_build_warmup_batch` assigns next.
    predecessors: dict[int, Call] = {}
    calls: list[Call] = []
    for sid in request_ids:
        call_id = CallId(requests._batch_id + 1, len(calls))
        call = prompt_op(sid, call_id, (0,))
        calls.append(call)

    # A following decode reads the prefill's token output, so the output
    # must outlive the prefill batch.
    _execute_warmup(
        requests,
        _build_warmup_batch(
            requests,
            admissions=tuple(admissions[sid] for sid in request_ids),
            calls=tuple(calls),
        ),
        retain_device_outputs=ForwardMode.DECODE in variants,
    )
    predecessors.update(zip(request_ids, calls, strict=True))

    if ForwardMode.DECODE in variants:
        for batch_size in batch_sizes:
            selected = request_ids[:batch_size]
            calls = []
            for sid in selected:
                call_id = CallId(requests._batch_id + 1, len(calls))
                calls.append(decode_op(sid, call_id, predecessors[sid]))

            _execute_warmup(
                requests,
                _build_warmup_batch(
                    requests,
                    admissions=(),
                    calls=tuple(calls),
                ),
                retain_device_outputs=True,
            )

            # The decode has consumed its predecessors' outputs; its own stay
            # retained as the predecessors of any later decode.
            requests.free_products(
                tuple(
                    output.buffer_id
                    for sid in selected
                    for output in predecessors[sid].tensor_outputs()
                )
            )
            predecessors.update(zip(selected, calls, strict=True))

    device = torch.device(requests.worker.worker_config.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    for sid in request_ids:
        requests.drop_request(sid)


def _warmup_canvas(requests: _WarmupRequests) -> None:
    """Exercise the canvas steps and block commits of a generating denoiser.

    Admits one request per row of the largest sampler chunk
    (``CanvasSlots.step_rows``) with canvas sampling, prefills one prompt
    token each, runs one canvas step over each row count up to a whole
    chunk, and commits one canvas-length block. A step splits its rows into
    whole chunks and one remainder, so these are all the chunk shapes a
    step can run: the sampler kernels load and its self-conditioning
    product tunes each shape's algorithm, and the canvas pass's attention
    and expert kernels and the commit's prefill kernels are chosen, before
    any request arrives. Returns without running when the worker already
    holds requests. Drops its requests afterwards.
    """
    from uniserve.sampling import SamplingParams
    from uniserve_worker.protocol.batch import (
        CanvasSampling,
        GenerationParams,
        NewRequest,
    )
    from uniserve_worker.protocol.call import (
        Bounds,
        Call,
        CallCoordinates,
        CanvasStep,
    )

    slots = requests.worker.canvas_slots
    if slots is None or requests.worker.requests.request_ids():
        return

    length = slots.canvas_length
    request_ids = tuple(range(1, slots.step_rows + 1))
    keys = {sid: RequestKey(0, sid, 1) for sid in request_ids}
    # Row ``i`` runs one step in each call over at least ``i + 1`` rows.
    sampling = CanvasSampling(
        canvas_length=length,
        max_steps=len(request_ids),
        entropy_bound=0.1,
        t_min=0.4,
        t_max=0.8,
        confidence_threshold=0.005,
        # The served stability, so warmup stages the history requests use.
        stability_threshold=slots.history_depth,
    )
    admissions = tuple(
        NewRequest(
            keys[sid],
            request_pool_idx=sid,
            generation=GenerationParams(
                sampling=SamplingParams(seed=sid), canvas=sampling
            ),
        )
        for sid in request_ids
    )

    def prefill(sid, call_id, start, tokens):
        """A prefill of ``tokens`` that writes KV and samples nothing."""
        return Call(
            request_key=keys[sid],
            call_id=call_id,
            coordinates=CallCoordinates(start, start, start),
            kind=ForwardMode.PREFILL,
            bounds=Bounds(max_tokens=len(tokens)),
            input_token_ids=tokens,
        )

    def step(sid, call_id, number):
        """Step ``number`` of the request's first canvas after its token."""
        return Call(
            request_key=keys[sid],
            call_id=call_id,
            coordinates=CallCoordinates(1, 1, 1),
            kind=ForwardMode.TOKEN_DENOISING,
            bounds=Bounds(max_tokens=length, max_completion_bytes=4 * length),
            canvas=CanvasStep(0, number),
        )

    batch = requests._batch_id + 1
    _execute_warmup(
        requests,
        _build_warmup_batch(
            requests,
            admissions=admissions,
            calls=tuple(
                prefill(sid, CallId(batch, index), 0, (0,))
                for index, sid in enumerate(request_ids)
            ),
        ),
    )

    # Every row count from one to a whole chunk.
    steps = dict.fromkeys(request_ids, 0)
    for rows in range(1, len(request_ids) + 1):
        batch = requests._batch_id + 1
        calls = []
        for index, sid in enumerate(request_ids[:rows]):
            calls.append(step(sid, CallId(batch, index), steps[sid]))
            steps[sid] += 1
        _execute_warmup(
            requests,
            _build_warmup_batch(requests, admissions=(), calls=tuple(calls)),
        )

    batch = requests._batch_id + 1
    _execute_warmup(
        requests,
        _build_warmup_batch(
            requests,
            admissions=(),
            calls=(
                prefill(request_ids[0], CallId(batch, 0), 1, (0,) * length),
            ),
        ),
    )

    device = torch.device(requests.worker.worker_config.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for sid in request_ids:
        requests.drop_request(sid)


def _warmup_flow(requests: _WarmupRequests) -> None:
    """Drive chained denoise quanta through the real flow forward path.

    For each configured guidance-branch count, admits a batch of two-step
    image requests, publishes their conditioning KV, prepares their initial
    latents, runs both denoising steps as separate DENOISING calls, and
    drops the requests. A batch admits at most as many requests as the
    worker's request slots hold: every slot for single-branch generation,
    half of them for guided generation, whose requests each also hold a slot
    for their alternative guidance prefix. Returns without running when the
    worker lacks LATENT_PREPARATION, DENOISING, or an image builder, or
    already holds requests. Image decoding is not exercised.
    """
    from uniserve_worker.protocol.batch import NewRequest
    from uniserve_worker.protocol.call import (
        Bounds,
        Call,
        CallCoordinates,
        DrawLayout,
        Rng,
    )
    from uniserve_worker.protocol.tensor import (
        DeviceDim,
        DType,
        ShapeBound,
        StaticDim,
        TensorRef,
    )

    generation = requests.worker.runner.image_builder
    if (
        not {
            MediaCall.LATENT_PREPARATION,
            MediaCall.DENOISING,
        }.issubset(requests.worker.info.supported_calls)
        or generation is None
    ):
        return

    if requests.worker.requests.request_ids():
        return

    # One scenario per configured guidance-branch count, at the last capture
    # shape with that count or else one image of `_warmup_image_size`. The
    # model executor's capture has already run before these requests.
    configured = tuple(
        next(
            (
                shape
                for shape in reversed(requests.worker.runner.flow_captures)
                if shape.cfg_branches == branches
            ),
            DiffusionShape(1, *_warmup_image_size(requests), branches),
        )
        for branches in requests.worker.runner.flow_cfg_branches
    )

    # Warmup identities and generations are private to this bounded
    # startup sequence.
    next_request_id = 1
    next_generation = 1
    request_slots = int(requests.worker.info.request_slots)
    for bucket in configured:
        height = bucket.height
        width = bucket.width
        cfg_branches = bucket.cfg_branches
        # The engine leases every guided request a second slot for its
        # alternative prefix (``ensure_flow_prefix``), so no guided batch it
        # forms holds more than half the slots. The scenario is bounded the
        # same way: its prefix slots, taken downward from the highest slot,
        # then stay clear of the rows' slots taken upward from slot one. A
        # capture bucket with more rows is warmed at the largest batch the
        # slots hold; serving never replays it with more.
        batch_size = min(
            bucket.rows,
            request_slots // 2 if cfg_branches > 1 else request_slots,
        )
        if batch_size < 1:
            continue

        request_ids = tuple(
            range(next_request_id, next_request_id + batch_size)
        )
        next_request_id += batch_size
        keys = tuple(RequestKey(0, request_id, 1) for request_id in request_ids)
        admissions = tuple(
            NewRequest(
                key,
                request_pool_idx=index,
                image=capture_image_parameters(
                    cfg_branches,
                    steps=2,
                    height=height,
                    width=width,
                ),
            )
            for index, key in enumerate(keys, start=1)
        )
        roots = tuple(CallId(0, 0) for key in keys)

        # Publish one conditioning KV product per request, in the batch that
        # admits the requests.
        conditionings: list[BufferId] = []
        publications: list[Call] = []
        for key, root in zip(keys, roots, strict=True):
            call_id = CallId(requests._batch_id + 1, len(publications))
            conditioning = BufferId(
                owner=key,
                producer_call_id=call_id,
                output_index=0,
                generation=next_generation,
            )
            next_generation += 1
            conditionings.append(conditioning)
            publications.append(
                Call(
                    request_key=key,
                    call_id=call_id,
                    coordinates=CallCoordinates(),
                    kind=TransferMode.KV_PUBLISH,
                    bounds=Bounds(max_transfer_bytes=1 << 20),
                    kv_output=conditioning,
                )
            )
        _execute_warmup(
            requests,
            _build_warmup_batch(
                requests,
                admissions=admissions,
                calls=tuple(publications),
                image_size=(height, width),
            ),
        )

        # Latents are declared flat, as the element count of the image
        # ``latent_shape``; byte budgets assume BF16, 2 bytes per element.
        max_latent_elements = max(
            1,
            math.prod(
                generation.denoiser.latent_shape(
                    "image", ImageConfig(height, width)
                )
            ),
        )

        # Prepare each request's initial latent and its U8 readiness flag.
        initial_latents: list[TensorRef] = []
        transitions: list[Call] = []
        for key, root, conditioning in zip(
            keys, roots, conditionings, strict=True
        ):
            call_id = CallId(requests._batch_id + 1, len(transitions))
            initial_latent = TensorRef(
                request_key=key,
                producer_call_id=call_id,
                output_index=0,
                generation=next_generation,
                dtype=DType.BF16,
                shape_bound=ShapeBound((DeviceDim(max_latent_elements),)),
            )
            next_generation += 1
            ready = TensorRef(
                request_key=key,
                producer_call_id=call_id,
                output_index=1,
                generation=next_generation,
                dtype=DType.U8,
                shape_bound=ShapeBound((StaticDim(1),)),
            )
            next_generation += 1
            initial_latents.append(initial_latent)
            transitions.append(
                Call(
                    request_key=key,
                    call_id=call_id,
                    coordinates=CallCoordinates(),
                    kind=MediaCall.LATENT_PREPARATION,
                    bounds=Bounds(
                        max_tokens=1,
                        max_latent_bytes=max_latent_elements * 2,
                    ),
                    kv_input=conditioning,
                    latent_output=initial_latent,
                    completion_output=ready,
                    rng=Rng(
                        seed=0,
                        semantic_index_base=1,
                        draw_layout=DrawLayout.FLOW_NOISE,
                    ),
                )
            )

        _execute_warmup(
            requests,
            _build_warmup_batch(
                requests,
                admissions=(),
                calls=tuple(transitions),
                image_size=(height, width),
            ),
        )

        current_latents = tuple(initial_latents)
        flow_predecessors = dict(zip(request_ids, transitions, strict=True))

        # Run the two scheduled steps as chained DENOISING calls: each quantum
        # covers one step, enters at the step its index names, and consumes
        # the previous quantum's latent.
        for quantum in range(2):
            outputs: list[TensorRef] = []
            flows: list[Call] = []
            for request_id, key, conditioning, current in zip(
                request_ids, keys, conditionings, current_latents, strict=True
            ):
                call_id = CallId(requests._batch_id + 1, len(flows))
                output = TensorRef(
                    request_key=key,
                    producer_call_id=call_id,
                    output_index=0,
                    generation=next_generation,
                    dtype=DType.BF16,
                    shape_bound=ShapeBound((DeviceDim(max_latent_elements),)),
                )
                next_generation += 1
                outputs.append(output)
                flows.append(
                    Call(
                        request_key=key,
                        call_id=call_id,
                        coordinates=CallCoordinates(flow_step=quantum),
                        kind=MediaCall.DENOISING,
                        bounds=Bounds(
                            max_tokens=1,
                            max_latent_bytes=max_latent_elements * 2,
                        ),
                        kv_input=conditioning,
                        latent_input=current,
                        latent_output=output,
                    )
                )

            _execute_warmup(
                requests,
                _build_warmup_batch(
                    requests,
                    admissions=(),
                    calls=tuple(flows),
                    image_size=(height, width),
                ),
            )

            requests.free_products(
                tuple(product.buffer_id for product in current_latents)
            )
            current_latents = tuple(outputs)
            flow_predecessors.update(zip(request_ids, flows, strict=True))

        for request_id in request_ids:
            requests.drop_request(request_id)
