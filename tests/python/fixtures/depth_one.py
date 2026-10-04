"""Depth-one ``Call``/``Batch`` builders.

These are builders for worker forward-behavior tests.

Each builder produces the records the scheduler supplies at depth one: an
:class:`NewRequest`, an :class:`Call` whose ``predecessor``
names accepted execution progress, and the host-staged token payload
consumed by token work.
"""

from __future__ import annotations

import time
import weakref
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from uniserve.sampling import SamplingParams
from uniserve_worker._uniserve_ipc import Submission

if TYPE_CHECKING:
    from uniserve_worker.worker import Worker

from uniserve_worker.protocol.batch import (
    Batch,
    BatchCommand,
    BlockTable,
    BufferAllocation,
    CacheUnitAllocation,
    GenerationParams,
    LatentParams,
    NewRequest,
    Start,
    TensorPublication,
)
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    CallStatus,
    DrawLayout,
    ForwardMode,
    ImageParams,
    MediaCall,
    Rng,
    TransferMode,
    VisionInput,
)
from uniserve_worker.protocol.identity import BufferId, CallId, RequestKey
from uniserve_worker.protocol.output import BatchOutput, RequestOutput
from uniserve_worker.protocol.tensor import (
    DeviceDim,
    DType,
    ShapeBound,
    TensorRef,
)
from uniserve_worker.protocol.transfer import KvTransfer

AUTHORITY = 0
_BLOCK_TABLES: dict[RequestKey, list[int]] = {}
_REQUEST_POOL_INDICES: dict[RequestKey, int] = {}
_PAGES_TO_ZERO: dict[tuple[RequestKey, CallId], tuple[int, ...]] = {}
_UNBOUND_PAGES: dict[RequestKey, list[int]] = {}
_IMAGE_PARAMS: dict[RequestKey, ImageParams] = {}
_CALL_KV_LENGTHS: dict[
    tuple[RequestKey, CallId], tuple[int, int, int, int]
] = {}
_CALL_KV_RESULTS: dict[tuple[RequestKey, CallId], int] = {}
# Where each request currently stands on one worker, which its next call
# states. This is the engine's role: it projects a call's effect when it
# submits the call and replaces the projection with the rank's report when the
# call completes. The record is per worker, because a request admitted into two
# workers stands at two independent positions.
_LEDGERS: weakref.WeakKeyDictionary[object, _Ledger]
_LEDGERS = weakref.WeakKeyDictionary()
# What each built call is projected to leave behind. A call whose result only a
# device selection resolves records nothing and waits for its completion.
_CALL_PROJECTIONS: dict[tuple[RequestKey, CallId], tuple[str, int]] = {}
# Runs this fixture assembled. A worker also submits its own warmup batches,
# which state their own coordinates and are left untouched. Each entry holds
# its run so the identity stays valid until the run is submitted.
_FIXTURE_BATCHES: dict[int, Batch] = {}
_LATENT_STEPS: dict[TensorRef, int] = {}
_MAX_CFG_BRANCHES = 1
_REQUEST_POOL_SIZE = 1
_CACHE_PAGES = 1
_BLOCK_SIZE = 1
_COMMIT_MARKER_TOKENS = 1
_LATENT_PAGE_UNITS = 1
_LATENT_DOWNSAMPLE = 1
# Each request's admitted negative prompt, which the engine keeps in request
# state and sizes the text-unconditional CFG prefix from on every later call.
_NEGATIVE_PROMPTS: dict[RequestKey, tuple[int, ...]] = {}
_ALTERNATIVE_SLOTS: dict[RequestKey, int] = {}
_ALTERNATIVE_PAGES: dict[RequestKey, tuple[int, ...]] = {}
# Denoising calls that use their request's alternative prefix, and the
# requests whose prefix a successful denoising call has materialized. Once
# materialized, the worker retains the prefix slot's table and KV, so later
# calls state only their branch rows; an image's final step retires both.
_PREFIX_CALLS: set[tuple[RequestKey, CallId]] = set()
_MATERIALIZED_PREFIXES: set[RequestKey] = set()
_BUFFER_ALLOCATIONS: dict[BufferId, BufferAllocation] = {}


def configure_physical_pool(
    *,
    cache_pages: int,
    request_pool_size: int,
    block_size: int,
    commit_marker_tokens: int,
    max_cfg_branches: int,
    latent_page_units: int,
    latent_downsample: int,
) -> None:
    global _CACHE_PAGES, _REQUEST_POOL_SIZE, _BLOCK_SIZE, _COMMIT_MARKER_TOKENS
    global _MAX_CFG_BRANCHES, _LATENT_PAGE_UNITS, _LATENT_DOWNSAMPLE
    _CACHE_PAGES = max(1, int(cache_pages))
    _REQUEST_POOL_SIZE = max(1, int(request_pool_size))
    _BLOCK_SIZE = max(1, int(block_size))
    _COMMIT_MARKER_TOKENS = max(0, int(commit_marker_tokens))
    _MAX_CFG_BRANCHES = max(1, int(max_cfg_branches))
    _LATENT_PAGE_UNITS = max(1, int(latent_page_units))
    _LATENT_DOWNSAMPLE = max(1, int(latent_downsample))


def _reset_request(rk: RequestKey) -> None:
    _BLOCK_TABLES[rk] = []
    _REQUEST_POOL_INDICES.pop(rk, None)
    _UNBOUND_PAGES[rk] = []
    _NEGATIVE_PROMPTS.pop(rk, None)
    _discard_flow_prefix(rk)
    _IMAGE_PARAMS.pop(rk, None)
    for ledger in _LEDGERS.values():
        ledger.current.pop(rk, None)
        for identity in tuple(
            identity for identity in ledger.submitted if identity[0] == rk
        ):
            ledger.submitted.pop(identity, None)
    for table in (
        _PAGES_TO_ZERO,
        _CALL_KV_LENGTHS,
        _CALL_KV_RESULTS,
        _CALL_PROJECTIONS,
    ):
        for identity in tuple(
            identity for identity in table if identity[0] == rk
        ):
            table.pop(identity, None)
    _FIXTURE_BATCHES.clear()
    for ledger in _LEDGERS.values():
        ledger.detached.difference_update(
            identity for identity in tuple(ledger.detached) if identity[0] == rk
        )
    for product in tuple(
        product for product in _LATENT_STEPS if product.request_key == rk
    ):
        _LATENT_STEPS.pop(product, None)
    for buffer in tuple(
        buffer for buffer in _BUFFER_ALLOCATIONS if buffer.owner == rk
    ):
        _BUFFER_ALLOCATIONS.pop(buffer, None)
    _CALL_KV_RESULTS[(rk, CallId(0, 0))] = 0


def _discard_flow_prefix(rk: RequestKey) -> None:
    """Forget a request's alternative prefix slot, pages, and progress."""
    _ALTERNATIVE_SLOTS.pop(rk, None)
    _ALTERNATIVE_PAGES.pop(rk, None)
    _MATERIALIZED_PREFIXES.discard(rk)
    _PREFIX_CALLS.difference_update(
        identity for identity in tuple(_PREFIX_CALLS) if identity[0] == rk
    )


def _kv_page(value: int) -> int:
    return int(value) + 1


def _latent_params(call: Call) -> LatentParams:
    image = _IMAGE_PARAMS[call.request_key]
    latent_units = max(
        1,
        (int(image.height) // _LATENT_DOWNSAMPLE)
        * (int(image.width) // _LATENT_DOWNSAMPLE),
    )
    page_count = (latent_units + _LATENT_PAGE_UNITS - 1) // _LATENT_PAGE_UNITS
    latent_input = call.latent_input
    start_step = (
        0 if latent_input is None else _LATENT_STEPS.get(latent_input, 0)
    )
    return LatentParams(
        request_key=call.request_key,
        call_id=call.call_id,
        page_table=tuple(range(1, page_count + 1)),
        latent_units=latent_units,
        height=int(image.height),
        width=int(image.width),
        start_step=start_step,
        step_count=(
            int(call.bounds.max_tokens)
            if call.kind is MediaCall.DENOISING
            else 0
        ),
    )


def _parent_kv_length(rk: RequestKey, predecessor: CallId) -> int:
    return _CALL_KV_RESULTS.get((rk, predecessor), 0)


def _ledger(worker: object) -> _Ledger:
    """The per-worker record of where each of its requests stands."""
    return _LEDGERS.setdefault(worker, _Ledger())


def _project_tokens(rk: RequestKey, call_id: CallId, tokens: int) -> None:
    """Record a call that adds `tokens` positions and initializes them."""
    _CALL_PROJECTIONS[(rk, call_id)] = ("tokens", int(tokens))


def _project_flow_step(rk: RequestKey, call_id: CallId, flow_step: int) -> None:
    """Record that a call leaves the request's trajectory at `flow_step`."""
    _CALL_PROJECTIONS[(rk, call_id)] = ("flow_step", int(flow_step))


class _Ledger:
    """One worker's view of its own requests.

    Holds where each request stands and where each submitted call entered from.
    """

    def __init__(self) -> None:
        self.current: dict[RequestKey, CallCoordinates] = {}
        self.submitted: dict[tuple[RequestKey, CallId], CallCoordinates] = {}
        # A media branch call around the request's state chain carries no
        # coordinates, so its completion reports none and leaves the request
        # where it was.
        self.detached: set[tuple[RequestKey, CallId]] = set()
        self.media: set[RequestKey] = set()


def _projected(
    entry: CallCoordinates, rule: tuple[str, int]
) -> CallCoordinates:
    """Apply one call's recorded effect to where the request stands."""
    name, value = rule
    if name == "flow_step":
        return replace(entry, flow_step=value)
    return CallCoordinates(
        logical_position=entry.logical_position + value,
        kv_visible_len=entry.kv_visible_len + value,
        kv_computed_len=entry.kv_visible_len + value,
        flow_step=entry.flow_step,
    )


def record_kv_result(
    rk: RequestKey, call_id: CallId, visible_length: int
) -> None:
    _CALL_KV_RESULTS[(rk, call_id)] = int(visible_length)


def bind_request_allocation(
    rk: RequestKey,
    *,
    request_pool_idx: int,
    page_ids: Sequence[int],
) -> None:
    pages = [int(page) for page in page_ids]
    if int(request_pool_idx) < 1 or any(page < 1 for page in pages):
        raise ValueError("request params identifiers must be positive")
    if len(set(pages)) != len(pages):
        raise ValueError("request params repeats a KV page")
    _REQUEST_POOL_INDICES[rk] = int(request_pool_idx)
    _BLOCK_TABLES[rk] = pages
    _UNBOUND_PAGES[rk] = []


def _record_existing_kv(
    rk: RequestKey,
    call_id: CallId,
    predecessor: CallId,
    input_length: int,
) -> int:
    prefix = _parent_kv_length(rk, predecessor)
    resulting = prefix + int(input_length)
    block_table = _BLOCK_TABLES.get(rk, ())
    _CALL_KV_LENGTHS[(rk, call_id)] = (
        prefix,
        int(input_length),
        prefix,
        resulting,
    )
    _CALL_KV_RESULTS[(rk, call_id)] = resulting
    return len(block_table)


def _alternative_slot(rk: RequestKey) -> int:
    existing = _ALTERNATIVE_SLOTS.get(rk)
    if existing is not None:
        return existing
    occupied = set(_REQUEST_POOL_INDICES.values()) | set(
        _ALTERNATIVE_SLOTS.values()
    )
    slot = next(
        (
            candidate
            for candidate in range(_REQUEST_POOL_SIZE, 0, -1)
            if candidate not in occupied
        ),
        None,
    )
    if slot is None:
        raise RuntimeError(
            "test scheduler has no request slot for a flow prefix"
        )
    _ALTERNATIVE_SLOTS[rk] = slot
    return slot


def _alternative_pages(rk: RequestKey, tokens: int) -> tuple[int, ...]:
    needed = (int(tokens) + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    existing = _ALTERNATIVE_PAGES.get(rk, ())
    if len(existing) >= needed:
        return existing[:needed]
    occupied = {
        page
        for pages in (*_BLOCK_TABLES.values(), *_ALTERNATIVE_PAGES.values())
        for page in pages
    }
    selected = tuple(
        candidate
        for candidate in range(_CACHE_PAGES - 1, 0, -1)
        if candidate not in occupied
    )[:needed]
    if len(selected) != needed:
        raise RuntimeError(
            "test scheduler has no ordinary KV pages for a flow prefix"
        )
    _ALTERNATIVE_PAGES[rk] = selected
    return selected


def stamp_batch(worker: object, batch: Batch) -> Batch:
    """State every call's coordinates from the worker's request ledger.

    The engine stamps coordinates when it assembles a batch, from the request
    state it owns, and carries its own projection forward so the next call it
    submits states where this one leaves off.
    """
    _FIXTURE_BATCHES.pop(id(batch), None)
    ledger = _ledger(worker).current
    submitted = _ledger(worker).submitted
    detached = _ledger(worker).detached
    for command in batch.commands:
        admission = getattr(command, "request", None)
        if admission is None or not isinstance(admission, NewRequest):
            continue
        # Admission starts the request at the position it was admitted with.
        prefix = (
            0
            if admission.generation is None
            else int(admission.generation.initial_position)
        )
        ledger[admission.request_key] = CallCoordinates(
            logical_position=prefix,
            kv_visible_len=prefix,
            kv_computed_len=prefix,
        )
        if admission.diffusion is not None:
            _ledger(worker).media.add(admission.request_key)
    stamped = []
    for call in batch.calls:
        identity = (call.request_key, call.call_id)
        entry = ledger.get(call.request_key, CallCoordinates())
        submitted[identity] = entry
        media = call.request_key in _ledger(worker).media
        if media and not call.advances_state:
            detached.add(identity)
        rule = _CALL_PROJECTIONS.get(identity)
        if rule is not None:
            ledger[call.request_key] = _projected(entry, rule)
        stamped.append(replace(call, coordinates=entry))
    return replace(batch, calls=tuple(stamped))


def submitted_batch(worker: object, batch: Batch) -> Batch:
    """State the coordinates of a run this fixture assembled.

    A worker also submits its own warmup batches, which state their own
    coordinates and pass through untouched.
    """
    if _FIXTURE_BATCHES.get(id(batch)) is None:
        return batch
    return stamp_batch(worker, batch)


def observe_completions(worker: object, report: BatchOutput) -> None:
    """Advance each request to where its accepted calls left it.

    A rank chains a call from the call it last accepted, and this is the
    same record the engine keeps from the completions it observes. A failed
    call accepts nothing, so it leaves the request where it was.
    """
    ledger = _ledger(worker).current
    submitted = _ledger(worker).submitted
    detached = _ledger(worker).detached
    for record in report.completions:
        # A successful denoising call has materialized its request's
        # alternative prefix. The image's final step retires the prefix on
        # the worker, as the engine then frees it, so the next image of the
        # request allocates and materializes its own.
        if (
            record.status is CallStatus.OK
            and (record.request_key, record.call_id) in _PREFIX_CALLS
        ):
            steps = int(_IMAGE_PARAMS[record.request_key].steps)
            if int(record.num_completed_steps) >= steps:
                _discard_flow_prefix(record.request_key)
            else:
                _MATERIALIZED_PREFIXES.add(record.request_key)
        if (record.request_key, record.call_id) in detached:
            continue
        if record.status is CallStatus.ERROR:
            # A failed call accepts nothing, so the request stays where it
            # entered that call.
            entry = submitted.get((record.request_key, record.call_id))
            if entry is not None:
                ledger[record.request_key] = entry
            continue
        ledger[record.request_key] = CallCoordinates(
            logical_position=record.position,
            kv_visible_len=record.kv_visible_len,
            kv_computed_len=record.kv_computed_len,
            flow_step=record.num_completed_steps,
        )


def execution_batch(
    *,
    batch_id: int,
    admissions: Sequence[NewRequest] = (),
    calls: Sequence[Call] = (),
    input_products: Sequence[TensorPublication] = (),
    kv_inputs: Sequence[KvTransfer] = (),
    commands: Sequence[BatchCommand] = (),
    block_tables: Sequence[BlockTable] = (),
    new_cache_units: Sequence[CacheUnitAllocation] = (),
) -> Batch:
    """Build scheduler columns and physical allocations.

    The columns and allocations drive observable worker behavior.

    A submission that carries calls takes its identity from them, because a
    call's ``call_id`` already names the batch that carries it. ``batch_id``
    names a command-only submission and orders every submission's
    collectives, so successive submissions to one worker must advance it.
    """
    for admission in admissions:
        if admission.image is not None:
            _IMAGE_PARAMS[admission.request_key] = admission.image
        # Admission starts the request's state on this worker, as the engine's
        # running state begins: the admitted negative prompt, and no
        # alternative prefix until the first denoising call allocates one.
        _NEGATIVE_PROMPTS[admission.request_key] = (
            ()
            if admission.generation is None
            else admission.generation.negative_token_ids
        )
        _discard_flow_prefix(admission.request_key)
        _REQUEST_POOL_INDICES[admission.request_key] = int(
            admission.request_pool_idx
        )
    for call in calls:
        for product in (
            *call.buffer_inputs(),
            *call.buffer_outputs(),
        ):
            if product.buffer_id in _BUFFER_ALLOCATIONS:
                continue
            required = int(product.max_bytes)
            offset = 0
            for params in sorted(
                _BUFFER_ALLOCATIONS.values(), key=lambda value: value.offset
            ):
                offset = (offset + 255) & ~255
                if offset + required <= params.offset:
                    break
                offset = max(offset, params.offset + params.bytes)
            offset = (offset + 255) & ~255
            _BUFFER_ALLOCATIONS[product.buffer_id] = BufferAllocation(
                product.buffer_id,
                offset,
                required,
            )
    explicit_tables = {
        (int(table.request_pool_idx), int(table.group_id)): table
        for table in block_tables
    }

    def table_for(call: Call) -> BlockTable | None:
        slot = _REQUEST_POOL_INDICES.get(
            call.request_key,
            int(call.request_key.request_id) + 1,
        )
        explicit = explicit_tables.get((slot, 0))
        if explicit is not None:
            return explicit
        lengths = _CALL_KV_LENGTHS.get((call.request_key, call.call_id))
        if lengths is None:
            return None
        pages = tuple(_BLOCK_TABLES.get(call.request_key, ()))
        return BlockTable(slot, 0, 0, pages, len(pages) * _BLOCK_SIZE)

    tables: dict[tuple[int, int], BlockTable] = {}
    allocations: dict[tuple[int, int], set[int]] = {}
    forward_call_indices: list[int] = []
    request_pool_indices: list[int] = []
    seq_lens: list[int] = []
    query_lens: list[int] = []
    write_kv: list[bool] = []
    for call_index, call in enumerate(calls):
        table = table_for(call)
        if table is not None:
            identity = (table.request_pool_idx, table.group_id)
            tables[identity] = table
            pages = _PAGES_TO_ZERO.get((call.request_key, call.call_id), ())
            if pages:
                allocations.setdefault(identity, set()).update(pages)
        lengths = _CALL_KV_LENGTHS.get((call.request_key, call.call_id))
        if lengths is not None and lengths[1] > 0:
            forward_call_indices.append(call_index)
            request_pool_indices.append(_REQUEST_POOL_INDICES[call.request_key])
            seq_lens.append(lengths[2] + lengths[1])
            query_lens.append(lengths[1])
            write_kv.append(True)
        if call.kind is MediaCall.DENOISING:
            image = _IMAGE_PARAMS[call.request_key]
            main_slot = _REQUEST_POOL_INDICES[call.request_key]
            main_len = 0 if lengths is None else lengths[2]
            text_off = abs(float(image.cfg_text_scale) - 1.0) <= 1e-6
            image_off = abs(float(image.cfg_img_scale) - 1.0) <= 1e-6
            branches = (
                1
                if text_off and image_off
                else 2
                if text_off or image_off
                else 3
            )
            branches = min(branches, _MAX_CFG_BRANCHES)
            query_len = (
                max(1, int(image.height) // _LATENT_DOWNSAMPLE)
                * max(1, int(image.width) // _LATENT_DOWNSAMPLE)
                + _COMMIT_MARKER_TOKENS
            )
            alternative: tuple[int, int] | None = None
            if branches > 1:
                # The text-unconditional branch owns a request slot sized for
                # the admitted negative prompt. Its pages are new, and so
                # zeroed, only on the call that allocates them; the slot's
                # table and prefix row are restated until a denoising call
                # succeeds and the worker retains the materialized prefix.
                negative = _NEGATIVE_PROMPTS.get(call.request_key, ())
                allocating = call.request_key not in _ALTERNATIVE_SLOTS
                alt_slot = _alternative_slot(call.request_key)
                alt_pages = _alternative_pages(call.request_key, len(negative))
                if allocating and alt_pages:
                    allocations.setdefault((alt_slot, 0), set()).update(
                        alt_pages
                    )
                _PREFIX_CALLS.add((call.request_key, call.call_id))
                if call.request_key not in _MATERIALIZED_PREFIXES:
                    tables[(alt_slot, 0)] = BlockTable(
                        alt_slot,
                        0,
                        0,
                        alt_pages,
                        len(alt_pages) * _BLOCK_SIZE,
                    )
                    if negative:
                        forward_call_indices.append(call_index)
                        request_pool_indices.append(alt_slot)
                        seq_lens.append(len(negative))
                        query_lens.append(len(negative))
                        write_kv.append(True)
                alternative = (alt_slot, len(negative))
            for branch in range(branches):
                slot, seq_len = (
                    (main_slot, main_len)
                    if branch == 0 or alternative is None
                    else alternative
                )
                forward_call_indices.append(call_index)
                request_pool_indices.append(slot)
                seq_lens.append(seq_len + query_len)
                query_lens.append(query_len)
                write_kv.append(False)
    for allocation in new_cache_units:
        identity = (allocation.request_pool_idx, allocation.group_id)
        allocations.setdefault(identity, set()).update(allocation.unit_ids)
    # A call's identity names the batch that carries it, so a submission that
    # carries calls takes its identity from them; `batch_id` names a
    # command-only submission and orders every submission's collectives.
    run = Batch(
        batch_id=calls[0].call_id.batch_id if calls else int(batch_id),
        collective_seq=int(batch_id) * 1024 + 2,
        calls=tuple(calls),
        block_tables=tuple(tables.values()),
        new_cache_units=tuple(
            CacheUnitAllocation(slot, group, tuple(sorted(pages)))
            for (slot, group), pages in allocations.items()
            if pages
        ),
        forward_call_indices=tuple(forward_call_indices),
        request_pool_indices=tuple(request_pool_indices),
        seq_lens=tuple(seq_lens),
        query_lens=tuple(query_lens),
        write_kv=tuple(write_kv),
        latent_params=tuple(
            _latent_params(call)
            for call in calls
            if call.kind in {MediaCall.LATENT_PREPARATION, MediaCall.DENOISING}
            or call.latent_input is not None
        ),
        buffer_allocations=tuple(
            {
                product.buffer_id: _BUFFER_ALLOCATIONS[product.buffer_id]
                for call in calls
                for product in (
                    *call.buffer_inputs(),
                    *call.buffer_outputs(),
                )
            }.values()
        ),
        input_products=tuple(input_products),
        kv_inputs=tuple(kv_inputs),
        commands=tuple(Start(request) for request in admissions)
        + tuple(commands),
    )
    _FIXTURE_BATCHES[id(run)] = run
    return run


def request_key(request_id: int, request_epoch: int = 1) -> RequestKey:
    return RequestKey(AUTHORITY, request_id, request_epoch)


def ar_params(
    request_id: int,
    *,
    block_ids: Sequence[int] = (),
    prefix_len: int = 0,
    request_epoch: int = 1,
    sampling: SamplingParams | None = None,
    input_images: int = 0,
) -> NewRequest:
    rk = request_key(request_id, request_epoch)
    _reset_request(rk)
    _BLOCK_TABLES[rk] = [_kv_page(value) for value in block_ids]
    _UNBOUND_PAGES[rk] = list(_BLOCK_TABLES[rk])
    _REQUEST_POOL_INDICES[rk] = request_id + 1
    _CALL_KV_RESULTS[(rk, CallId(0, 0))] = int(prefix_len)
    return NewRequest(
        rk,
        request_pool_idx=request_id + 1,
        generation=GenerationParams(
            sampling=(
                sampling
                if sampling is not None
                else SamplingParams(temperature=0.0, ignore_eos=True)
            ),
            initial_position=int(prefix_len),
        ),
        input_images=input_images,
    )


def umm_params(
    request_id: int, image: ImageParams, *, request_epoch: int = 1
) -> NewRequest:
    rk = request_key(request_id, request_epoch)
    _reset_request(rk)
    _IMAGE_PARAMS[rk] = image
    _REQUEST_POOL_INDICES[rk] = request_id + 1
    return NewRequest(
        rk,
        request_pool_idx=request_id + 1,
        image=image,
    )


def root_parent(admission: NewRequest) -> CallId:
    """The ordering sentinel for a request's first state call."""
    return CallId(0, 0)


def finalized_report(worker: Worker, state: Submission) -> BatchOutput:
    """Drive the public Worker interface until the batch returns its result."""
    deadline = time.monotonic() + 10.0
    while True:
        worker.advance()
        output = worker.poll(state)
        if output is not None:
            observe_completions(worker, output)
            return output
        if time.monotonic() >= deadline:
            raise TimeoutError("worker completion did not become query-ready")
        time.sleep(0.00005)


def record_completion(call: Call, report: BatchOutput) -> RequestOutput:
    """Observe accepted output and carry its visible KV extent.

    The extent is carried into the next test input.
    """
    resolved = report
    matches = tuple(
        record
        for record in resolved.completions
        if record.request_key == call.request_key
        and record.call_id == call.call_id
    )
    if len(matches) != 1 or matches[0].status is not CallStatus.OK:
        raise ValueError("call has no unique successful completion")
    record = matches[0]
    record_kv_result(call.request_key, call.call_id, record.kv_visible_len)
    return record


def token_call(
    rk: RequestKey,
    *,
    call_id: CallId,
    predecessor: CallId,
    mode: ForwardMode,
    tokens: Sequence[int],
    block_table_delta: Sequence[int] = (),
    predicate: TensorRef | None = None,
    logprobs: bool = False,
    rng: Rng | None = None,
) -> Call:
    """Build a token computation with its actual model input IDs."""
    block_table = _BLOCK_TABLES.setdefault(rk, [])
    added = [_kv_page(value) for value in block_table_delta]
    if set(added) & set(block_table):
        raise ValueError("KV block-table delta repeats an existing page")
    block_table.extend(added)
    pending = _UNBOUND_PAGES.setdefault(rk, [])
    pending.extend(added)
    _PAGES_TO_ZERO[(rk, call_id)] = tuple(pending)
    pending.clear()
    prefix_length = _parent_kv_length(rk, predecessor)
    input_length = len(tokens)
    _CALL_KV_LENGTHS[(rk, call_id)] = (
        prefix_length,
        input_length,
        prefix_length,
        prefix_length + input_length,
    )
    if mode is not ForwardMode.VERIFY:
        _CALL_KV_RESULTS[(rk, call_id)] = prefix_length + input_length
        _project_tokens(rk, call_id, input_length)

    token_output = TensorRef(
        request_key=rk,
        producer_call_id=call_id,
        output_index=0,
        generation=call_id.batch_id * 4 + 1,
        dtype=DType.I64,
        shape_bound=ShapeBound(),
    )
    call = Call(
        request_key=rk,
        call_id=call_id,
        coordinates=CallCoordinates(),
        kind=mode,
        bounds=Bounds(
            max_tokens=max(1, len(tokens)),
            max_kv_units=len(added),
            max_completion_bytes=((1 << 16) - 1 if logprobs else 0),
        ),
        input_token_ids=tuple(int(value) for value in tokens),
        token_output=token_output,
        predicate=predicate,
        rng=rng,
    )
    return call


def encode_call(
    rk: RequestKey,
    *,
    call_id: CallId,
    predecessor: CallId,
    image_base64: str | None,
    encoder_handle: int,
    mode: MediaCall = MediaCall.VISION_ENCODING,
    source_product: TensorRef | None = None,
    component: str = "model",
) -> Call:
    """An encoder computation with an encoded image or a resident image source.

    The scheduler stamps the encode output reference's ``generation`` with the
    content-stable encoder handle; the worker echoes it in
    ``completion.product_generations``.
    """
    if (image_base64 is None) == (source_product is None):
        raise ValueError("encode call requires exactly one image source")
    output_ref = TensorRef(
        request_key=rk,
        producer_call_id=call_id,
        output_index=0,
        generation=int(encoder_handle),
        dtype=DType.BF16,
        shape_bound=ShapeBound((DeviceDim(4_096),)),
    )
    call = Call(
        request_key=rk,
        call_id=call_id,
        coordinates=CallCoordinates(),
        kind=mode,
        # Which component serves a call is the model's, not the call's: a model
        # that encodes images from its own component names that component,
        # while one that encodes from its language backbone names "model".
        component=component,
        bounds=Bounds(max_tokens=64, max_latent_bytes=8_192),
        input_image=image_base64,
        image_input=source_product,
        encoder_output=output_ref,
    )
    return call


def diffusion_prepare_call(
    rk: RequestKey,
    *,
    call_id: CallId,
    predecessor: CallId,
    conditioning: BufferId,
    seed: int = 29,
    image_index: int = 1,
) -> tuple[Call, TensorRef]:
    image = _IMAGE_PARAMS[rk]
    latent = TensorRef(
        request_key=rk,
        producer_call_id=call_id,
        output_index=0,
        generation=call_id.batch_id * 3 + 1,
        dtype=DType.BF16,
        shape_bound=ShapeBound(
            (DeviceDim(3 * int(image.height) * int(image.width)),)
        ),
    )
    ready = TensorRef(
        request_key=rk,
        producer_call_id=call_id,
        output_index=1,
        generation=call_id.batch_id * 3 + 2,
        dtype=DType.U8,
        shape_bound=ShapeBound(),
    )
    _record_existing_kv(rk, call_id, predecessor, 0)
    _project_flow_step(rk, call_id, 0)
    call = Call(
        request_key=rk,
        call_id=call_id,
        coordinates=CallCoordinates(),
        kind=MediaCall.LATENT_PREPARATION,
        bounds=Bounds(max_tokens=1, max_latent_bytes=latent.max_bytes),
        kv_input=conditioning,
        latent_output=latent,
        completion_output=ready,
        rng=Rng(
            seed=int(seed),
            semantic_index_base=int(image_index),
            draw_layout=DrawLayout.FLOW_NOISE,
        ),
    )
    _LATENT_STEPS[latent] = 0
    return call, latent


def diffusion_step_call(
    rk: RequestKey,
    *,
    call_id: CallId,
    predecessor: CallId,
    conditioning: BufferId,
    latent: TensorRef,
    steps: int,
) -> tuple[Call, TensorRef]:
    output = TensorRef(
        request_key=rk,
        producer_call_id=call_id,
        output_index=0,
        generation=call_id.batch_id * 3 + 1,
        dtype=DType.BF16,
        shape_bound=latent.shape_bound,
    )
    _record_existing_kv(rk, call_id, predecessor, 0)
    _project_flow_step(rk, call_id, _LATENT_STEPS.get(latent, 0) + int(steps))
    call = Call(
        request_key=rk,
        call_id=call_id,
        coordinates=CallCoordinates(),
        kind=MediaCall.DENOISING,
        bounds=Bounds(max_tokens=int(steps), max_latent_bytes=output.max_bytes),
        kv_input=conditioning,
        latent_input=latent,
        latent_output=output,
    )
    _LATENT_STEPS[output] = _LATENT_STEPS.get(latent, 0) + int(steps)
    return call, output


def kv_publication_call(
    rk: RequestKey,
    *,
    call_id: CallId,
    predecessor: CallId,
) -> tuple[Call, BufferId]:
    product = BufferId(
        owner=rk,
        producer_call_id=call_id,
        output_index=0,
        generation=call_id.batch_id * 3 + 1,
    )
    _record_existing_kv(rk, call_id, predecessor, 0)
    call = Call(
        request_key=rk,
        call_id=call_id,
        coordinates=CallCoordinates(),
        kind=TransferMode.KV_PUBLISH,
        bounds=Bounds(max_transfer_bytes=1 << 20),
        kv_output=product,
    )
    return call, product


def diffusion_finalize_call(
    rk: RequestKey,
    *,
    call_id: CallId,
    predecessor: CallId,
    latent: TensorRef,
    feedback_source: bool = False,
) -> Call:
    image_output = None
    if feedback_source:
        image_output = TensorRef(
            request_key=rk,
            producer_call_id=call_id,
            output_index=1,
            generation=call_id.batch_id * 3 + 1,
            dtype=DType.BF16,
            shape_bound=ShapeBound((DeviceDim(3 * 16 * 16),)),
        )
    _project_flow_step(rk, call_id, 0)
    return Call(
        request_key=rk,
        call_id=call_id,
        coordinates=CallCoordinates(),
        kind=MediaCall.IMAGE_DECODING,
        bounds=Bounds(
            max_latent_bytes=(3 * 16 * 16 * 2 if feedback_source else 0),
            max_completion_bytes=65_536,
        ),
        latent_input=latent,
        image_output=image_output,
    )


def visual_state_call(
    rk: RequestKey,
    *,
    call_id: CallId,
    predecessor: CallId,
    feature: TensorRef,
    sample_continuation: bool,
    max_tokens: int,
) -> Call:
    completion = TensorRef(
        request_key=rk,
        producer_call_id=call_id,
        output_index=0,
        generation=call_id.batch_id * 3,
        dtype=DType.U8,
        shape_bound=ShapeBound(),
    )
    token = None
    if sample_continuation:
        token = TensorRef(
            request_key=rk,
            producer_call_id=call_id,
            output_index=1,
            generation=call_id.batch_id * 3 + 1,
            dtype=DType.I64,
            shape_bound=ShapeBound(),
        )
    _record_existing_kv(rk, call_id, predecessor, max_tokens)
    return Call(
        request_key=rk,
        call_id=call_id,
        coordinates=CallCoordinates(),
        kind=ForwardMode.PREFILL,
        bounds=Bounds(max_tokens=max_tokens),
        vision_inputs=(VisionInput(0, feature),),
        completion_output=completion,
        token_output=token,
    )


__all__ = [
    "AUTHORITY",
    "bind_request_allocation",
    "record_completion",
    "encode_call",
    "execution_batch",
    "finalized_report",
    "diffusion_step_call",
    "diffusion_prepare_call",
    "umm_params",
    "diffusion_finalize_call",
    "kv_publication_call",
    "observe_completions",
    "stamp_batch",
    "submitted_batch",
    "record_kv_result",
    "request_key",
    "root_parent",
    "token_call",
    "ar_params",
    "visual_state_call",
]
