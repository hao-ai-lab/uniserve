"""Bounded request, sampling, and product startup scenarios."""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

import torch

from uniserve.math import ceil_div
from uniserve.media.image import Config as ImageConfig
from uniserve_worker.protocol.call import (
    ForwardMode,
    MediaCall,
    TransferMode,
)
from uniserve_worker.protocol.identity import (
    BufferId,
    CallId,
    RequestKey,
)

from ..execution.diffusion_state import resolve_prefix
from ..execution.flow import image_state
from ..execution.graph_inputs import DiffusionShape
from ..execution.model_runner import capture_image_parameters
from ..foundation.errors import invalid_descriptor
from ..protocol.batch import (
    Batch,
    BatchCommand,
    BlockTable,
    BufferAllocation,
    CachePageAllocation,
    Finish,
    Free,
    LatentParams,
    NewRequest,
    Start,
    TensorPublication,
)
from ..protocol.call import Call, CallStatus
from ..protocol.output import BatchOutput, RequestOutput
from ..protocol.tensor import TensorRef

if TYPE_CHECKING:
    from .worker import Worker

logger = logging.getLogger(__name__)


class _WarmupRequests:
    """Track synthetic request allocations.

    Allocations borrow Worker execution resources.
    """

    def __init__(self, worker: Worker) -> None:
        """Borrow startup-owned worker resources.

        The borrowed resources exercise every execution shape.
        """
        self.worker = worker
        self._kv_pages: dict[tuple[RequestKey, int], list[int]] = {}
        self._prefix_pages: dict[RequestKey, list[int]] = {}
        self._prefix_slots: dict[RequestKey, int] = {}
        self._latent_pages: dict[RequestKey, list[int]] = {}
        self._buffers: dict[BufferId, BufferAllocation] = {}
        # Free extents of the synthetic persistent buffer pool, as
        # (offset, bytes).
        self._free_buffer_ranges: list[tuple[int, int]] = [
            (0, int(self.worker.info.buffer_pool_bytes))
        ]
        self._batch_id = 0

    def drop_request(self, request_id: int) -> None:
        """Release warmup state for one identifier.

        Covers request, runtime, cache, latent, and product state.
        """
        request = self.worker.requests.peek(int(request_id))
        if request is None:
            return

        self._execute_controls((Finish(request.request_key),))
        # Serving keeps a terminal row until its slot is reassigned. Synthetic
        # requests have no further scheduler messages and can leave the table.
        self.worker.requests.drop(request_id)

        for group_id in range(
            0
            if self.worker.kv_cache is None
            else self.worker.kv_cache.group_count
        ):
            self._kv_pages.pop((request.request_key, group_id), None)
        self._prefix_pages.pop(request.request_key, None)
        self._prefix_slots.pop(request.request_key, None)
        self._latent_pages.pop(request.request_key, None)

        released = tuple(
            buffer
            for buffer, allocation in self._buffers.items()
            if int(allocation.buffer.owner.request_id) == int(request_id)
        )
        self._release_buffer_allocations(released)

    def free_products(self, buffers: tuple[BufferId, ...]) -> None:
        """Release warmup products.

        Recycles their synthetic persistent-buffer allocations.
        """
        if not buffers:
            return
        self._execute_controls(tuple(Free(buffer) for buffer in buffers))
        self._release_buffer_allocations(buffers)

    def _execute_controls(self, commands: tuple[BatchCommand, ...]) -> None:
        """Wait for the physical retirement acknowledgement.

        The same acknowledgement serving uses.
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
        """Allocate warmup persistent storage for a product.

        The slice is deterministic and aligned.
        """
        existing = self._buffers.get(product.buffer_id)
        if existing is not None:
            return existing

        # First-fit over the free extents, honoring the pool's 256-byte
        # alignment.
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
    new_cache_pages: dict[
        tuple[RequestKey, CallId], tuple[CachePageAllocation, ...]
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
    flattens them into the Batch's parallel row arrays.
    """
    return Batch(
        batch_id=batch_id,
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
        new_cache_pages=tuple(
            allocation
            for call in calls
            for allocation in new_cache_pages.get(
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
    """Declare an encoded int64 token relay for warmup sampling."""
    from ..protocol.tensor import DType, ShapeBound

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
    """Execute a runtime scenario.

    Optionally retains the scenario's published outputs.
    """
    worker = requests.worker
    state = worker.submit(batch, propagate_errors=True)

    # A batch is one call on one component, so it returns one result.
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
    """Derive allocations for a warmup submission.

    Covers cache, latent, buffer, and row allocations.
    """
    requests._batch_id += 1
    admissions_by_key = {
        admission.request_key: admission for admission in admissions
    }
    occupied_blocks = {
        page for pages in requests._kv_pages.values() for page in pages
    }
    request_pool_indices: dict[RequestKey, int] = {}
    block_tables: dict[tuple[RequestKey, CallId], tuple[BlockTable, ...]] = {}
    new_cache_pages: dict[
        tuple[RequestKey, CallId], tuple[CachePageAllocation, ...]
    ] = {}
    forward_inputs: dict[
        tuple[RequestKey, CallId],
        tuple[
            tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[bool, ...]
        ],
    ] = {}
    latent_params: dict[tuple[RequestKey, CallId], LatentParams] = {}
    buffer_allocations: dict[BufferId, BufferAllocation] = {}
    # Persistent products reserve stable buffer allocations before lane
    # construction.
    for call in calls:
        for product in (
            *call.buffer_inputs(),
            *call.buffer_outputs(),
        ):
            allocation = requests.buffer_allocation(product)
            buffer_allocations[allocation.buffer] = allocation
    # Bind request slots and grow reusable KV leases to each call's
    # maximum shape.
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

        tables: list[BlockTable] = []
        allocations: list[CachePageAllocation] = []
        if requests.worker.kv_cache is None:
            raise invalid_descriptor("warmup KV call requires cache storage")

        for group_id in range(requests.worker.kv_cache.group_count):
            lease_key = (call.request_key, group_id)
            block_table = requests._kv_pages.setdefault(lease_key, [])
            target_pages = ceil_div(
                visible + input_length,
                int(requests.worker.kv_cache.info.block_size),
            )

            # Leases only grow: earlier captured graphs still reference
            # these pages.
            missing = target_pages - len(block_table)
            if missing < 0:
                raise invalid_descriptor(
                    "warmup call regresses its KV capacity"
                )

            allocated = tuple(
                candidate
                for candidate in requests.worker.kv_cache.page_ids(group_id)
                if candidate not in occupied_blocks
            )[:missing]
            if len(allocated) != missing:
                raise invalid_descriptor(
                    "warmup KV allocation exceeds resident capacity: "
                    f"request={call.request_key.request_id}, "
                    f"group={group_id}, "
                    f"required_pages={missing}, "
                    f"available_pages={len(allocated)}, "
                    "resident_pages="
                    f"{len(requests.worker.kv_cache.page_ids(group_id))}, "
                    f"leased_pages={len(occupied_blocks)}"
                )

            block_table.extend(allocated)
            occupied_blocks.update(allocated)
            tables.append(
                BlockTable(
                    request_pool_idx=request_pool_indices[call.request_key],
                    group_id=group_id,
                    page_ids=tuple(block_table),
                    allocated_tokens=len(block_table)
                    * requests.worker.kv_cache.info.block_size,
                )
            )
            if allocated:
                allocations.append(
                    CachePageAllocation(
                        request_pool_idx=request_pool_indices[call.request_key],
                        group_id=group_id,
                        page_ids=allocated,
                    )
                )
        identity = (call.request_key, call.call_id)
        block_tables[identity] = tuple(tables)
        new_cache_pages[identity] = tuple(allocations)
        if input_length > 0:
            forward_inputs[identity] = (
                (request_pool_indices[call.request_key],),
                (visible + input_length,),
                (input_length,),
                (True,),
            )
    height, width = image_size or _warmup_image_size(requests)
    # Latents occupy a grid of (height/downsample) x (width/downsample) units.
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
            new_cache_pages[identity] = (
                *new_cache_pages.get(identity, ()),
                *extra_allocations,
            )
            forward_inputs[identity] = flow_rows

    return _warmup_batch(
        batch_id=requests._batch_id,
        admissions=admissions,
        calls=calls,
        block_tables=block_tables,
        new_cache_pages=new_cache_pages,
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
    tuple[CachePageAllocation, ...],
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[bool, ...]],
]:
    """Build KV tables and forward rows for all active CFG branches.

    The tables cover alternative prefixes.
    """
    request = requests.worker.requests.get(call.request_key.request_id)
    image = request.image
    generation = requests.worker.runner.image_builder
    if image is None or generation is None:
        raise invalid_descriptor(
            "generation warmup has no admitted image runtime"
        )
    trajectory = image_state(generation, ImageConfig(height, width), image)
    branches = trajectory.guidance.branches(
        trajectory.schedule, request.accepted_progress.flow_step
    )

    runtime = request.accepted_progress
    query = generation.sequence_length(ImageConfig(height, width))
    image_prompt = image.image_prompts[0] if image.image_prompts else ""
    # Branches either reuse the conditioned request slot or share one
    # alternative prefix.
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

    if requests.worker.kv_cache is None:
        raise invalid_descriptor("warmup flow requires KV cache storage")

    required = ceil_div(
        len(alternative), requests.worker.kv_cache.info.block_size
    )
    lease = requests._prefix_pages.setdefault(call.request_key, [])
    missing = required - len(lease)
    occupied = {
        page
        for request_key, pages in requests._prefix_pages.items()
        if request_key != call.request_key
        for page in pages
    }
    occupied.update(
        page for pages in requests._kv_pages.values() for page in pages
    )
    # Prefix pages persist across warmup shapes so graph capture observes
    # stable tables.
    allocated = tuple(
        page
        for page in requests.worker.kv_cache.page_ids(0)
        if page not in occupied
    )[:missing]
    if len(allocated) != missing:
        raise invalid_descriptor(
            "warmup alternative prefix exceeds KV capacity"
        )
    lease.extend(allocated)

    tables: tuple[BlockTable, ...] = ()
    allocations: tuple[CachePageAllocation, ...] = ()
    alternative_slot = main_slot
    request_pool_indices: list[int] = []
    seq_lens: list[int] = []
    query_lens: list[int] = []
    write_kv: list[bool] = []

    if alternative:
        alternative_slot = requests._prefix_slots.setdefault(
            call.request_key,
            int(requests.worker.info.request_slots)
            - len(requests._prefix_slots),
        )
        if alternative_slot == main_slot or alternative_slot < 1:
            raise invalid_descriptor(
                "warmup has no request slot for an alternative prefix"
            )
        tables = (
            BlockTable(
                request_pool_idx=alternative_slot,
                group_id=0,
                page_ids=tuple(lease),
                allocated_tokens=len(lease)
                * requests.worker.kv_cache.info.block_size,
            ),
        )
        if allocated:
            allocations = (
                CachePageAllocation(
                    request_pool_idx=alternative_slot,
                    group_id=0,
                    page_ids=allocated,
                ),
            )
        request_pool_indices.append(alternative_slot)
        seq_lens.append(len(alternative))
        query_lens.append(len(alternative))
        write_kv.append(True)

    # Append every column in the exact guidance-branch evaluation order.
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

    Successful scenarios retire their requests before returning. If execution
    fails, leave resource release to the enclosing Worker scope instead of
    issuing more execution commands that could replace the startup error.
    """
    requests = _WarmupRequests(worker)

    if torch.device(worker.worker_config.device).type == "cuda":
        if ForwardMode.PREFILL in worker.info.supported_calls:
            _warmup_tokens(requests)
            logger.info("completed token runtime warmup")

        if worker.runner.image_builder is not None:
            _warmup_flow(requests)
            logger.info("completed flow runtime warmup")


def _warmup_image_size(requests: _WarmupRequests) -> tuple[int, int]:
    """Derive the largest square image for warmup.

    The image's latent grid must fit the declared capacity.
    """
    builder = requests.worker.runner.image_builder
    downsample = 1 if builder is None else builder.denoiser.downsample
    capacity = int(requests.worker.info.latent_capacity_units)
    if builder is not None:
        capacity = min(capacity, builder.max_tokens + builder.framing)

    side = max(1, math.isqrt(max(1, capacity)))
    return side * downsample, side * downsample


def _warmup_tokens(requests: _WarmupRequests) -> None:
    """Exercise extend-to-decode token handoff.

    Releases its synthetic request afterwards.
    """
    from uniserve.sampling import SamplingParams

    from ..protocol.batch import GenerationParams, NewRequest
    from ..protocol.call import (
        Bounds,
        Call,
        CallCoordinates,
    )

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

    predecessors: dict[int, Call] = {}
    calls: list[Call] = []
    for sid in request_ids:
        call_id = CallId(requests._batch_id + 1, len(calls))
        call = prompt_op(sid, call_id, (0,))
        calls.append(call)

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


def _warmup_flow(requests: _WarmupRequests) -> None:
    """Drive chained denoise quanta through the real flow forward path."""
    from ..protocol.batch import NewRequest
    from ..protocol.call import (
        Bounds,
        Call,
        CallCoordinates,
        DrawLayout,
        Rng,
    )
    from ..protocol.tensor import (
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

    # One scenario per CFG execution branch; numerical shape catalogs belong
    # to the execution owners and are already resident before these requests.
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
    for bucket in configured:
        batch_size = bucket.rows
        height = bucket.height
        width = bucket.width
        cfg_branches = bucket.cfg_branches
        if batch_size > int(requests.worker.info.request_slots):
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

        # Publish one conditioning KV product per request.
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

        max_latent_elements = max(
            1,
            math.prod(
                generation.denoiser.latent_shape(
                    "image", ImageConfig(height, width)
                )
            ),
        )
        # Byte budgets below assume BF16 latents: 2 bytes per element.

        # Prepare each request's initial latent and its readiness product.
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

        # Chain two denoise quanta so back-to-back execution shapes batch. Each
        # quantum covers one step, so it enters at the step its index names.
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
