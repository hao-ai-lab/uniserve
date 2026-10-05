"""Token-denoising calls: canvases over a cached prompt, read or stepped.

A readout ``TOKEN_DENOISING`` call carries one or more token canvases back
to back in ``input_token_ids`` and the canvas tokens whose logits it reads
(``Call.readout``). The engine describes each canvas as one read-only
forward row of the call: its prefix is the request's visible KV and its
query is the canvas. ``prepare_rows`` turns the call into one ``CanvasRow``
per canvas, and ``publish`` captures the rows' candidate log-probabilities
into the batch's output buffer and stages the call's outcome.

A generating call (``Call.canvas``) runs one denoising step of the canvas
its request keeps in its slot; ``prepare_step`` turns it into one
``CanvasStepRow``, and ``publish_steps`` captures every stepped row's stop
flag and tokens of a batch together, which each completion reports as its
committed tokens once its step stops the canvas. A canvas pass writes no
KV, so the request's coordinates stay where the call found them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from uniserve.tensors import adjacent_view
from uniserve_worker._uniserve_ipc import BatchState
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution import calls
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.model_executor.input_batch import (
    CanvasRow,
    CanvasStepRow,
    int64_bits,
)
from uniserve_worker.protocol.call import Call, ForwardMode
from uniserve_worker.storage.canvas_slots import STEP_CONTINUED

if TYPE_CHECKING:
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.canvas_slots import CanvasSlots
    from uniserve_worker.storage.tensor_store import TensorStore


def prepare_rows(
    call: Call,
    *,
    state: BatchState,
    request_tables: BlockTables | None,
) -> tuple[CanvasRow, ...]:
    """Split a token-denoising call into its canvas rows.

    Every canvas starts at the request's logical position, where the next
    prompt token would stand, and reads the request's visible KV. Slots and
    candidates go to the canvas that holds them.

    Raises:
        WorkerError: ``invalid_descriptor`` when the call carries no readout,
            its forward rows do not partition its canvas tokens into
            read-only rows over the request's visible KV, or the call's
            cache coordinates are rejected.
    """
    readout = call.readout
    if call.kind is not ForwardMode.TOKEN_DENOISING or readout is None:
        raise invalid_descriptor("a canvas pass requires a readout")

    request = state.pending_output(call.request_key.request_id)
    slot, visible, _capacity = calls.cache_coordinates(
        request, tables=request_tables
    )

    # Each forward row of the call is one canvas over the visible prefix.
    inputs = state.batch
    descriptors = state.forward_indices.get(calls.call_identity(call), ())
    lengths = tuple(inputs.query_lens[index] for index in descriptors)
    if (
        not descriptors
        or sum(lengths) != len(call.input_token_ids)
        or any(
            inputs.write_kv[index]
            or inputs.request_pool_indices[index] != slot
            or inputs.seq_lens[index] - inputs.query_lens[index] != visible
            for index in descriptors
        )
    ):
        raise invalid_descriptor(
            "canvas rows must partition the call's tokens over its prefix"
        )

    start = int(request.progress.logical_position)
    # The token tuple converts through NumPy, which reads Python ints far
    # faster than a tensor construction does.
    tokens = torch.from_numpy(np.asarray(call.input_token_ids, dtype=np.int64))
    offsets = readout.candidate_offsets
    # Every canvas of the call starts at the same position, so canvases of
    # one length share one read-only position vector.
    positions: dict[int, torch.Tensor] = {}
    rows, first, slot_index = [], 0, 0
    for length in lengths:
        # Slots are sorted by token index, so each canvas takes the run of
        # slots inside its span.
        end = first + length
        last = slot_index
        while (
            last < len(readout.slot_tokens) and readout.slot_tokens[last] < end
        ):
            last += 1
        if last == slot_index:
            raise invalid_descriptor("every canvas row must read a slot")

        rows.append(
            CanvasRow(
                forward_mode=ForwardMode.TOKEN_DENOISING,
                request_pool_idx=slot,
                positions=positions.setdefault(
                    length, torch.arange(start, start + length)
                ),
                seq_len=visible,
                write_kv=False,
                causal=False,
                token_ids=tokens[first:end],
                slot_tokens=tuple(
                    token - first
                    for token in readout.slot_tokens[slot_index:last]
                ),
                candidate_offsets=tuple(
                    offset - offsets[slot_index]
                    for offset in offsets[slot_index : last + 1]
                ),
                candidate_ids=readout.candidate_ids[
                    offsets[slot_index] : offsets[last]
                ],
            )
        )
        first, slot_index = end, last
    return tuple(rows)


def publish(
    call: Call,
    values: list[torch.Tensor],
    *,
    state: BatchState,
    request_tables: BlockTables | None,
) -> PendingOutput:
    """Capture a call's candidate log-probabilities and stage its outcome.

    ``values`` holds each canvas row's FP32 log-probabilities in row order.
    Their FP32 bit patterns are captured as sign-extended int64 words, which
    ``PendingOutput.materialize`` decodes. The outcome reports the request's
    coordinates unchanged, since the pass wrote no KV.

    Raises:
        WorkerError: ``invalid_descriptor`` when the rows' values do not
            cover the call's candidates.
    """
    readout = call.readout
    assert readout is not None
    request = state.pending_output(call.request_key.request_id)
    logprobs = torch.cat(values) if len(values) > 1 else values[0]
    if logprobs.dtype != torch.float32 or logprobs.numel() != len(
        readout.candidate_ids
    ):
        raise invalid_descriptor(
            "canvas readout does not cover the call's candidates"
        )
    request.set_candidates(
        state.output_buffer.capture(logprobs.contiguous().view(torch.int32))
    )

    cache = calls.cache_coordinates(request, tables=request_tables)
    request.progress = calls.execution_runtime(request, cache)
    return request


def prepare_step(
    call: Call,
    *,
    state: BatchState,
    request_tables: BlockTables | None,
    canvas_slots: CanvasSlots | None,
) -> CanvasStepRow:
    """Build the row of one step of a request's generating canvas.

    The canvas starts at the request's logical position and reads the
    request's visible KV, like a readout canvas; its length and sampler
    constants follow the request's admitted block-diffusion sampling
    (``CanvasSlots.sampling``), and the step continues the canvas its slot
    ran last (``CanvasSlots.advance``).

    Raises:
        WorkerError: ``invalid_descriptor`` when the worker keeps no
            generating canvases, the request was admitted without canvas
            sampling or a seed, the sampling does not fit the model's
            canvases, the call's forward row is not one read-only canvas of
            that length over the request's visible KV, or the step does not
            continue the slot's canvas.
    """
    step = call.canvas
    if call.kind is not ForwardMode.TOKEN_DENOISING or step is None:
        raise invalid_descriptor("a canvas step requires its step")
    if canvas_slots is None:
        raise invalid_descriptor("this worker keeps no generating canvases")

    request = state.pending_output(call.request_key.request_id)
    admitted = request.request.admission.generation
    sampling = None if admitted is None else admitted.canvas
    seed = None if admitted is None else admitted.sampling.seed
    if sampling is None or seed is None:
        raise invalid_descriptor(
            "a canvas step requires a request admitted with seeded canvas "
            "sampling"
        )
    constants = canvas_slots.sampling(sampling)
    slot, visible, _capacity = calls.cache_coordinates(
        request, tables=request_tables
    )

    inputs = state.batch
    descriptors = state.forward_indices.get(calls.call_identity(call), ())
    length = sampling.canvas_length
    if (
        len(descriptors) != 1
        or call.bounds.max_tokens != length
        or any(
            inputs.write_kv[index]
            or inputs.request_pool_indices[index] != slot
            or inputs.query_lens[index] != length
            or inputs.seq_lens[index] - length != visible
            for index in descriptors
        )
    ):
        raise invalid_descriptor(
            "a canvas step is one read-only canvas row over its prefix"
        )
    if step.step >= constants.steps:
        raise invalid_descriptor("the canvas step exceeds its step limit")
    canvas_slots.advance(slot, step.block, step.step)

    start = int(request.progress.logical_position)
    return CanvasStepRow(
        forward_mode=ForwardMode.TOKEN_DENOISING,
        request_pool_idx=slot,
        positions=torch.arange(start, start + length),
        seq_len=visible,
        write_kv=False,
        causal=False,
        canvas_length=length,
        seed=int64_bits(int(seed)),
        block=step.block,
        step=step.step,
        sampling=constants,
    )


def publish_steps(
    steps: Sequence[tuple[Call, torch.Tensor]],
    *,
    state: BatchState,
    request_tables: BlockTables | None,
    tensor_store: TensorStore,
) -> None:
    """Capture a batch's canvas step outcomes and tokens and stage them.

    ``steps`` pairs each canvas step call of the batch with its row's int64
    ``[1 + canvas]`` vector (``CanvasRunner.step``): its outcome and its
    truncated argmax canvas, in the forward's row order. The rows are
    adjacent views of one ``[rows, 1 + canvas]`` block. ``CanvasRunner.step``
    unbinds one result tensor, and a replayed graph's output clone keeps
    same-dtype rows in one allocation. So the batch captures every row with
    one copy into its output buffer, which reaches the host with the batch's
    one completion copy, and each row's span follows the last.
    ``PendingOutput.materialize`` reports a row's tokens as its call's
    committed tokens when the step stopped the block, and a skipped step as
    predicated.

    Each call's completion output receives, on the device, whether its block
    continues after the step: the predicate of a step queued behind it. One
    publication covers every row, so each row's completion becomes visible
    together with its values, after the same producer point on the stream.
    The outcomes report the requests' coordinates unchanged, since a step
    writes no KV.

    Raises:
        WorkerError: ``invalid_descriptor`` when a value does not hold its
            call's canvas, or the rows are not adjacent rows of one block.
    """
    if not steps:
        return
    width = 1 + steps[0][0].bounds.max_tokens
    if any(
        value.dtype != torch.int64
        or value.shape != (1 + call.bounds.max_tokens,)
        or value.shape != (width,)
        for call, value in steps
    ):
        raise invalid_descriptor("a canvas step does not cover its canvas")
    block = adjacent_view(tuple(value for _call, value in steps))
    if block is None:
        raise invalid_descriptor(
            "the canvas steps of a batch are not adjacent rows of one result"
        )

    offset, _count = state.output_buffer.capture(block)
    writes = []
    written_rows = []
    for row, (call, _value) in enumerate(steps):
        request = state.pending_output(call.request_key.request_id)
        request.set_canvas((offset + row * width, width))
        if request.completion_write is not None:
            writes.append(request.completion_write)
            written_rows.append(row)

        cache = calls.cache_coordinates(request, tables=request_tables)
        request.progress = calls.execution_runtime(request, cache)

    if writes:
        outcome_column = block.view(len(steps), width)[:, 0]
        if len(written_rows) != len(steps):
            outcome_column = outcome_column[written_rows]
        (view, *_views) = tensor_store.producer_write_views(tuple(writes))
        tensor_store.publish_writes(
            tuple(writes),
            (outcome_column == STEP_CONTINUED).to(view.dtype),
        )
