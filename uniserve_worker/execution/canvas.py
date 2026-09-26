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
``CanvasStepRow``, and ``publish_step`` captures the row's stop flag and
tokens, which the completion reports as its committed tokens once the step
stops the canvas. A canvas pass writes no KV, so the request's coordinates
stay where the call found them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution import calls
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.model_executor.input_batch import CanvasRow, CanvasStepRow
from uniserve_worker.protocol.call import Call, CallStatus, ForwardMode
from uniserve_worker.protocol.output import FinishFlags

if TYPE_CHECKING:
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.canvas_slots import CanvasSlots


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

    start = int(calls.require_progress(request).logical_position)
    tokens = torch.tensor(call.input_token_ids, dtype=torch.int64)
    offsets = readout.candidate_offsets
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
                positions=torch.arange(start, start + length),
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
    request.token.candidate_range = state.output_buffer.capture(
        logprobs.contiguous().view(torch.int32)
    )

    cache = calls.cache_coordinates(request, tables=request_tables)
    request.status = CallStatus.OK
    request.progress = calls.execution_runtime(request, cache)
    request.finish_flags = FinishFlags()
    request.product_generations = calls.output_generations(call)
    request.token.committed_tokens = ()
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

    start = int(calls.require_progress(request).logical_position)
    return CanvasStepRow(
        forward_mode=ForwardMode.TOKEN_DENOISING,
        request_pool_idx=slot,
        positions=torch.arange(start, start + length),
        seq_len=visible,
        write_kv=False,
        causal=False,
        canvas_length=length,
        seed=int(seed),
        block=step.block,
        step=step.step,
        sampling=constants,
    )


def publish_step(
    call: Call,
    value: torch.Tensor,
    *,
    state: BatchState,
    request_tables: BlockTables | None,
) -> PendingOutput:
    """Capture a canvas step's stop flag and tokens and stage its outcome.

    ``value`` is the row's int64 ``[1 + canvas]`` vector (``CanvasRunner``):
    its stop flag and its truncated argmax canvas. Both are captured into
    the batch's output buffer, which reaches the host with the batch's one
    completion copy; ``PendingOutput.materialize`` reports the tokens as the
    call's committed tokens when the flag is set. The outcome reports the
    request's coordinates unchanged, since the step wrote no KV.

    Raises:
        WorkerError: ``invalid_descriptor`` when the value does not hold the
            call's canvas.
    """
    request = state.pending_output(call.request_key.request_id)
    if value.dtype != torch.int64 or value.shape != (
        1 + call.bounds.max_tokens,
    ):
        raise invalid_descriptor("a canvas step does not cover its canvas")
    request.token.canvas_range = state.output_buffer.capture(value)

    cache = calls.cache_coordinates(request, tables=request_tables)
    request.status = CallStatus.OK
    request.progress = calls.execution_runtime(request, cache)
    request.finish_flags = FinishFlags()
    request.product_generations = calls.output_generations(call)
    request.token.committed_tokens = ()
    return request
