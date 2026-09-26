"""Token-denoising calls: canvas rows over a cached prompt and their readout.

A ``TOKEN_DENOISING`` call carries one or more token canvases back to back
in ``input_token_ids`` and, as a readout, the canvas tokens whose logits it
reads (``Call.readout``). The engine describes each canvas as one read-only
forward row of the call: its prefix is the request's visible KV and its
query is the canvas. ``prepare_rows`` turns the call into one ``CanvasRow``
per canvas, and ``publish`` captures the rows' candidate log-probabilities
into the batch's output buffer and stages the call's outcome. A canvas pass
writes no KV, so the request's coordinates stay where the call found them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution import calls
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.model_executor.input_batch import CanvasRow
from uniserve_worker.protocol.call import Call, CallStatus, ForwardMode
from uniserve_worker.protocol.output import FinishFlags

if TYPE_CHECKING:
    from uniserve_worker.storage.block_tables import BlockTables


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
