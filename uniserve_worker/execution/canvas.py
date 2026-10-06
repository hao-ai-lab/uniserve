"""Numerical canvas inputs and completion columns.

Readout calls borrow a cached prompt and evaluate read-only canvas rows.
Generating calls advance their resident canvas through a numerical step.
The native executor captures candidate scores and canvas outcomes, binds
continuation writes, and retains the request's unchanged KV coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from uniserve.tensors import adjacent_view
from uniserve_worker._uniserve_ipc import BatchState
from uniserve_worker.errors import invalid_descriptor
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
    slot, visible, _capacity = request.cache_coordinates(request_tables)

    # Each forward row of the call is one canvas over the visible prefix.
    inputs = state.batch
    descriptors = state.forward_rows(call.request_key.request_id)
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


def readout_values(values: Sequence[torch.Tensor], count: int) -> torch.Tensor:
    """Pack FP32 candidate scores as integer words for completion capture."""
    logprobs = torch.cat(tuple(values)) if len(values) > 1 else values[0]
    if logprobs.dtype != torch.float32 or logprobs.numel() != count:
        raise invalid_descriptor(
            "canvas readout does not cover the call's candidates"
        )
    return logprobs.contiguous().view(torch.int32)


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
    slot, visible, _capacity = request.cache_coordinates(request_tables)

    inputs = state.batch
    descriptors = state.forward_rows(call.request_key.request_id)
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


def step_values(
    values: Sequence[torch.Tensor], widths: Sequence[int]
) -> torch.Tensor:
    """Borrow adjacent result rows for one completion copy.

    Each int64 row contains the stop outcome followed by its canvas tokens.
    Graph outputs preserve these rows together when copying reusable storage.
    """
    width = widths[0]
    if any(
        value.dtype != torch.int64
        or value.shape != (expected,)
        or expected != width
        for value, expected in zip(values, widths, strict=True)
    ):
        raise invalid_descriptor("a canvas step does not cover its canvas")
    block = adjacent_view(tuple(values))
    if block is None:
        raise invalid_descriptor(
            "the canvas steps of a batch are not adjacent rows of one result"
        )
    return block


def step_continuations(
    block: torch.Tensor, width: int, rows: Sequence[int], dtype: torch.dtype
) -> torch.Tensor:
    """Select device continuation flags for the reserved output rows."""
    outcomes = block.view(-1, width)[:, 0]
    if len(rows) != outcomes.numel():
        outcomes = outcomes[list(rows)]
    return (outcomes == STEP_CONTINUED).to(dtype)
