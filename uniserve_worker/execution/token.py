"""Numerical token and visual inputs, prompt scores and device coordinates.

The native executor owns sampling dispatch, accepted progress and output
capture. This module builds borrowed model rows and computes prompt scores
and speculative device coordinates with ordinary tensor operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from uniserve.sampling import SamplingParams
from uniserve_worker._uniserve_ipc import BatchState
from uniserve_worker.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.execution import image
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.protocol.call import (
    Call,
    ForwardMode,
    SamplingState,
    VisionInput,
)
from uniserve_worker.sampling import sampler as sampling
from uniserve_worker.sampling.metadata import TokenSelection
from uniserve_worker.sampling.result import LogprobValues, SamplerRow
from uniserve_worker.storage.tensor_store import FeatureMetadata

if TYPE_CHECKING:
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.model_executor.diffusion_inputs import (
        DiffusionRow,
    )
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.decode_state import DecodeState
    from uniserve_worker.storage.tensor_store import TensorStore


def prepare_forward(
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
    decode_state: DecodeState | None,
) -> TokenRow | DiffusionRow:
    """Pack one prefill, decode or verify call into a model-forward row.

    A prefill of generated-image feedback or of an input image's latent
    becomes a visual-state row; a context prefill with vision blocks is
    packed by ``prepare_context``. Other rows start at the request's current
    logical position: a
    prefill selects all logits when prompt log probabilities are requested
    and the last logits otherwise, a decode feeds one token, and a verify
    feeds the current token followed by the draft tokens and selects all
    logits.

    Raises:
        WorkerError: When, for example, the request has no admitted sampling
            state, or the call's input tokens or device token continuation
            are missing or malformed.
    """
    request = state.pending_output(call.request_key.request_id)
    if request.request.sampling is None:
        raise invalid_descriptor("sequence call has no admitted sampling state")

    mode = call.kind if isinstance(call.kind, ForwardMode) else None
    if call.writes_context():
        raise invalid_descriptor("a context prefill packs one row per segment")
    if mode is ForwardMode.PREFILL and call.writes_visual_state():
        return _prepare_visual(
            call,
            request,
            tensor_store=tensor_store,
            request_tables=request_tables,
            model_runner=model_runner,
            state=state,
        )

    start = int(request.progress.logical_position)
    tokens: tuple[int | torch.Tensor, ...]
    current: int | torch.Tensor
    if mode is ForwardMode.PREFILL:
        if call.predicate is not None and not call.input_token_ids:
            tokens = (
                resolve_decode_token(call, request, decode_state=decode_state),
            )
        else:
            tokens = call.input_token_ids
        if not tokens:
            raise invalid_descriptor("text extension requires input tokens")

        sampling = require_sampling(request)
        scores_prompt = bool(
            sampling.return_prompt_logprobs
            or int(sampling.n_prompt_logprobs) > 0
        )
        # A chunk that neither samples its next token nor scores its prompt
        # only writes the K/V cache (no result logits are needed).
        task = token_task(
            call,
            request,
            tokens,
            tuple(range(start, start + len(tokens))),
            TokenSelection.ALL_LOGITS
            if scores_prompt
            else TokenSelection.LAST_LOGITS
            if call.token_output is not None
            else TokenSelection.CACHE,
            request_tables=request_tables,
        )
    elif mode is ForwardMode.DECODE:
        # Indexed decode reads its token and position directly from
        # request-indexed device state instead of host-supplied values. It
        # requires CUDA decode state beside the unit tables and a tagged (I64
        # relay) predicate carrying the predecessor's device decision.
        indexed = (
            decode_state is not None
            and request_tables is not None
            and decode_state.device.type == "cuda"
            and request_tables.unit_tables.device == decode_state.device
            and request.predicate is not None
            and request.predicate[1]
        )
        task = token_task(
            call,
            request,
            None
            if indexed
            else (
                resolve_decode_token(call, request, decode_state=decode_state),
            ),
            None if indexed else (start,),
            TokenSelection.LAST_LOGITS,
            request_tables=request_tables,
            request_indexed_decode=indexed,
        )
    else:
        # A predicated verify takes its current token from the device relay
        # and `input_token_ids` holds only the draft; otherwise the ids are
        # the current token followed by the draft.
        if call.predicate is not None:
            current = resolve_decode_token(
                call, request, decode_state=decode_state
            )
            draft = call.input_token_ids
        else:
            input_tokens = call.input_token_ids
            if len(input_tokens) < 2:
                raise invalid_descriptor(
                    "fixed-parent verification requires current and draft "
                    "tokens"
                )
            current = int(input_tokens[0])
            draft = input_tokens[1:]

        tokens = (current, *draft)
        task = token_task(
            call,
            request,
            tokens,
            tuple(range(start, start + len(tokens))),
            TokenSelection.ALL_LOGITS,
            request_tables=request_tables,
        )
    return task


def _prepare_visual(
    call: Call,
    request: PendingOutput,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
) -> TokenRow | DiffusionRow:
    """Build a visual-state row from a prefill's image features.

    Consumes the call's single vision or latent feature tensor onto its first
    device and builds the row at the request's logical position. A vision
    feature is generated-image feedback, whose row closes the image.

    Raises:
        WorkerError: When, for example, the call carries other than exactly
            one feature input, the tensor lacks ``FeatureMetadata``, or the
            row exceeds the call's token bound.
    """
    features = tuple(block.feature for block in call.vision_inputs)
    if call.latent_feature_input is not None:
        features += (call.latent_feature_input,)
    if len(features) != 1:
        raise invalid_descriptor(
            "visual extend requires exactly one feature tensor"
        )
    (reference,) = features
    read = tensor_store.consume(
        reference,
        consumer_call_id=call.call_id,
        device=model_runner.call_devices(call)[0],
    )
    request.feature_reads.append(read)
    metadata = read.metadata
    if not isinstance(metadata, FeatureMetadata):
        raise invalid_descriptor(
            "visual input requires encoder feature metadata"
        )

    position = int(request.progress.logical_position)
    sample_token = call.token_output is not None
    task: TokenRow | DiffusionRow
    if call.vision_inputs:
        task = image.vision_state_row(
            call,
            read.tensor,
            metadata.height,
            metadata.width,
            position,
            seq_len=request.cache_coordinates(request_tables)[1],
            close_image=True,
            logits=sample_token,
            request_tables=request_tables,
            model_runner=model_runner,
            state=state,
        )
    else:
        task = image.latent_state_row(
            call,
            read.tensor,
            metadata.height,
            metadata.width,
            position,
            request_tables=request_tables,
            model_runner=model_runner,
            state=state,
        )

    if task.query_tokens > int(call.bounds.max_tokens):
        raise invalid_descriptor(
            "image state query span exceeds the call token bound"
        )
    return task


def next_position(row: TokenRow) -> int:
    """Return the logical position after a context row.

    A row occupies the positions of its temporal axis: a prompt run one per
    token, and a vision block one per feature at consecutive positions, or
    one when its features share a temporal position.
    """
    positions = row.positions
    if positions is None:
        raise RuntimeError("a context row has no positions")
    temporal = positions if positions.ndim == 1 else positions[0]
    return int(temporal.max()) + 1


def prepare_context(
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
) -> tuple[TokenRow, ...]:
    """Pack a context prefill into one row per segment, in context order.

    The call's prompt tokens run between its vision blocks: each block enters
    at its ``VisionInput.offset`` into the call's tokens as one non-causal
    row over its encoder features (``image.vision_state_row``), and each run
    of prompt tokens before, between or after the blocks as one causal row.
    Each row continues where the previous one ends, in KV and in logical
    position (``next_position``). The rows form one numerical call whose
    attention writes every row's KV before any row reads it, so each row
    attends to the request's prefix and the rows before it, and a block's
    features attend to each other in both directions.

    Prompt runs select all logits when prompt log probabilities are
    requested, and vision blocks select their final logits to score the
    following text token. Otherwise the last row selects its last logits
    when the call samples a token, and every other row only writes KV. The
    block features are consumed onto the call's first device.

    Raises:
        WorkerError: ``invalid_descriptor`` when the request has no admitted
            sampling, the blocks are out of order or outside the call's
            tokens, a feature lacks ``FeatureMetadata``, or the rows
            disagree with the call's token
            bound or its forward rows.
    """
    request = state.pending_output(call.request_key.request_id)
    parameters = require_sampling(request)
    scores_prompt = bool(
        parameters.return_prompt_logprobs
        or int(parameters.n_prompt_logprobs) > 0
    )
    tokens = call.input_token_ids

    # Segments in context order: (start, stop) spans of the call's tokens
    # and the blocks at their offsets.
    segments: list[tuple[int, int] | VisionInput] = []
    start = 0
    for block in call.vision_inputs:
        offset = int(block.offset)
        if offset < start or offset > len(tokens):
            raise invalid_descriptor(
                "context blocks must lie in order within the call's tokens"
            )
        if offset > start:
            segments.append((start, offset))
        segments.append(block)
        start = offset
    if len(tokens) > start:
        segments.append((start, len(tokens)))
    slot, visible, _capacity = request.cache_coordinates(request_tables)
    position = int(request.progress.logical_position)
    device = model_runner.call_devices(call)[0]
    rows: list[TokenRow] = []
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if isinstance(segment, VisionInput):
            read = tensor_store.consume(
                segment.feature, consumer_call_id=call.call_id, device=device
            )
            request.feature_reads.append(read)
            metadata = read.metadata
            if not isinstance(metadata, FeatureMetadata):
                raise invalid_descriptor(
                    "a vision block requires encoder feature metadata"
                )
            row = image.vision_state_row(
                call,
                read.tensor,
                metadata.height,
                metadata.width,
                position,
                seq_len=visible,
                close_image=False,
                logits=scores_prompt
                or (last and call.token_output is not None),
                request_tables=request_tables,
                model_runner=model_runner,
                state=state,
            )
        else:
            begin, stop = segment
            row = TokenRow(
                forward_mode=ForwardMode.PREFILL,
                token_ids=torch.tensor(tokens[begin:stop], dtype=torch.long),
                positions=torch.arange(
                    position, position + stop - begin, dtype=torch.long
                ),
                selection=TokenSelection.ALL_LOGITS
                if scores_prompt
                else TokenSelection.LAST_LOGITS
                if last and call.token_output is not None
                else TokenSelection.CACHE,
                request_pool_idx=slot,
                seq_len=visible,
                write_kv=True,
                causal=True,
            )
        rows.append(row)
        position = next_position(row)
        visible += row.query_tokens

    # The engine sizes the call's KV and forward rows from the blocks'
    # declared lengths; each row must be the one it declared.
    inputs = state.batch
    descriptors = state.forward_rows(call.request_key.request_id)
    if (
        sum(row.query_tokens for row in rows) > int(call.bounds.max_tokens)
        or len(descriptors) != len(rows)
        or any(
            inputs.request_pool_indices[descriptor] != slot
            or inputs.query_lens[descriptor] != row.query_tokens
            or inputs.seq_lens[descriptor] != row.seq_len + row.query_tokens
            for descriptor, row in zip(descriptors, rows, strict=True)
        )
    ):
        raise invalid_descriptor(
            "context rows disagree with the call's forward rows"
        )
    return tuple(rows)


def prompt_logprobs(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    previous: torch.Tensor | None,
    parameters: SamplingParams,
) -> tuple[torch.Tensor, LogprobValues | None]:
    """Score a prompt run and return the logits for the next run.

    The executor supplies preceding logits for a continued run. Without
    them, the first token has no prediction and is excluded from scoring.
    All score tensors remain on the logits' device until output capture.
    """
    tokens = tokens.reshape(-1).to(device=logits.device, dtype=torch.long)
    if logits.ndim != 2 or int(logits.shape[0]) != int(tokens.numel()):
        raise invalid_descriptor(
            "prompt scoring logits do not align with input tokens"
        )

    if previous is None:
        score_logits, targets = logits[:-1], tokens[1:]
    else:
        previous = previous.reshape(1, -1).to(
            device=logits.device, dtype=logits.dtype
        )
        score_logits = torch.cat((previous, logits[:-1]), dim=0)
        targets = tokens

    retained = logits[-1].detach()
    if targets.numel() == 0:
        return retained, None

    indexes = torch.arange(
        int(targets.numel()), dtype=torch.long, device=score_logits.device
    )
    details = sampling.logprob_details(
        score_logits.float(),
        indexes,
        targets,
        (parameters,) * int(targets.numel()),
    )
    return retained, details


def speculative_positions(
    sampled: SamplerRow, visible: int, position: int, sampling_position: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Offset device-selected acceptance counts from their host coordinates.

    Accepted drafts include a correction or bonus token unless a draft
    finished the request. The selected extent stays on device for queued
    execution; native result materialization resolves it after completion.
    """
    count = sampled.accepted_token_count
    if count is None:
        accepted = sampled.accepted_draft_count
        if accepted is None:
            raise RuntimeError("speculative sampling lost its selected point")
        count = accepted.to(dtype=torch.int32) + 1
    return count + visible, count + position, count + sampling_position


def token_task(
    call: Call,
    request: PendingOutput,
    token_ids: tuple[int | torch.Tensor, ...] | None,
    positions: tuple[int, ...] | torch.Tensor | None,
    selection: TokenSelection,
    *,
    seq_len: int | None = None,
    request_indexed_decode: bool = False,
    request_tables: BlockTables | None,
) -> TokenRow:
    """Build a ``TokenRow`` from token coordinates and cache coordinates.

    With ``request_indexed_decode``, ``token_ids`` and ``positions`` must be
    None because the row borrows both from ``DecodeState``. Otherwise they
    must be non-empty and equally long; a single tensor token keeps its
    device view. ``seq_len``, when given, must equal the request's accepted
    visible KV length.

    Raises:
        WorkerError: ``invalid_descriptor`` when any of these checks fails,
            or the native cache query rejects the request's coordinates.
    """
    if request_indexed_decode:
        if (
            call.kind is not ForwardMode.DECODE
            or token_ids is not None
            or positions is not None
        ):
            raise invalid_descriptor(
                "indexed decode borrows its token and position from request "
                "state"
            )
        token_values = None
        position_values = None
    else:
        if (
            token_ids is None
            or positions is None
            or len(token_ids) != len(positions)
            or not token_ids
        ):
            raise invalid_descriptor("token task ids and positions must align")
        token_values = (
            token_ids[0].reshape(1)
            if len(token_ids) == 1 and isinstance(token_ids[0], torch.Tensor)
            else torch.tensor(
                tuple(int(value) for value in token_ids), dtype=torch.long
            )
        )
        position_values = (
            positions.reshape(-1)
            if isinstance(positions, torch.Tensor)
            else torch.tensor(positions, dtype=torch.long)
        )

    predicate_value = request.predicate
    sampling_state = call.sampling_state or SamplingState()

    # (request slot, accepted visible length, capacity).
    cache = request.cache_coordinates(request_tables)
    visible = cache[1] if seq_len is None else int(seq_len)
    if visible != cache[1]:
        raise invalid_descriptor(
            "token row visibility disagrees with call metadata"
        )
    return TokenRow(
        forward_mode=cast(ForwardMode, call.kind),
        token_ids=token_values,
        positions=position_values,
        selection=selection,
        request_pool_idx=cache[0],
        seq_len=visible,
        write_kv=True,
        causal=True,
        request_indexed_decode=request_indexed_decode,
        decode_predicate=None
        if predicate_value is None
        else predicate_value[0],
        decode_predicate_tagged=False
        if predicate_value is None
        else predicate_value[1],
        decode_force_finish=bool(sampling_state.force_finish),
    )


def resolve_decode_token(
    call: Call,
    request: PendingOutput,
    *,
    decode_state: DecodeState | None,
) -> int | torch.Tensor:
    """Resolve one decode input token from the call or the device relay.

    A predicated call reads the token its predecessor left in
    ``DecodeState.future_input_tokens`` as an int64 view of shape ``[1]``
    into that state, without host synchronization. Otherwise the first of
    ``call.input_token_ids`` is returned.

    Raises:
        WorkerError: When the device continuation is not registered, there
            is no decode state, or the call carries no input token.
    """
    if call.predicate is not None:
        predicate = request.predicate
        if predicate is None:
            raise invalid_descriptor(
                "device token continuation is not registered"
            )
        states = decode_state
        if states is None:
            raise unsupported_setup(
                "device continuation has no request runtime state"
            )
        slot = int(request.request.request_pool_idx)
        return states.future_input_tokens[slot, :1]

    tokens = call.input_token_ids
    if not tokens:
        raise invalid_descriptor(
            "last-sampled token source has no committed token"
        )
    return int(tokens[0])


def require_sampling(request: PendingOutput) -> SamplingParams:
    """Return the sampling parameters required by an autoregressive request."""
    if request.request.sampling is None:
        raise invalid_descriptor(
            "sequence execution requires admitted sampling parameters"
        )
    return request.request.sampling


__all__ = ["prepare_context", "prepare_forward"]
