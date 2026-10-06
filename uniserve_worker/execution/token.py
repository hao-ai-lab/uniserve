"""Token and visual-state packing and export.

This module is the autoregressive half of the numerical path that
``forward`` drives. ``prepare_forward`` packs a prefill, decode or verify
call into a ``TokenRow``, or into a visual-state row when a prefill writes
generated-image feedback or an input image's latent. ``prepare_context``
packs a context prefill, whose prompt tokens run between input-image vision
blocks, into one row per segment, and ``finish_context`` reads its rows'
outputs. ``prepare_sampling`` turns the model's logits into
``SamplingMetadata`` for the sampler, or into a direct outcome when the call
samples nothing. ``publish_sample`` records the sampled result and the
request progress it implies on the call's ``PendingOutput``.

Device state is only bound here: the views in ``PendingOutput.token_update``
are applied to ``DecodeState`` by the native executor at batch commit. A verify
call leaves its accepted span on device; ``PendingOutput`` materialization
resolves it on the host against the base coordinates recorded here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from uniserve.sampling import SamplingParams
from uniserve.tensors import adjacent_view
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
from uniserve_worker.sampling.metadata import SamplingMetadata, TokenSelection
from uniserve_worker.sampling.result import (
    SamplerRow,
    sample_columns,
)
from uniserve_worker.storage.tensor_store import Buffer, FeatureMetadata

if TYPE_CHECKING:
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.model_executor.diffusion_inputs import (
        DiffusionRow,
        ImageBuilder,
    )
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.decode_state import DecodeState
    from uniserve_worker.storage.tensor_store import TensorStore


def _writes_visual_state(call: Call) -> bool:
    """Whether a prefill writes one visual-state row of image features.

    That is generated-image feedback, whose single vision or latent feature
    closes the image, or an input image's latent block; a context prefill's
    vision blocks are rows of its segments instead.
    """
    return not call.writes_context() and (
        bool(call.vision_inputs) or call.latent_feature_input is not None
    )


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
    if mode is ForwardMode.PREFILL and _writes_visual_state(call):
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
        # only writes the K/V cache (``prepare_sampling`` reads no logits).
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


def prepare_sampling(
    call: Call,
    task: TokenRow | DiffusionRow,
    output: torch.Tensor,
    *,
    state: BatchState,
    request_pool_index: torch.Tensor,
    tensor_store: TensorStore,
    image_builder: ImageBuilder | None,
    request_tables: BlockTables | None,
    decode_state: DecodeState | None,
) -> SamplingMetadata | PendingOutput:
    """Turn a call's forward output into sampling work or a direct outcome.

    A prefill commits every query token and, with a ``DecodeState``, retains
    the resulting runtime cache length; a decode commits one token without
    preparing it; a verify commits nothing here.
    Returns a finished ``PendingOutput`` when a prefill declares no token
    output or a visual call samples nothing. Otherwise returns
    ``SamplingMetadata`` positioned after the tokens the call computed; a
    verify samples one position per row.
    """
    request = state.pending_output(call.request_key.request_id)
    start = int(request.progress.logical_position)
    mode = call.kind

    if _writes_visual_state(call):
        return _prepare_visual_sampling(
            call,
            task,
            output,
            request_pool_index=request_pool_index,
            image_builder=image_builder,
            request_tables=request_tables,
            decode_state=decode_state,
            state=state,
        )

    logits = output
    if mode is ForwardMode.PREFILL:
        count = task.query_tokens
        commit_kv(
            task,
            count,
            request,
            request_tables=request_tables,
            decode_state=decode_state,
        )
        if call.token_output is None:
            return token_outcome(
                call,
                tokens=0,
                request_tables=request_tables,
                state=state,
            )

        sample = SamplingMetadata.for_call(
            call,
            logits[-1],
            request,
            positions=(start + count,),
            request_pool_index=request_pool_index,
            decode_state=decode_state,
        )
    elif mode is ForwardMode.DECODE:
        commit_kv(
            task,
            1,
            request,
            publish_runtime=False,
            request_tables=request_tables,
            decode_state=decode_state,
        )
        sample = SamplingMetadata.for_call(
            call,
            logits[-1],
            request,
            positions=(start + 1,),
            request_pool_index=request_pool_index,
            decode_state=decode_state,
        )
    else:
        draft = (
            call.input_token_ids
            if call.predicate is not None
            else call.input_token_ids[1:]
        )
        sample = SamplingMetadata.for_call(
            call,
            logits,
            request,
            positions=tuple(range(start + 1, start + len(draft) + 2)),
            draft_token_ids=draft,
            request_pool_index=request_pool_index,
            decode_state=decode_state,
        )
    return sample


def publish_sample(
    call: Call,
    task: TokenRow | DiffusionRow,
    logits: torch.Tensor,
    sample_work: SamplingMetadata | None,
    sampled: SamplerRow,
    *,
    state: BatchState,
    image_builder: ImageBuilder | None,
    request_tables: BlockTables | None,
    decode_state: DecodeState | None,
) -> PendingOutput:
    """Record a sampled selection and the request progress it implies.

    A prefill or decode advances the logical position by its computed tokens
    and the RNG counter by one; a prefill's prompt scoring adds to the ranges
    already recorded. A context prefill's last text row publishes as a prompt
    chunk; a last vision row advances past its temporal positions without
    scoring image placeholders. A verify binds device tensors offset from
    its base coordinates and records those coordinates
    in the native pending output so materialization can resolve the accepted
    span. A visual call advances the RNG counter by one and its position as
    ``_finish_visual`` does.

    ``sample_work`` may be None only for a decode whose selection came from
    graph replay (``sample_graph``).
    """
    request = state.pending_output(call.request_key.request_id)
    start = int(request.progress.logical_position)
    mode = call.kind
    if sample_work is None and mode is not ForwardMode.DECODE:
        raise RuntimeError("only captured decode may omit sampling inputs")
    penalty_base = None if sample_work is None else sample_work.penalty_base

    if _writes_visual_state(call):
        request.advance_tokens(0, sampled=True)
        progress = request.progress
        flow = image_builder
        logical_position = start + (
            max(1, 1 if flow is None else int(flow.rope_advance))
            if call.completion_output is not None
            else 1
        )
        publish_runtime_sample(
            request,
            sampled,
            penalty_base=penalty_base,
            logical_position=logical_position,
            sampling_position=progress.rng_counter,
            decode_state=decode_state,
        )
        return _finish_visual(
            call,
            task,
            image_builder=image_builder,
            request_tables=request_tables,
            state=state,
        )

    if mode in (ForwardMode.PREFILL, ForwardMode.DECODE):
        # Only visual extension builds a diffusion row, and it returned above.
        assert isinstance(task, TokenRow)
        if mode is ForwardMode.PREFILL and task.causal:
            parameters = require_sampling(request)
            if (
                parameters.return_prompt_logprobs
                or int(parameters.n_prompt_logprobs) > 0
            ):
                request.add_prompt_logprobs(
                    prompt_logprob_details(
                        request,
                        start,
                        cast(torch.Tensor, task.token_ids),
                        logits,
                        decode_state=decode_state,
                        state=state,
                    )
                )

        count = task.query_tokens if mode is ForwardMode.PREFILL else 1
        token_outcome(
            call,
            request=request,
            task=task if mode is ForwardMode.DECODE else None,
            tokens=count,
            logical_position=_next_position(task)
            if call.writes_context()
            else None,
            sampled=True,
            request_tables=request_tables,
            state=state,
        )
        progress = request.progress
        publish_runtime_sample(
            request,
            sampled,
            penalty_base=penalty_base,
            logical_position=progress.logical_position,
            sampling_position=progress.rng_counter,
            decode_increment=mode is ForwardMode.DECODE,
            decode_state=decode_state,
        )
        return request
    else:
        assert sample_work is not None
        draft = sample_work.draft_token_ids
        initialized = task.seq_len + task.query_tokens

        # The verifier selects the accepted span on device; host completion
        # resolves it later against these base coordinates. The accepted
        # token count is the accepted drafts plus a correction or bonus token,
        # which is omitted when acceptance reaches the terminal draft prefix.
        # It stays a device tensor, so the pending cache length and positions
        # below are device values as well.
        device_selected = sampled.accepted_token_count
        if device_selected is None:
            accepted_device = sampled.accepted_draft_count
            if accepted_device is None:
                raise RuntimeError(
                    "speculative sampling lost its selected point"
                )
            device_selected = accepted_device.to(dtype=torch.int32) + 1

        request.token_update.cache_length = device_selected + int(task.seq_len)
        publish_runtime_sample(
            request,
            sampled,
            penalty_base=penalty_base,
            logical_position=device_selected + start,
            sampling_position=(
                device_selected + int(request.progress.rng_counter)
            ),
            decode_state=decode_state,
        )

        request.set_speculation(
            draft,
            sample_work.terminal_draft_prefix,
            int(task.seq_len),
            initialized,
        )
        return request


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


def _prepare_visual_sampling(
    call: Call,
    task: TokenRow | DiffusionRow,
    output: torch.Tensor,
    *,
    state: BatchState,
    request_pool_index: torch.Tensor,
    image_builder: ImageBuilder | None,
    request_tables: BlockTables | None,
    decode_state: DecodeState | None,
) -> SamplingMetadata | PendingOutput:
    """Commit a visual row's KV, then sample its next token or finish it.

    When the call declares a token output, returns ``SamplingMetadata`` for
    the request's logical position advanced by the image builder's RoPE
    advance (at least one); only vision rows produce logits. Otherwise
    finishes the call through ``_finish_visual``.
    """
    request = state.pending_output(call.request_key.request_id)

    value = output if call.vision_inputs else None
    commit_kv(
        task,
        task.query_tokens,
        request,
        request_tables=request_tables,
        decode_state=decode_state,
    )

    if call.token_output is not None:
        assert value is not None
        generation = image_builder
        sample = SamplingMetadata.for_call(
            call,
            value[-1],
            request,
            positions=(
                int(request.progress.logical_position)
                + max(
                    1, 1 if generation is None else int(generation.rope_advance)
                ),
            ),
            request_pool_index=request_pool_index,
            decode_state=decode_state,
        )
        return sample
    return _finish_visual(
        call,
        task,
        image_builder=image_builder,
        request_tables=request_tables,
        state=state,
    )


def _next_position(row: TokenRow) -> int:
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


def _finish_visual(
    call: Call,
    task: TokenRow | DiffusionRow,
    *,
    state: BatchState,
    image_builder: ImageBuilder | None,
    request_tables: BlockTables | None,
) -> PendingOutput:
    """Advance the logical position past a visual row and record its outcome.

    A call that closes the image advances by the image builder's RoPE advance
    (at least one); an input image's latent row keeps its position.
    """
    request = state.pending_output(call.request_key.request_id)
    if call.completion_output is not None:
        flow = image_builder
        request.advance_tokens(
            max(1, 1 if flow is None else int(flow.rope_advance))
        )
    return image.state_outcome(call, request_tables=request_tables, state=state)


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
    position (``_next_position``). The rows form one numerical call whose
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
        position = _next_position(row)
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


def finish_context(
    call: Call,
    rows: tuple[tuple[TokenRow, torch.Tensor, torch.Tensor], ...],
    *,
    state: BatchState,
    request_tables: BlockTables | None,
    decode_state: DecodeState | None,
) -> tuple[TokenRow, torch.Tensor, SamplingMetadata] | PendingOutput:
    """Commit a context prefill's KV and read its prompt runs' outputs.

    ``rows`` are the call's ``(row, value, request_pool_index)`` triples from
    ``prepare_context``, in context order. The call commits every row's KV.
    Prompt runs are scored in order when prompt log probabilities are
    requested (``prompt_logprob_details``). Vision blocks score nothing;
    their final logits predict the following text token.

    A call that samples nothing records its outcome at the position after its
    last row. A sampling call reads the final row's last logits, whether
    that row is text or vision. A final prompt run publishes as a prompt
    chunk from its starting position; a final vision block contributes no
    prompt scores. ``publish_sample`` advances past the final row's temporal
    positions and publishes the sampled token.

    Raises:
        WorkerError: From ``prompt_logprob_details`` or
            ``SamplingMetadata.for_call``.
        RuntimeError: When the KV commit exceeds the request's page tables.
    """
    request = state.pending_output(call.request_key.request_id)
    parameters = require_sampling(request)
    scores_prompt = bool(
        parameters.return_prompt_logprobs
        or int(parameters.n_prompt_logprobs) > 0
    )
    last, last_value, sampling_index = rows[-1]
    commit_kv(
        last,
        last.query_tokens,
        request,
        request_tables=request_tables,
        decode_state=decode_state,
    )

    # A sampled call's last run is scored when its sample publishes; prompt
    # runs are the causal rows.
    samples = call.token_output is not None
    scored = rows[:-1] if samples and last.causal else rows
    if scores_prompt:
        for row, value, _index in scored:
            if not row.causal:
                request.set_prompt_logits(value[-1].detach())
                continue
            request.add_prompt_logprobs(
                prompt_logprob_details(
                    request,
                    int(cast(torch.Tensor, row.positions)[0]),
                    cast(torch.Tensor, row.token_ids),
                    value,
                    decode_state=decode_state,
                    state=state,
                )
            )

    if not samples:
        return token_outcome(
            call,
            request=request,
            task=last,
            tokens=last.query_tokens,
            logical_position=_next_position(last),
            request_tables=request_tables,
            state=state,
        )

    if last.causal:
        start = int(cast(torch.Tensor, last.positions)[0])
        request.advance_tokens(0, position=start)
    sample = SamplingMetadata.for_call(
        call,
        last_value[-1],
        request,
        positions=(_next_position(last),),
        request_pool_index=sampling_index,
        decode_state=decode_state,
    )
    return last, last_value, sample


def prompt_logprob_details(
    request: PendingOutput,
    start: int,
    tokens: torch.Tensor,
    logits: torch.Tensor,
    *,
    state: BatchState,
    decode_state: DecodeState | None,
) -> tuple[tuple[int, int, int], ...]:
    """Score one prompt chunk and capture its log-probability ranges.

    The first chunk (``start == 0``) scores every token after the first. A
    continued chunk also scores its first token, using the last logits of the
    previous chunk from the request's retained runtime logits or
    ``DecodeState.prompt_logits``. This chunk's last logits are retained for
    the next chunk and ``prompt_logits_ready`` is set.

    Returns:
        One ``(offset, count, row)`` completion-buffer span per scored token.

    Raises:
        WorkerError: When, for example, the logits do not align with
            ``tokens``, there is no decode state, or a continued chunk has no
            preceding logits.
    """
    # logits is [num_tokens, vocab]; each token is scored by the logits of the
    # preceding position.
    tokens = tokens.reshape(-1).to(device=logits.device, dtype=torch.long)
    if logits.ndim != 2 or int(logits.shape[0]) != int(tokens.numel()):
        raise invalid_descriptor(
            "prompt scoring logits do not align with input tokens"
        )

    states = decode_state
    if states is None:
        raise unsupported_setup(
            "prompt scoring has no request-indexed runtime state"
        )
    slot = int(request.request.request_pool_idx)

    if start == 0:
        score_logits = logits[:-1]
        targets = tokens[1:]
    else:
        # Continued prompts prepend the carried-over logits of the last token
        # of the previous chunk so its first token is also scored.
        if not request.progress.prompt_logits_ready:
            raise invalid_descriptor(
                "continued prompt scoring has no preceding logits"
            )
        pending = request.token_update.prompt_logits
        if pending is None:
            pending = states.prompt_logits[slot]
        previous = pending.reshape(1, -1).to(
            device=logits.device,
            dtype=logits.dtype,
        )
        score_logits = torch.cat((previous, logits[:-1]), dim=0)
        targets = tokens

    request.set_prompt_logits(logits[-1].detach())
    if int(targets.numel()) == 0:
        return ()

    parameters = require_sampling(request)
    prompt_parameters = parameters.replace(
        return_logprobs=True,
        n_logprobs=int(parameters.n_prompt_logprobs),
    )
    indexes = torch.arange(
        int(targets.numel()),
        dtype=torch.long,
        device=score_logits.device,
    )
    details = sampling.logprob_details(
        score_logits.float(),
        indexes,
        targets,
        (prompt_parameters,) * int(targets.numel()),
    )
    captured = state.output_buffer.capture_logprobs(details)
    return tuple(captured[index] for index in range(int(targets.numel())))


def token_outcome(
    call: Call,
    *,
    state: BatchState,
    request: PendingOutput | None = None,
    task: TokenRow | None = None,
    tokens: int,
    logical_position: int | None = None,
    sampled: bool = False,
    request_tables: BlockTables | None,
) -> PendingOutput:
    """Project progress for an ordinary prefill or decode.

    KV length comes from the numerical update when present, otherwise from
    the forward's extent or the accepted request length. Speculative calls
    defer their selected progress to native completion.
    """
    if request is None:
        request = state.pending_output(call.request_key.request_id)

    cache = request.cache_coordinates(request_tables)
    published_length = request.token_update.cache_length
    if published_length is None:
        published_length = (
            int(task.seq_len) + int(tokens) if task is not None else cache[1]
        )
    if isinstance(published_length, torch.Tensor):
        raise RuntimeError("dynamic KV length requires a speculative selection")

    request.advance_tokens(
        tokens,
        cache_length=int(published_length),
        position=logical_position,
        sampled=sampled,
    )
    return request


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


def commit_kv(
    task: TokenRow | DiffusionRow,
    tokens: int,
    request: PendingOutput,
    *,
    publish_runtime: bool = True,
    request_tables: BlockTables | None,
    decode_state: DecodeState | None,
) -> None:
    """Validate a row's computed KV extent after ``tokens`` query tokens.

    The count must lie within the row's query span and the resulting extent
    within the request's allocated page table. With ``publish_runtime`` and a
    ``DecodeState``, the extent is recorded as
    ``request.token_update.cache_length`` for the commit. A zero count
    leaves the update unchanged and skips the page-table check.

    Raises:
        RuntimeError: When the count or extent is out of range, there are no
            page tables, or, when preparing inputs, the row belongs to another
            request slot.
    """
    count = int(tokens)
    if count < 0 or count > task.query_tokens:
        raise RuntimeError("KV commit count is outside the task query span")
    if count == 0:
        return

    resulting = int(task.seq_len) + count
    page_tables = request_tables
    if page_tables is None:
        raise RuntimeError("KV commit requires request page tables")
    if resulting > page_tables.allocated_length(task.request_pool_idx):
        raise RuntimeError("KV task exceeds its scheduler block table")

    if publish_runtime and decode_state is not None:
        if int(task.request_pool_idx) != int(request.request.request_pool_idx):
            raise RuntimeError("token KV update crossed request slots")
        request.token_update.cache_length = resulting


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


def publish_runtime_sample(
    request: PendingOutput,
    sample: SamplerRow,
    *,
    penalty_base: torch.Tensor | None,
    logical_position: int | torch.Tensor,
    sampling_position: int | torch.Tensor,
    decode_increment: bool = False,
    decode_state: DecodeState | None,
) -> None:
    """Bind one call's selection for the batch's device state update.

    Does nothing without a ``DecodeState``. The native executor applies
    the bound values when the batch commits.
    """
    if decode_state is None:
        return
    request.token_update.sampled = sample
    request.token_update.logical_position = logical_position
    request.token_update.sampling_position = sampling_position
    request.token_update.penalty_base = penalty_base
    request.token_update.decode_increment = decode_increment


def publish_token_products(
    calls: tuple[Call, ...],
    samples: tuple[SamplerRow, ...],
    *,
    state: BatchState,
    tensor_store: TensorStore,
) -> None:
    """Publish sampled token and transition values into their reserved writes.

    Transition products are published in one pass and token products in a
    second; each pass publishes every call that reserved that write through
    one ``TensorStore.write_scalars`` call. Token products receive the
    tagged token values. Transition payloads are concatenated unless they
    already form one adjacent view.
    """
    for transitions in (True, False):
        writes: list[Buffer] = []
        selected: list[SamplerRow] = []
        for call, sample in zip(calls, samples, strict=True):
            request = state.pending_output(call.request_key.request_id)
            write = (
                request.transition_write if transitions else request.token_write
            )
            if write is not None:
                writes.append(write)
                selected.append(sample)

        if not writes:
            continue

        if transitions:
            transition_values = tuple(sample.transition for sample in selected)
            if any(value is None for value in transition_values):
                raise RuntimeError(
                    "sampling result lost a declared transition output"
                )
            tensors = tuple(
                cast(torch.Tensor, value) for value in transition_values
            )
            values = adjacent_view(tensors)
            if values is None:
                values = torch.cat(tensors, dim=0)
        else:
            (values,) = sample_columns(selected, ("tagged_tokens",))
        tensor_store.write_scalars(tuple(writes), values.reshape(-1))


def require_sampling(request: PendingOutput) -> SamplingParams:
    """Return the sampling parameters required by an autoregressive request."""
    if request.request.sampling is None:
        raise invalid_descriptor(
            "sequence execution requires admitted sampling parameters"
        )
    return request.request.sampling


__all__ = [
    "finish_context",
    "prepare_context",
    "prepare_forward",
    "prepare_sampling",
    "publish_sample",
]
