"""Token and visual-state packing and publication."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch

from uniserve.nn.rng import DRAW_LAYOUT_TARGET, sampling_key, sampling_uniform
from uniserve.sampling import SamplingParams
from uniserve.tensors import adjacent_view
from uniserve_worker.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.execution import calls, image
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.output import PendingOutput, capture_logprobs
from uniserve_worker.model_executor.input_batch import InputRow, TokenRow
from uniserve_worker.protocol.call import (
    Call,
    CallStatus,
    DrawLayout,
    ForwardMode,
    SamplingState,
)
from uniserve_worker.protocol.output import FinishFlags
from uniserve_worker.sampling import sampler as sampling
from uniserve_worker.sampling.metadata import SamplingMetadata, TokenSelection
from uniserve_worker.sampling.result import (
    SamplerOutput,
    SamplerRow,
    sample_columns,
)
from uniserve_worker.sampling.sampler import broadcast_selection
from uniserve_worker.storage.tensor_store import FeatureMetadata, TensorRecord

if TYPE_CHECKING:
    from uniserve.distributed.mesh import Communicator
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.model_executor.diffusion_inputs import (
        DiffusionRow,
        ImageBuilder,
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
    """Pack autoregressive extension, decode.

    or verification work into model-forward rows.
    """
    request = state.pending_output(call.request_key.request_id)
    if request.request.sampling is None:
        raise invalid_descriptor("sequence call has no admitted sampling state")

    mode = call.kind if isinstance(call.kind, ForwardMode) else None
    if mode is ForwardMode.PREFILL and (
        call.vision_input is not None or call.latent_feature_input is not None
    ):
        return _prepare_visual(
            call,
            request,
            tensor_store=tensor_store,
            request_tables=request_tables,
            model_runner=model_runner,
            state=state,
        )

    start = int(calls.require_progress(request).logical_position)
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
        task = token_task(
            call,
            request,
            tokens,
            tuple(range(start, start + len(tokens))),
            TokenSelection.ALL_LOGITS
            if scores_prompt
            else TokenSelection.LAST_LOGITS,
            request_tables=request_tables,
        )
    elif mode is ForwardMode.DECODE:
        # Indexed decode reads its token and position directly from
        # request-indexed device state instead of host-supplied values.
        indexed = (
            decode_state is not None
            and request_tables is not None
            and decode_state.device.type == "cuda"
            and request_tables.page_tables.device == decode_state.device
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
    """Convert token-model outputs into sampling work, prompt log probabilities.

    or direct outcomes.
    """
    request = state.pending_output(call.request_key.request_id)
    start = int(calls.require_progress(request).logical_position)
    mode = call.kind

    if call.vision_input is not None or call.latent_feature_input is not None:
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
                committed_tokens=(),
                request_tables=request_tables,
                state=state,
            )

        sample = build_sampling_metadata(
            call,
            logits[-1],
            request,
            positions=(start + count,),
            request_pool_index=request_pool_index,
            decode_state=decode_state,
            state=state,
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
        sample = build_sampling_metadata(
            call,
            logits[-1],
            request,
            positions=(start + 1,),
            request_pool_index=request_pool_index,
            decode_state=decode_state,
            state=state,
        )
    else:
        draft = (
            call.input_token_ids
            if call.predicate is not None
            else call.input_token_ids[1:]
        )
        sample = build_sampling_metadata(
            call,
            logits,
            request,
            positions=tuple(range(start + 1, start + len(draft) + 2)),
            draft_token_ids=draft,
            request_pool_index=request_pool_index,
            decode_state=decode_state,
            state=state,
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
    """Publish sampled tokens, speculative selections.

    and request runtime transitions.
    """
    request = state.pending_output(call.request_key.request_id)
    start = int(calls.require_progress(request).logical_position)
    mode = call.kind
    if sample_work is None and mode is not ForwardMode.DECODE:
        raise RuntimeError("only captured decode may omit sampling inputs")
    penalty_base = None if sample_work is None else sample_work.penalty_base

    if call.vision_input is not None or call.latent_feature_input is not None:
        request.progress = replace(
            calls.require_progress(request),
            rng_counter=calls.require_progress(request).rng_counter + (1),
        )
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
            sampling_position=calls.require_progress(request).rng_counter,
            decode_state=decode_state,
        )
        return _finish_visual(
            call,
            image_builder=image_builder,
            request_tables=request_tables,
            state=state,
        )

    if mode in (ForwardMode.PREFILL, ForwardMode.DECODE):
        # Only visual extension builds a diffusion row, and it returned above.
        assert isinstance(task, TokenRow)
        if mode is ForwardMode.PREFILL:
            parameters = require_sampling(request)
            if (
                parameters.return_prompt_logprobs
                or int(parameters.n_prompt_logprobs) > 0
            ):
                request.token.prompt_logprob_ranges = prompt_logprob_details(
                    request,
                    start,
                    cast(torch.Tensor, task.token_ids),
                    logits,
                    decode_state=decode_state,
                    state=state,
                )

        count = task.query_tokens if mode is ForwardMode.PREFILL else 1
        progress = calls.require_progress(request)
        logical_position = start + count
        rng_counter = progress.rng_counter + 1
        publish_runtime_sample(
            request,
            sampled,
            penalty_base=penalty_base,
            logical_position=logical_position,
            sampling_position=rng_counter,
            decode_increment=mode is ForwardMode.DECODE,
            decode_state=decode_state,
        )
        return token_outcome(
            call,
            request=request,
            task=task if mode is ForwardMode.DECODE else None,
            tokens=count,
            logical_position=logical_position,
            rng_counter=rng_counter,
            request_tables=request_tables,
            state=state,
        )
    else:
        assert sample_work is not None
        draft = sample_work.draft_token_ids
        initialized = task.seq_len + task.query_tokens

        # The verifier selects the accepted span on device; host completion
        # resolves it later against these base coordinates.
        device_selected = sampled.accepted_token_count
        if device_selected is None:
            accepted_device = sampled.accepted_draft_count
            if accepted_device is None:
                raise RuntimeError(
                    "speculative sampling lost its selected point"
                )
            device_selected = accepted_device.to(dtype=torch.int32) + 1

        request.token.runtime_cache_length = device_selected + int(task.seq_len)
        publish_runtime_sample(
            request,
            sampled,
            penalty_base=penalty_base,
            logical_position=device_selected + start,
            sampling_position=(
                device_selected
                + int(calls.require_progress(request).rng_counter)
            ),
            decode_state=decode_state,
        )

        request.token.draft_tokens = draft
        request.token.terminal_prefix = sample_work.terminal_draft_prefix
        request.token.base_logical_position = start
        request.token.base_rng_counter = calls.require_progress(
            request
        ).rng_counter
        request.token.base_kv_visible = initialized - task.query_tokens
        request.token.initialized_kv = initialized
        return token_outcome(
            call,
            tokens=0,
            request_tables=request_tables,
            state=state,
        )


def _prepare_visual(
    call: Call,
    request: PendingOutput,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
) -> TokenRow | DiffusionRow:
    """Resolve image features and interleave them with prompt tokens for model.

    execution.
    """
    reference = call.vision_input or call.latent_feature_input
    if reference is None or (
        call.vision_input is not None and call.latent_feature_input is not None
    ):
        raise invalid_descriptor(
            "visual extend requires exactly one feature tensor"
        )
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

    position = int(calls.require_progress(request).logical_position)
    close_image = call.completion_output is not None
    sample_token = call.token_output is not None
    task: TokenRow | DiffusionRow
    if call.vision_input is not None:
        task = image.vision_state_row(
            call,
            read.tensor,
            metadata.height,
            metadata.width,
            position,
            close_image=close_image,
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
    """Publish encoded visual features and update the request feature.

    reference.
    """
    request = state.pending_output(call.request_key.request_id)

    value = output if call.vision_input is not None else None
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
        sample = build_sampling_metadata(
            call,
            value[-1],
            request,
            positions=(
                int(calls.require_progress(request).logical_position)
                + max(
                    1, 1 if generation is None else int(generation.rope_advance)
                ),
            ),
            request_pool_index=request_pool_index,
            decode_state=decode_state,
            state=state,
        )
        return sample
    return _finish_visual(
        call,
        image_builder=image_builder,
        request_tables=request_tables,
        state=state,
    )


def _finish_visual(
    call: Call,
    *,
    state: BatchState,
    image_builder: ImageBuilder | None,
    request_tables: BlockTables | None,
) -> PendingOutput:
    """Finalize visual feature publication and advance the image call.

    state.
    """
    request = state.pending_output(call.request_key.request_id)
    position = int(calls.require_progress(request).logical_position)
    if call.completion_output is not None:
        flow = image_builder
        request.progress = replace(
            calls.require_progress(request),
            logical_position=position
            + max(1, 1 if flow is None else int(flow.rope_advance)),
        )
    elif call.vision_input is not None:
        request.progress = replace(
            calls.require_progress(request), logical_position=position + 1
        )
    return image.state_outcome(call, request_tables=request_tables, state=state)


def graph_decode_samples(
    calls: tuple[Call, ...],
    requests: tuple[PendingOutput, ...],
    tasks: tuple[InputRow, ...],
    output: SamplerOutput | None,
    *,
    sampling_group: Communicator | None,
    request_pool_indices: torch.Tensor,
) -> tuple[SamplerRow, ...] | None:
    """Validate and synchronize graph selections before common token.

    publication.
    """
    if output is None:
        return None
    count = len(calls)

    # Any shape mismatch disqualifies the graph selection; the caller falls
    # back to eager sampling for the whole group on None.
    columns = (requests, tasks)
    vectors = (
        request_pool_indices,
        output.tokens,
        output.valid,
        output.active,
        output.finish,
        output.continuation,
        output.tagged_tokens,
    )
    if (
        count == 0
        or any(len(values) != count for values in columns)
        or any(
            value is None or int(value.numel()) != count for value in vectors
        )
        or int(output.completion.numel())
        != sampling.SAMPLING_COMPLETION_FIELDS * count
    ):
        return None

    finish_sets: list[tuple[int, ...]] = []
    forced: list[bool] = []
    for call, request, task in zip(calls, requests, tasks, strict=True):
        parameters = request.request.sampling
        sampling_state = call.sampling_state or SamplingState()
        finish_token_ids = (
            request.request.finish_token_ids
            if not sampling_state.finish_token_ids
            else sampling_state.finish_token_ids
            if not request.request.finish_token_ids
            else tuple(
                sorted(
                    {
                        *request.request.finish_token_ids,
                        *sampling_state.finish_token_ids,
                    }
                )
            )
        )

        write = request.token_write
        # Graph selection only covers single-token greedy decode without
        # per-row host constraints; anything richer needs eager sampling.
        if (
            call.kind is not ForwardMode.DECODE
            or parameters is None
            or not sampling.device_greedy_parameters(parameters)
            or parameters.allowed_token_ids is not None
            or bool(parameters.forced_token_ids)
            or sampling_state.allowed_token_ids is not None
            or bool(sampling_state.suppressed_token_ids)
            or bool(sampling_state.transition_token_ids)
            or request.transition_write is not None
            or not isinstance(task, TokenRow)
            or task.decode_predicate is None
            or not task.decode_predicate_tagged
            or bool(sampling_state.force_finish)
            != bool(task.decode_force_finish)
            or write is None
        ):
            return None
        finish_sets.append(finish_token_ids)
        forced.append(bool(sampling_state.force_finish))

    broadcast_selection(sampling_group, output.tokens)

    if any(finish_sets):
        # Graph selection already resolved logits and device activity. Apply the
        # same terminal policy as eager sampling without selecting those logits
        # again; new continuation tensors preserve the borrowed graph storage.
        finish = sampling.sampled_finish_values(
            tuple(finish_sets),
            tuple(forced),
            output.tokens,
            output.valid & output.active,
        )
        continuation = output.valid & output.active & ~finish
        output = replace(
            output,
            finish=finish,
            continuation=continuation,
            tagged_tokens=sampling.tagged_token_values(
                output.tokens, continuation
            ),
        )
    return tuple(
        output.row(
            index, request_pool_index=request_pool_indices[index : index + 1]
        )
        for index in range(count)
    )


def prompt_logprob_details(
    request: PendingOutput,
    start: int,
    tokens: torch.Tensor,
    logits: torch.Tensor,
    *,
    state: BatchState,
    decode_state: DecodeState | None,
) -> tuple[tuple[int, int, int], ...]:
    """Create per-position log-probability rows for a prompt logits tensor."""
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
        if not calls.require_progress(request).prompt_logits_ready:
            raise invalid_descriptor(
                "continued prompt scoring has no preceding logits"
            )
        pending = request.token.runtime_prompt_logits
        if pending is None:
            pending = states.prompt_logits[slot]
        previous = pending.reshape(1, -1).to(
            device=logits.device,
            dtype=logits.dtype,
        )
        score_logits = torch.cat((previous, logits[:-1]), dim=0)
        targets = tokens

    request.token.runtime_prompt_logits = logits[-1].detach()
    request.progress = replace(
        calls.require_progress(request), prompt_logits_ready=True
    )
    if int(targets.numel()) == 0:
        return ()

    parameters = require_sampling(request)
    prompt_parameters = replace(
        parameters,
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
    captured = capture_logprobs(details, state.output_buffer)
    return tuple(captured[index] for index in range(int(targets.numel())))


def token_outcome(
    call: Call,
    *,
    state: BatchState,
    request: PendingOutput | None = None,
    task: TokenRow | None = None,
    tokens: int,
    logical_position: int | None = None,
    rng_counter: int | None = None,
    committed_tokens: tuple[int, ...] = (),
    request_tables: BlockTables | None,
) -> PendingOutput:
    """Stage request progress and pending payloads for one autoregressive.

    completion.
    """
    if request is None:
        request = state.pending_output(call.request_key.request_id)

    cache = calls.cache_coordinates(request, tables=request_tables)
    initialized = cache[2]
    if request.token.draft_tokens is None:
        published_length = request.token.runtime_cache_length
        if published_length is None:
            published_length = (
                int(task.seq_len) + int(tokens)
                if task is not None
                else cache[2]
            )
        if isinstance(published_length, torch.Tensor):
            raise RuntimeError(
                "dynamic KV length requires a speculative selection"
            )
        visible_value = int(published_length)
        initialized = visible_value
    else:
        visible_value = int(request.token.base_kv_visible)
        initialized = request.token.initialized_kv

    progress = calls.require_progress(request)
    # Publish one complete projection. Device-selected verifier acceptance stays
    # unresolved until host completion, with the initialized KV extent retained.
    # Verification keeps the base logical position so acceptance can advance it
    # by the accepted span resolved at host completion.
    request.progress = replace(
        progress,
        logical_position=(
            request.token.base_logical_position
            if request.token.draft_tokens is not None
            else progress.logical_position
            if logical_position is None
            else logical_position
        ),
        rng_counter=progress.rng_counter
        if rng_counter is None
        else rng_counter,
        kv_visible_len=visible_value,
        kv_computed_len=initialized,
    )

    request.status = CallStatus.OK
    request.finish_flags = FinishFlags()
    request.product_generations = calls.output_generations(call)
    request.token.committed_tokens = committed_tokens
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
    """Build an autoregressive row from runtime and token coordinates."""
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
    cache = calls.cache_coordinates(request, tables=request_tables)
    visible = cache[2] if seq_len is None else int(seq_len)
    if visible != cache[2]:
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
        group_id=cache[1],
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
    """Advance computed KV length and optionally publish the updated request.

    runtime.
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
        request.token.runtime_cache_length = resulting


def resolve_decode_token(
    call: Call,
    request: PendingOutput,
    *,
    decode_state: DecodeState | None,
) -> int | torch.Tensor:
    """Resolve one decode input token from an explicit value or device relay.

    product.
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
    """Bind one call's selection for the batch's device state update."""
    if decode_state is None:
        return
    request.token.sampled = sample
    request.token.runtime_logical_position = logical_position
    request.token.runtime_sampling_position = sampling_position
    request.token.runtime_penalty_base = penalty_base
    request.token.runtime_decode_increment = decode_increment


def publish_token_products(
    calls: tuple[Call, ...],
    samples: tuple[SamplerRow, ...],
    *,
    state: BatchState,
    tensor_store: TensorStore,
) -> None:
    """Publish numerical token and transition columns through their storage.

    owner.

    Rows retain their numerical batches. Token publication reads their selected
    spans directly; variable transition payloads retain their own tensor
    layouts.
    """
    for transitions in (True, False):
        writes: list[TensorRecord] = []
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
        tensor_store.publish_writes(tuple(writes), values.reshape(-1))


def build_sampling_metadata(
    call: Call,
    logits: torch.Tensor,
    request: PendingOutput,
    *,
    state: BatchState,
    positions: tuple[int, ...],
    request_pool_index: torch.Tensor,
    draft_token_ids: tuple[int, ...] = (),
    decode_state: DecodeState | None,
) -> SamplingMetadata:
    """Build sampling rows, penalties, RNG coordinates, predicates.

    and device-product bindings.
    """
    parameters = require_sampling(request)
    sampling_state = call.sampling_state or SamplingState()
    allowed_token_ids = (
        sampling_state.allowed_token_ids
        if sampling_state.allowed_token_ids is not None
        else parameters.allowed_token_ids
    )
    if not sampling_state.finish_token_ids:
        finish_token_ids = request.request.finish_token_ids
    elif not request.request.finish_token_ids:
        finish_token_ids = sampling_state.finish_token_ids
    else:
        finish_token_ids = tuple(
            sorted(
                {
                    *request.request.finish_token_ids,
                    *sampling_state.finish_token_ids,
                }
            )
        )

    # Stochastic sampling must draw from the call's registered target
    # layout so device and host evaluations agree on the Philox coordinates.
    rng = call.rng
    if float(parameters.temperature) > 0.0:
        if rng is None or rng.draw_layout is not DrawLayout.TARGET_SAMPLING:
            raise invalid_descriptor(
                "stochastic sampling requires target-sampling RNG coordinates"
            )
        if int(rng.seed) != int(parameters.seed or 0):
            raise invalid_descriptor(
                "call RNG seed disagrees with admitted sampling"
            )
        expected_positions = tuple(
            range(
                int(rng.semantic_index_base),
                int(rng.semantic_index_base) + len(positions),
            )
        )
        if positions != expected_positions:
            raise invalid_descriptor(
                "sampling positions disagree with registered semantic RNG "
                "coordinates"
            )

    rng_seed = 0 if rng is None else int(rng.seed)
    stochastic = float(parameters.temperature) > 0.0
    draw_key = (
        sampling_key(
            rng_seed,
            int(call.request_key.engine_id),
            int(call.request_key.request_id),
            int(call.request_key.request_epoch),
            DRAW_LAYOUT_TARGET,
        )
        if stochastic
        else 0
    )

    # rows is [len(positions), vocab]; a single position arrives as 1-D logits.
    rows = logits.reshape(1, -1) if logits.ndim == 1 else logits
    if rows.ndim != 2 or int(rows.shape[0]) != len(positions):
        raise invalid_descriptor(
            "sampling task positions do not align with its logits"
        )
    vocab = int(rows.shape[1])

    uses_penalties = (
        parameters.repetition_penalty != 1.0
        or parameters.frequency_penalty != 0.0
        or parameters.presence_penalty != 0.0
    )
    penalty_base = (
        _request_penalty_base(
            request, vocab, rows.device, decode_state=decode_state
        )
        if uses_penalties
        else None
    )
    penalty_view = (
        None
        if penalty_base is None
        else _candidate_penalty_counts(request, penalty_base)
    )

    forced_token_ids = parameters.forced_token_ids
    penalty_counts: list[torch.Tensor | None] = []
    allowed: list[tuple[int, ...] | None] = []
    uniform_draws: list[float] = []
    for index, position in enumerate(positions):
        if penalty_view is None:
            row_counts = None
        elif index == 0 or not draft_token_ids:
            row_counts = penalty_view
        else:
            # Verification rows accumulate penalty counts for the draft
            # tokens accepted before them in the same span.
            row_counts = penalty_view.clone()
            for token_id in draft_token_ids[:index]:
                row_counts[int(token_id)] += 1

        # Processor step 2 forced-token constraint: point `index` of the
        # call's span narrows selection to `forced_token_ids[index]`,
        # overriding any allowed-token whitelist for that point.
        row_allowed = (
            (int(forced_token_ids[index]),)
            if index < len(forced_token_ids)
            else allowed_token_ids
        )
        penalty_counts.append(row_counts)
        allowed.append(row_allowed)
        uniform_draws.append(
            sampling_uniform(draw_key, int(position)) if stochastic else 0.0
        )

    device_greedy = (
        not draft_token_ids
        and sampling.device_greedy_parameters(parameters)
        and all(value is None for value in allowed)
    )
    if device_greedy:
        draws = None
        parameter_values = None
    else:
        # Upload semantic Philox draws once; no device observation is needed to
        # construct the call's RNG coordinates or filtering parameters.
        draws = torch.tensor(
            uniform_draws, dtype=torch.float32, device=rows.device
        )
        parameter_values = torch.tensor(
            [
                (
                    float(parameters.temperature),
                    float(parameters.top_p),
                    float(parameters.min_p),
                )
            ]
            * len(positions),
            dtype=torch.float32,
            device=rows.device,
        )

    predicate_value = request.predicate
    finish_set = set(finish_token_ids)
    terminal_draft_prefix = next(
        (
            index + 1
            for index, token_id in enumerate(draft_token_ids)
            if token_id in finish_set
        ),
        None,
    )
    return SamplingMetadata(
        logits=rows,
        parameters=parameters,
        penalty_counts=tuple(penalty_counts),
        allowed=tuple(allowed),
        suppress=sampling_state.suppressed_token_ids,
        finish_token_ids=finish_token_ids,
        transition_token_ids=sampling_state.transition_token_ids,
        force_finish=sampling_state.force_finish,
        draws=draws,
        parameter_values=parameter_values,
        draft_token_ids=tuple(int(value) for value in draft_token_ids),
        terminal_draft_prefix=terminal_draft_prefix,
        return_transition=request.transition_write is not None,
        predicate=None if predicate_value is None else predicate_value[0],
        tagged_predicate=False
        if predicate_value is None
        else predicate_value[1],
        request_pool_index=request_pool_index,
        penalty_base=penalty_base,
    )


def _request_penalty_base(
    request: PendingOutput,
    vocab: int,
    device: torch.device,
    *,
    decode_state: DecodeState | None,
) -> torch.Tensor:
    """Return the fixed request-indexed committed penalty-count row."""
    states = decode_state
    if states is None:
        raise RuntimeError("token sampling has no request runtime-state owner")
    if states.vocab_size != int(vocab) or states.device != device:
        raise unsupported_setup(
            "sampling vocabulary or device disagrees with request runtime state"
        )
    return states.penalty_counts[int(request.request.request_pool_idx)]


def _candidate_penalty_counts(
    request: PendingOutput, committed: torch.Tensor
) -> torch.Tensor:
    """Include a selected token that has not yet reached committed penalty.

    storage.
    """
    sampled = request.token.sampled
    if sampled is None:
        return committed

    # Strip the continuation tag bit; the token counts only when its row is
    # both valid and predicate-active.
    counts = committed.clone()
    token = sampled.tokens.reshape(-1)[:1].bitwise_and(
        sampling.TOKEN_VALUE_MASK
    )
    weight = (
        sampled.valid.reshape(-1)[:1] & sampled.active.reshape(-1)[:1]
    ).to(dtype=counts.dtype)
    counts.scatter_add_(0, token.to(dtype=torch.int64), weight)
    return counts


def require_sampling(request: PendingOutput) -> SamplingParams:
    """Return the sampling parameters required by an autoregressive request."""
    if request.request.sampling is None:
        raise invalid_descriptor(
            "sequence execution requires admitted sampling parameters"
        )
    return request.request.sampling


__all__ = ["prepare_forward", "prepare_sampling", "publish_sample"]
