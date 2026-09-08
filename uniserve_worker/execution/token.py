"""Token and visual-state packing and publication."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch

from uniserve_worker.execution.batch import (
    DeviceSelected,
    DrawLayout,
    FinishFlags,
    LogicalLengths,
    OpCode,
    Operation,
    OpStatus,
    ProductKind,
    ProductPayload,
    ProductRef,
    SamplingParams,
    SamplingState,
    TokenMode,
    TokenSpan,
)
from uniserve_worker.execution.output import (
    LogprobOutputRow,
    LogprobPayload,
    SamplingOutputRow,
)
from uniserve_worker.foundation.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.runtime.device_products import (
    DeviceProductWrite,
)
from uniserve_worker.runtime.request import RequestDraft

from . import encode
from . import sample as sampling
from .cuda_graph import GraphGreedyOutput
from .forward_batch import ModelPhase, TokenSelection, packed_tensor_views
from .rng import DRAW_LAYOUT_TARGET, sampling_key, sampling_uniform
from .rows import (
    DecodeRuntimePublication,
    ForwardRow,
    LaneState,
    OperationState,
    Outcome,
    PromptLogitsPublication,
    RuntimePublication,
    SampleResult,
    SampleRow,
    SampleWork,
    SpeculativeSelection,
    StateOutcome,
)

if TYPE_CHECKING:
    from ..worker.worker import Worker


def pack_forward(runtime: Worker, state: OperationState) -> tuple[ForwardRow, ...]:
    """Pack autoregressive extension, decode, or verification work into model-forward rows."""

    if state.phase != "initial":
        return ()

    operation = state.operation
    scope = state.lane
    request = runtime.request_row(scope, operation.request_key.request_id)
    if request.request.sampling is None:
        raise invalid_descriptor("sequence operation has no admitted sampling state")
    mode = operation.kind.token_mode
    if mode is TokenMode.EXTEND and any(
        reference.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}
        for reference in operation.inputs
    ):
        return _pack_visual(runtime, state, request)
    start = int(request.logical_position)
    tokens: tuple[int | torch.Tensor, ...]
    if mode is TokenMode.EXTEND:
        if isinstance(operation.state_parent.point, DeviceSelected) and not any(
            reference.kind is ProductKind.TOKEN for reference in operation.inputs
        ):
            tokens = (resolve_decode_token(runtime, operation, request, scope),)
        else:
            tokens = operation_token_ids(runtime, operation, scope)
        sampling = require_sampling(request)
        scores_prompt = bool(sampling.return_prompt_logprobs or int(sampling.n_prompt_logprobs) > 0)
        task = token_task(
            runtime,
            operation,
            request,
            tokens,
            tuple(range(start, start + len(tokens))),
            TokenSelection.ALL_LOGITS if scores_prompt else TokenSelection.LAST_LOGITS,
            scope,
        )
        state.data.update(
            start=start,
            tokens=tokens,
            task=task,
            scores_prompt=scores_prompt,
            mode="extend",
        )
    elif mode is TokenMode.DECODE:
        current = resolve_decode_token(runtime, operation, request, scope)
        task = token_task(
            runtime,
            operation,
            request,
            (current,),
            (start,),
            TokenSelection.LAST_LOGITS,
            scope,
        )
        state.data.update(start=start, task=task, mode="decode")
    else:
        if isinstance(operation.state_parent.point, DeviceSelected):
            current = resolve_decode_token(runtime, operation, request, scope)
            draft = operation_token_ids(runtime, operation, scope)
        else:
            input_tokens = operation_token_ids(runtime, operation, scope)
            if len(input_tokens) < 2:
                raise invalid_descriptor(
                    "fixed-parent verification requires current and draft tokens"
                )
            current = int(input_tokens[0])
            draft = input_tokens[1:]
        tokens = (current, *draft)
        task = token_task(
            runtime,
            operation,
            request,
            tokens,
            tuple(range(start, start + len(tokens))),
            TokenSelection.ALL_LOGITS,
            scope,
        )
        state.data.update(
            start=start,
            draft=draft,
            task=task,
            mode="verify",
        )
    state.rows = (task,)
    state.phase = "forward_pending"
    return state.rows


def consume_forward(
    runtime: Worker,
    state: OperationState,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    """Convert token-model outputs into sampling work, prompt log probabilities, or direct outcomes."""

    if state.phase != "forward_pending" or len(outputs) != 1:
        raise RuntimeError("token forward result is not aligned")
    operation = state.operation
    scope = state.lane
    task = state.data["task"]
    mode = state.data["mode"]
    request = runtime.request_row(state.lane, state.operation.request_key.request_id)
    start = state.data["start"]
    if mode == "visual":
        _consume_visual(runtime, state, outputs[0])
        return
    logits = token_logits(outputs[0])
    if mode == "extend":
        tokens = state.data["tokens"]
        commit_kv(runtime, task, len(tokens), scope)
        if not any(output.kind is ProductKind.TOKEN for output in operation.outputs):
            state.outcome = token_outcome(
                runtime,
                operation,
                scope,
                base=start,
                tokens=0,
                committed_tokens=(),
            )
            state.phase = "done"
            return
        sample = build_sample_work(
            runtime,
            operation,
            logits[-1],
            request,
            scope,
            positions=(start + len(tokens),),
            request_pool_index=sampling.request_pool_index(task),
        )
        state.data["logits"] = logits
    elif mode == "decode":
        commit_kv(runtime, task, 1, scope)
        sample = build_sample_work(
            runtime,
            operation,
            logits[-1],
            request,
            scope,
            positions=(start + 1,),
            request_pool_index=sampling.request_pool_index(task),
        )
    else:
        draft = state.data["draft"]
        sample = build_sample_work(
            runtime,
            operation,
            logits,
            request,
            scope,
            positions=tuple(range(start + 1, start + len(draft) + 2)),
            draft_token_ids=draft,
            request_pool_index=sampling.request_pool_index(task),
        )
    state.sample = sample
    state.phase = "sample"


def pack_sample(state: OperationState) -> object | None:
    """Return the sampling task previously prepared for an autoregressive operation."""

    if state.phase != "sample":
        return None
    state.phase = "sample_pending"
    return state.sample


def consume_sample(runtime: Worker, state: OperationState, value: object) -> None:
    """Publish sampled tokens, speculative selections, and request runtime transitions."""

    if state.phase != "sample_pending":
        raise RuntimeError("token sample result has no pending selection")
    sampled = sampling.sample_result(value)
    operation = state.operation
    scope = state.lane
    request = runtime.request_row(state.lane, state.operation.request_key.request_id)
    start = state.data["start"]
    task = state.data["task"]
    mode = state.data["mode"]
    sample_work = state.sample
    if sample_work is None:
        raise RuntimeError("token sample result lost its pending sampling work")
    if mode == "visual":
        publish_token_product(runtime, operation, sampled, scope)
        request.rng_counter += 1
        flow = runtime.model.generation
        logical_position = start + (
            max(1, 1 if flow is None else int(flow.rope_advance))
            if state.data["close_image"]
            else 1
        )
        publish_runtime_samples(
            runtime,
            (operation,),
            (request,),
            (sampled,),
            scope=scope,
            sample_tasks=(sample_work,),
            logical_positions=(logical_position,),
            sampling_positions=(request.rng_counter,),
        )
        state.data["state_outcome"] = StateOutcome(
            sampling=sampled.completion,
            products=sample_product_payloads(operation, sampled),
        )
        _finish_visual(runtime, state)
        return
    if mode == "extend":
        if state.data["scores_prompt"]:
            sampled = replace(
                sampled,
                prompt_logprobs=prompt_logprob_details(
                    runtime,
                    request,
                    start,
                    cast(torch.Tensor, task.token_ids),
                    state.data["logits"],
                    scope,
                ),
            )
        publish_token_product(runtime, operation, sampled, scope)
        request.rng_counter += 1
        request.logical_position = start + len(state.data["tokens"])
        publish_runtime_samples(
            runtime,
            (operation,),
            (request,),
            (sampled,),
            scope=scope,
            sample_tasks=(sample_work,),
            logical_positions=(request.logical_position,),
            sampling_positions=(request.rng_counter,),
        )
        state.outcome = token_outcome(
            runtime,
            operation,
            scope,
            base=start,
            tokens=len(state.data["tokens"]),
            sampling=sampled.completion,
            sample=sampled,
        )
    elif mode == "decode":
        request.rng_counter += 1
        request.logical_position = start + 1
        publish_token_product(runtime, operation, sampled, scope)
        publish_runtime_samples(
            runtime,
            (operation,),
            (request,),
            (sampled,),
            scope=scope,
            sample_tasks=(sample_work,),
            logical_positions=(request.logical_position,),
            sampling_positions=(request.rng_counter,),
        )
        state.outcome = token_outcome(
            runtime,
            operation,
            scope,
            base=start,
            tokens=1,
            sampling=sampled.completion,
            sample=sampled,
        )
    else:
        draft = state.data["draft"]
        initialized = task.seq_len + task.query_tokens
        publish_token_product(runtime, operation, sampled, scope)
        sampling_row = replace(
            sampled.completion,
            draft_tokens=draft,
            terminal_prefix=sample_work.terminal_draft_prefix,
            logical_base=start,
            kv_base=initialized - task.query_tokens,
        )
        device_selected = sampled.device_selected_point
        if device_selected is None:
            accepted_device = sampled.device_accepted_tokens
            if accepted_device is None:
                raise RuntimeError("speculative sampling lost its selected point")
            device_selected = accepted_device.to(dtype=torch.int32) + 1
        scope.runtime_cache_lengths[int(request.request.request_pool_idx)] = device_selected + int(
            task.seq_len
        )
        publish_runtime_samples(
            runtime,
            (operation,),
            (request,),
            (sampled,),
            scope=scope,
            sample_tasks=(sample_work,),
            logical_positions=(device_selected + start,),
            sampling_positions=(device_selected + int(request.rng_counter),),
        )
        state.outcome = token_outcome(
            runtime,
            operation,
            scope,
            base=start,
            tokens=0,
            sampling=sampling_row,
            sample=sampled,
            selection=SpeculativeSelection(
                completion=sampling_row,
                draft_tokens=draft,
                terminal_prefix=sample_work.terminal_draft_prefix,
                base_logical_position=start,
                base_rng_counter=request.rng_counter,
                base_kv_visible=initialized - task.query_tokens,
                initialized_kv=initialized,
            ),
        )
    state.phase = "done"


def _pack_visual(
    runtime: Worker, state: OperationState, request: RequestDraft
) -> tuple[ForwardRow, ...]:
    """Resolve image features and interleave them with prompt tokens for model execution."""

    operation = state.operation
    scope = state.lane
    references = tuple(
        reference
        for reference in operation.inputs
        if reference.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}
    )
    if len(references) != 1:
        raise invalid_descriptor("visual extend requires exactly one feature product")
    reference = references[0]
    read = runtime.encoder_cache.consume(
        reference,
        consumer_op_id=operation.op_id,
        device=runtime.operation_device(operation),
    )
    scope.encoder_reads.append(read)
    position = int(request.logical_position)
    close_image = any(output.kind is ProductKind.COMPLETION for output in operation.outputs)
    sample_token = any(output.kind is ProductKind.TOKEN for output in operation.outputs)
    if reference.kind is ProductKind.VISION_FEATURE:
        task = encode.vision_state_row(
            runtime,
            operation,
            read.tensor,
            read.metadata.height,
            read.metadata.width,
            position,
            scope,
            close_image=close_image,
            logits=sample_token,
        )
        variant = OpCode.ENCODER_VISION
    else:
        task = encode.latent_state_row(
            runtime,
            operation,
            read.tensor,
            read.metadata.height,
            read.metadata.width,
            position,
            scope,
        )
        variant = OpCode.ENCODER_LATENT
    if task.query_tokens > int(operation.bounds.max_tokens):
        raise invalid_descriptor("image state query span exceeds the operation token bound")
    state.data.update(
        start=position,
        task=task,
        mode="visual",
        variant=variant,
        reference_kind=reference.kind,
        close_image=close_image,
        sample_token=sample_token,
    )
    state.rows = (task,)
    state.phase = "forward_pending"
    return state.rows


def _consume_visual(runtime: Worker, state: OperationState, output: torch.Tensor) -> None:
    """Publish encoded visual features and update the request feature reference."""

    from . import flow

    request = runtime.request_row(state.lane, state.operation.request_key.request_id)

    task = state.data["task"]
    scope = state.lane
    if state.data["variant"] is OpCode.ENCODER_VISION:
        value = token_logits_or_hidden(output)
    else:
        flow.prediction(output)
        value = None
    commit_kv(runtime, task, task.query_tokens, scope)
    if state.data["sample_token"]:
        assert value is not None
        generation = runtime.model.generation
        sample = build_sample_work(
            runtime,
            state.operation,
            value[-1],
            request,
            scope,
            positions=(
                state.data["start"]
                + max(1, 1 if generation is None else int(generation.rope_advance)),
            ),
            request_pool_index=sampling.request_pool_index(task),
        )
        state.sample = sample
        state.phase = "sample"
        return
    state.data["state_outcome"] = StateOutcome()
    _finish_visual(runtime, state)


def _finish_visual(runtime: Worker, state: OperationState) -> None:
    """Finalize visual feature publication and advance the encode operation state."""

    request = runtime.request_row(state.lane, state.operation.request_key.request_id)
    position = state.data["start"]
    if state.data["close_image"]:
        flow = runtime.model.generation
        request.logical_position = position + max(1, 1 if flow is None else int(flow.rope_advance))
    elif state.data["reference_kind"] is ProductKind.VISION_FEATURE:
        request.logical_position = position + 1
    state.outcome = encode.state_outcome(
        runtime,
        state.operation,
        state.data["state_outcome"],
        state.lane,
        base=position,
    )
    state.phase = "done"


def decode_batch(
    runtime: Worker,
    operations: tuple[Operation, ...],
    scope: LaneState,
) -> tuple[Outcome, ...]:
    """Build a request-indexed decode forward and sampling group for compatible operations."""

    build_started = time.perf_counter_ns()
    starts: list[int] = []
    layout = scope.layout
    if layout is None:
        raise RuntimeError("scope lost its aligned request-row view")
    if layout.operations == operations:
        requests = layout.requests
        seq_lens = layout.seq_lens
        weights = layout.weights
    else:
        aligned = {
            identity: (request, seq_len, weight)
            for identity, request, seq_len, weight in zip(
                layout.identities,
                layout.requests,
                layout.seq_lens,
                layout.weights,
                strict=True,
            )
        }
        selected = tuple(aligned[runtime.operation_identity(operation)] for operation in operations)
        requests = tuple(value[0] for value in selected)
        seq_lens = tuple(value[1] for value in selected)
        weights = tuple(value[2] for value in selected)
    starts.extend(int(request.logical_position) for request in requests)
    tasks = _decode_forward_tasks(
        runtime,
        operations,
        requests,
        seq_lens,
        weights,
        starts,
        scope,
    )

    runtime.record_component(scope, "text_build_batch", build_started)
    forward_started = time.perf_counter_ns()
    forward_result = runtime.run_observed_forward_group(tasks, scope)
    outputs = forward_result.values
    graph_greedy = forward_result.greedy
    if forward_result.output_event is not None:
        torch.cuda.current_stream(runtime.phase_device(tasks[0].phase)).wait_event(
            forward_result.output_event
        )
    runtime.record_component(scope, "text_model_forward", forward_started)
    for task in tasks:
        commit_kv(runtime, task, 1, scope, publish_runtime=False)

    sample_started = time.perf_counter_ns()
    graph_outcomes = project_graph_decode(
        runtime,
        operations,
        requests,
        seq_lens,
        tasks,
        starts,
        graph_greedy,
        scope,
    )
    if graph_outcomes is not None:
        runtime.record_component(scope, "text_sample", sample_started)
        return graph_outcomes
    logits = tuple(token_logits(output)[-1] for output in outputs)
    sample_tasks = tuple(
        build_sample_work(
            runtime,
            operation,
            row_logits,
            request,
            scope,
            positions=(start + 1,),
            request_pool_index=sampling.request_pool_index(task),
        )
        for operation, request, task, start, row_logits in zip(
            operations,
            requests,
            tasks,
            starts,
            logits,
            strict=True,
        )
    )
    sampling_reads = tuple(scope.device_reads)
    samples = (
        sampling.sample_device_greedy_group(
            sample_tasks,
            scope.completion,
            apply_suppression=False,
            device_products=runtime.device_products,
            device_reads=sampling_reads,
            selection_broadcast=runtime.broadcast_tp_selection,
            preselected=graph_greedy,
        )
        if sampling.graph_greedy_compatible(sample_tasks, tasks, graph_greedy)
        else sampling.sample(
            sample_tasks,
            scope.completion,
            device_products=runtime.device_products,
            device_reads=sampling_reads,
            selection_broadcast=runtime.broadcast_tp_selection,
        )
    )
    runtime.record_component(scope, "text_sample", sample_started)
    finalize_started = time.perf_counter_ns()
    publish_token_products(runtime, operations, samples, scope)
    outcomes: list[Outcome] = []
    for operation, request, seq_len, task, start, sampled in zip(
        operations,
        requests,
        seq_lens,
        tasks,
        starts,
        samples,
        strict=True,
    ):
        request.rng_counter += 1
        request.logical_position = start + 1
        # Keep the sampled token in the output row: materializing it here (``int()``)
        # blocks on the copy event and stalls the decode pipeline. It is
        # finalized when the response is serialized, after the next forward
        # has launched.
        outcomes.append(
            token_outcome(
                runtime,
                operation,
                scope,
                request=request,
                task=task,
                base=start,
                tokens=1,
                sampling=sampled.completion,
                sample=sampled,
            )
        )
    publish_runtime_samples(
        runtime,
        operations,
        requests,
        samples,
        scope=scope,
        sample_tasks=sample_tasks,
        logical_positions=tuple(start + 1 for start in starts),
        sampling_positions=tuple(request.rng_counter for request in requests),
        decode_increment=True,
    )
    runtime.record_component(scope, "text_finalize", finalize_started)
    return tuple(outcomes)


def _decode_forward_tasks(
    runtime: Worker,
    operations: tuple[Operation, ...],
    requests: tuple[RequestDraft, ...],
    seq_lens: tuple[int, ...],
    weights: tuple[WeightSet, ...],
    starts: list[int],
    scope: LaneState,
) -> tuple[ForwardRow, ...]:
    """Build paged-decode forward rows and stage their runtime-owned scalar columns."""

    states = runtime.runtime_states
    page_tables = runtime.req_to_token_pool
    if page_tables is None:
        raise RuntimeError("token decode requires request page tables")
    predicates = tuple(
        scope.predicate_values.get(runtime.operation_identity(operation))
        for operation in operations
    )
    request_indexed = (
        states is not None
        and states.device.type == "cuda"
        and page_tables.page_tables.device == states.device
        and all(value is not None and value[1] for value in predicates)
    )
    if request_indexed:
        assert states is not None
        placeholder = states.future_input_tokens[0, :1]
        tasks: list[ForwardRow] = []
        for operation, request, seq_len, weight, predicate in zip(
            operations,
            requests,
            seq_lens,
            weights,
            predicates,
            strict=True,
        ):
            if request.request.sampling is None or predicate is None:
                raise RuntimeError("request-indexed decode lost its aligned row state")
            sampling_state = scope.sampling_states.get(
                runtime.operation_identity(operation), SamplingState()
            )
            tasks.append(
                ForwardRow(
                    operation=operation,
                    request=request,
                    weights=weight,
                    phase=ModelPhase.TEXT,
                    token_ids=placeholder,
                    positions=placeholder,
                    selection=TokenSelection.LAST_LOGITS,
                    request_pool_idx=int(request.request.request_pool_idx),
                    seq_len=int(seq_len),
                    write_kv=True,
                    causal=True,
                    decode_predicate=predicate[0],
                    decode_predicate_tagged=True,
                    decode_force_finish=bool(sampling_state.force_finish),
                    request_indexed_decode=True,
                )
            )
        return tuple(tasks)

    current_tokens = resolve_decode_tokens(runtime, operations, scope)
    position_values = torch.tensor(starts, dtype=torch.long)
    return tuple(
        token_task(
            runtime,
            operation,
            request,
            (current,),
            position_values[index : index + 1],
            TokenSelection.LAST_LOGITS,
            scope,
            seq_len=seq_len,
            weights=weight,
        )
        for index, (operation, request, seq_len, weight, current) in enumerate(
            zip(
                operations,
                requests,
                seq_lens,
                weights,
                current_tokens,
                strict=True,
            )
        )
    )


def project_graph_decode(
    runtime: Worker,
    operations: tuple[Operation, ...],
    requests: tuple[RequestDraft, ...],
    seq_lens: tuple[int, ...],
    tasks: tuple[ForwardRow, ...],
    starts: list[int],
    output: GraphGreedyOutput | None,
    scope: LaneState,
) -> tuple[Outcome, ...] | None:
    """Project a captured greedy decision directly into request state."""

    if output is None:
        return None
    count = len(operations)
    columns = (requests, seq_lens, tasks, starts)
    vectors = (
        output.request_pool_indices,
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
        or any(int(value.numel()) != count for value in vectors)
        or int(output.completion.numel()) != sampling.SAMPLING_COMPLETION_FIELDS * count
    ):
        return None

    token_writes: list[DeviceProductWrite] = []
    for operation, request, task in zip(operations, requests, tasks, strict=True):
        parameters = request.request.sampling
        state = scope.sampling_states.get(runtime.operation_identity(operation), SamplingState())
        finish_token_ids = (
            request.request.finish_token_ids
            if not state.finish_token_ids
            else state.finish_token_ids
            if not request.request.finish_token_ids
            else tuple(sorted({*request.request.finish_token_ids, *state.finish_token_ids}))
        )
        write = scope.token_writes.get(runtime.operation_identity(operation))
        if (
            parameters is None
            or not sampling.device_greedy_parameters(parameters)
            or parameters.allowed_token_ids is not None
            or bool(parameters.forced_token_ids)
            or state.allowed_token_ids is not None
            or bool(state.suppressed_token_ids)
            or bool(finish_token_ids)
            or bool(state.transition_token_ids)
            or runtime.operation_identity(operation) in scope.transition_writes
            or task.decode_predicate is None
            or not task.decode_predicate_tagged
            or bool(state.force_finish) != bool(task.decode_force_finish)
            or write is None
        ):
            return None
        token_writes.append(write)

    token_batch = runtime.device_products.producer_scalar_batch(tuple(token_writes))
    runtime.broadcast_tp_selection(output.tokens)
    span = sampling.capture_preselected_span(output, scope.completion)
    if token_batch is not None:
        token_batch.tensor.copy_(output.tagged_tokens)
        runtime.device_products.publish_scalar_group(
            (token_batch,),
            after_reads=tuple(scope.device_reads),
        )
    else:
        sampling.publish_device_writes(
            tuple(token_writes),
            output.tagged_tokens,
            runtime.device_products,
            tuple(scope.device_reads),
        )
    states = runtime.runtime_states
    if states is None:
        raise RuntimeError("graph decode has no request runtime-state owner")
    slots = tuple(int(request.request.request_pool_idx) for request in requests)
    if any(
        operation.request_key != request.request.request_key
        for operation, request in zip(
            operations,
            requests,
            strict=True,
        )
    ):
        raise RuntimeError("graph decode crossed request rows")
    scope.runtime_publications.append(
        DecodeRuntimePublication(
            slots=slots,
            device_slots=output.request_pool_indices,
            tokens=output.tokens,
            predicates=output.continuation,
            selected_points=None,
            penalty_bases=(None,) * count,
            valid=output.valid,
            active=output.active,
        )
    )

    outcomes: list[Outcome] = []
    for index, (operation, request, seq_len, task, start) in enumerate(
        zip(operations, requests, seq_lens, tasks, starts, strict=True)
    ):
        request.rng_counter += 1
        request.logical_position = int(start) + 1
        cache = runtime.cache_coordinates(operation, scope)
        cache = (cache[0], cache[1], int(seq_len) + 1, cache[3])
        lengths = runtime.logical_lengths(
            operation,
            request,
            cache,
            computed_len=int(task.seq_len) + 1,
        )
        outcomes.append(
            Outcome(
                status=OpStatus.OK,
                selected_point=1,
                logical_lengths=replace(lengths, token_len=request.logical_position),
                token_span=TokenSpan(base=int(start), len=1),
                finish_flags=FinishFlags(),
                product_generations=runtime.output_generations(operation),
                sampling=SamplingOutputRow(span, index),
            )
        )
    return tuple(outcomes)


def prompt_logprob_details(
    runtime: Worker,
    request: RequestDraft,
    start: int,
    tokens: torch.Tensor,
    logits: torch.Tensor,
    scope: LaneState,
) -> tuple[LogprobOutputRow, ...]:
    """Create per-position log-probability rows for a prompt logits tensor."""

    tokens = tokens.reshape(-1).to(device=logits.device, dtype=torch.long)
    if logits.ndim != 2 or int(logits.shape[0]) != int(tokens.numel()):
        raise invalid_descriptor("prompt scoring logits do not align with input tokens")
    states = runtime.runtime_states
    if states is None:
        raise unsupported_setup("prompt scoring has no request-indexed runtime state")
    slot = int(request.request.request_pool_idx)
    if start == 0:
        score_logits = logits[:-1]
        targets = tokens[1:]
    else:
        if not request.prompt_logits_ready:
            raise invalid_descriptor("continued prompt scoring has no preceding logits")
        pending = next(
            (
                publication.logits
                for publication in reversed(scope.prompt_logits_publications)
                if publication.slot == slot
            ),
            states.prompt_logits[slot],
        )
        previous = pending.reshape(1, -1).to(
            device=logits.device,
            dtype=logits.dtype,
        )
        score_logits = torch.cat((previous, logits[:-1]), dim=0)
        targets = tokens
    scope.prompt_logits_publications.append(
        PromptLogitsPublication(slot=slot, logits=logits[-1].detach())
    )
    request.prompt_logits_ready = True
    if int(targets.numel()) == 0:
        return ()
    parameters = require_sampling(request)
    prompt_parameters = replace(
        parameters,
        return_logprobs=True,
        n_logprobs=int(parameters.n_prompt_logprobs),
    )
    rows = tuple(
        SampleRow(
            parameters=prompt_parameters,
            penalty_counts=None,
            allowed=None,
            suppress=(),
            draw=0.0,
            n_logprobs=int(parameters.n_prompt_logprobs),
        )
        for _ in range(int(targets.numel()))
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
        rows,
        scope.completion,
    )
    return tuple(details[index] for index in range(int(targets.numel())))


def token_outcome(
    runtime: Worker,
    operation: Operation,
    scope: LaneState,
    *,
    request: RequestDraft | None = None,
    task: ForwardRow | None = None,
    base: int,
    tokens: int,
    committed_tokens: tuple[int, ...] = (),
    sampling: SamplingOutputRow | None = None,
    sample: SampleResult | None = None,
    selection: SpeculativeSelection | None = None,
) -> Outcome:
    """Record logical lengths and pending payloads for one autoregressive completion."""

    if request is None:
        request = runtime.request_row(scope, operation.request_key.request_id)
    cache = runtime.cache_coordinates(operation, scope)
    initialized = cache[2]
    if selection is None:
        published_length = scope.runtime_cache_lengths.get(
            cache[0],
            int(task.seq_len) + int(tokens) if task is not None else cache[2],
        )
        if isinstance(published_length, torch.Tensor):
            raise RuntimeError("dynamic KV length requires a speculative selection")
        visible_value = int(published_length)
        initialized = visible_value
    else:
        visible_value = int(selection.base_kv_visible)
        initialized = selection.initialized_kv
    lengths = runtime.logical_lengths(
        operation,
        request,
        (cache[0], cache[1], cache[2] if selection is not None else int(visible_value), cache[3]),
        computed_len=initialized,
    )
    visible = visible_value
    token_len = request.logical_position if selection is None else selection.base_logical_position
    if sample is not None and selection is not None:
        _publish_selection_products(
            runtime,
            operation,
            sample,
            scope,
        )
    return Outcome(
        status=OpStatus.OK,
        selected_point=(1 if selection is None else 0),
        logical_lengths=LogicalLengths(
            token_len=token_len,
            kv_visible_len=visible,
            kv_computed_len=lengths.kv_computed_len,
        ),
        token_span=TokenSpan(base=base, len=tokens),
        finish_flags=FinishFlags(),
        product_generations=runtime.output_generations(operation),
        committed_tokens=committed_tokens,
        sampling=sampling,
        products=sample_product_payloads(operation, sample),
        selection=selection,
    )


def token_task(
    runtime: Worker,
    operation: Operation,
    request: RequestDraft,
    token_ids: tuple[int | torch.Tensor, ...],
    positions: tuple[int, ...] | torch.Tensor,
    selection: TokenSelection,
    scope: LaneState,
    *,
    seq_len: int | None = None,
    weights: WeightSet | None = None,
) -> ForwardRow:
    """Build one staged autoregressive forward row from request runtime and token coordinates."""

    if len(token_ids) != len(positions) or not token_ids:
        raise invalid_descriptor("token task ids and positions must align")
    if len(token_ids) == 1 and isinstance(token_ids[0], torch.Tensor):
        token_values = token_ids[0].reshape(1).to(dtype=torch.long)
    else:
        token_values = torch.tensor(
            tuple(int(value) for value in token_ids),
            dtype=torch.long,
        )
    predicate_value = scope.predicate_values.get(runtime.operation_identity(operation))
    sampling_state = scope.sampling_states.get(
        runtime.operation_identity(operation), SamplingState()
    )
    cache = runtime.cache_coordinates(operation, scope)
    visible = cache[2] if seq_len is None else int(seq_len)
    if visible != cache[2]:
        raise invalid_descriptor("token row visibility disagrees with operation metadata")
    return ForwardRow(
        operation=operation,
        request=request,
        weights=runtime.weights if weights is None else weights,
        phase=ModelPhase.TEXT,
        token_ids=token_values,
        positions=(
            positions.reshape(-1).to(dtype=torch.long)
            if isinstance(positions, torch.Tensor)
            else torch.tensor(positions, dtype=torch.long)
        ),
        selection=selection,
        request_pool_idx=cache[0],
        seq_len=visible,
        group_id=cache[1],
        write_kv=True,
        causal=True,
        decode_predicate=None if predicate_value is None else predicate_value[0],
        decode_predicate_tagged=False if predicate_value is None else predicate_value[1],
        decode_force_finish=bool(sampling_state.force_finish),
    )


def commit_kv(
    runtime: Worker,
    task: ForwardRow,
    tokens: int,
    scope: LaneState,
    *,
    publish_runtime: bool = True,
) -> None:
    """Advance computed KV length and optionally publish the updated request runtime."""

    count = int(tokens)
    if count < 0 or count > task.query_tokens:
        raise RuntimeError("KV commit count is outside the task query span")
    if count == 0:
        return
    resulting = int(task.seq_len) + count
    page_tables = runtime.req_to_token_pool
    if page_tables is None:
        raise RuntimeError("KV commit requires request page tables")
    if resulting > page_tables.allocated_length(task.request_pool_idx):
        raise RuntimeError("KV task exceeds its scheduler block table")
    if publish_runtime and runtime.runtime_states is not None:
        scope.runtime_cache_lengths[int(task.request_pool_idx)] = resulting


def operation_token_ids(
    runtime: Worker,
    operation: Operation,
    scope: LaneState,
) -> tuple[int, ...]:
    """Read the token id values a token operation names as an input product.

    Host-known prompt and draft tokens arrive through a declared host-staging
    product. Device-rooted decode tokens are read from request-indexed runtime
    state.
    """

    for reference in operation.inputs:
        if reference.kind is not ProductKind.TOKEN:
            continue
        values = scope.input_tokens.get(reference)
        if values is not None:
            return values
    raise invalid_descriptor("token operation has no input token product")


def resolve_decode_token(
    runtime: Worker,
    operation: Operation,
    request: RequestDraft,
    scope: LaneState,
) -> int | torch.Tensor:
    """Resolve one decode input token from an explicit value or device relay product."""

    point = operation.state_parent.point
    if isinstance(point, DeviceSelected):
        predicate = scope.predicate_values.get(runtime.operation_identity(operation))
        if predicate is None:
            raise invalid_descriptor("device token continuation is not registered")
        states = runtime.runtime_states
        if states is None:
            raise unsupported_setup("device continuation has no request runtime state")
        slot = int(request.request.request_pool_idx)
        pending = _pending_runtime_token(runtime, slot, scope)
        if pending is not None:
            return pending.reshape(-1)[:1].bitwise_and(sampling.TOKEN_VALUE_MASK)
        return states.future_input_tokens[slot, :1]
    tokens = operation_token_ids(runtime, operation, scope)
    if not tokens:
        raise invalid_descriptor("last-sampled token source has no committed token")
    return int(tokens[0])


def resolve_decode_tokens(
    runtime: Worker,
    operations: tuple[Operation, ...],
    scope: LaneState,
) -> tuple[int | torch.Tensor, ...]:
    """Resolve and validate decode tokens for a batch of operations."""

    resolved: list[int | torch.Tensor | None] = [None] * len(operations)
    for index, operation in enumerate(operations):
        point = operation.state_parent.point
        if isinstance(point, DeviceSelected):
            predicate = scope.predicate_values.get(runtime.operation_identity(operation))
            if predicate is None:
                raise invalid_descriptor("device token continuation is not registered")
            states = runtime.runtime_states
            if states is None:
                raise unsupported_setup("device continuation has no request runtime state")
            request = runtime.request_row(scope, operation.request_key.request_id)
            slot = int(request.request.request_pool_idx)
            pending = _pending_runtime_token(runtime, slot, scope)
            resolved[index] = (
                states.future_input_tokens[slot, :1]
                if pending is None
                else pending.reshape(-1)[:1].bitwise_and(sampling.TOKEN_VALUE_MASK)
            )
            continue
        tokens = operation_token_ids(runtime, operation, scope)
        if not tokens:
            raise invalid_descriptor("last-sampled token source has no committed token")
        resolved[index] = int(tokens[0])
    if any(value is None for value in resolved):
        raise RuntimeError("decode token resolution left an operation without input")
    return tuple(cast(int | torch.Tensor, value) for value in resolved)


def _pending_runtime_token(
    runtime: Worker,
    slot: int,
    scope: LaneState,
) -> torch.Tensor | None:
    """Read the staged future token for a request while tracking its device lease."""

    for publication in reversed(scope.runtime_publications):
        if isinstance(publication, RuntimePublication):
            if publication.slot == slot:
                return publication.token
            continue
        try:
            index = publication.slots.index(slot)
        except ValueError:
            continue
        return publication.tokens[index : index + 1]
    return None


def publish_token_product(
    runtime: Worker,
    operation: Operation,
    sample: SampleResult,
    scope: LaneState,
) -> None:
    """Publish a sampled token as a device product when declared by the operation."""

    if sample.device_product_published:
        return
    write = scope.token_writes.get(runtime.operation_identity(operation))
    if write is None:
        return
    device_token = sample.device_token
    if device_token is None:
        device_token = torch.tensor(
            sample.completion.materialize()[0][-1:],
            dtype=torch.long,
            device=runtime.operation_device(operation),
        )
    continuation = sample.device_continuation
    if continuation is None:
        continuation = torch.ones_like(device_token, dtype=torch.bool)
    runtime.device_products.publish_write(
        write,
        sampling.tagged_token_values(device_token, continuation),
    )


def publish_runtime_samples(
    runtime: Worker,
    operations: Sequence[Operation],
    requests: Sequence[RequestDraft],
    samples: Sequence[SampleResult],
    *,
    scope: LaneState,
    sample_tasks: Sequence[SampleWork],
    logical_positions: Sequence[int | torch.Tensor],
    sampling_positions: Sequence[int | torch.Tensor],
    decode_increment: bool = False,
) -> None:
    """Commit device-selected token transitions into request runtime storage and semantic history."""

    states = runtime.runtime_states
    if states is None:
        return
    columns = (
        operations,
        requests,
        samples,
        sample_tasks,
        logical_positions,
        sampling_positions,
    )
    if len({len(values) for values in columns}) != 1:
        raise RuntimeError("runtime sampling publication columns are not aligned")
    if decode_increment:
        if any(
            operation.request_key != request.request.request_key
            for operation, request in zip(operations, requests, strict=True)
        ):
            raise RuntimeError("runtime sampling publication crossed request rows")
        batch_vectors = samples[0].device_batch if samples else None
        if batch_vectors is not None and all(
            sample.device_batch is batch_vectors and sample.device_batch_index == index
            for index, sample in enumerate(samples)
        ):
            if (
                int(batch_vectors.tokens.numel()) != len(samples)
                or int(batch_vectors.valid.numel()) != len(samples)
                or int(batch_vectors.active.numel()) != len(samples)
                or int(batch_vectors.continuation.numel()) != len(samples)
                or int(batch_vectors.request_pool_indices.numel()) != len(samples)
                or len(batch_vectors.penalty_bases) != len(samples)
            ):
                raise RuntimeError("batched runtime sampling publication is not row-aligned")
            scope.runtime_publications.append(
                DecodeRuntimePublication(
                    slots=tuple(int(request.request.request_pool_idx) for request in requests),
                    device_slots=batch_vectors.request_pool_indices,
                    tokens=batch_vectors.tokens,
                    predicates=batch_vectors.continuation,
                    selected_points=batch_vectors.selected_points,
                    penalty_bases=batch_vectors.penalty_bases,
                    valid=batch_vectors.valid,
                    active=batch_vectors.active,
                )
            )
            return
        selected_values = tuple(sample.device_selected_point for sample in samples)
        accepted_values = tuple(sample.device_accepted_tokens for sample in samples)
        selected_points = (
            None
            if all(value is None for value in (*selected_values, *accepted_values))
            else sampling.runtime_selected_points(selected_values, accepted_values)
        )
        scope.runtime_publications.append(
            DecodeRuntimePublication(
                slots=tuple(int(request.request.request_pool_idx) for request in requests),
                device_slots=sampling.sample_request_pool_indices(sample_tasks),
                tokens=sampling.sample_result_vector(samples, "device_token"),
                predicates=sampling.sample_result_vector(samples, "device_continuation"),
                selected_points=selected_points,
                penalty_bases=tuple(task.penalty_base for task in sample_tasks),
                valid=sampling.sample_result_vector(samples, "device_valid"),
                active=sampling.sample_result_vector(samples, "device_active"),
            )
        )
        return
    for operation, request, sample, sample_task, logical, sampling_position in zip(
        operations,
        requests,
        samples,
        sample_tasks,
        logical_positions,
        sampling_positions,
        strict=True,
    ):
        if operation.request_key != request.request.request_key:
            raise RuntimeError("runtime sampling publication crossed request rows")
        token = sample.device_token
        predicate = sample.device_continuation
        valid = sample.device_valid
        active = sample.device_active
        if token is None or predicate is None or valid is None or active is None:
            raise RuntimeError("runtime sampling publication lost device state")
        selected = sample.device_selected_point
        if selected is None:
            accepted = sample.device_accepted_tokens
            selected = (
                torch.ones_like(token, dtype=torch.int32)
                if accepted is None
                else accepted.to(dtype=torch.int32) + 1
            )
        slot = int(request.request.request_pool_idx)
        scope.runtime_publications.append(
            RuntimePublication(
                slot=slot,
                token=token,
                predicate=predicate,
                selected_point=selected,
                logical_position=logical,
                sampling_position=sampling_position,
                penalty_base=sample_task.penalty_base,
                valid=valid,
                active=active,
            )
        )


def publish_token_products(
    runtime: Worker,
    operations: tuple[Operation, ...],
    samples: tuple[SampleResult, ...],
    scope: LaneState,
) -> None:
    """Publish sampled token and selected-checkpoint products for all operations in a group."""

    if all(sample.device_product_published for sample in samples):
        return
    writes: list[DeviceProductWrite] = []
    device_tokens: list[torch.Tensor] = []
    for operation, sample in zip(operations, samples, strict=True):
        write = scope.token_writes.get(runtime.operation_identity(operation))
        if write is None or sample.device_token is None:
            for candidate_operation, candidate_sample in zip(
                operations,
                samples,
                strict=True,
            ):
                publish_token_product(
                    runtime,
                    candidate_operation,
                    candidate_sample,
                    scope,
                )
            return
        writes.append(write)
        device_tokens.append(sample.device_token)
    packed = packed_tensor_views(tuple(device_tokens))
    if packed is None:
        for operation, sample in zip(operations, samples, strict=True):
            publish_token_product(runtime, operation, sample, scope)
        return
    continuation_flags = tuple(
        sample.device_continuation for sample in samples if sample.device_continuation is not None
    )
    if len(continuation_flags) != len(samples):
        raise RuntimeError("sampled token publication lost its continuation state")
    packed_flags = packed_tensor_views(continuation_flags)
    if packed_flags is None:
        packed_flags = torch.cat(continuation_flags, dim=0)
    runtime.device_products.publish_writes(
        tuple(writes),
        sampling.tagged_token_values(packed, packed_flags),
    )


def _publish_selection_products(
    runtime: Worker,
    operation: Operation,
    sample: SampleResult,
    scope: LaneState,
) -> None:
    """Publish sampled token and speculative selection products from one result."""

    device_token = sample.device_token
    if device_token is None:
        raise RuntimeError("device selection products require a device token")
    accepted = sample.device_accepted_tokens
    if accepted is None:
        accepted = torch.zeros_like(device_token, dtype=torch.long)
    selected_point = sample.device_selected_point
    if selected_point is None:
        selected_point = accepted.to(dtype=torch.long) + 1
    operation_identity = runtime.operation_identity(operation)
    selected_write = scope.selected_point_writes.get(operation_identity)
    if selected_write is None:
        raise invalid_descriptor("token operation is missing its selected-point product")
    runtime.device_products.publish_write(selected_write, selected_point)


def build_sample_work(
    runtime: Worker,
    operation: Operation,
    logits: torch.Tensor,
    request: RequestDraft,
    scope: LaneState,
    *,
    positions: tuple[int, ...],
    request_pool_index: torch.Tensor,
    draft_token_ids: tuple[int, ...] = (),
) -> SampleWork:
    """Build sampling rows, penalties, RNG coordinates, predicates, and device-product bindings."""

    parameters = require_sampling(request)
    state = scope.sampling_states.get(runtime.operation_identity(operation), SamplingState())
    allowed_token_ids = (
        state.allowed_token_ids
        if state.allowed_token_ids is not None
        else parameters.allowed_token_ids
    )
    if not state.finish_token_ids:
        finish_token_ids = request.request.finish_token_ids
    elif not request.request.finish_token_ids:
        finish_token_ids = state.finish_token_ids
    else:
        finish_token_ids = tuple(
            sorted(
                {
                    *request.request.finish_token_ids,
                    *state.finish_token_ids,
                }
            )
        )
    rng = operation.rng
    if float(parameters.temperature) > 0.0:
        if rng is None or rng.draw_layout is not DrawLayout.TARGET_SAMPLING:
            raise invalid_descriptor("stochastic sampling requires target-sampling RNG coordinates")
        if int(rng.seed) != int(parameters.seed or 0):
            raise invalid_descriptor("operation RNG seed disagrees with admitted sampling")
        expected_positions = tuple(
            range(int(rng.semantic_index_base), int(rng.semantic_index_base) + len(positions))
        )
        if positions != expected_positions:
            raise invalid_descriptor(
                "sampling positions disagree with registered semantic RNG coordinates"
            )
    rng_seed = 0 if rng is None else int(rng.seed)
    stochastic = float(parameters.temperature) > 0.0
    draw_key = (
        sampling_key(
            rng_seed,
            int(operation.request_key.authority_id),
            int(operation.request_key.request_id),
            int(operation.request_key.epoch),
            DRAW_LAYOUT_TARGET,
        )
        if stochastic
        else 0
    )
    rows = logits.reshape(1, -1) if logits.ndim == 1 else logits
    if rows.ndim != 2 or int(rows.shape[0]) != len(positions):
        raise invalid_descriptor("sampling task positions do not align with its logits")
    vocab = int(rows.shape[1])
    uses_penalties = (
        parameters.repetition_penalty != 1.0
        or parameters.frequency_penalty != 0.0
        or parameters.presence_penalty != 0.0
    )
    penalty_base = (
        _request_penalty_base(runtime, request, vocab, rows.device) if uses_penalties else None
    )
    penalty_view = (
        None
        if penalty_base is None
        else _candidate_penalty_counts(runtime, request, penalty_base, scope)
    )
    forced_token_ids = parameters.forced_token_ids
    descriptors: list[SampleRow] = []
    for index, position in enumerate(positions):
        if penalty_view is None:
            row_counts = None
        elif index == 0 or not draft_token_ids:
            row_counts = penalty_view
        else:
            row_counts = penalty_view.clone()
            for token_id in draft_token_ids[:index]:
                row_counts[int(token_id)] += 1
        # Processor step 2 forced-token constraint: point `index` of the
        # operation's span narrows selection to `forced_token_ids[index]`,
        # overriding any allowed-token whitelist for that point.
        row_allowed = (
            (int(forced_token_ids[index]),) if index < len(forced_token_ids) else allowed_token_ids
        )
        descriptors.append(
            SampleRow(
                parameters=parameters,
                penalty_counts=row_counts,
                allowed=row_allowed,
                suppress=state.suppressed_token_ids,
                finish_token_ids=finish_token_ids,
                transition_token_ids=state.transition_token_ids,
                force_finish=state.force_finish,
                draw=(sampling_uniform(draw_key, int(position)) if stochastic else 0.0),
                n_logprobs=int(parameters.n_logprobs),
            )
        )
    descriptor_rows = tuple(descriptors)
    operation_identity = runtime.operation_identity(operation)
    device_greedy = not draft_token_ids and all(
        sampling.device_greedy_row(row) for row in descriptor_rows
    )
    if device_greedy:
        draws = None
        penalty_token_ids = None
        penalty_counts = None
        parameter_values = None
    else:
        draws = sampling.semantic_sampling_draws(
            descriptor_rows,
            device=rows.device,
        )
        penalty_token_ids, penalty_counts, parameter_values = sampling.sampling_task_tensors(
            descriptor_rows,
            vocab=vocab,
            device=rows.device,
        )
    token_product = scope.token_writes.get(operation_identity)
    transition_product = scope.transition_writes.get(operation_identity)
    predicate_value = scope.predicate_values.get(operation_identity)
    finish_set = set(descriptor_rows[0].finish_token_ids)
    terminal_draft_prefix = next(
        (index + 1 for index, token_id in enumerate(draft_token_ids) if token_id in finish_set),
        None,
    )
    return SampleWork(
        operation=operation,
        logits=rows,
        rows=descriptor_rows,
        draws=draws,
        penalty_token_ids=penalty_token_ids,
        penalty_counts=penalty_counts,
        parameter_values=parameter_values,
        draft_token_ids=tuple(int(value) for value in draft_token_ids),
        terminal_draft_prefix=terminal_draft_prefix,
        token_product=token_product,
        transition_product=transition_product,
        predicate=None if predicate_value is None else predicate_value[0],
        tagged_predicate=False if predicate_value is None else predicate_value[1],
        request_pool_index=request_pool_index,
        penalty_base=penalty_base,
    )


def _request_penalty_base(
    runtime: Worker,
    request: RequestDraft,
    vocab: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the fixed request-indexed committed penalty-count row."""

    states = runtime.runtime_states
    if states is None:
        raise RuntimeError("token sampling has no request runtime-state owner")
    if states.vocab_size != int(vocab) or states.device != device:
        raise unsupported_setup("sampling geometry disagrees with request runtime state")
    return states.penalty_counts[int(request.request.request_pool_idx)]


def _candidate_penalty_counts(
    runtime: Worker,
    request: RequestDraft,
    committed: torch.Tensor,
    scope: LaneState,
) -> torch.Tensor:
    """Build penalty counts that combine committed history with speculative candidates."""

    slot = int(request.request.request_pool_idx)
    counts = committed.clone()
    found = False
    for publication in scope.runtime_publications:
        if isinstance(publication, RuntimePublication):
            if publication.slot != slot:
                continue
            token = publication.token.reshape(-1)[:1]
            valid = publication.valid.reshape(-1)[:1]
            active = publication.active.reshape(-1)[:1]
        else:
            try:
                index = publication.slots.index(slot)
            except ValueError:
                continue
            token = publication.tokens[index : index + 1]
            valid = publication.valid[index : index + 1]
            active = publication.active[index : index + 1]
        found = True
        token = token.bitwise_and(sampling.TOKEN_VALUE_MASK)
        weight = (valid & active).to(dtype=counts.dtype)
        counts.scatter_add_(0, token.to(dtype=torch.int64), weight)
    return counts if found else committed


def _logprob_product_ref(operation: Operation) -> ProductRef | None:
    """Return the declared log-probability product for an autoregressive operation."""

    matches = tuple(output for output in operation.outputs if output.kind is ProductKind.LOGPROB)
    if len(matches) > 1:
        raise invalid_descriptor("operation declares multiple logprob products")
    return matches[0] if matches else None


def sample_product_payloads(
    operation: Operation,
    sample: SampleResult | None,
) -> tuple[ProductPayload, ...]:
    """Return wire payloads for declared token and selected-point products."""

    if sample is None or (sample.logprobs is None and not sample.prompt_logprobs):
        return ()
    reference = _logprob_product_ref(operation)
    if reference is None:
        raise invalid_descriptor("sampler produced undeclared logprob output")
    payload = LogprobPayload(
        sample.logprobs,
        sample.prompt_logprobs,
    )
    return (ProductPayload(product=reference, payload=cast(bytes, payload)),)


def token_logits(output: torch.Tensor) -> torch.Tensor:
    """Normalize model output to a two-dimensional token-by-vocabulary logits tensor."""

    if not isinstance(output, torch.Tensor) or output.ndim < 2:
        raise invalid_descriptor("token route did not return logits")
    return output


def require_sampling(request: RequestDraft) -> SamplingParams:
    """Return the sampling parameters required by an autoregressive request."""

    if request.request.sampling is None:
        raise invalid_descriptor("sequence execution requires admitted sampling parameters")
    return request.request.sampling


def token_logits_or_hidden(output: torch.Tensor) -> torch.Tensor:
    """Normalize model output to a two-dimensional token-by-feature tensor."""

    if not isinstance(output, torch.Tensor) or output.ndim < 2:
        raise invalid_descriptor("token route did not return a token tensor")
    return output


__all__ = ["consume_forward", "consume_sample", "decode_batch", "pack_forward", "pack_sample"]
