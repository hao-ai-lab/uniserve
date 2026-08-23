"""Token and visual-state packing and publication."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from typing import cast

import torch

from uniserve_worker.batch import (
    DevicePoint,
    DrawLayout,
    FinishFlags,
    LogicalLengths,
    Operation,
    OpStatus,
    ProductKind,
    ProductPayload,
    ProductRef,
    SamplingParams,
    SamplingState,
    TokenMode,
    TokenSpan,
    WorkVariant,
)
from uniserve_worker.foundation.errors import capability_mismatch, invalid_descriptor
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.runtime.cache_pool import CacheRow
from uniserve_worker.runtime.device_products import (
    DeviceProductScalarBatch,
    DeviceProductWrite,
)
from uniserve_worker.server.completion import (
    _CompletionDerivedInteger,
    _CompletionInteger,
    _CompletionLogprobPayload,
    _CompletionSampleToken,
    _CompletionSpeculativePoint,
    _CompletionSpeculativeTokens,
    _CompletionToken,
    _CompletionTopLogprobs,
)
from uniserve_worker.server.request_state import RequestRow

from . import encode
from . import sample as sampling
from . import step as ops
from .cuda_graph import GraphGreedyOutput
from .forward_batch import ModelPhase, TokenSelection, packed_tensor_views
from .rng import DRAW_LAYOUT_TARGET, sampling_key, sampling_uniform
from .rows import (
    DecodeRuntimePublication,
    ForwardRow,
    OperationState,
    Outcome,
    PartitionState,
    PromptLogitsPublication,
    RuntimePublication,
    SampleResult,
    SampleRow,
    SampleWork,
    SpeculativeSelection,
    StateOutcome,
)


def pack_forward(runtime: object, state: OperationState) -> tuple[object, ...]:
    if state.phase != "initial":
        return ()
    from . import step as ops

    operation = state.operation
    partition = state.partition
    session = ops._request_row(runtime, partition, operation.request_key.session_id)
    if session.sampling is None:
        raise invalid_descriptor("sequence operation has no admitted sampling state")
    mode = operation.work.mode
    if mode == TokenMode.EXTEND.value and any(
        reference.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}
        for reference in operation.inputs
    ):
        return _pack_visual(runtime, state, session)
    start = int(session.logical_position)
    if mode == TokenMode.EXTEND.value:
        if isinstance(operation.parent.point, DevicePoint) and not any(
            reference.kind is ProductKind.TOKEN for reference in operation.inputs
        ):
            tokens = (resolve_decode_token(runtime, operation, session, partition),)
        else:
            tokens = operation_token_ids(runtime, operation, partition)
        sampling = require_sampling(session)
        scores_prompt = bool(sampling.return_prompt_logprobs or int(sampling.n_prompt_logprobs) > 0)
        task = token_task(
            runtime,
            operation,
            session,
            tokens,
            tuple(range(start, start + len(tokens))),
            TokenSelection.ALL_LOGITS if scores_prompt else TokenSelection.LAST_LOGITS,
            partition,
        )
        state.data.update(
            session=session,
            start=start,
            tokens=tokens,
            task=task,
            scores_prompt=scores_prompt,
            mode="extend",
        )
    elif mode == TokenMode.DECODE.value:
        current = resolve_decode_token(runtime, operation, session, partition)
        task = token_task(
            runtime,
            operation,
            session,
            (current,),
            (start,),
            TokenSelection.LAST_LOGITS,
            partition,
        )
        state.data.update(session=session, start=start, task=task, mode="decode")
    else:
        if isinstance(operation.parent.point, DevicePoint):
            current = resolve_decode_token(runtime, operation, session, partition)
            draft = operation_token_ids(runtime, operation, partition)
        else:
            input_tokens = operation_token_ids(runtime, operation, partition)
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
            session,
            tokens,
            tuple(range(start, start + len(tokens))),
            TokenSelection.ALL_LOGITS,
            partition,
        )
        state.data.update(
            session=session,
            start=start,
            draft=draft,
            task=task,
            mode="verify",
        )
    state.rows = (task,)
    state.phase = "forward_pending"
    return state.rows


def consume_forward(
    runtime: object,
    state: OperationState,
    outputs: tuple[torch.Tensor, ...],
) -> None:

    if state.phase != "forward_pending" or len(outputs) != 1:
        raise RuntimeError("token forward result is not aligned")
    operation = state.operation
    partition = state.partition
    task = state.data["task"]
    mode = state.data["mode"]
    session = state.data["session"]
    start = state.data["start"]
    if mode == "visual":
        _consume_visual(runtime, state, outputs[0])
        return
    logits = token_logits(outputs[0])
    if mode == "extend":
        tokens = state.data["tokens"]
        commit_kv(runtime, task, len(tokens), partition)
        if not any(output.kind is ProductKind.TOKEN for output in operation.outputs):
            state.outcome = token_outcome(
                runtime,
                operation,
                partition,
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
            session,
            partition,
            positions=(start + len(tokens),),
            request_pool_index=sampling.request_pool_index(task),
        )
        state.data["logits"] = logits
    elif mode == "decode":
        commit_kv(runtime, task, 1, partition)
        sample = build_sample_work(
            runtime,
            operation,
            logits[-1],
            session,
            partition,
            positions=(start + 1,),
            request_pool_index=sampling.request_pool_index(task),
        )
    else:
        draft = state.data["draft"]
        sample = build_sample_work(
            runtime,
            operation,
            logits,
            session,
            partition,
            positions=tuple(range(start + 1, start + len(draft) + 2)),
            draft_token_ids=draft,
            request_pool_index=sampling.request_pool_index(task),
        )
    state.sample = sample
    state.phase = "sample"


def pack_sample(state: OperationState) -> object | None:
    if state.phase != "sample":
        return None
    state.phase = "sample_pending"
    return state.sample


def consume_sample(runtime: object, state: OperationState, value: object) -> None:

    if state.phase != "sample_pending":
        raise RuntimeError("token sample result has no pending selection")
    sampled = sampling.sample_result(value)
    operation = state.operation
    partition = state.partition
    session = state.data["session"]
    start = state.data["start"]
    task = state.data["task"]
    mode = state.data["mode"]
    sample_work = state.sample
    if mode == "visual":
        publish_token_product(runtime, operation, sampled, partition)
        session.rng_counter += 1
        flow = runtime.model.generation
        logical_position = start + (
            max(1, 1 if flow is None else int(flow.rope_advance))
            if state.data["close_image"]
            else 1
        )
        publish_runtime_samples(
            runtime,
            (operation,),
            (session,),
            (sampled,),
            scope=partition,
            sample_tasks=(sample_work,),
            logical_positions=(logical_position,),
            sampling_positions=(session.rng_counter,),
        )
        state.data["state_outcome"] = StateOutcome(
            (sampled.token_id,), sample_product_payloads(operation, sampled)
        )
        _finish_visual(runtime, state)
        return
    if mode == "extend":
        if state.data["scores_prompt"]:
            sampled = replace(
                sampled,
                prompt_logprobs=prompt_logprob_details(
                    runtime,
                    session,
                    start,
                    cast(torch.Tensor, task.token_ids),
                    state.data["logits"],
                    partition,
                ),
            )
        publish_token_product(runtime, operation, sampled, partition)
        session.rng_counter += 1
        session.logical_position = start + len(state.data["tokens"])
        publish_runtime_samples(
            runtime,
            (operation,),
            (session,),
            (sampled,),
            scope=partition,
            sample_tasks=(sample_work,),
            logical_positions=(session.logical_position,),
            sampling_positions=(session.rng_counter,),
        )
        state.outcome = token_outcome(
            runtime,
            operation,
            partition,
            base=start,
            tokens=len(state.data["tokens"]),
            committed_tokens=(sampled.token_id,),
            sample=sampled,
        )
    elif mode == "decode":
        session.rng_counter += 1
        session.logical_position = start + 1
        publish_token_product(runtime, operation, sampled, partition)
        publish_runtime_samples(
            runtime,
            (operation,),
            (session,),
            (sampled,),
            scope=partition,
            sample_tasks=(sample_work,),
            logical_positions=(session.logical_position,),
            sampling_positions=(session.rng_counter,),
        )
        state.outcome = token_outcome(
            runtime,
            operation,
            partition,
            base=start,
            tokens=1,
            committed_tokens=(sampled.token_id,),
            sample=sampled,
        )
    else:
        draft = state.data["draft"]
        if task.entry is None:
            raise RuntimeError("verification task has no scheduler KV row")
        initialized = task.entry.initialize(task.query_tokens)
        publish_token_product(runtime, operation, sampled, partition)
        accepted = sampled.num_accepted_tokens
        selected_point = _CompletionSpeculativePoint(accepted, sample_work.terminal_draft_prefix)
        committed_tokens = _CompletionSpeculativeTokens(
            draft, accepted, sampled.token_id, sample_work.terminal_draft_prefix
        )
        device_selected = sampled.device_selected_point
        if device_selected is None:
            accepted_device = sampled.device_accepted_tokens
            if accepted_device is None:
                raise RuntimeError("speculative sampling lost its selected point")
            device_selected = accepted_device.to(dtype=torch.int32) + 1
        partition.runtime_cache_lengths[int(session.request_pool_idx)] = device_selected + int(
            initialized - task.query_tokens
        )
        publish_runtime_samples(
            runtime,
            (operation,),
            (session,),
            (sampled,),
            scope=partition,
            sample_tasks=(sample_work,),
            logical_positions=(device_selected + start,),
            sampling_positions=(device_selected + int(session.rng_counter),),
        )
        state.outcome = token_outcome(
            runtime,
            operation,
            partition,
            base=start,
            tokens=selected_point,
            committed_tokens=cast(tuple[object, ...], committed_tokens),
            sample=sampled,
            selection=SpeculativeSelection(
                accepted=accepted,
                selected_point=selected_point,
                draft_tokens=draft,
                terminal_prefix=sample_work.terminal_draft_prefix,
                base_logical_position=start,
                base_rng_counter=session.rng_counter,
                base_kv_visible=initialized - task.query_tokens,
                initialized_kv=initialized,
            ),
        )
    state.phase = "done"


def _pack_visual(runtime: object, state: OperationState, session: object) -> tuple[object, ...]:
    from . import step as ops

    operation = state.operation
    partition = state.partition
    references = tuple(
        reference
        for reference in operation.inputs
        if reference.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}
    )
    if len(references) != 1:
        raise invalid_descriptor("visual extend requires exactly one feature product")
    reference = references[0]
    read = ops._consume_encoder_feature(
        runtime,
        reference,
        partition,
        consumer_op_id=operation.op_id,
        device=ops._operation_device(runtime, operation),
    )
    partition.encoder_reads.append(read)
    position = int(session.logical_position)
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
            partition,
            close_image=close_image,
            logits=sample_token,
        )
        variant = WorkVariant.ENCODE_VISION
    else:
        task = encode.latent_state_row(
            runtime,
            operation,
            read.tensor,
            read.metadata.height,
            read.metadata.width,
            position,
            partition,
        )
        variant = WorkVariant.ENCODE_LATENT
    if task.query_tokens > int(operation.bounds.max_tokens):
        raise invalid_descriptor("image state query span exceeds the operation token bound")
    state.data.update(
        session=session,
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


def _consume_visual(runtime: object, state: OperationState, output: torch.Tensor) -> None:
    from . import flow

    task = state.data["task"]
    partition = state.partition
    if state.data["variant"] is WorkVariant.ENCODE_VISION:
        value = token_logits_or_hidden(output)
    else:
        flow.prediction(output)
        value = None
    commit_kv(runtime, task, task.query_tokens, partition)
    if state.data["sample_token"]:
        assert value is not None
        flow = runtime.model.generation
        sample = build_sample_work(
            runtime,
            state.operation,
            value[-1],
            state.data["session"],
            partition,
            positions=(
                state.data["start"] + max(1, 1 if flow is None else int(flow.rope_advance)),
            ),
            request_pool_index=sampling.request_pool_index(task),
        )
        state.sample = sample
        state.phase = "sample"
        return
    state.data["state_outcome"] = StateOutcome()
    _finish_visual(runtime, state)


def _finish_visual(runtime: object, state: OperationState) -> None:

    session = state.data["session"]
    position = state.data["start"]
    if state.data["close_image"]:
        flow = runtime.model.generation
        session.logical_position = position + max(1, 1 if flow is None else int(flow.rope_advance))
    elif state.data["reference_kind"] is ProductKind.VISION_FEATURE:
        session.logical_position = position + 1
    state.outcome = encode.state_outcome(
        runtime,
        state.operation,
        state.data["state_outcome"],
        state.partition,
        base=position,
    )
    state.phase = "done"


def decode_batch(
    runtime,
    operations: tuple[Operation, ...],
    scope: PartitionState,
) -> tuple[Outcome, ...]:
    build_started = time.perf_counter_ns()
    starts: list[int] = []
    layout = scope.layout
    if layout is None:
        raise RuntimeError("partition lost its aligned request-row view")
    if layout.operations == operations:
        requests = layout.requests
        cache_rows = layout.cache_rows
        weights = layout.weights
    else:
        aligned = {
            identity: (request, cache_row, weight)
            for identity, request, cache_row, weight in zip(
                layout.identities,
                layout.requests,
                layout.cache_rows,
                layout.weights,
                strict=True,
            )
        }
        selected = tuple(aligned[ops._operation_identity(operation)] for operation in operations)
        requests = tuple(value[0] for value in selected)
        cache_rows = tuple(value[1] for value in selected)
        weights = tuple(value[2] for value in selected)
    starts.extend(int(request.logical_position) for request in requests)
    tasks = _decode_forward_tasks(
        runtime,
        operations,
        requests,
        cache_rows,
        weights,
        starts,
        scope,
    )

    ops._record_component(scope, "text_build_batch", build_started)
    forward_started = time.perf_counter_ns()
    forward_result = ops._run_observed_forward_group(runtime, tasks, scope)
    outputs = forward_result.values
    graph_greedy = forward_result.greedy
    if forward_result.output_event is not None:
        torch.cuda.current_stream(ops._phase_device(runtime, tasks[0].phase)).wait_event(
            forward_result.output_event
        )
    ops._record_component(scope, "text_model_forward", forward_started)
    for task in tasks:
        commit_kv(runtime, task, 1, scope, publish_runtime=False)

    sample_started = time.perf_counter_ns()
    graph_outcomes = project_graph_decode(
        runtime,
        operations,
        requests,
        cache_rows,
        tasks,
        starts,
        graph_greedy,
        scope,
    )
    if graph_outcomes is not None:
        ops._record_component(scope, "text_sample", sample_started)
        return graph_outcomes
    logits = tuple(token_logits(output)[-1] for output in outputs)
    sample_tasks = tuple(
        build_sample_work(
            runtime,
            operation,
            row_logits,
            session,
            scope,
            positions=(start + 1,),
            request_pool_index=sampling.request_pool_index(task),
        )
        for operation, session, task, start, row_logits in zip(
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
            selection_broadcast=partial(ops._broadcast_tp_selection, runtime),
            preselected=graph_greedy,
        )
        if sampling.graph_greedy_compatible(sample_tasks, tasks, graph_greedy)
        else sampling.sample(
            sample_tasks,
            scope.completion,
            device_products=runtime.device_products,
            device_reads=sampling_reads,
            selection_broadcast=partial(ops._broadcast_tp_selection, runtime),
        )
    )
    ops._record_component(scope, "text_sample", sample_started)
    finalize_started = time.perf_counter_ns()
    publish_token_products(runtime, operations, samples, scope)
    outcomes: list[Outcome] = []
    for operation, session, entry, start, sampled in zip(
        operations,
        requests,
        cache_rows,
        starts,
        samples,
        strict=True,
    ):
        session.rng_counter += 1
        session.logical_position = start + 1
        # Keep the sampled token deferred: materializing it here (``int()``)
        # blocks on the copy event and stalls the decode pipeline. It is
        # finalized when the response is serialized, after the next forward
        # has launched.
        outcomes.append(
            token_outcome(
                runtime,
                operation,
                scope,
                session=session,
                kv_entry=entry,
                base=start,
                tokens=1,
                committed_tokens=(sampled.token_id,),
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
    ops._record_component(scope, "text_finalize", finalize_started)
    return tuple(outcomes)


def _decode_forward_tasks(
    runtime,
    operations: tuple[Operation, ...],
    requests: tuple[RequestRow, ...],
    cache_rows: tuple[CacheRow | None, ...],
    weights: tuple[WeightSet, ...],
    starts: list[int],
    scope: PartitionState,
) -> tuple[ForwardRow, ...]:
    states = runtime.runtime_states
    predicates = tuple(
        scope.predicate_values.get(ops._operation_identity(operation)) for operation in operations
    )
    entries = tuple(row for row in cache_rows if row is not None)
    request_indexed = (
        states is not None
        and states.device.type == "cuda"
        and runtime.cache_pool.request_page_tables.device == states.device
        and len(entries) == len(operations)
        and len({row.group_id for row in entries}) == 1
        and all(value is not None and value[1] for value in predicates)
    )
    if request_indexed:
        placeholder = states.future_input_tokens[0, :1]
        tasks: list[ForwardRow] = []
        for operation, request, entry, weight, predicate in zip(
            operations,
            requests,
            cache_rows,
            weights,
            predicates,
            strict=True,
        ):
            if request.sampling is None or entry is None or predicate is None:
                raise RuntimeError("request-indexed decode lost its aligned row state")
            sampling_state = scope.sampling_states.get(
                ops._operation_identity(operation), SamplingState()
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
                    entry=entry,
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
            entry=entry,
            weights=weight,
        )
        for index, (operation, request, entry, weight, current) in enumerate(
            zip(
                operations,
                requests,
                cache_rows,
                weights,
                current_tokens,
                strict=True,
            )
        )
    )


def project_graph_decode(
    runtime,
    operations: tuple[Operation, ...],
    requests: tuple[RequestRow, ...],
    cache_rows: tuple[CacheRow | None, ...],
    tasks: tuple[ForwardRow, ...],
    starts: list[int],
    output: GraphGreedyOutput | None,
    scope: PartitionState,
) -> tuple[Outcome, ...] | None:
    """Project a captured greedy decision directly into protocol state."""

    if output is None:
        return None
    count = len(operations)
    columns = (requests, cache_rows, tasks, starts)
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
        parameters = request.sampling
        state = scope.sampling_states.get(ops._operation_identity(operation), SamplingState())
        finish_token_ids = (
            request.finish_token_ids
            if not state.finish_token_ids
            else state.finish_token_ids
            if not request.finish_token_ids
            else tuple(sorted({*request.finish_token_ids, *state.finish_token_ids}))
        )
        write = scope.token_writes.get(ops._operation_identity(operation))
        if (
            parameters is None
            or not sampling.device_greedy_parameters(parameters)
            or parameters.allowed_token_ids is not None
            or bool(parameters.forced_token_ids)
            or state.allowed_token_ids is not None
            or bool(state.suppressed_token_ids)
            or bool(finish_token_ids)
            or bool(state.transition_token_ids)
            or ops._operation_identity(operation) in scope.transition_writes
            or task.decode_predicate is None
            or not task.decode_predicate_tagged
            or bool(state.force_finish) != bool(task.decode_force_finish)
            or write is None
        ):
            return None
        token_writes.append(write)

    token_batch = runtime.device_products.producer_scalar_batch(tuple(token_writes))
    finish_indexes = tuple(
        index
        for index, operation in enumerate(operations)
        if ops._operation_identity(operation) in scope.finish_writes
    )
    finish_batch: DeviceProductScalarBatch | None = None
    if finish_indexes:
        finish_writes = tuple(
            scope.finish_writes[ops._operation_identity(operations[index])]
            for index in finish_indexes
        )
        finish_batch = runtime.device_products.producer_scalar_batch(finish_writes)

    ops._broadcast_tp_selection(runtime, output.tokens)
    span = sampling.capture_preselected_span(output, scope.completion)
    if token_batch is not None and (not finish_indexes or finish_batch is not None):
        token_batch.tensor.copy_(output.tagged_tokens)
        batches: tuple[DeviceProductScalarBatch, ...] = (token_batch,)
        if finish_batch is not None:
            finish_batch.tensor.copy_(sampling.select_device_values(output.finish, finish_indexes))
            batches = (finish_batch, token_batch)
        runtime.device_products.publish_scalar_group(
            batches,
            after_reads=tuple(scope.device_reads),
        )
    else:
        sampling.publish_device_writes(
            tuple(token_writes),
            output.tagged_tokens,
            runtime.device_products,
            tuple(scope.device_reads),
        )
    if finish_indexes and not (token_batch is not None and finish_batch is not None):
        finish_writes = tuple(
            scope.finish_writes[ops._operation_identity(operations[index])]
            for index in finish_indexes
        )
        sampling.publish_device_writes(
            finish_writes,
            sampling.select_device_values(output.finish, finish_indexes),
            runtime.device_products,
            tuple(scope.device_reads),
        )

    states = runtime.runtime_states
    if states is None:
        raise RuntimeError("graph decode has no request runtime-state owner")
    slots = tuple(int(request.request_pool_idx) for request in requests)
    if any(
        operation.request_key != request.request_key
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
    for index, (operation, request, cache_row, start) in enumerate(
        zip(operations, requests, cache_rows, starts, strict=True)
    ):
        if cache_row is None:
            raise RuntimeError("graph decode lost its scheduler cache row")
        request.rng_counter += 1
        request.logical_position = int(start) + 1
        extents = cache_row.extents()
        outcomes.append(
            Outcome(
                status=OpStatus.OK,
                selected_point=1,
                logical_lengths=LogicalLengths(
                    token_len=request.logical_position,
                    kv_visible_len=extents.visible,
                    latent_len=request.flow_step,
                    kv_reserved_len=extents.reserved,
                    kv_initialized_len=extents.initialized,
                    kv_committed_len=extents.committed,
                    kv_published_len=extents.published,
                ),
                token_span=TokenSpan(base=int(start), len=1),
                finish_flags=FinishFlags(),
                product_generations=ops._output_generations(operation),
                committed_tokens=(_CompletionSampleToken(span, index),),
            )
        )
    return tuple(outcomes)


def prompt_logprob_details(
    runtime,
    session: RequestRow,
    start: int,
    tokens: torch.Tensor,
    logits: torch.Tensor,
    scope: PartitionState,
) -> tuple[
    tuple[tuple[int, float, int], ...] | _CompletionTopLogprobs,
    ...,
]:
    tokens = tokens.reshape(-1).to(device=logits.device, dtype=torch.long)
    if logits.ndim != 2 or int(logits.shape[0]) != int(tokens.numel()):
        raise invalid_descriptor("prompt scoring logits do not align with input tokens")
    states = runtime.runtime_states
    if states is None:
        raise capability_mismatch("prompt scoring has no request-indexed runtime state")
    slot = int(session.request_pool_idx)
    if start == 0:
        score_logits = logits[:-1]
        targets = tokens[1:]
    else:
        if not session.prompt_logits_ready:
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
    session.prompt_logits_ready = True
    if int(targets.numel()) == 0:
        return ()
    parameters = require_sampling(session)
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
    return tuple(details[index][1] for index in range(int(targets.numel())))


def token_outcome(
    runtime,
    operation: Operation,
    scope: PartitionState,
    *,
    session: RequestRow | None = None,
    kv_entry: CacheRow | None = None,
    base: int,
    tokens: int | _CompletionDerivedInteger | _CompletionSpeculativePoint,
    committed_tokens: tuple[int | _CompletionToken, ...],
    sample: SampleResult | None = None,
    selection: SpeculativeSelection | None = None,
) -> Outcome:
    if session is None:
        session = ops._request_row(runtime, scope, operation.request_key.session_id)
    row = ops._cache_row(runtime, operation, scope) if kv_entry is None else kv_entry
    extents = row.extents()
    visible = (
        extents.visible
        if selection is None
        else _CompletionDerivedInteger(
            cast(_CompletionInteger, selection.selected_point),
            selection.base_kv_visible,
        )
    )
    token_len = (
        session.logical_position
        if selection is None
        else _CompletionDerivedInteger(
            cast(_CompletionInteger, selection.selected_point),
            selection.base_logical_position,
        )
    )
    if sample is not None and selection is not None:
        _publish_selection_products(
            runtime,
            operation,
            sample,
            scope,
            logical_position=session.logical_position,
            kv_visible=extents.visible,
            selection=selection,
        )
    return Outcome(
        status=OpStatus.OK,
        selected_point=(1 if selection is None else selection.selected_point),
        logical_lengths=LogicalLengths(
            token_len=cast(int, token_len),
            kv_visible_len=cast(int, visible),
            kv_reserved_len=extents.reserved,
            kv_initialized_len=extents.initialized,
            kv_committed_len=extents.committed,
            kv_published_len=extents.published,
        ),
        token_span=TokenSpan(base=base, len=cast(int, tokens)),
        finish_flags=FinishFlags(),
        product_generations=ops._output_generations(operation),
        committed_tokens=committed_tokens,
        products=sample_product_payloads(operation, sample),
        selection=selection,
    )


def token_task(
    runtime,
    operation: Operation,
    session: RequestRow,
    token_ids: tuple[int | torch.Tensor, ...],
    positions: tuple[int, ...] | torch.Tensor,
    selection: TokenSelection,
    scope: PartitionState,
    *,
    entry: CacheRow | None = None,
    weights: WeightSet | None = None,
) -> ForwardRow:
    if len(token_ids) != len(positions) or not token_ids:
        raise invalid_descriptor("token task ids and positions must align")
    if len(token_ids) == 1 and isinstance(token_ids[0], torch.Tensor):
        token_values = token_ids[0].reshape(1).to(dtype=torch.long)
    else:
        token_values = torch.tensor(
            tuple(int(value) for value in token_ids),
            dtype=torch.long,
        )
    predicate_value = scope.predicate_values.get(ops._operation_identity(operation))
    sampling_state = scope.sampling_states.get(ops._operation_identity(operation), SamplingState())
    return ForwardRow(
        operation=operation,
        request=session,
        weights=ops._weights(
            runtime,
        )
        if weights is None
        else weights,
        phase=ModelPhase.TEXT,
        token_ids=token_values,
        positions=(
            positions.reshape(-1).to(dtype=torch.long)
            if isinstance(positions, torch.Tensor)
            else torch.tensor(positions, dtype=torch.long)
        ),
        selection=selection,
        entry=ops._cache_row(runtime, operation, scope) if entry is None else entry,
        write_kv=True,
        causal=True,
        decode_predicate=None if predicate_value is None else predicate_value[0],
        decode_predicate_tagged=False if predicate_value is None else predicate_value[1],
        decode_force_finish=bool(sampling_state.force_finish),
    )


def commit_kv(
    runtime,
    task: ForwardRow,
    tokens: int,
    scope: PartitionState,
    *,
    publish_runtime: bool = True,
) -> None:
    count = int(tokens)
    if count < 0 or count > task.query_tokens:
        raise RuntimeError("KV commit count is outside the task query span")
    if count == 0:
        return
    if task.entry is None:
        raise RuntimeError("KV task has no scheduler cache row")
    task.entry.advance(count)
    if publish_runtime and runtime.runtime_states is not None:
        slot = int(task.request.request_pool_idx)
        scope.runtime_cache_lengths[slot] = int(task.entry.length)


def operation_token_ids(
    runtime,
    operation: Operation,
    scope: PartitionState,
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
    runtime,
    operation: Operation,
    session: RequestRow,
    scope: PartitionState,
) -> int | torch.Tensor:
    point = operation.parent.point
    if isinstance(point, DevicePoint):
        predicate = scope.predicate_values.get(ops._operation_identity(operation))
        if predicate is None:
            raise invalid_descriptor("device token continuation is not registered")
        states = runtime.runtime_states
        if states is None:
            raise capability_mismatch("device continuation has no request runtime state")
        slot = int(session.request_pool_idx)
        pending = _pending_runtime_token(runtime, slot, scope)
        if pending is not None:
            return pending.reshape(-1)[:1].bitwise_and(sampling.TOKEN_VALUE_MASK)
        return states.future_input_tokens[slot, :1]
    tokens = operation_token_ids(runtime, operation, scope)
    if not tokens:
        raise invalid_descriptor("last-sampled token source has no committed token")
    return int(tokens[0])


def resolve_decode_tokens(
    runtime,
    operations: tuple[Operation, ...],
    scope: PartitionState,
) -> tuple[int | torch.Tensor, ...]:
    resolved: list[int | torch.Tensor | None] = [None] * len(operations)
    for index, operation in enumerate(operations):
        point = operation.parent.point
        if isinstance(point, DevicePoint):
            predicate = scope.predicate_values.get(ops._operation_identity(operation))
            if predicate is None:
                raise invalid_descriptor("device token continuation is not registered")
            states = runtime.runtime_states
            if states is None:
                raise capability_mismatch("device continuation has no request runtime state")
            session = ops._request_row(runtime, scope, operation.request_key.session_id)
            slot = int(session.request_pool_idx)
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
    runtime,
    slot: int,
    scope: PartitionState,
) -> torch.Tensor | None:
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
    runtime,
    operation: Operation,
    sample: SampleResult,
    scope: PartitionState,
) -> None:
    if sample.device_product_published:
        return
    write = scope.token_writes.get(ops._operation_identity(operation))
    if write is None:
        return
    device_token = sample.device_token
    if device_token is None:
        device_token = torch.tensor(
            (int(sample.token_id),),
            dtype=torch.long,
            device=ops._operation_device(runtime, operation),
        )
    continuation = sample.device_continuation
    if continuation is None:
        continuation = torch.ones_like(device_token, dtype=torch.bool)
    runtime.device_products.publish_write(
        write,
        sampling.tagged_token_values(device_token, continuation),
    )


def publish_runtime_samples(
    runtime,
    operations: Sequence[Operation],
    sessions: Sequence[RequestRow],
    samples: Sequence[SampleResult],
    *,
    scope: PartitionState,
    sample_tasks: Sequence[SampleWork],
    logical_positions: Sequence[int | torch.Tensor],
    sampling_positions: Sequence[int | torch.Tensor],
    decode_increment: bool = False,
) -> None:
    states = runtime.runtime_states
    if states is None:
        return
    columns = (
        operations,
        sessions,
        samples,
        sample_tasks,
        logical_positions,
        sampling_positions,
    )
    if len({len(values) for values in columns}) != 1:
        raise RuntimeError("runtime sampling publication columns are not aligned")
    if decode_increment:
        if any(
            operation.request_key != session.request_key
            for operation, session in zip(operations, sessions, strict=True)
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
                    slots=tuple(int(session.request_pool_idx) for session in sessions),
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
                slots=tuple(int(session.request_pool_idx) for session in sessions),
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
    for operation, session, sample, sample_task, logical, sampling_position in zip(
        operations,
        sessions,
        samples,
        sample_tasks,
        logical_positions,
        sampling_positions,
        strict=True,
    ):
        if operation.request_key != session.request_key:
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
        slot = int(session.request_pool_idx)
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
    runtime,
    operations: tuple[Operation, ...],
    samples: tuple[SampleResult, ...],
    scope: PartitionState,
) -> None:
    if all(sample.device_product_published for sample in samples):
        return
    writes: list[DeviceProductWrite] = []
    device_tokens: list[torch.Tensor] = []
    for operation, sample in zip(operations, samples, strict=True):
        write = scope.token_writes.get(ops._operation_identity(operation))
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
    runtime,
    operation: Operation,
    sample: SampleResult,
    scope: PartitionState,
    *,
    logical_position: int,
    kv_visible: int,
    selection: SpeculativeSelection | None,
) -> None:
    device_token = sample.device_token
    if device_token is None:
        raise RuntimeError("device selection products require a device token")
    accepted = sample.device_accepted_tokens
    if accepted is None:
        accepted = torch.zeros_like(device_token, dtype=torch.long)
    selected_point = sample.device_selected_point
    if selected_point is None:
        selected_point = accepted.to(dtype=torch.long) + 1
    semantic_token = (
        device_token.reshape(-1).to(dtype=torch.long).bitwise_and(sampling.TOKEN_VALUE_MASK)
    )
    operation_identity = ops._operation_identity(operation)
    selected_write = scope.selected_point_writes.get(operation_identity)
    span_write = scope.accepted_span_writes.get(operation_identity)
    continuation_write = scope.state_continuation_writes.get(operation_identity)
    if selected_write is None:
        raise invalid_descriptor("token operation is missing its selected-point product")
    runtime.device_products.publish_write(selected_write, selected_point)
    if span_write is None and continuation_write is None and int(operation.bounds.max_points) == 1:
        return
    if span_write is None or continuation_write is None:
        raise invalid_descriptor("multi-point token operation is missing its branch products")
    if selection is None:
        candidates = semantic_token
        logical = torch.full_like(selected_point, int(logical_position))
        visible = torch.full_like(selected_point, int(kv_visible))
    else:
        draft = torch.tensor(
            selection.draft_tokens,
            dtype=torch.long,
            device=device_token.device,
        )
        candidates = torch.cat((draft, semantic_token))
        logical = selected_point + int(selection.base_logical_position)
        visible = selected_point + int(selection.base_kv_visible)
    max_points = int(operation.bounds.max_points)
    if int(candidates.numel()) != max_points:
        raise RuntimeError("selection candidates do not match the operation point bound")
    indexes = torch.arange(max_points, dtype=torch.long, device=device_token.device)
    visible_tokens = torch.where(
        indexes < selected_point.reshape(()),
        candidates,
        torch.zeros_like(candidates),
    )
    runtime.device_products.publish_write(
        span_write,
        torch.cat((selected_point.reshape(-1), visible_tokens)),
    )
    runtime.device_products.publish_write(
        continuation_write,
        torch.stack(
            (
                semantic_token[0],
                selected_point.reshape(-1)[0],
                visible.reshape(-1)[0],
                logical.reshape(-1)[0],
            )
        ),
    )


def build_sample_work(
    runtime,
    operation: Operation,
    logits: torch.Tensor,
    session: RequestRow,
    scope: PartitionState,
    *,
    positions: tuple[int, ...],
    request_pool_index: torch.Tensor,
    draft_token_ids: tuple[int, ...] = (),
) -> SampleWork:
    parameters = require_sampling(session)
    state = scope.sampling_states.get(ops._operation_identity(operation), SamplingState())
    allowed_token_ids = (
        state.allowed_token_ids
        if state.allowed_token_ids is not None
        else parameters.allowed_token_ids
    )
    if not state.finish_token_ids:
        finish_token_ids = session.finish_token_ids
    elif not session.finish_token_ids:
        finish_token_ids = state.finish_token_ids
    else:
        finish_token_ids = tuple(
            sorted(
                {
                    *session.finish_token_ids,
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
            int(operation.request_key.session_id),
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
        _session_penalty_base(runtime, session, vocab, rows.device) if uses_penalties else None
    )
    penalty_view = (
        None
        if penalty_base is None
        else _candidate_penalty_counts(runtime, session, penalty_base, scope)
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
    operation_identity = ops._operation_identity(operation)
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
    finish_product = scope.finish_writes.get(operation_identity)
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
        finish_product=finish_product,
        transition_product=transition_product,
        predicate=None if predicate_value is None else predicate_value[0],
        tagged_predicate=False if predicate_value is None else predicate_value[1],
        request_pool_index=request_pool_index,
        penalty_base=penalty_base,
    )


def _session_penalty_base(
    runtime,
    session: RequestRow,
    vocab: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the fixed request-indexed committed penalty-count row."""

    states = runtime.runtime_states
    if states is None:
        raise RuntimeError("token sampling has no request runtime-state owner")
    if states.vocab_size != int(vocab) or states.device != device:
        raise capability_mismatch("sampling geometry disagrees with request runtime state")
    return states.penalty_counts[int(session.request_pool_idx)]


def _candidate_penalty_counts(
    runtime,
    session: RequestRow,
    committed: torch.Tensor,
    scope: PartitionState,
) -> torch.Tensor:
    slot = int(session.request_pool_idx)
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
    matches = tuple(output for output in operation.outputs if output.kind is ProductKind.LOGPROB)
    if len(matches) > 1:
        raise invalid_descriptor("operation declares multiple logprob products")
    return matches[0] if matches else None


def sample_product_payloads(
    operation: Operation,
    sample: SampleResult | None,
) -> tuple[ProductPayload, ...]:
    if sample is None or (
        (sample.logprob is None or sample.top_logprobs is None) and not sample.prompt_logprobs
    ):
        return ()
    reference = _logprob_product_ref(operation)
    if reference is None:
        raise invalid_descriptor("sampler produced undeclared logprob output")
    payload = _CompletionLogprobPayload(
        sample.logprob,
        sample.top_logprobs,
        sample.prompt_logprobs,
    )
    return (ProductPayload(product=reference, payload=cast(bytes, payload)),)


def token_logits(output: torch.Tensor) -> torch.Tensor:
    if not isinstance(output, torch.Tensor) or output.ndim < 2:
        raise invalid_descriptor("token route did not return logits")
    return output


def require_sampling(session: RequestRow) -> SamplingParams:
    if session.sampling is None:
        raise invalid_descriptor("sequence execution requires admitted sampling parameters")
    return session.sampling


def token_logits_or_hidden(output: torch.Tensor) -> torch.Tensor:
    if not isinstance(output, torch.Tensor) or output.ndim < 2:
        raise invalid_descriptor("token route did not return a token tensor")
    return output


__all__ = ["consume_forward", "consume_sample", "decode_batch", "pack_forward", "pack_sample"]
