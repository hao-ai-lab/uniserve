"""Batched GPU token sampling and device-resident sample publication."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import cast

import torch

from uniserve_worker.backends.triton import triton_available
from uniserve_worker.execution.batch import SamplingParams
from uniserve_worker.execution.forward_batch import packed_tensor_views
from uniserve_worker.foundation.errors import unsupported_setup, invalid_descriptor
from uniserve_worker.foundation.math import bucketed_length
from uniserve_worker.runtime.device_products import (
    DeviceProductRead,
    DeviceProducts,
    DeviceProductScalarBatch,
    DeviceProductWrite,
)
from uniserve_worker.execution.output import (
    LogprobCapture,
    LogprobOutputRow,
    SamplingCapture,
    SamplingOutputRow,
    OutputBuffer,
)

from .cuda_graph import GraphGreedyOutput
from .rows import (
    ForwardRow,
    SampleBatchVectors,
    SampleResult,
    SampleRow,
    SampleWork,
)
from .top_k_sampling import SamplingParameters, sample_top_k

SAMPLING_COMPLETION_FIELDS = 4
TOKEN_CONTINUATION_BIT = 1 << 31
TOKEN_VALUE_MASK = TOKEN_CONTINUATION_BIT - 1


def sample_result(value: object) -> SampleResult:
    """Validate and return the single tensor produced by a sampling operator."""

    if not isinstance(value, SampleResult):
        raise RuntimeError("sampling task returned an invalid result")
    return value


def sampling_task_tensors(
    rows: Sequence[SampleRow],
    *,
    vocab: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack row sampling controls, suppression masks, and penalty counts into device tensors."""

    # Penalties are applied densely from the device-resident count base in the
    # general sampling path; the fused top-k path never receives a penalty-
    # bearing operation. The sparse penalty tensors are therefore always empty
    # and exist only to satisfy the shared batch layout for the fused path.
    width = min(vocab, bucketed_length(0))
    padding: list[int] = []
    candidate = vocab - 1
    while len(padding) < width:
        padding.append(candidate)
        candidate -= 1
    token_ids = torch.tensor(
        tuple(padding),
        dtype=torch.long,
        device=device,
    ).reshape(1, width)
    counts = torch.zeros((1, width), dtype=torch.float32, device=device)
    row_count = len(rows)
    parameter_values = torch.tensor(
        [
            (
                float(row.parameters.temperature),
                float(row.parameters.top_p),
                float(row.parameters.min_p),
                float(row.parameters.repetition_penalty),
                float(row.parameters.frequency_penalty),
                float(row.parameters.presence_penalty),
            )
            for row in rows
        ],
        dtype=torch.float32,
        device=device,
    )
    return (
        token_ids.expand(row_count, width),
        counts.expand(row_count, width),
        parameter_values,
    )


def device_greedy_parameters(parameters: SamplingParams) -> bool:
    """Return whether sampling parameters reduce exactly to unpenalized greedy selection."""

    return (
        float(parameters.temperature) <= 0.0
        and not parameters.return_logprobs
        and int(parameters.n_logprobs) == 0
        and not parameters.logprob_token_ids
        and not parameters.logit_bias
        and parameters.repetition_penalty == 1.0
        and parameters.frequency_penalty == 0.0
        and parameters.presence_penalty == 0.0
    )


def device_greedy_row(row: SampleRow) -> bool:
    """Return whether one sampling row can use the captured device-greedy path."""

    return (
        device_greedy_parameters(row.parameters)
        and int(row.n_logprobs) == 0
        and row.allowed is None
    )


def graph_greedy_compatible(
    tasks: tuple[SampleWork, ...],
    forward_tasks: Sequence[ForwardRow],
    output: GraphGreedyOutput | None,
) -> bool:
    """Validate that graph-produced greedy outputs match every sampling task and forward row."""

    if output is None or len(tasks) != len(forward_tasks):
        return False
    count = len(tasks)
    vectors = (
        output.request_pool_indices,
        output.tokens,
        output.valid,
        output.active,
        output.finish,
        output.continuation,
        output.tagged_tokens,
    )
    return (
        count > 0
        and all(int(value.numel()) == count for value in vectors)
        and int(output.completion.numel()) == SAMPLING_COMPLETION_FIELDS * count
        and all(
            not task.draft_token_ids
            and len(task.rows) == 1
            and device_greedy_row(task.rows[0])
            and not task.rows[0].suppress
            and not task.rows[0].finish_token_ids
            and not task.rows[0].transition_token_ids
            and task.transition_product is None
            and task.request_pool_index is not None
            and forward_task.decode_predicate is not None
            and forward_task.decode_predicate_tagged
            and bool(task.rows[0].force_finish) == bool(forward_task.decode_force_finish)
            for task, forward_task in zip(tasks, forward_tasks, strict=True)
        )
    )


@torch.inference_mode()
def sample(
    tasks: Sequence[SampleWork],
    completion: OutputBuffer | None = None,
    *,
    device_products: DeviceProducts | None = None,
    device_reads: tuple[DeviceProductRead, ...] = (),
    selection_broadcast: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[SampleResult, ...]:
    """Shape and draw every compatible sampling row in each device batch."""

    grouped: dict[tuple[torch.device, int, int, int], list[tuple[int, SampleWork]]] = defaultdict(
        list
    )
    for index, task in enumerate(tasks):
        if (
            task.logits.ndim != 2
            or not task.logits.is_floating_point()
            or int(task.logits.shape[0]) < 1
            or int(task.logits.shape[1]) < 1
            or int(task.logits.shape[0]) != len(task.rows)
        ):
            raise invalid_descriptor("sampling task logits must be shaped [rows, vocab]")
        device_greedy = not task.draft_token_ids and all(
            device_greedy_row(row) for row in task.rows
        )
        if device_greedy:
            if (
                any(
                    value is not None
                    for value in (
                        task.draws,
                        task.penalty_token_ids,
                        task.penalty_counts,
                        task.parameter_values,
                    )
                )
                or task.draft_token_ids
                or len(task.rows) != 1
            ):
                raise invalid_descriptor("greedy sampling task has shaped metadata")
        else:
            draws = cast(torch.Tensor, task.draws)
            penalty_token_ids = cast(torch.Tensor, task.penalty_token_ids)
            penalty_counts = cast(torch.Tensor, task.penalty_counts)
            parameter_values = cast(torch.Tensor, task.parameter_values)
            if (
                draws.device != task.logits.device
                or tuple(draws.shape) != (len(task.rows),)
                or not draws.is_floating_point()
            ):
                raise invalid_descriptor("sampling task draws must align with its rows")
            if (
                penalty_token_ids.device != task.logits.device
                or penalty_counts.device != task.logits.device
                or penalty_token_ids.ndim != 2
                or penalty_counts.shape != penalty_token_ids.shape
                or int(penalty_token_ids.shape[0]) != len(task.rows)
            ):
                raise invalid_descriptor("sampling task penalty tensors do not align")
            if parameter_values.device != task.logits.device or parameter_values.shape != (
                len(task.rows),
                6,
            ):
                raise invalid_descriptor("sampling task parameter vectors do not align")
        if task.draft_token_ids:
            if len(task.rows) != len(task.draft_token_ids) + 1:
                raise invalid_descriptor("speculative sampling rows do not cover the draft chain")
        elif len(task.rows) != 1:
            raise invalid_descriptor("ordinary sampling tasks must contain exactly one row")
        vocab = int(task.logits.shape[1])
        if vocab > TOKEN_VALUE_MASK:
            raise unsupported_setup("vocabulary exceeds the device token decision range")
        if any(value < 0 or value >= vocab for value in task.draft_token_ids):
            raise invalid_descriptor("speculative draft token is outside the model vocabulary")
        sampling_path = (
            (-2 if any(row.suppress for row in task.rows) else -1)
            if device_greedy
            else _fused_top_k(task, vocab)
        )
        penalty_width = (
            int(cast(torch.Tensor, task.penalty_token_ids).shape[1]) if sampling_path > 0 else 0
        )
        grouped[(task.logits.device, vocab, sampling_path, penalty_width)].append((index, task))

    result: list[SampleResult | None] = [None] * len(tasks)
    for (_device, _vocab, sampling_path, _penalty_width), compatible in grouped.items():
        indexes, group = zip(*compatible, strict=True)
        if sampling_path < 0:
            sampled_group = sample_device_greedy_group(
                tuple(group),
                completion,
                apply_suppression=sampling_path == -2,
                device_products=device_products,
                device_reads=device_reads,
                selection_broadcast=selection_broadcast,
            )
        elif sampling_path > 0:
            sampled_group = _sample_fused_top_k_group(
                tuple(group),
                sampling_path,
                completion,
                device_products=device_products,
                device_reads=device_reads,
                selection_broadcast=selection_broadcast,
            )
        else:
            sampled_group = _sample_task_group(
                tuple(group),
                completion,
                device_products=device_products,
                device_reads=device_reads,
                selection_broadcast=selection_broadcast,
            )
        for index, sampled in zip(indexes, sampled_group, strict=True):
            result[index] = sampled
    return tuple(cast(SampleResult, value) for value in result)


def sample_device_greedy_group(
    tasks: tuple[SampleWork, ...],
    completion: OutputBuffer | None,
    *,
    apply_suppression: bool,
    device_products: DeviceProducts | None,
    device_reads: tuple[DeviceProductRead, ...],
    selection_broadcast: Callable[[torch.Tensor], torch.Tensor] | None,
    preselected: GraphGreedyOutput | None = None,
) -> tuple[SampleResult, ...]:
    """Resolve predicates, greedy tokens, finish state, publications, and completions for a task group."""

    # Materialize one row-major logit matrix unless graph replay already
    # supplied its device-resident token and predicate decisions.
    selection_logits: torch.Tensor | None = None
    if preselected is None:
        logits = packed_tensor_views(tuple(task.logits for task in tasks))
        if logits is None:
            logits = torch.cat(tuple(task.logits for task in tasks), dim=0)
        else:
            logits = logits.reshape(len(tasks), -1)
        selection_logits = logits
        if apply_suppression:
            selection_logits = logits.to(dtype=torch.float32, copy=True)
            vocab = int(selection_logits.shape[1])
            for row_index, task in enumerate(tasks):
                for token_id in dict.fromkeys(int(value) for value in task.rows[0].suppress):
                    if 0 <= token_id < vocab:
                        selection_logits[row_index, token_id].fill_(float("-inf"))
    # Scalar product arenas allow selection and later publication to share the
    # same storage. A heterogeneous binding falls back to independent writes.
    products = tuple(task.token_product for task in tasks)
    bound_products = device_products is not None and all(
        product is not None for product in products
    )
    product_table = cast(DeviceProducts, device_products) if bound_products else None
    product_writes = (
        tuple(cast(DeviceProductWrite, product) for product in products)
        if product_table is not None
        else ()
    )
    product_batch = (
        product_table.producer_scalar_batch(product_writes) if product_table is not None else None
    )
    packed_output = product_batch.tensor if product_batch is not None else None
    transition_writes = tuple(
        cast(DeviceProductWrite, task.transition_product)
        for task in tasks
        if task.transition_product is not None
    )
    transition_batch: DeviceProductScalarBatch | None = None
    if transition_writes and device_products is not None:
        transition_batch = device_products.producer_scalar_batch(transition_writes)
    grouped_publication = (
        product_table is not None
        and product_batch is not None
        and (not transition_writes or transition_batch is not None)
    )
    # Select directly into publication storage when all token products form one
    # linked scalar batch; otherwise retain a transient result tensor.
    if preselected is not None:
        device_tokens = preselected.tokens
        max_values = None
    elif packed_output is None or int(packed_output.numel()) != len(tasks):
        assert selection_logits is not None
        device_tokens = torch.argmax(selection_logits, dim=-1)
        max_values = None
    elif grouped_publication:
        assert selection_logits is not None
        device_tokens = packed_output
        max_values = torch.empty(
            len(tasks),
            dtype=selection_logits.dtype,
            device=selection_logits.device,
        )
        torch.max(selection_logits, dim=-1, out=(max_values, device_tokens))
    else:
        assert selection_logits is not None
        device_tokens = packed_output
        max_values = None
        torch.argmax(selection_logits, dim=-1, out=device_tokens)
    valid = (
        preselected.valid
        if preselected is not None
        else torch.isfinite(max_values)
        if max_values is not None
        else (
            ~torch.isnan(cast(torch.Tensor, selection_logits)).any(dim=-1)
            & ~torch.isposinf(cast(torch.Tensor, selection_logits)).any(dim=-1)
            & torch.isfinite(cast(torch.Tensor, selection_logits)).any(dim=-1)
        )
    )
    if selection_broadcast is not None:
        selection_broadcast(device_tokens)

    # Finish, continuation, and transition predicates are derived from the same
    # selected token vector so they cannot observe different sampling outcomes.
    empty_finish = grouped_publication and all(
        not task.rows[0].force_finish and not task.rows[0].finish_token_ids for task in tasks
    )
    active = (
        preselected.active
        if preselected is not None
        else _sample_predicates(tasks, device_tokens.device)
    )
    device_finish: torch.Tensor | None
    if preselected is not None:
        device_finish = preselected.finish
        continuation_values = preselected.continuation
    elif grouped_publication:
        if empty_finish:
            device_finish = torch.zeros_like(valid, dtype=torch.bool)
            continuation_values = active & valid
        else:
            device_finish = _device_finish_values(tasks, device_tokens, valid & active)
            continuation_values = active & valid & ~device_finish
    else:
        device_finish, continuation_values = _resolve_sampled_finish_values(
            tasks,
            device_tokens,
            valid,
            active,
            torch.zeros_like(active, dtype=torch.bool),
        )
    resolved_transition_writes, transition_values = _sampled_transition_values(
        tasks,
        device_tokens,
        valid,
        active,
        destination=(
            transition_batch.tensor
            if grouped_publication
            and transition_batch is not None
            and transition_batch.tensor.dtype is torch.bool
            else None
        ),
    )
    if len(resolved_transition_writes) != len(transition_writes) or any(
        resolved is not expected
        for resolved, expected in zip(
            resolved_transition_writes,
            transition_writes,
            strict=True,
        )
    ):
        raise RuntimeError("sampling transition publication lost its output alignment")
    if transition_writes and device_products is None:
        raise RuntimeError("sampling transition outputs have no device-product owner")
    if grouped_publication and transition_batch is not None:
        if transition_values is None:
            raise RuntimeError("sampling transition publication lost its device values")
        target = transition_batch.tensor.reshape(-1)
        aliases_target = (
            transition_values.device == target.device
            and transition_values.dtype == target.dtype
            and transition_values.untyped_storage().data_ptr()
            == target.untyped_storage().data_ptr()
            and int(transition_values.storage_offset()) == int(target.storage_offset())
        )
        if not aliases_target:
            target.copy_(transition_values)
    if transition_writes and not grouped_publication:
        if transition_values is None:
            raise RuntimeError("sampling transition publication lost its device values")
        publish_device_writes(
            transition_writes,
            transition_values,
            cast(DeviceProducts, device_products),
            device_reads,
        )
    # Capture host-facing token metadata before publishing products. The output
    # buffer owns the asynchronous device-to-host lifetime from this point.
    span = (
        capture_preselected_span(preselected, completion)
        if preselected is not None
        else _capture_sample_span(
            valid,
            active,
            device_tokens,
            cast(torch.Tensor, device_finish) if empty_finish else torch.zeros_like(device_tokens),
            completion,
        )
    )
    tagged_tokens = (
        preselected.tagged_tokens
        if preselected is not None
        else tagged_token_values(
            device_tokens,
            continuation_values,
            in_place=packed_output is not None and int(packed_output.numel()) == len(tasks),
        )
    )
    if preselected is not None and packed_output is not None:
        packed_output.copy_(tagged_tokens)
    # Group publication records one dependency edge for every linked scalar
    # output, preserving atomic visibility after all consumed reads complete.
    published = product_table is not None
    if product_table is not None:
        if grouped_publication:
            side_batches: tuple[DeviceProductScalarBatch, ...] = (
                (transition_batch,) if transition_batch is not None else ()
            )
            product_table.publish_scalar_group(
                (*side_batches, cast(DeviceProductScalarBatch, product_batch)),
                after_reads=device_reads,
            )
            device_finish = None
        elif product_batch is None:
            product_table.publish_writes(
                product_writes,
                tagged_tokens,
            )
        else:
            product_table.publish_scalar_batch(
                product_batch,
                after_reads=device_reads,
            )
    batch_vectors = (
        SampleBatchVectors(
            request_pool_indices=(
                preselected.request_pool_indices
                if preselected is not None
                else sample_request_pool_indices(tasks)
            ),
            tokens=device_tokens,
            valid=valid,
            active=active,
            continuation=continuation_values,
            selected_points=None,
            penalty_bases=tuple(task.penalty_base for task in tasks),
        )
        if all(task.request_pool_index is not None for task in tasks)
        else None
    )
    return tuple(
        SampleResult(
            completion=SamplingOutputRow(span, index),
            device_token=device_tokens[index : index + 1],
            logprobs=None,
            device_valid=valid[index : index + 1],
            device_active=active[index : index + 1],
            device_finish=(None if device_finish is None else device_finish[index : index + 1]),
            device_continuation=continuation_values[index : index + 1],
            device_product_published=published,
            device_batch=batch_vectors,
            device_batch_index=index if batch_vectors is not None else -1,
        )
        for index, task in enumerate(tasks)
    )


def _fused_top_k(task: SampleWork, vocab: int) -> int:
    """Return a fused top-k width when the task satisfies the kernel's sampling contract."""

    if task.logits.device.type != "cuda":
        return 0
    if task.draft_token_ids:
        return 0
    row = task.rows[0]
    parameters = row.parameters
    top_k = int(parameters.top_k)
    wants_logprobs = (
        parameters.return_logprobs or int(row.n_logprobs) > 0 or bool(parameters.logprob_token_ids)
    )
    uses_penalties = (
        parameters.repetition_penalty != 1.0
        or parameters.frequency_penalty != 0.0
        or parameters.presence_penalty != 0.0
    )
    if (
        wants_logprobs
        or uses_penalties
        or row.allowed is not None
        or row.suppress
        or parameters.logit_bias
        or float(parameters.typical_p) < 1.0
        or top_k <= 0
        or top_k > 128
        or top_k >= vocab
    ):
        return 0
    return top_k


def _sample_fused_top_k_group(
    tasks: tuple[SampleWork, ...],
    top_k: int,
    completion: OutputBuffer | None,
    *,
    device_products: DeviceProducts | None,
    device_reads: tuple[DeviceProductRead, ...],
    selection_broadcast: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[SampleResult, ...]:
    """Sample a homogeneous fused-top-k task group and publish its device-resident results."""

    # The compiled kernel consumes one contiguous column for each sampling
    # input, so compatible task rows are packed before a single launch.
    rows = tuple(task.rows[0] for task in tasks)
    logits = torch.cat(tuple(task.logits for task in tasks), dim=0)
    draws = torch.cat(tuple(cast(torch.Tensor, task.draws) for task in tasks), dim=0)
    penalty_token_ids = torch.cat(
        tuple(cast(torch.Tensor, task.penalty_token_ids) for task in tasks), dim=0
    )
    penalty_counts = torch.cat(
        tuple(cast(torch.Tensor, task.penalty_counts) for task in tasks), dim=0
    )
    parameters = torch.cat(
        tuple(cast(torch.Tensor, task.parameter_values) for task in tasks), dim=0
    )
    tokens, valid = _run_fused_top_k_sampling(
        logits,
        draws,
        penalty_token_ids,
        penalty_counts,
        parameters,
        top_k,
    )

    # Tensor-parallel peers share the selected tokens; finish and transition
    # policy is then evaluated from the common selection on every rank.
    if selection_broadcast is not None:
        selection_broadcast(tokens)
    active = _sample_predicates(tasks, tokens.device)
    device_finish, continuation_values = _resolve_sampled_finish_values(
        tasks,
        tokens,
        valid,
        active,
        torch.zeros_like(active, dtype=torch.bool),
    )
    _publish_sampled_transition_values(
        tasks,
        tokens,
        valid,
        active,
        device_products,
        device_reads,
    )

    # Completion storage retains the host-visible sampling fields while token
    # products remain device-resident for successor operations.
    published = _publish_sampled_device_values(
        tasks,
        "token_product",
        tagged_token_values(tokens, continuation_values),
        device_products,
        device_reads,
    )
    span = _capture_sample_span(valid, active, tokens, torch.zeros_like(tokens), completion)
    return tuple(
        SampleResult(
            SamplingOutputRow(span, index),
            tokens[index : index + 1],
            None,
            device_valid=valid[index : index + 1],
            device_active=active[index : index + 1],
            device_finish=(None if device_finish is None else device_finish[index : index + 1]),
            device_continuation=continuation_values[index : index + 1],
            device_product_published=published,
        )
        for index in range(len(rows))
    )


def _run_fused_top_k_sampling(
    logits: torch.Tensor,
    draws: torch.Tensor,
    penalty_token_ids: torch.Tensor,
    penalty_counts: torch.Tensor,
    parameters: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the compiled fixed-top-k sampler after validating CUDA toolchain availability."""

    if not triton_available(logits.device):
        raise unsupported_setup(
            "fused top-k sampling requires a supported CUDA compile toolchain"
        )
    return sample_top_k(
        logits,
        draws,
        penalty_token_ids,
        penalty_counts,
        SamplingParameters.from_columns(parameters),
        int(top_k),
    )


def _sample_task_group(
    tasks: tuple[SampleWork, ...],
    completion: OutputBuffer | None,
    *,
    device_products: DeviceProducts | None,
    device_reads: tuple[DeviceProductRead, ...],
    selection_broadcast: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[SampleResult, ...]:
    """Sample arbitrary compatible tasks, resolve speculative acceptance, and capture outputs."""

    # Flatten task-local candidate rows into one sampling matrix while retaining
    # offsets needed to restore one selected result per operation.
    device = tasks[0].logits.device
    vocab = int(tasks[0].logits.shape[1])
    offsets: list[int] = []
    offset = 0
    for task in tasks:
        offsets.append(offset)
        offset += len(task.rows)
    rows = tuple(row for task in tasks for row in task.rows)
    logits = torch.cat(tuple(task.logits.float() for task in tasks), dim=0)
    draws = torch.cat(tuple(cast(torch.Tensor, task.draws) for task in tasks), dim=0)
    work, valid = _shape_sampling_logits_batch(logits, rows)

    # Temperature-zero rows use deterministic argmax; remaining rows invert the
    # categorical CDF with their precomputed RNG draw.
    temperatures = torch.tensor(
        [float(row.parameters.temperature) for row in rows],
        dtype=work.dtype,
        device=device,
    )
    probabilities = torch.softmax(work, dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    sampled_tokens = (
        (cumulative < draws.to(dtype=cumulative.dtype).unsqueeze(1))
        .sum(dim=-1)
        .clamp_max(vocab - 1)
    )
    row_tokens = torch.where(
        temperatures > 0.0,
        sampled_tokens,
        torch.argmax(work, dim=-1),
    )

    # Speculative tasks accept the longest matching draft prefix. A terminal
    # prefix selects its final draft token without adding a continuation point.
    accepted_counts: list[torch.Tensor] = []
    selected_points: list[torch.Tensor] = []
    terminal_finishes: list[torch.Tensor] = []
    output_rows: list[torch.Tensor] = []
    terminal_tokens: list[torch.Tensor] = []
    for task, row_offset in zip(tasks, offsets, strict=True):
        if task.draft_token_ids:
            draft = torch.tensor(task.draft_token_ids, dtype=row_tokens.dtype, device=device)
            matches = row_tokens[row_offset : row_offset + len(task.draft_token_ids)] == draft
            raw_accepted = torch.cumprod(matches.to(torch.long), dim=0).sum()
        else:
            draft = torch.empty((0,), dtype=row_tokens.dtype, device=device)
            raw_accepted = torch.zeros((), dtype=torch.long, device=device)
        if task.terminal_draft_prefix is None:
            accepted = raw_accepted
            terminal = torch.zeros((), dtype=torch.bool, device=device)
            terminal_token = torch.zeros((), dtype=row_tokens.dtype, device=device)
        else:
            terminal_prefix = int(task.terminal_draft_prefix)
            accepted = raw_accepted.clamp_max(terminal_prefix)
            terminal = raw_accepted >= terminal_prefix
            terminal_token = draft[terminal_prefix - 1]
        accepted_counts.append(accepted)
        selected_points.append(accepted + (~terminal).to(dtype=torch.long))
        terminal_finishes.append(terminal)
        output_rows.append(accepted + row_offset)
        terminal_tokens.append(terminal_token)
    output_indexes = torch.stack(output_rows)
    task_tokens = row_tokens.index_select(0, output_indexes)
    counts = torch.stack(accepted_counts)
    points = torch.stack(selected_points)
    terminal_finish = torch.stack(terminal_finishes)
    task_tokens = torch.where(terminal_finish, torch.stack(terminal_tokens), task_tokens)

    # Broadcast the complete selection state so all tensor-parallel ranks
    # advance request state from identical speculative decisions.
    if selection_broadcast is not None:
        selection = torch.stack(
            (
                task_tokens.to(dtype=torch.int64),
                counts.to(dtype=torch.int64),
                points.to(dtype=torch.int64),
                terminal_finish.to(dtype=torch.int64),
            )
        )
        selection_broadcast(selection)
        task_tokens = selection[0].to(dtype=task_tokens.dtype)
        counts = selection[1].to(dtype=counts.dtype)
        points = selection[2].to(dtype=points.dtype)
        terminal_finish = selection[3].to(dtype=torch.bool)

    # A task is valid only when every consumed speculative point retained a
    # legal vocabulary candidate under its sampling constraints.
    task_valid = torch.stack(
        tuple(
            valid[offsets[index] : offsets[index] + len(task.rows)][
                torch.arange(len(task.rows), device=device) < points[index]
            ].all()
            for index, task in enumerate(tasks)
        )
    )
    active = _sample_predicates(tasks, task_tokens.device)
    device_finish, continuation_values = _resolve_sampled_finish_values(
        tasks,
        task_tokens,
        task_valid,
        active,
        terminal_finish,
    )
    _publish_sampled_transition_values(
        tasks,
        task_tokens,
        task_valid,
        active,
        device_products,
        device_reads,
    )

    # Publish device-side dependencies before capturing the compact host
    # completion and any requested log-probability details.
    published = _publish_sampled_device_values(
        tasks,
        "token_product",
        tagged_token_values(task_tokens, continuation_values),
        device_products,
        device_reads,
    )
    span = _capture_sample_span(task_valid, active, task_tokens, counts, completion)
    details = logprob_details(
        work,
        output_indexes - terminal_finish.to(dtype=torch.long),
        task_tokens,
        tuple(task.rows[0] for task in tasks),
        completion,
    )
    return tuple(
        SampleResult(
            completion=SamplingOutputRow(span, index),
            device_token=task_tokens[index : index + 1],
            logprobs=details.get(index),
            device_accepted_tokens=counts[index : index + 1],
            device_selected_point=points[index : index + 1],
            device_valid=task_valid[index : index + 1],
            device_active=active[index : index + 1],
            device_finish=(None if device_finish is None else device_finish[index : index + 1]),
            device_continuation=continuation_values[index : index + 1],
            device_product_published=published,
        )
        for index in range(len(tasks))
    )


def _capture_sample_span(
    valid: torch.Tensor,
    active: torch.Tensor,
    tokens: torch.Tensor,
    accepted: torch.Tensor,
    completion: OutputBuffer | None,
) -> SamplingCapture:
    """Pack validity, activity, token, and acceptance vectors into completion storage."""

    count = int(tokens.numel())
    if (
        int(valid.numel()) != count
        or int(active.numel()) != count
        or int(accepted.numel()) != count
    ):
        raise RuntimeError("sampling completion vectors do not align")
    metadata = torch.cat(
        (
            valid.reshape(-1),
            active.reshape(-1),
            tokens.reshape(-1),
            accepted.reshape(-1),
        )
    )
    if int(metadata.numel()) != SAMPLING_COMPLETION_FIELDS * count:
        raise RuntimeError("sampling completion field count exceeds its fixed capacity")
    owns_completion = completion is None
    if metadata.device.type != "cuda":
        values = tuple(int(value) for value in metadata.tolist())
        if owns_completion and not all(
            bool(values[index]) or not bool(values[count + index]) for index in range(count)
        ):
            raise invalid_descriptor("sampling policy masked every vocabulary entry")
        return SamplingCapture(None, count, values)
    if completion is None:
        raise RuntimeError("CUDA sampling requires a server completion lease")
    span = SamplingCapture(completion.capture(metadata), count)
    return span


def capture_preselected_span(
    output: GraphGreedyOutput,
    completion: OutputBuffer | None,
) -> SamplingCapture:
    """Capture graph-selected token, validity, activity, and acceptance vectors into host output."""

    count = int(output.tokens.numel())
    if int(output.completion.numel()) != SAMPLING_COMPLETION_FIELDS * count:
        raise RuntimeError("graph sampling completion vectors do not align")
    if output.completion.device.type != "cuda":
        values = tuple(int(value) for value in output.completion.tolist())
        return SamplingCapture(None, count, values)
    if completion is None:
        raise RuntimeError("CUDA graph sampling requires a server completion lease")
    return SamplingCapture(completion.capture(output.completion), count)


def _device_finish_values(
    tasks: tuple[SampleWork, ...],
    device_tokens: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Evaluate per-row terminal token policies entirely on the sampling device."""

    count = len(tasks)
    tokens = device_tokens.reshape(-1)
    validity = valid.reshape(-1)
    if int(tokens.numel()) != count or int(validity.numel()) != count:
        raise RuntimeError("sampling finish vectors do not align")
    if count == 0:
        return validity.to(dtype=torch.bool)

    rows = tuple(task.rows[0] for task in tasks)
    first = rows[0]
    if all(
        row.force_finish == first.force_finish and row.finish_token_ids == first.finish_token_ids
        for row in rows[1:]
    ):
        if first.force_finish:
            return validity.to(dtype=torch.bool)
        if not first.finish_token_ids:
            return torch.zeros_like(validity, dtype=torch.bool)
        matched = tokens == first.finish_token_ids[0]
        for token_id in first.finish_token_ids[1:]:
            matched |= tokens == token_id
        return matched & validity

    values: list[torch.Tensor] = []
    for index, row in enumerate(rows):
        selected = tokens[index]
        if row.force_finish:
            finish = torch.ones((), dtype=torch.bool, device=tokens.device)
        elif row.finish_token_ids:
            finish_ids = torch.tensor(
                row.finish_token_ids,
                dtype=tokens.dtype,
                device=tokens.device,
            )
            finish = (finish_ids == selected).any()
        else:
            finish = torch.zeros((), dtype=torch.bool, device=tokens.device)
        values.append(finish & validity[index])
    return torch.stack(values)


def _sample_predicates(
    tasks: tuple[SampleWork, ...],
    device: torch.device,
) -> torch.Tensor:
    """Collect one active predicate per task, decoding tagged continuation values when needed."""

    if tasks and all(task.predicate is not None and task.tagged_predicate for task in tasks):
        predicates = tuple(cast(torch.Tensor, task.predicate).reshape(-1)[:1] for task in tasks)
        packed = packed_tensor_views(predicates)
        if packed is None:
            packed = torch.cat(predicates, dim=0)
        return packed.reshape(-1).ge(TOKEN_CONTINUATION_BIT)
    values = tuple(
        (
            torch.ones((1,), dtype=torch.bool, device=device)
            if task.predicate is None
            else (
                task.predicate.reshape(-1)[:1].ge(TOKEN_CONTINUATION_BIT)
                if task.tagged_predicate
                else task.predicate.reshape(-1)[:1].to(device=device, dtype=torch.bool)
            )
        )
        for task in tasks
    )
    return torch.cat(values, dim=0)


def tagged_token_values(
    tokens: torch.Tensor,
    continuation: torch.Tensor,
    *,
    in_place: bool = False,
) -> torch.Tensor:
    """Pack token identifiers with continuation flags into signed 64-bit relay values."""

    if int(tokens.numel()) != int(continuation.numel()):
        raise RuntimeError("token continuation vector does not align with selected tokens")
    tags = torch.where(continuation.reshape(-1), TOKEN_CONTINUATION_BIT, 0)
    target = tokens.reshape(-1) if in_place else tokens.reshape(-1).clone()
    target.bitwise_or_(tags)
    return target


def copy_runtime_scalar(target: torch.Tensor, value: int | torch.Tensor) -> None:
    """Copy a scalar or one-element tensor into fixed device runtime storage."""

    if isinstance(value, torch.Tensor):
        target.copy_(value.reshape(-1)[:1].to(device=target.device, dtype=target.dtype))
    else:
        target.fill_(int(value))


def request_pool_index(task: ForwardRow) -> torch.Tensor:
    """Return the validated scalar request-pool index carried by a forward row."""

    index = task.request_pool_index
    if index is None:
        raise RuntimeError("forward task has no staged request slot")
    return index


def _packed_required_views(
    values: Sequence[torch.Tensor | None],
    message: str,
) -> torch.Tensor:
    """Return one packed view of required aligned tensors, concatenating when necessary."""

    if not values or any(value is None for value in values):
        raise RuntimeError(message)
    tensors = tuple(cast(torch.Tensor, value).reshape(-1) for value in values)
    packed = packed_tensor_views(tensors)
    return torch.cat(tensors, dim=0) if packed is None else packed


def sample_request_pool_indices(tasks: Sequence[SampleWork]) -> torch.Tensor:
    """Collect one validated request-pool index for each sampling task."""

    return _packed_required_views(
        tuple(task.request_pool_index for task in tasks),
        "sample batch lost its staged request slots",
    )


def sample_result_vector(
    samples: Sequence[SampleResult],
    field_name: str,
) -> torch.Tensor:
    """Concatenate an optional device-result field across sampling rows."""

    return _packed_required_views(
        tuple(cast(torch.Tensor | None, getattr(sample, field_name)) for sample in samples),
        f"sample batch lost device field {field_name}",
    )


def runtime_selected_points(
    selected_values: Sequence[torch.Tensor | None],
    accepted_values: Sequence[torch.Tensor | None],
) -> torch.Tensor:
    """Resolve selected speculative points from explicit indices or acceptance counts."""

    if len(selected_values) != len(accepted_values):
        raise RuntimeError("runtime selected-point columns are not aligned")
    points: list[torch.Tensor] = []
    for selected, accepted in zip(selected_values, accepted_values, strict=True):
        if selected is not None:
            points.append(selected.reshape(-1).to(dtype=torch.int32))
        elif accepted is not None:
            points.append(accepted.reshape(-1).to(dtype=torch.int32) + 1)
        else:
            raise RuntimeError("runtime selected-point publication lost device state")
    packed = packed_tensor_views(points)
    return torch.cat(points, dim=0) if packed is None else packed


def _publish_sampled_device_values(
    tasks: tuple[SampleWork, ...],
    product_field: str,
    device_values: torch.Tensor,
    device_products: DeviceProducts | None,
    device_reads: tuple[DeviceProductRead, ...],
    *,
    producer_event: torch.cuda.Event | None = None,
) -> bool:
    """Publish one aligned sampled-value vector when every task declares the product."""

    products = tuple(getattr(task, product_field) for task in tasks)
    if device_products is None or not all(product is not None for product in products):
        return False
    writes = tuple(cast(DeviceProductWrite, product) for product in products)
    publish_device_writes(
        writes,
        device_values.reshape(-1),
        device_products,
        device_reads,
        producer_event=producer_event,
    )
    return True


def _publish_sampled_transition_values(
    tasks: tuple[SampleWork, ...],
    device_tokens: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    device_products: DeviceProducts | None,
    device_reads: tuple[DeviceProductRead, ...],
) -> None:
    """Publish transition-token matches for tasks that declare transition products."""

    writes, transitions = _sampled_transition_values(
        tasks,
        device_tokens,
        valid,
        active,
    )
    if not writes:
        return
    if device_products is None:
        raise RuntimeError("sampling transition outputs have no device-product owner")
    if transitions is None:
        raise RuntimeError("sampling transition publication lost its device values")
    publish_device_writes(
        writes,
        transitions,
        device_products,
        device_reads,
    )


def _sampled_transition_values(
    tasks: tuple[SampleWork, ...],
    device_tokens: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    *,
    destination: torch.Tensor | None = None,
) -> tuple[tuple[DeviceProductWrite, ...], torch.Tensor | None]:
    """Match selected tokens against per-task transition sets and return aligned writes."""

    # Only operations declaring a transition product participate; indexes keep
    # their token and eligibility rows aligned after filtering.
    selected = tuple(
        (index, task, task.transition_product)
        for index, task in enumerate(tasks)
        if task.transition_product is not None
    )
    if not selected:
        return (), None
    tokens = device_tokens.reshape(-1)
    eligibility = valid.reshape(-1).to(dtype=torch.bool) & active.reshape(-1).to(dtype=torch.bool)
    if int(tokens.numel()) != len(tasks) or int(eligibility.numel()) != len(tasks):
        raise RuntimeError("sampling transition vectors do not align")
    indexes = tuple(index for index, _task, _write in selected)
    writes = tuple(cast(DeviceProductWrite, write) for _index, _task, write in selected)
    selected_tokens = select_device_values(tokens, indexes)
    selected_eligibility = select_device_values(eligibility, indexes)

    # Captured graphs may supply fixed destination storage. Its one-bit row
    # contract must exactly match the selected product subset.
    target: torch.Tensor | None = None
    if destination is not None:
        target = destination.reshape(-1)
        if (
            int(target.numel()) != len(selected)
            or target.device != selected_tokens.device
            or target.dtype is not torch.bool
        ):
            raise RuntimeError("sampling transition destination does not align")

    # Shared transition sets avoid materializing a per-row token matrix; mixed
    # policies use a padded matrix with -1 as the non-token sentinel.
    transition_sets = tuple(task.rows[0].transition_token_ids for _index, task, _write in selected)
    first = transition_sets[0]
    if all(values == first for values in transition_sets[1:]):
        if not first:
            transitions = (
                target.zero_()
                if target is not None
                else torch.zeros_like(selected_eligibility, dtype=torch.bool)
            )
        else:
            if target is None:
                transitions = selected_tokens == int(first[0])
            else:
                torch.eq(selected_tokens, int(first[0]), out=target)
                transitions = target
            for token_id in first[1:]:
                transitions.logical_or_(selected_tokens == int(token_id))
    else:
        width = max(1, *(len(values) for values in transition_sets))
        transition_ids = torch.tensor(
            tuple((*values, *((-1,) * (width - len(values)))) for values in transition_sets),
            dtype=selected_tokens.dtype,
            device=selected_tokens.device,
        )
        transitions = selected_tokens.unsqueeze(1).eq(transition_ids).any(dim=1)
        if target is not None:
            target.copy_(transitions)
            transitions = target
    transitions.logical_and_(selected_eligibility)
    return writes, transitions


def _resolve_sampled_finish_values(
    tasks: tuple[SampleWork, ...],
    device_tokens: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    terminal_finish: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine token, speculative-terminal, validity, and activity finish policies."""

    finish_values = _device_finish_values(tasks, device_tokens, valid & active) | (
        terminal_finish.reshape(-1).to(dtype=torch.bool) & valid & active
    )
    continuation_values = active & valid & ~finish_values
    return finish_values, continuation_values


def select_device_values(values: torch.Tensor, indexes: tuple[int, ...]) -> torch.Tensor:
    """Gather device values at a validated tuple of host-selected indices."""

    flat = values.reshape(-1)
    if len(indexes) == int(flat.numel()) and all(
        index == expected for expected, index in enumerate(indexes)
    ):
        return flat
    views = tuple(flat[index : index + 1] for index in indexes)
    if len(views) == 1:
        return views[0]
    packed = packed_tensor_views(views)
    return torch.cat(views, dim=0) if packed is None else packed.reshape(-1)


def publish_device_writes(
    writes: tuple[DeviceProductWrite, ...],
    device_values: torch.Tensor,
    device_products: DeviceProducts,
    device_reads: tuple[DeviceProductRead, ...],
    *,
    producer_event: torch.cuda.Event | None = None,
) -> torch.cuda.Event | None:
    """Publish a batch of device-resident scalar products with shared producer synchronization."""

    values = device_values.reshape(-1)
    product_batch = device_products.producer_scalar_batch(writes)
    if product_batch is not None and int(product_batch.tensor.numel()) == len(writes):
        product_batch.tensor.reshape(-1).copy_(values)
        device_products.publish_scalar_batch(
            product_batch,
            after_reads=device_reads,
            producer_event=producer_event,
        )
    else:
        device_products.publish_writes(
            writes,
            values,
            producer_event=producer_event,
        )
    return writes[0].producer_event


def semantic_sampling_draws(
    rows: Sequence[SampleRow],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Materialize deterministic uniform RNG draws for sampling rows on the target device."""

    # Each row's draw is the canonical Philox uniform for its semantic
    # coordinate, computed from host-known identity when the descriptor was
    # built. Greedy rows carry a zero draw. Uploading the host-resident vector
    # keeps the request path free of any device-to-host observation.
    return torch.tensor(
        [float(row.draw) for row in rows],
        dtype=torch.float32,
        device=device,
    )


def _shape_sampling_logits_batch(
    logits: torch.Tensor,
    rows: Sequence[SampleRow],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the canonical shaping and truncation order to a logits matrix."""

    work = logits.to(dtype=torch.float32, copy=True)
    row_count, vocab = (int(value) for value in work.shape)
    if row_count != len(rows):
        raise invalid_descriptor("sampling parameters do not align with logits rows")

    allowed_rows: list[int] = []
    allowed_flat: list[int] = []
    for row_index, row in enumerate(rows):
        if row.allowed is None:
            continue
        allowed = tuple(
            dict.fromkeys(int(value) for value in row.allowed if 0 <= int(value) < vocab)
        )
        allowed_rows.append(row_index)
        allowed_flat.extend(row_index * vocab + value for value in allowed)
    if allowed_rows:
        mask = torch.ones_like(work, dtype=torch.bool)
        mask.index_fill_(
            0,
            torch.tensor(allowed_rows, dtype=torch.long, device=work.device),
            False,
        )
        mask.reshape(-1)[torch.tensor(allowed_flat, dtype=torch.long, device=work.device)] = True
        work.masked_fill_(~mask, float("-inf"))

    suppressed_flat = tuple(
        row_index * vocab + value
        for row_index, row in enumerate(rows)
        for value in dict.fromkeys(int(token) for token in row.suppress if 0 <= int(token) < vocab)
    )
    if suppressed_flat:
        work.reshape(-1).index_fill_(
            0,
            torch.tensor(suppressed_flat, dtype=torch.long, device=work.device),
            float("-inf"),
        )

    # Penalties over the device-resident committed count base. Each penalty row
    # carries a dense per-vocabulary count vector (committed generated tokens
    # plus any speculative prefix); repetition is multiplicative and sign-aware,
    # frequency scales with the count, and presence is a flat once-appeared
    # subtraction. Masked (-inf) entries are preserved.
    penalty_rows = [
        row_index for row_index, row in enumerate(rows) if row.penalty_counts is not None
    ]
    if penalty_rows:
        row_index_tensor = torch.tensor(penalty_rows, dtype=torch.long, device=work.device)
        counts = torch.stack(
            [cast(torch.Tensor, rows[row_index].penalty_counts) for row_index in penalty_rows]
        ).to(dtype=work.dtype)
        params = torch.tensor(
            [
                (
                    float(rows[row_index].parameters.repetition_penalty),
                    float(rows[row_index].parameters.frequency_penalty),
                    float(rows[row_index].parameters.presence_penalty),
                )
                for row_index in penalty_rows
            ],
            dtype=work.dtype,
            device=work.device,
        )
        repetition = params[:, 0].unsqueeze(1)
        frequency = params[:, 1].unsqueeze(1)
        presence = params[:, 2].unsqueeze(1)
        values = work.index_select(0, row_index_tensor)
        seen = counts > 0
        repeated = torch.where(values > 0.0, values / repetition, values * repetition)
        adjusted = repeated - frequency * counts - presence
        apply = seen & ~torch.isneginf(values)
        work.index_copy_(0, row_index_tensor, torch.where(apply, adjusted, values))

    bias_indexes: list[int] = []
    bias_values: list[float] = []
    for row_index, row in enumerate(rows):
        for token_id, bias in row.parameters.logit_bias:
            index = int(token_id)
            if 0 <= index < vocab:
                bias_indexes.append(row_index * vocab + index)
                bias_values.append(float(bias))
    if bias_indexes:
        work.reshape(-1).index_put_(
            (
                torch.tensor(
                    bias_indexes,
                    dtype=torch.long,
                    device=work.device,
                ),
            ),
            torch.tensor(bias_values, dtype=work.dtype, device=work.device),
            accumulate=True,
        )

    parameter_values = torch.tensor(
        [
            (
                float(row.parameters.temperature),
                float(row.parameters.min_p),
                float(row.parameters.top_p),
            )
            for row in rows
        ],
        dtype=work.dtype,
        device=work.device,
    )
    temperatures = parameter_values[:, 0]
    divisors = torch.where(
        temperatures > 0.0,
        temperatures,
        torch.ones((), dtype=work.dtype, device=work.device),
    )
    work.div_(divisors.unsqueeze(1))

    top_k_groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        top_k = int(row.parameters.top_k)
        if 0 < top_k < vocab:
            top_k_groups[top_k].append(index)
    for top_k, row_indexes in top_k_groups.items():
        indexes = torch.tensor(row_indexes, dtype=torch.long, device=work.device)
        subset = work.index_select(0, indexes)
        values, token_indexes = torch.topk(
            subset,
            top_k,
            dim=-1,
            sorted=False,
        )
        ordered, order = torch.sort(values, dim=-1, descending=True)
        token_indexes = token_indexes.gather(1, order)
        top_p = parameter_values.index_select(0, indexes)[:, 2]
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        over = cumulative > top_p.unsqueeze(1)
        drop = torch.cat(
            (
                torch.zeros(
                    (len(row_indexes), 1),
                    dtype=torch.bool,
                    device=work.device,
                ),
                over[:, :-1],
            ),
            dim=1,
        )
        ordered.masked_fill_(drop, float("-inf"))
        truncated = torch.full_like(subset, float("-inf"))
        truncated.scatter_(1, token_indexes, ordered)
        work.index_copy_(0, indexes, truncated)

    top_p_rows = tuple(
        index
        for index, row in enumerate(rows)
        if (0.0 < float(row.parameters.top_p) < 1.0 and not 0 < int(row.parameters.top_k) < vocab)
    )
    if top_p_rows:
        indexes = torch.tensor(top_p_rows, dtype=torch.long, device=work.device)
        subset = work.index_select(0, indexes)
        ordered, token_indexes = torch.sort(subset, dim=-1, descending=True)
        top_p = parameter_values.index_select(0, indexes)[:, 2]
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        over = cumulative > top_p.unsqueeze(1)
        drop = torch.cat(
            (
                torch.zeros(
                    (len(top_p_rows), 1),
                    dtype=torch.bool,
                    device=work.device,
                ),
                over[:, :-1],
            ),
            dim=1,
        )
        ordered.masked_fill_(drop, float("-inf"))
        truncated = torch.full_like(subset, float("-inf"))
        truncated.scatter_(1, token_indexes, ordered)
        work.index_copy_(0, indexes, truncated)

    min_p_rows = tuple(index for index, row in enumerate(rows) if float(row.parameters.min_p) > 0.0)
    if min_p_rows:
        indexes = torch.tensor(min_p_rows, dtype=torch.long, device=work.device)
        subset = work.index_select(0, indexes)
        min_p = parameter_values.index_select(0, indexes)[:, 1]
        min_threshold = subset.max(dim=-1).values + torch.log(min_p)
        subset.masked_fill_(subset < min_threshold.unsqueeze(1), float("-inf"))
        work.index_copy_(0, indexes, subset)

    typical_rows = tuple(
        index for index, row in enumerate(rows) if float(row.parameters.typical_p) < 1.0
    )
    if typical_rows:
        indexes = torch.tensor(typical_rows, dtype=torch.long, device=work.device)
        subset = work.index_select(0, indexes)
        typical_p = torch.tensor(
            [float(rows[index].parameters.typical_p) for index in typical_rows],
            dtype=work.dtype,
            device=work.device,
        )
        probs = torch.softmax(subset, dim=-1)
        log_probs = probs.log()
        entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1, keepdim=True)
        scores = ((-log_probs) - entropy).abs()
        order = torch.argsort(scores, dim=-1)
        cumulative = probs.gather(1, order).cumsum(dim=-1)
        over = cumulative >= typical_p.unsqueeze(1)
        drop = torch.cat(
            (
                torch.zeros((len(typical_rows), 1), dtype=torch.bool, device=work.device),
                over[:, :-1],
            ),
            dim=1,
        )
        subset.scatter_(1, order, subset.gather(1, order).masked_fill(drop, float("-inf")))
        work.index_copy_(0, indexes, subset)

    valid = (
        ~torch.isnan(work).any(dim=-1)
        & ~torch.isposinf(work).any(dim=-1)
        & torch.isfinite(work).any(dim=-1)
    )
    return work, valid


def logprob_details(
    work: torch.Tensor,
    output_rows: torch.Tensor,
    output_tokens: torch.Tensor,
    rows: Sequence[SampleRow],
    completion: OutputBuffer | None,
) -> Mapping[int, LogprobOutputRow]:
    """Compute selected-token, top-k, and requested-token log probabilities and ranks."""

    vocab = int(work.shape[1])
    requested_rows = tuple(
        index
        for index, row in enumerate(rows)
        if row.parameters.return_logprobs
        or int(row.n_logprobs) > 0
        or bool(row.parameters.logprob_token_ids)
    )
    if not requested_rows:
        return {}
    request_indexes = torch.tensor(
        requested_rows,
        dtype=torch.long,
        device=work.device,
    )
    score_rows = output_rows.index_select(0, request_indexes)
    selected_tokens = output_tokens.index_select(0, request_indexes)
    scores = torch.log_softmax(work.index_select(0, score_rows), dim=-1)
    selected_values = scores.gather(1, selected_tokens.unsqueeze(1))[:, 0]
    selected_ranks = (scores > selected_values.unsqueeze(1)).sum(dim=-1, dtype=torch.long) + 1

    counts = tuple(min(max(0, int(rows[index].n_logprobs)), vocab) for index in requested_rows)
    max_count = max(counts, default=0)
    if max_count:
        top_values, top_indexes = torch.topk(scores, max_count, dim=-1, sorted=True)
        positions = torch.arange(
            1,
            max_count + 1,
            dtype=torch.long,
            device=work.device,
        ).unsqueeze(0)
        starts = torch.cat(
            (
                torch.ones(
                    (len(requested_rows), 1),
                    dtype=torch.bool,
                    device=work.device,
                ),
                top_values[:, 1:] < top_values[:, :-1],
            ),
            dim=1,
        )
        top_ranks = torch.where(starts, positions, 0).cummax(dim=1).values
    else:
        top_values = torch.empty(
            (len(requested_rows), 0),
            dtype=scores.dtype,
            device=work.device,
        )
        top_indexes = torch.empty(
            (len(requested_rows), 0),
            dtype=torch.long,
            device=work.device,
        )
        top_ranks = torch.empty_like(top_indexes)

    requested_ids = tuple(
        tuple(
            dict.fromkeys(
                int(value)
                for value in rows[index].parameters.logprob_token_ids
                if 0 <= int(value) < vocab
            )
        )
        for index in requested_rows
    )
    max_requested = max((len(value) for value in requested_ids), default=0)
    if max_requested:
        candidate_indexes = torch.zeros(
            (len(requested_rows), max_requested),
            dtype=torch.long,
            device=work.device,
        )
        for row_index, values in enumerate(requested_ids):
            if values:
                candidate_indexes[row_index, : len(values)] = torch.tensor(
                    values,
                    dtype=torch.long,
                    device=work.device,
                )
        candidate_values = scores.gather(1, candidate_indexes)
        candidate_ranks = torch.stack(
            tuple(
                (scores > candidate_values[:, index].unsqueeze(1)).sum(dim=-1, dtype=torch.long) + 1
                for index in range(max_requested)
            ),
            dim=1,
        )
    else:
        candidate_values = torch.empty(
            (len(requested_rows), 0),
            dtype=scores.dtype,
            device=work.device,
        )
        candidate_ranks = torch.empty(
            (len(requested_rows), 0),
            dtype=torch.long,
            device=work.device,
        )

    def float_bits(values: torch.Tensor) -> torch.Tensor:
        """Encode float32 values as unsigned-preserving integer bit patterns."""

        return values.to(dtype=torch.float32).contiguous().view(torch.int32).to(torch.long)

    packed = torch.cat(
        (
            selected_tokens.reshape(-1).to(torch.long),
            float_bits(selected_values.reshape(-1)),
            selected_ranks.reshape(-1).to(torch.long),
            top_indexes.reshape(-1).to(torch.long),
            float_bits(top_values.reshape(-1)),
            top_ranks.reshape(-1).to(torch.long),
            float_bits(candidate_values.reshape(-1)),
            candidate_ranks.reshape(-1).to(torch.long),
        )
    )
    if packed.device.type != "cuda":
        values = tuple(int(value) for value in packed.tolist())
        batch = LogprobCapture(
            None,
            requested_rows,
            counts,
            requested_ids,
            max_count,
            max_requested,
            values,
        )
    else:
        if completion is None:
            raise RuntimeError("CUDA logprob materialization requires a server completion lease")
        batch = LogprobCapture(
            completion.capture(packed),
            requested_rows,
            counts,
            requested_ids,
            max_count,
            max_requested,
        )
    return {
        result_index: LogprobOutputRow(batch, result_index)
        for result_index in requested_rows
    }
