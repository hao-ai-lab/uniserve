"""Batched numerical token selection, predicates, and speculative acceptance."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import cast

import torch

from uniserve_worker.backends.triton import triton_available
from uniserve_worker.execution.forward_batch import packed_tensor_views
from uniserve_worker.foundation.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.protocol.batch import SamplingParams

from ..nn.mesh import Communicator
from .sampling import LogprobValues, SamplerOutput, SamplerRow, SamplingMetadata
from .top_k_sampling import sample_top_k

SAMPLING_COMPLETION_FIELDS = 4
TOKEN_CONTINUATION_BIT = 1 << 31
TOKEN_VALUE_MASK = TOKEN_CONTINUATION_BIT - 1


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


@torch.inference_mode()
def sample(
    tasks: Sequence[SamplingMetadata],
    *,
    selection_broadcast: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[SamplerRow, ...]:
    """Shape and draw every compatible sampling row in each device batch."""

    grouped: dict[tuple[torch.device, int, int], list[tuple[int, SamplingMetadata]]] = defaultdict(
        list
    )
    for index, task in enumerate(tasks):
        if (
            task.logits.ndim != 2
            or not task.logits.is_floating_point()
            or int(task.logits.shape[0]) < 1
            or int(task.logits.shape[1]) < 1
            or int(task.logits.shape[0]) != len(task.allowed)
            or len(task.penalty_counts) != len(task.allowed)
        ):
            raise invalid_descriptor("sampling task logits must be shaped [rows, vocab]")
        device_greedy = (
            not task.draft_token_ids
            and device_greedy_parameters(task.parameters)
            and all(value is None for value in task.allowed)
        )
        if device_greedy:
            if (
                any(
                    value is not None
                    for value in (
                        task.draws,
                        task.parameter_values,
                    )
                )
                or task.draft_token_ids
                or len(task.allowed) != 1
            ):
                raise invalid_descriptor("greedy sampling task has shaped metadata")
        else:
            draws = cast(torch.Tensor, task.draws)
            parameter_values = cast(torch.Tensor, task.parameter_values)
            if (
                draws.device != task.logits.device
                or tuple(draws.shape) != (len(task.allowed),)
                or not draws.is_floating_point()
            ):
                raise invalid_descriptor("sampling task draws must align with its rows")
            if parameter_values.device != task.logits.device or parameter_values.shape != (
                len(task.allowed),
                3,
            ):
                raise invalid_descriptor("sampling task parameter vectors do not align")
        if task.draft_token_ids:
            if len(task.allowed) != len(task.draft_token_ids) + 1:
                raise invalid_descriptor("speculative sampling rows do not cover the draft chain")
        elif len(task.allowed) != 1:
            raise invalid_descriptor("ordinary sampling tasks must contain exactly one row")
        vocab = int(task.logits.shape[1])
        if vocab > TOKEN_VALUE_MASK:
            raise unsupported_setup("vocabulary exceeds the device token decision range")
        if any(value < 0 or value >= vocab for value in task.draft_token_ids):
            raise invalid_descriptor("speculative draft token is outside the model vocabulary")
        sampling_path = (
            (-2 if task.suppress else -1) if device_greedy else _fused_top_k(task, vocab)
        )
        grouped[(task.logits.device, vocab, sampling_path)].append((index, task))

    result: list[SamplerRow | None] = [None] * len(tasks)
    for (_device, _vocab, sampling_path), compatible in grouped.items():
        indexes, group = zip(*compatible, strict=True)
        if sampling_path < 0:
            sampled_group = sample_device_greedy_group(
                tuple(group),
                apply_suppression=sampling_path == -2,
                selection_broadcast=selection_broadcast,
            )
        elif sampling_path > 0:
            sampled_group = _sample_fused_top_k_group(
                tuple(group),
                sampling_path,
                selection_broadcast=selection_broadcast,
            )
        else:
            sampled_group = _sample_task_group(
                tuple(group),
                selection_broadcast=selection_broadcast,
            )
        for index, sampled in zip(indexes, sampled_group, strict=True):
            result[index] = sampled
    return tuple(cast(SamplerRow, value) for value in result)


def sample_device_greedy_group(
    tasks: tuple[SamplingMetadata, ...],
    *,
    apply_suppression: bool,
    selection_broadcast: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[SamplerRow, ...]:
    """Resolve predicates, greedy tokens, finish state, and completion values for a group."""

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
            for token_id in dict.fromkeys(int(value) for value in task.suppress):
                if 0 <= token_id < vocab:
                    selection_logits[row_index, token_id].fill_(float("-inf"))
    device_tokens = torch.argmax(selection_logits, dim=-1)
    valid = (
        ~torch.isnan(selection_logits).any(dim=-1)
        & ~torch.isposinf(selection_logits).any(dim=-1)
        & torch.isfinite(selection_logits).any(dim=-1)
    )
    if selection_broadcast is not None:
        selection_broadcast(device_tokens)
    active = _sample_predicates(tasks, device_tokens.device)
    device_finish, continuation_values = _resolve_sampled_finish_values(
        tasks, device_tokens, valid, active, torch.zeros_like(active, dtype=torch.bool)
    )
    transitions = _sampled_transition_values(tasks, device_tokens, valid, active)
    span = sampling_columns(valid, active, device_tokens, torch.zeros_like(device_tokens))
    tagged_tokens = tagged_token_values(device_tokens, continuation_values, in_place=False)
    output = SamplerOutput(
        tokens=device_tokens,
        valid=valid,
        active=active,
        finish=device_finish,
        continuation=continuation_values,
        tagged_tokens=tagged_tokens,
        completion=span,
    )
    return tuple(
        output.row(
            index,
            request_pool_index=task.request_pool_index,
            transition=transitions.get(index),
        )
        for index, task in enumerate(tasks)
    )


def _fused_top_k(task: SamplingMetadata, vocab: int) -> int:
    """Return a fused top-k width when the task satisfies the kernel's sampling contract."""

    if task.logits.device.type != "cuda":
        return 0
    if task.draft_token_ids:
        return 0
    parameters = task.parameters
    top_k = int(parameters.top_k)
    wants_logprobs = (
        parameters.return_logprobs
        or int(parameters.n_logprobs) > 0
        or bool(parameters.logprob_token_ids)
    )
    uses_penalties = (
        parameters.repetition_penalty != 1.0
        or parameters.frequency_penalty != 0.0
        or parameters.presence_penalty != 0.0
    )
    if (
        wants_logprobs
        or uses_penalties
        or task.allowed[0] is not None
        or task.suppress
        or parameters.logit_bias
        or float(parameters.typical_p) < 1.0
        or top_k <= 0
        or top_k > 128
        or top_k >= vocab
    ):
        return 0
    return top_k


def _sample_fused_top_k_group(
    tasks: tuple[SamplingMetadata, ...],
    top_k: int,
    *,
    selection_broadcast: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[SamplerRow, ...]:
    """Sample a homogeneous fused-top-k group and return its numerical selections."""

    # The compiled kernel consumes one contiguous column for each sampling
    # input, so compatible task rows are packed before a single launch.
    logits = torch.cat(tuple(task.logits for task in tasks), dim=0)
    draws = torch.cat(tuple(cast(torch.Tensor, task.draws) for task in tasks), dim=0)
    parameters = torch.cat(
        tuple(cast(torch.Tensor, task.parameter_values) for task in tasks), dim=0
    )
    tokens, valid = _run_fused_top_k_sampling(
        logits,
        draws,
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
    transitions = _sampled_transition_values(tasks, tokens, valid, active)
    tagged_tokens = tagged_token_values(tokens, continuation_values)
    span = sampling_columns(valid, active, tokens, torch.zeros_like(tokens))
    output = SamplerOutput(
        tokens=tokens,
        valid=valid,
        active=active,
        finish=device_finish,
        continuation=continuation_values,
        tagged_tokens=tagged_tokens,
        completion=span,
    )
    return tuple(
        output.row(
            index,
            request_pool_index=task.request_pool_index,
            transition=transitions.get(index),
        )
        for index, task in enumerate(tasks)
    )


def _run_fused_top_k_sampling(
    logits: torch.Tensor,
    draws: torch.Tensor,
    parameters: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the compiled fixed-top-k sampler on a CUDA device."""

    if not triton_available(logits.device):
        raise unsupported_setup("fused top-k sampling requires a CUDA device")
    return sample_top_k(
        logits,
        draws,
        parameters,
        int(top_k),
    )


def _sample_task_group(
    tasks: tuple[SamplingMetadata, ...],
    *,
    selection_broadcast: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[SamplerRow, ...]:
    """Sample arbitrary compatible tasks, resolve speculative acceptance, and capture outputs."""

    # Flatten task-local candidate rows into one sampling matrix while retaining
    # offsets needed to restore one selected result per operation.
    device = tasks[0].logits.device
    vocab = int(tasks[0].logits.shape[1])
    offsets: list[int] = []
    offset = 0
    for task in tasks:
        offsets.append(offset)
        offset += len(task.allowed)
    parameters = tuple(task.parameters for task in tasks for _ in task.allowed)
    logits = torch.cat(tuple(task.logits.float() for task in tasks), dim=0)
    draws = torch.cat(tuple(cast(torch.Tensor, task.draws) for task in tasks), dim=0)
    work, valid = _shape_sampling_logits_batch(
        logits,
        parameters,
        tuple(value for task in tasks for value in task.allowed),
        tuple(task.suppress for task in tasks for _ in task.allowed),
        tuple(value for task in tasks for value in task.penalty_counts),
    )

    # Temperature-zero rows use deterministic argmax; remaining rows invert the
    # categorical CDF with their precomputed RNG draw.
    temperatures = torch.tensor(
        [float(value.temperature) for value in parameters],
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
    accepted_token_counts: list[torch.Tensor] = []
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
        accepted_token_counts.append(accepted + (~terminal).to(dtype=torch.long))
        terminal_finishes.append(terminal)
        output_rows.append(accepted + row_offset)
        terminal_tokens.append(terminal_token)
    output_indexes = torch.stack(output_rows)
    task_tokens = row_tokens.index_select(0, output_indexes)
    counts = torch.stack(accepted_counts)
    points = torch.stack(accepted_token_counts)
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
            valid[offsets[index] : offsets[index] + len(task.allowed)][
                torch.arange(len(task.allowed), device=device) < points[index]
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
    transitions = _sampled_transition_values(tasks, task_tokens, task_valid, active)
    tagged_tokens = tagged_token_values(task_tokens, continuation_values)
    span = sampling_columns(task_valid, active, task_tokens, counts)
    details = logprob_details(
        work,
        output_indexes - terminal_finish.to(dtype=torch.long),
        task_tokens,
        tuple(task.parameters for task in tasks),
    )
    output = SamplerOutput(
        tokens=task_tokens,
        valid=task_valid,
        active=active,
        finish=device_finish,
        continuation=continuation_values,
        tagged_tokens=tagged_tokens,
        completion=span,
        logprobs=details,
        accepted_draft_count=counts,
        accepted_token_count=points,
    )
    return tuple(
        output.row(
            index,
            request_pool_index=task.request_pool_index,
            transition=transitions.get(index),
        )
        for index, task in enumerate(tasks)
    )


def sampling_columns(
    valid: torch.Tensor,
    active: torch.Tensor,
    tokens: torch.Tensor,
    accepted: torch.Tensor,
) -> torch.Tensor:
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
    return metadata


def sampled_finish_values(
    finish_token_ids: tuple[tuple[int, ...], ...],
    force_finish: tuple[bool, ...],
    device_tokens: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Evaluate per-row terminal token policies entirely on the sampling device."""

    count = len(finish_token_ids)
    tokens = device_tokens.reshape(-1)
    validity = valid.reshape(-1)
    if len(force_finish) != count or int(tokens.numel()) != count or int(validity.numel()) != count:
        raise RuntimeError("sampling finish vectors do not align")
    if count == 0:
        return validity.to(dtype=torch.bool)

    first_ids = finish_token_ids[0]
    first_force = force_finish[0]
    if all(
        forced == first_force and ids == first_ids
        for forced, ids in zip(force_finish[1:], finish_token_ids[1:], strict=True)
    ):
        if first_force:
            return validity.to(dtype=torch.bool)
        if not first_ids:
            return torch.zeros_like(validity, dtype=torch.bool)
        matched = tokens == first_ids[0]
        for token_id in first_ids[1:]:
            matched |= tokens == token_id
        return matched & validity

    values: list[torch.Tensor] = []
    for index, (ids, forced) in enumerate(zip(finish_token_ids, force_finish, strict=True)):
        selected = tokens[index]
        if forced:
            finish = torch.ones((), dtype=torch.bool, device=tokens.device)
        elif ids:
            finish_ids = torch.tensor(
                ids,
                dtype=tokens.dtype,
                device=tokens.device,
            )
            finish = (finish_ids == selected).any()
        else:
            finish = torch.zeros((), dtype=torch.bool, device=tokens.device)
        values.append(finish & validity[index])
    return torch.stack(values)


def _sample_predicates(
    tasks: tuple[SamplingMetadata, ...],
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


def _sampled_transition_values(
    tasks: tuple[SamplingMetadata, ...],
    device_tokens: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    *,
    destination: torch.Tensor | None = None,
) -> dict[int, torch.Tensor]:
    """Match selected tokens against requested transition sets and return numerical row views."""

    # Only operations declaring a transition product participate; indexes keep
    # their token and eligibility rows aligned after filtering.
    selected = tuple((index, task) for index, task in enumerate(tasks) if task.return_transition)
    if not selected:
        return {}
    tokens = device_tokens.reshape(-1)
    eligibility = valid.reshape(-1).to(dtype=torch.bool) & active.reshape(-1).to(dtype=torch.bool)
    if int(tokens.numel()) != len(tasks) or int(eligibility.numel()) != len(tasks):
        raise RuntimeError("sampling transition vectors do not align")
    indexes = tuple(index for index, _task in selected)
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
    transition_sets = tuple(task.transition_token_ids for _index, task in selected)
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
    return {index: transitions[row : row + 1] for row, (index, _task) in enumerate(selected)}


def _resolve_sampled_finish_values(
    tasks: tuple[SamplingMetadata, ...],
    device_tokens: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    terminal_finish: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine token, speculative-terminal, validity, and activity finish policies."""

    finish_values = sampled_finish_values(
        tuple(task.finish_token_ids for task in tasks),
        tuple(task.force_finish for task in tasks),
        device_tokens,
        valid & active,
    ) | (terminal_finish.reshape(-1).to(dtype=torch.bool) & valid & active)
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


def _shape_sampling_logits_batch(
    logits: torch.Tensor,
    parameters: Sequence[SamplingParams],
    allowed_tokens: Sequence[tuple[int, ...] | None],
    suppressed_tokens: Sequence[tuple[int, ...]],
    penalty_counts: Sequence[torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the canonical shaping and truncation order to a logits matrix."""

    work = logits.to(dtype=torch.float32, copy=True)
    row_count, vocab = (int(value) for value in work.shape)
    if any(
        row_count != len(column)
        for column in (parameters, allowed_tokens, suppressed_tokens, penalty_counts)
    ):
        raise invalid_descriptor("sampling parameters do not align with logits rows")

    allowed_rows: list[int] = []
    allowed_flat: list[int] = []
    for row_index, allowed_values in enumerate(allowed_tokens):
        if allowed_values is None:
            continue
        allowed = tuple(
            dict.fromkeys(int(value) for value in allowed_values if 0 <= int(value) < vocab)
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
        for row_index, suppressed in enumerate(suppressed_tokens)
        for value in dict.fromkeys(int(token) for token in suppressed if 0 <= int(token) < vocab)
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
        row_index for row_index, value in enumerate(penalty_counts) if value is not None
    ]
    if penalty_rows:
        row_index_tensor = torch.tensor(penalty_rows, dtype=torch.long, device=work.device)
        counts = torch.stack(
            [cast(torch.Tensor, penalty_counts[row_index]) for row_index in penalty_rows]
        ).to(dtype=work.dtype)
        params = torch.tensor(
            [
                (
                    float(parameters[row_index].repetition_penalty),
                    float(parameters[row_index].frequency_penalty),
                    float(parameters[row_index].presence_penalty),
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
    for row_index, row in enumerate(parameters):
        for token_id, bias in row.logit_bias:
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
                float(row.temperature),
                float(row.min_p),
                float(row.top_p),
            )
            for row in parameters
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
    for index, row in enumerate(parameters):
        top_k = int(row.top_k)
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
        for index, row in enumerate(parameters)
        if (0.0 < float(row.top_p) < 1.0 and not 0 < int(row.top_k) < vocab)
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

    min_p_rows = tuple(index for index, row in enumerate(parameters) if float(row.min_p) > 0.0)
    if min_p_rows:
        indexes = torch.tensor(min_p_rows, dtype=torch.long, device=work.device)
        subset = work.index_select(0, indexes)
        min_p = parameter_values.index_select(0, indexes)[:, 1]
        min_threshold = subset.max(dim=-1).values + torch.log(min_p)
        subset.masked_fill_(subset < min_threshold.unsqueeze(1), float("-inf"))
        work.index_copy_(0, indexes, subset)

    typical_rows = tuple(
        index for index, row in enumerate(parameters) if float(row.typical_p) < 1.0
    )
    if typical_rows:
        indexes = torch.tensor(typical_rows, dtype=torch.long, device=work.device)
        subset = work.index_select(0, indexes)
        typical_p = torch.tensor(
            [float(parameters[index].typical_p) for index in typical_rows],
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
    parameters: Sequence[SamplingParams],
) -> LogprobValues | None:
    """Compute selected-token, top-k, and requested-token log probabilities and ranks."""

    vocab = int(work.shape[1])
    requested_rows = tuple(
        index
        for index, row in enumerate(parameters)
        if row.return_logprobs or int(row.n_logprobs) > 0 or bool(row.logprob_token_ids)
    )
    if not requested_rows:
        return None
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

    counts = tuple(
        min(max(0, int(parameters[index].n_logprobs)), vocab) for index in requested_rows
    )
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
                for value in parameters[index].logprob_token_ids
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
    return packed, requested_rows, counts, requested_ids, max_count, max_requested


def broadcast_selection(group: Communicator | None, value: torch.Tensor) -> torch.Tensor:
    """Publish selected tokens through the bound tensor-parallel communicator."""

    return value if group is None else group.broadcast(value, src=0)
