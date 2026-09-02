"""Model-runner integration for round-level sampling tasks through the real forward.

Sampling is device postprocessing inside the token modes, so inline rows from
one submission round share a single batched sampling task, and a verifier
submits its position rows as one task. The committed tokens are the serial
oracle the stub model defines through its deterministic next-token map.
"""

from __future__ import annotations

import struct
import time
from dataclasses import replace
from typing import cast

import pytest

from tests.python.fixtures.depth_one import (
    commit_for_completion,
    execution_batch,
    root_parent,
    token_operation,
    und_admission,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.execution.batch import (
    CompletionReport,
    DrawLayout,
    DType,
    ErrorCode,
    NewRequest,
    Operation,
    OpStatus,
    PointRange,
    ProductKind,
    ProductPayload,
    ProductRef,
    Rng,
    SamplingParams,
    SamplingState,
    ShapeBound,
    StaticDim,
    StorageClass,
    TokenMode,
    UndAdmission,
    encode_sampling_state_bytes,
)
from uniserve_worker.execution.output import (
    completion_report_ready,
    finalize_completion_report,
)
from uniserve_worker.models.stub import STUB_IMG_START_TOKEN_ID, _next_token

pytestmark = pytest.mark.integration


def _logprob_positions(payload: bytes) -> tuple[tuple[tuple[int, float, int], ...], ...]:
    offset = 0
    sampled = payload[offset]
    offset += 1 + (4 if sampled else 0)

    def read_u32() -> int:
        nonlocal offset
        value = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        return value

    def read_entries() -> tuple[tuple[int, float, int], ...]:
        nonlocal offset
        entries = []
        for _ in range(read_u32()):
            token_id, logprob, rank = struct.unpack_from("<IfI", payload, offset)
            offset += 12
            entries.append((token_id, logprob, rank))
        return tuple(entries)

    read_entries()
    positions = tuple(read_entries() for _ in range(read_u32()))
    assert offset == len(payload)
    return positions


def _materialize(report: CompletionReport) -> CompletionReport:
    deadline = time.monotonic() + 5.0
    while not completion_report_ready(report) and time.monotonic() < deadline:
        time.sleep(0.0001)
    assert completion_report_ready(report)
    return finalize_completion_report(report)


def _with_sampling_state(
    operation: Operation,
    state: SamplingState,
) -> tuple[Operation, ProductPayload]:
    payload = encode_sampling_state_bytes(state)
    reference = ProductRef(
        request_key=operation.request_key,
        producer_op_id=operation.op_id,
        output_index=(1 << 16) - 2,
        generation=operation.op_id * 3 + 2,
        kind=ProductKind.SAMPLING_STATE,
        storage_class=StorageClass.HOST_STAGING,
        dtype=DType.U8,
        shape_bound=ShapeBound((StaticDim(len(payload)),)),
        point_range=PointRange(),
    )
    registered = Operation.registered(
        request_key=operation.request_key,
        op_id=operation.op_id,
        parent=operation.parent,
        work=operation.work,
        route=operation.route,
        domain=operation.domain,
        bounds=operation.bounds,
        inputs=(*operation.inputs, reference),
        outputs=operation.outputs,
        predicate=operation.predicate,
        rng=operation.rng,
        control_seq=operation.control_seq,
    )
    return registered, ProductPayload(reference, payload)


def test_logprob_reporting_does_not_change_sample_selection() -> None:
    worker = execution_worker()
    sampling = SamplingParams(temperature=0.8, top_k=4, top_p=0.9, seed=71)
    first = und_admission(11, block_ids=(2,), sampling=sampling)
    second = und_admission(
        12,
        block_ids=(3,),
        sampling=replace(sampling, return_logprobs=True, n_logprobs=2),
    )
    first_op, first_input = token_operation(
        first.request_key,
        op_id=1,
        parent=root_parent(first),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
        rng=Rng(seed=71, semantic_index_base=2, draw_layout=DrawLayout.TARGET_SAMPLING),
    )
    second_op, second_input = token_operation(
        second.request_key,
        op_id=2,
        parent=root_parent(second),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
        logprobs=True,
        rng=Rng(seed=71, semantic_index_base=2, draw_layout=DrawLayout.TARGET_SAMPLING),
    )

    result = _materialize(
        worker.execute(
            execution_batch(
                step_id=1,
                admissions=(first, second),
                operations=(first_op, second_op),
                input_products=(first_input, second_input),
            )
        )
    )

    assert result.completions[0].committed_tokens == result.completions[1].committed_tokens


def test_batched_decode_produces_the_serial_oracle_tokens() -> None:
    worker = execution_worker()
    admissions = (und_admission(21, block_ids=(0,)), und_admission(22, block_ids=(1,)))
    primed: list[tuple[Operation, CompletionReport]] = []
    for index, admission in enumerate(admissions):
        extend, extend_input = token_operation(
            admission.request_key,
            op_id=1 + index,
            parent=root_parent(admission),
            mode=TokenMode.EXTEND,
            tokens=(3, 4),
        )
        report = worker.execute(
            execution_batch(
                step_id=1 + index,
                admissions=(admission,),
                operations=(extend,),
                input_products=(extend_input,),
            )
        )
        primed.append((extend, report))

    decode_ops = []
    decode_inputs = []
    commits = []
    for index, (admission, (extend, report)) in enumerate(zip(admissions, primed, strict=True)):
        commit = commit_for_completion(extend, report)
        operation, payload = token_operation(
            admission.request_key,
            op_id=3 + index,
            parent=commit.selected,
            mode=TokenMode.DECODE,
            tokens=(_next_token(4),),
            control_seq=commit.control_seq,
        )
        decode_ops.append(operation)
        decode_inputs.append(payload)
        commits.append(commit)
    result = _materialize(
        worker.execute(
            execution_batch(
                step_id=9,
                admissions=(),
                operations=tuple(decode_ops),
                controls=tuple(commits),
                input_products=tuple(decode_inputs),
            )
        )
    )

    expected = _next_token(_next_token(4))
    assert result.completions[0].committed_tokens == (expected,)
    assert result.completions[1].committed_tokens == (expected,)
    assert tuple(completion.selected_point for completion in result.completions) == (1, 1)


def test_sampling_batch_returns_serial_tokens_for_mixed_finish_policies() -> None:
    worker = execution_worker()
    first = und_admission(24, block_ids=(3,))
    second_base = und_admission(25, block_ids=(4,))
    expected = _next_token(4)
    assert second_base.und is not None
    second = NewRequest.create(
        second_base.request_key,
        request_pool_idx=second_base.request_pool_idx,
        und=replace(second_base.und, finish_token_ids=(expected,)),
    )
    first_op, first_input = token_operation(
        first.request_key,
        op_id=1,
        parent=root_parent(first),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    second_op, second_input = token_operation(
        second.request_key,
        op_id=2,
        parent=root_parent(second),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )

    result = _materialize(
        worker.execute(
            execution_batch(
                step_id=1,
                admissions=(first, second),
                operations=(first_op, second_op),
                input_products=(first_input, second_input),
            )
        )
    )

    assert result.completions[0].committed_tokens == (expected,)
    assert result.completions[1].committed_tokens == (expected,)
    assert result.completions[0].product_generations == tuple(
        output.generation for output in first_op.outputs
    )
    assert result.completions[1].product_generations == tuple(
        output.generation for output in second_op.outputs
    )


def test_verify_commits_every_accepted_position() -> None:
    worker = execution_worker()
    admission = und_admission(
        4, block_ids=(3,), sampling=SamplingParams(return_logprobs=True, n_logprobs=2, seed=31)
    )
    # Prime the request, then carry its selected token explicitly with the draft.
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
        logprobs=True,
    )
    prime = worker.execute(
        execution_batch(
            step_id=1, admissions=(admission,), operations=(extend,), input_products=(extend_input,)
        )
    )
    commit = commit_for_completion(extend, prime)
    verify, verify_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=commit.selected,
        mode=TokenMode.VERIFY,
        tokens=(1000, 1001, STUB_IMG_START_TOKEN_ID),
        logprobs=True,
        control_seq=commit.control_seq,
    )
    result = _materialize(
        worker.execute(
            execution_batch(
                step_id=2,
                admissions=(),
                operations=(verify,),
                controls=(commit,),
                input_products=(verify_input,),
            )
        )
    )
    committed = result.completions[0].committed_tokens

    assert committed == (1001, STUB_IMG_START_TOKEN_ID, 1002)
    assert result.completions[0].selected_point == 3
    assert result.completions[0].logical_lengths.kv_visible_len == 5


def test_verify_selects_the_exact_target_kv_prefix_from_the_initialized_span() -> None:
    worker = execution_worker()
    admission = und_admission(5, block_ids=(4,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    prime = _materialize(
        worker.execute(
            execution_batch(
                step_id=1,
                admissions=(admission,),
                operations=(extend,),
                input_products=(extend_input,),
            )
        )
    )
    commit = commit_for_completion(extend, prime)
    verify, verify_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=commit.selected,
        mode=TokenMode.VERIFY,
        tokens=(prime.completions[0].committed_tokens[0], 900, 901),
        control_seq=commit.control_seq,
    )

    result = _materialize(
        worker.execute(
            execution_batch(
                step_id=2,
                admissions=(),
                operations=(verify,),
                controls=(commit,),
                input_products=(verify_input,),
            )
        )
    )

    completion = result.completions[0]
    assert completion.committed_tokens == (_next_token(prime.completions[0].committed_tokens[0]),)
    assert completion.selected_point == 1
    assert completion.logical_lengths.kv_computed_len == 5
    assert completion.logical_lengths.kv_visible_len == 3


def test_verify_commits_the_accepted_terminal_draft_as_its_exact_prefix() -> None:
    worker = execution_worker()
    base = und_admission(6, block_ids=(5,))
    admission = NewRequest.create(
        base.request_key,
        request_pool_idx=base.request_pool_idx,
        und=replace(cast(UndAdmission, base.und), finish_token_ids=(1001,)),
    )
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    prime = worker.execute(
        execution_batch(
            step_id=1,
            admissions=(admission,),
            operations=(extend,),
            input_products=(extend_input,),
        )
    )
    commit = commit_for_completion(extend, prime)
    verify, verify_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=commit.selected,
        mode=TokenMode.VERIFY,
        tokens=(1000, 1001),
        control_seq=commit.control_seq,
    )

    result = _materialize(
        worker.execute(
            execution_batch(
                step_id=2,
                admissions=(),
                operations=(verify,),
                controls=(commit,),
                input_products=(verify_input,),
            )
        )
    )

    completion = result.completions[0]
    assert completion.committed_tokens == (1001,)
    assert completion.selected_point == 1


def test_chunked_prompt_logprobs_preserve_the_preceding_device_logits() -> None:
    worker = execution_worker()
    admission = und_admission(
        31,
        block_ids=(7,),
        sampling=SamplingParams(return_prompt_logprobs=True, n_prompt_logprobs=2),
    )
    first, first_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
        logprobs=True,
    )
    first_result = _materialize(
        worker.execute(
            execution_batch(
                step_id=1,
                admissions=(admission,),
                operations=(first,),
                input_products=(first_input,),
            )
        )
    )
    commit = commit_for_completion(first, first_result)
    second, second_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=commit.selected,
        mode=TokenMode.EXTEND,
        tokens=(5, 6),
        logprobs=True,
        control_seq=commit.control_seq,
    )
    second_result = _materialize(
        worker.execute(
            execution_batch(
                step_id=2,
                admissions=(),
                operations=(second,),
                controls=(commit,),
                input_products=(second_input,),
            )
        )
    )

    first_blob = next(
        product.payload
        for product in first_result.products
        if product.product.kind is ProductKind.LOGPROB
    )
    second_blob = next(
        product.payload
        for product in second_result.products
        if product.product.kind is ProductKind.LOGPROB
    )
    first_positions = _logprob_positions(first_blob)
    second_positions = _logprob_positions(second_blob)

    assert tuple(position[0][0] for position in first_positions) == (4,)
    assert tuple(position[0][0] for position in second_positions) == (5, 6)
    assert all(position[0][2] >= 1 for position in (*first_positions, *second_positions))
    assert all(position[0][1] <= 0.0 for position in (*first_positions, *second_positions))


def test_failed_prompt_chunk_preserves_the_preceding_logits() -> None:
    sampling = SamplingParams(return_prompt_logprobs=True, n_prompt_logprobs=2)
    admission = und_admission(32, block_ids=(8,), sampling=sampling)
    worker = execution_worker()
    first, first_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
        logprobs=True,
    )
    first_result = _materialize(
        worker.execute(
            execution_batch(
                step_id=1,
                admissions=(admission,),
                operations=(first,),
                input_products=(first_input,),
            )
        )
    )
    commit = commit_for_completion(first, first_result)
    invalid, invalid_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=commit.selected,
        mode=TokenMode.EXTEND,
        tokens=(5, 6),
        logprobs=True,
        control_seq=commit.control_seq,
    )
    invalid_outputs = tuple(
        replace(output, shape_bound=ShapeBound((StaticDim(1),)))
        if output.kind is ProductKind.LOGPROB
        else output
        for output in invalid.outputs
    )
    invalid = Operation.registered(
        request_key=invalid.request_key,
        op_id=invalid.op_id,
        parent=invalid.parent,
        work=invalid.work,
        route=invalid.route,
        domain=invalid.domain,
        bounds=invalid.bounds,
        inputs=invalid.inputs,
        outputs=invalid_outputs,
        predicate=invalid.predicate,
        rng=invalid.rng,
        control_seq=invalid.control_seq,
    )
    failed = _materialize(
        worker.execute(
            execution_batch(
                step_id=2,
                operations=(invalid,),
                controls=(commit,),
                input_products=(invalid_input,),
            )
        )
    )
    assert failed.completions[0].status is OpStatus.ERROR
    assert failed.completions[0].error_code is ErrorCode.INVALID_OPERATION

    continued, continued_input = token_operation(
        admission.request_key,
        op_id=3,
        parent=commit.selected,
        mode=TokenMode.EXTEND,
        tokens=(7, 8),
        logprobs=True,
        control_seq=commit.control_seq,
    )
    recovered = _materialize(
        worker.execute(
            execution_batch(
                step_id=3,
                operations=(continued,),
                input_products=(continued_input,),
            )
        )
    )

    oracle = execution_worker()
    oracle_first, oracle_first_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
        logprobs=True,
    )
    oracle_result = _materialize(
        oracle.execute(
            execution_batch(
                step_id=1,
                admissions=(admission,),
                operations=(oracle_first,),
                input_products=(oracle_first_input,),
            )
        )
    )
    oracle_commit = commit_for_completion(oracle_first, oracle_result)
    oracle_continued, oracle_continued_input = token_operation(
        admission.request_key,
        op_id=3,
        parent=oracle_commit.selected,
        mode=TokenMode.EXTEND,
        tokens=(7, 8),
        logprobs=True,
        control_seq=oracle_commit.control_seq,
    )
    expected = _materialize(
        oracle.execute(
            execution_batch(
                step_id=3,
                operations=(oracle_continued,),
                controls=(oracle_commit,),
                input_products=(oracle_continued_input,),
            )
        )
    )

    recovered_payload = next(
        product.payload
        for product in recovered.products
        if product.product.kind is ProductKind.LOGPROB
    )
    expected_payload = next(
        product.payload
        for product in expected.products
        if product.product.kind is ProductKind.LOGPROB
    )
    assert recovered_payload == expected_payload


def test_worker_samples_with_the_operation_branch_state() -> None:
    worker = execution_worker()
    admission = und_admission(41, block_ids=(9,))
    operation, token_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    operation, sampling_input = _with_sampling_state(
        operation,
        SamplingState(allowed_token_ids=(7,)),
    )

    result = _materialize(
        worker.execute(
            execution_batch(
                step_id=1,
                admissions=(admission,),
                operations=(operation,),
                input_products=(token_input, sampling_input),
            )
        )
    )

    assert result.completions[0].committed_tokens == (7,)


def test_forced_token_schedule_overrides_selection() -> None:
    worker = execution_worker()
    admission = und_admission(
        43,
        block_ids=(11,),
        sampling=SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            forced_token_ids=(7,),
        ),
    )
    operation, token_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )

    result = _materialize(
        worker.execute(
            execution_batch(
                step_id=1,
                admissions=(admission,),
                operations=(operation,),
                input_products=(token_input,),
            )
        )
    )

    assert result.completions[0].committed_tokens == (7,)


def test_all_masked_branch_state_produces_an_error_completion() -> None:
    worker = execution_worker()
    admission = und_admission(42, block_ids=(10,))
    operation, token_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    operation, sampling_input = _with_sampling_state(
        operation,
        SamplingState(allowed_token_ids=()),
    )

    result = _materialize(
        worker.execute(
            execution_batch(
                step_id=1,
                admissions=(admission,),
                operations=(operation,),
                input_products=(token_input, sampling_input),
            )
        )
    )

    completion = result.completions[0]
    assert completion.status is OpStatus.ERROR
    assert completion.error_code is ErrorCode.INVALID_OPERATION
    assert completion.committed_tokens == ()
