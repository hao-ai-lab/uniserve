"""Executor integration for round-level sampling tasks through the real forward.

Sampling is device postprocessing inside the token modes, so inline rows from
one submission round share a single batched sampling task, and a verifier
submits its position rows as one task. The committed tokens are the serial
oracle the stub model defines through its deterministic next-token map.
"""

from __future__ import annotations

import struct

import pytest

from tests.python.fixtures.depth_one import root_parent, token_operation, und_admission
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import (
    Batch,
    DrawLayout,
    DType,
    ErrorCode,
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
    encode_sampling_state_bytes,
)
from uniserve_worker.execution import executor as executor_module
from uniserve_worker.server.stub import STUB_IMG_START_TOKEN_ID, _next_token

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
        new_kv_blocks=operation.new_kv_blocks,
        predicate=operation.predicate,
        rng=operation.rng,
        control_seq=operation.control_seq,
    )
    return registered, ProductPayload(reference, payload)


def _observe_sample_batches(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    observed: list[tuple[int, int]] = []
    implementation = executor_module._sample_task_batch

    def wrapped(tasks, completion=None, *, device_products=None, device_reads=()):
        observed.append((len(tasks), sum(len(task.rows) for task in tasks)))
        return implementation(
            tasks,
            completion,
            device_products=device_products,
            device_reads=device_reads,
        )

    monkeypatch.setattr(executor_module, "_sample_task_batch", wrapped)
    return observed


def test_inline_rows_from_one_round_share_one_sampling_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = execution_worker()
    first = und_admission(
        1, block_ids=(0,), sampling=SamplingParams(temperature=0.7, top_k=4, top_p=0.9, seed=11)
    )
    second = und_admission(
        2, block_ids=(1,), sampling=SamplingParams(temperature=0.9, top_k=3, top_p=0.85, seed=17)
    )
    first_op, first_input = token_operation(
        first.request_key,
        op_id=1,
        parent=root_parent(first),
        mode=TokenMode.EXTEND,
        tokens=(8, 9),
        rng=Rng(seed=11, semantic_index_base=2, draw_layout=DrawLayout.TARGET_SAMPLING),
    )
    second_op, second_input = token_operation(
        second.request_key,
        op_id=2,
        parent=root_parent(second),
        mode=TokenMode.EXTEND,
        tokens=(12,),
        rng=Rng(seed=17, semantic_index_base=1, draw_layout=DrawLayout.TARGET_SAMPLING),
    )
    observed = _observe_sample_batches(monkeypatch)

    result = worker.execute(
        Batch(
            step_id=1,
            admissions=(first, second),
            operations=(first_op, second_op),
            input_products=(first_input, second_input),
        )
    )

    assert observed == [(2, 2)]
    assert result.completions[0].committed_tokens == (_next_token(9),)
    assert result.completions[1].committed_tokens == (_next_token(12),)


def test_batched_decode_shares_one_sampling_task(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = execution_worker()
    admissions = (und_admission(21, block_ids=(0,)), und_admission(22, block_ids=(1,)))
    for index, admission in enumerate(admissions):
        extend, extend_input = token_operation(
            admission.request_key,
            op_id=1 + index,
            parent=root_parent(admission),
            mode=TokenMode.EXTEND,
            tokens=(3, 4),
        )
        worker.execute(
            Batch(
                step_id=1 + index,
                admissions=(admission,),
                operations=(extend,),
                input_products=(extend_input,),
            )
        )

    decode_ops = []
    decode_inputs = []
    for index, admission in enumerate(admissions):
        session_id = admission.request_key.session_id
        operation, payload = token_operation(
            admission.request_key,
            op_id=3 + index,
            parent=worker.sessions.get(session_id).committed_version(),
            mode=TokenMode.DECODE,
            tokens=(_next_token(4),),
        )
        decode_ops.append(operation)
        decode_inputs.append(payload)
    observed = _observe_sample_batches(monkeypatch)

    result = worker.execute(
        Batch(
            step_id=9,
            admissions=(),
            operations=tuple(decode_ops),
            input_products=tuple(decode_inputs),
        )
    )

    assert observed == [(2, 2)]
    expected = _next_token(_next_token(4))
    assert result.completions[0].committed_tokens == (expected,)
    assert result.completions[1].committed_tokens == (expected,)


def test_verify_submits_its_position_rows_as_one_sampling_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    worker.execute(
        Batch(
            step_id=1, admissions=(admission,), operations=(extend,), input_products=(extend_input,)
        )
    )
    verify, verify_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=worker.sessions.get(4).committed_version(),
        mode=TokenMode.VERIFY,
        tokens=(1000, 1001, STUB_IMG_START_TOKEN_ID),
        logprobs=True,
    )
    observed = _observe_sample_batches(monkeypatch)

    result = worker.execute(
        Batch(step_id=2, admissions=(), operations=(verify,), input_products=(verify_input,))
    )
    committed = result.completions[0].committed_tokens

    assert observed == [(1, 3)]
    assert committed == (1001, STUB_IMG_START_TOKEN_ID, 1002)


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
    first_result = worker.execute(
        Batch(
            step_id=1,
            admissions=(admission,),
            operations=(first,),
            input_products=(first_input,),
        )
    )
    second, second_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=worker.sessions.get(31).committed_version(),
        mode=TokenMode.EXTEND,
        tokens=(5, 6),
        logprobs=True,
    )
    second_result = worker.execute(
        Batch(
            step_id=2,
            admissions=(),
            operations=(second,),
            input_products=(second_input,),
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
    assert all(
        position[0][1] <= 0.0 for position in (*first_positions, *second_positions)
    )


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

    result = worker.execute(
        Batch(
            step_id=1,
            admissions=(admission,),
            operations=(operation,),
            input_products=(token_input, sampling_input),
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

    result = worker.execute(
        Batch(
            step_id=1,
            admissions=(admission,),
            operations=(operation,),
            input_products=(token_input, sampling_input),
        )
    )

    completion = result.completions[0]
    assert completion.status is OpStatus.ERROR
    assert completion.error_code is ErrorCode.INVALID_OPERATION
    assert completion.committed_tokens == ()
