"""Model-runner integration for round-level sampling tasks through the real forward.

Sampling is device postprocessing inside the token modes, so inline rows from
one submission round share a single batched sampling task, and a verifier
submits its position rows as one task. The committed tokens are the serial
oracle the stub model defines through its deterministic next-token map.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch

from tests.python.fixtures.depth_one import (
    ar_params,
    execution_run,
    finalized_report,
    record_completion,
    root_parent,
    token_operation,
)
from tests.python.fixtures.execution_worker import execution_worker
from tests.python.fixtures.simulation import expected_successor
from uniserve_models.stub import STUB_IMG_START_TOKEN_ID
from uniserve_worker.protocol.batch import (
    ArRequestParams,
    BatchOutput,
    ComputationId,
    DrawLayout,
    ErrorCode,
    ForwardMode,
    NewRequest,
    OpStatus,
    Rng,
    SamplingParams,
    SamplingState,
    ScheduledRequest,
)

pytestmark = pytest.mark.integration


def test_logprob_reporting_does_not_change_sample_selection() -> None:
    worker = execution_worker()
    sampling = SamplingParams(temperature=0.8, top_k=4, top_p=0.9, seed=71)
    first = ar_params(11, block_ids=(2,), sampling=sampling)
    second = ar_params(
        12,
        block_ids=(3,),
        sampling=replace(sampling, return_logprobs=True, n_logprobs=2),
    )
    first_op = token_operation(
        first.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(first),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        rng=Rng(seed=71, semantic_index_base=2, draw_layout=DrawLayout.TARGET_SAMPLING),
    )
    second_op = token_operation(
        second.request_key,
        op_id=ComputationId(1, 1),
        predecessor=root_parent(second),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        logprobs=True,
        rng=Rng(seed=71, semantic_index_base=2, draw_layout=DrawLayout.TARGET_SAMPLING),
    )

    result = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=1,
                admissions=(first, second),
                operations=(first_op, second_op),
            )
        ),
    )

    assert result.completions[0].committed_tokens == result.completions[1].committed_tokens


@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)))
def test_batched_decode_produces_the_serial_oracle_tokens(device: str) -> None:
    with execution_worker(device=device) as worker:
        # The middle request asks for scores. Compatible sampling rows therefore
        # remain interleaved in operation order, with distinct tokens in every row.
        admissions = (
            ar_params(21, block_ids=(0,)),
            ar_params(22, block_ids=(1,), sampling=SamplingParams(return_logprobs=True)),
            ar_params(23, block_ids=(2,)),
        )
        prompt_ends = (4, 5, 6)
        primed: list[tuple[ScheduledRequest, BatchOutput]] = []
        for index, admission in enumerate(admissions):
            extend = token_operation(
                admission.request_key,
                op_id=ComputationId(1 + index, 0),
                predecessor=root_parent(admission),
                mode=ForwardMode.PREFILL,
                tokens=(3, prompt_ends[index]),
                logprobs=index == 1,
            )
            report = finalized_report(
                worker,
                worker.submit(
                    execution_run(
                        run_id=1 + index,
                        admissions=(admission,),
                        operations=(extend,),
                    )
                ),
            )
            completion = report.completions[0]
            assert (completion.position, completion.kv_visible_len, completion.kv_computed_len) == (
                2,
                2,
                2,
            )
            primed.append((extend, report))

        decode_ops = []
        commits = []
        for index, (admission, (extend, report)) in enumerate(zip(admissions, primed, strict=True)):
            observation = record_completion(extend, report)
            operation = token_operation(
                admission.request_key,
                op_id=ComputationId(4, index),
                predecessor=observation.op_id,
                mode=ForwardMode.DECODE,
                tokens=(expected_successor(prompt_ends[index]),),
                logprobs=index == 1,
            )
            decode_ops.append(operation)

            commits.append(observation)
        result = finalized_report(
            worker,
            worker.submit(
                execution_run(
                    run_id=9,
                    admissions=(),
                    operations=tuple(decode_ops),
                    commands=tuple(commits),
                )
            ),
        )

        for completion, prompt_end in zip(result.completions, prompt_ends, strict=True):
            assert completion.committed_tokens == (expected_successor(expected_successor(prompt_end)),)
            assert (completion.position, completion.kv_visible_len, completion.kv_computed_len) == (
                3,
                3,
                3,
            )


def test_sampling_batch_returns_serial_tokens_for_mixed_finish_policies() -> None:
    worker = execution_worker()
    first = ar_params(24, block_ids=(3,))
    second_base = ar_params(25, block_ids=(4,))
    expected = expected_successor(4)
    assert second_base.ar is not None
    second = NewRequest(
        second_base.request_key,
        request_pool_idx=second_base.request_pool_idx,
        ar=replace(second_base.ar, finish_token_ids=(expected,)),
    )
    first_op = token_operation(
        first.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(first),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    second_op = token_operation(
        second.request_key,
        op_id=ComputationId(1, 1),
        predecessor=root_parent(second),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )

    result = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=1,
                admissions=(first, second),
                operations=(first_op, second_op),
            )
        ),
    )

    assert result.completions[0].committed_tokens == (expected,)
    assert result.completions[1].committed_tokens == (expected,)
    assert result.completions[0].product_generations == tuple(
        output.generation for output in first_op.tensor_outputs()
    )
    assert result.completions[1].product_generations == tuple(
        output.generation for output in second_op.tensor_outputs()
    )


def test_verify_commits_every_accepted_position() -> None:
    worker = execution_worker()
    admission = ar_params(
        4, block_ids=(3,), sampling=SamplingParams(return_logprobs=True, n_logprobs=2, seed=31)
    )
    # Prime the request, then carry its selected token explicitly with the draft.
    extend = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        logprobs=True,
    )
    prime = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=1, admissions=(admission,), operations=(extend,), input_products=()
            )
        ),
    )
    observation = record_completion(extend, prime)
    verify = token_operation(
        admission.request_key,
        op_id=ComputationId(2, 0),
        predecessor=observation.op_id,
        mode=ForwardMode.VERIFY,
        tokens=(1000, 1001, STUB_IMG_START_TOKEN_ID),
        logprobs=True,
    )
    result = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=2,
                admissions=(),
                operations=(verify,),
                commands=(),
            )
        ),
    )
    committed = result.completions[0].committed_tokens

    assert committed == (1001, STUB_IMG_START_TOKEN_ID, 1002)
    assert result.completions[0].kv_visible_len == 5


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
        ),
    ],
)
def test_verify_selects_the_exact_target_kv_prefix_from_the_initialized_span(device: str) -> None:
    worker = execution_worker(device=device, pipeline_depth=2)
    admission = ar_params(5, block_ids=(4,))
    extend = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    prime = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(extend,),
            )
        ),
    )
    observation = record_completion(extend, prime)
    verify = token_operation(
        admission.request_key,
        op_id=ComputationId(2, 0),
        predecessor=observation.op_id,
        mode=ForwardMode.VERIFY,
        tokens=(prime.completions[0].committed_tokens[0], 900, 901),
    )

    result = finalized_report(worker, worker.submit(execution_run(run_id=2, operations=(verify,))))
    successor = replace(
        token_operation(
            admission.request_key,
            op_id=ComputationId(3, 0),
            predecessor=verify.op_id,
            mode=ForwardMode.DECODE,
            tokens=(0,),
            predicate=verify.token_output,
        ),
        input_token_ids=(),
    )
    # Reserve the verifier's full possible prefix, as a queued scheduler would.
    # The decode must use the accepted prefix rather than the initialized tail.
    successor_batch = replace(
        execution_run(run_id=3, operations=(successor,)),
        seq_lens=(len(extend.input_token_ids) + len(verify.input_token_ids) + 1,),
    )
    following = finalized_report(worker, worker.submit(successor_batch))

    completion = result.completions[0]
    assert completion.committed_tokens == (expected_successor(prime.completions[0].committed_tokens[0]),)
    assert completion.kv_computed_len == 5
    assert completion.kv_visible_len == 3

    assert following.completions[0].status is OpStatus.OK
    assert following.completions[0].committed_tokens == (
        expected_successor(completion.committed_tokens[0]),
    )
    assert following.completions[0].kv_visible_len == 4


def test_verify_commits_the_accepted_terminal_draft_as_its_exact_prefix() -> None:
    worker = execution_worker()
    base = ar_params(6, block_ids=(5,))
    admission = NewRequest(
        base.request_key,
        request_pool_idx=base.request_pool_idx,
        ar=replace(cast(ArRequestParams, base.ar), finish_token_ids=(1001,)),
    )
    extend = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    prime = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(extend,),
            )
        ),
    )
    observation = record_completion(extend, prime)
    verify = token_operation(
        admission.request_key,
        op_id=ComputationId(2, 0),
        predecessor=observation.op_id,
        mode=ForwardMode.VERIFY,
        tokens=(1000, 1001),
    )

    result = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=2,
                admissions=(),
                operations=(verify,),
                commands=(),
            )
        ),
    )

    completion = result.completions[0]
    assert completion.committed_tokens == (1001,)


def test_chunked_prompt_logprobs_preserve_the_preceding_device_logits() -> None:
    worker = execution_worker()
    admission = ar_params(
        31,
        block_ids=(7,),
        sampling=SamplingParams(return_prompt_logprobs=True, n_prompt_logprobs=2),
    )
    first = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        logprobs=True,
    )
    first_result = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(first,),
            )
        ),
    )
    observation = record_completion(first, first_result)
    second = token_operation(
        admission.request_key,
        op_id=ComputationId(2, 0),
        predecessor=observation.op_id,
        mode=ForwardMode.PREFILL,
        tokens=(5, 6),
        logprobs=True,
    )
    second_result = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=2,
                admissions=(),
                operations=(second,),
                commands=(),
            )
        ),
    )

    first_positions = first_result.completions[0].prompt_logprobs
    second_positions = second_result.completions[0].prompt_logprobs

    assert tuple(position[0][0] for position in first_positions) == (4,)
    assert tuple(position[0][0] for position in second_positions) == (5, 6)
    assert all(position[0][2] >= 1 for position in (*first_positions, *second_positions))
    assert all(position[0][1] <= 0.0 for position in (*first_positions, *second_positions))


def test_failed_prompt_chunk_preserves_the_preceding_logits() -> None:
    sampling = SamplingParams(return_prompt_logprobs=True, n_prompt_logprobs=2)
    admission = ar_params(32, block_ids=(8,), sampling=sampling)
    worker = execution_worker()
    first = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        logprobs=True,
    )
    first_result = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(first,),
            )
        ),
    )
    observation = record_completion(first, first_result)
    invalid = token_operation(
        admission.request_key,
        op_id=ComputationId(2, 0),
        predecessor=observation.op_id,
        mode=ForwardMode.PREFILL,
        tokens=(5, 6),
        logprobs=True,
    )
    invalid = replace(invalid, bounds=replace(invalid.bounds, max_completion_bytes=1))
    failed = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=2,
                operations=(invalid,),
                commands=(),
            )
        ),
    )
    assert failed.completions[0].status is OpStatus.ERROR
    assert failed.completions[0].error_code is ErrorCode.INVALID_OPERATION

    continued = token_operation(
        admission.request_key,
        op_id=ComputationId(3, 0),
        predecessor=observation.op_id,
        mode=ForwardMode.PREFILL,
        tokens=(7, 8),
        logprobs=True,
    )
    recovered = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=3,
                operations=(continued,),
            )
        ),
    )

    oracle = execution_worker()
    oracle_first = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        logprobs=True,
    )
    oracle_result = finalized_report(
        oracle,
        oracle.submit(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(oracle_first,),
            )
        ),
    )
    oracle_observation = record_completion(oracle_first, oracle_result)
    oracle_continued = token_operation(
        admission.request_key,
        op_id=ComputationId(3, 0),
        predecessor=oracle_observation.op_id,
        mode=ForwardMode.PREFILL,
        tokens=(7, 8),
        logprobs=True,
    )
    expected = finalized_report(
        oracle,
        oracle.submit(
            execution_run(
                run_id=3,
                operations=(oracle_continued,),
                commands=(),
            )
        ),
    )

    recovered_output = recovered.completions[0]
    expected_output = expected.completions[0]
    assert recovered_output.sampled_logprob == expected_output.sampled_logprob
    assert recovered_output.top_logprobs == expected_output.top_logprobs
    assert recovered_output.prompt_logprobs == expected_output.prompt_logprobs


def test_worker_samples_with_the_operation_branch_state() -> None:
    worker = execution_worker()
    admission = ar_params(41, block_ids=(9,))
    operation = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    operation = replace(
        operation,
        sampling_state=SamplingState(allowed_token_ids=(7,)),
    )

    result = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(operation,),
            )
        ),
    )

    assert result.completions[0].committed_tokens == (7,)


def test_forced_token_schedule_overrides_selection() -> None:
    worker = execution_worker()
    admission = ar_params(
        43,
        block_ids=(11,),
        sampling=SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            forced_token_ids=(7,),
        ),
    )
    operation = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )

    result = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(operation,),
            )
        ),
    )

    assert result.completions[0].committed_tokens == (7,)


def test_all_masked_branch_state_produces_an_error_completion() -> None:
    worker = execution_worker()
    admission = ar_params(42, block_ids=(10,))
    operation = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    operation = replace(
        operation,
        sampling_state=SamplingState(allowed_token_ids=()),
    )

    result = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(operation,),
            )
        ),
    )

    completion = result.completions[0]
    assert completion.status is OpStatus.ERROR
    assert completion.error_code is ErrorCode.INVALID_OPERATION
    assert completion.committed_tokens == ()
