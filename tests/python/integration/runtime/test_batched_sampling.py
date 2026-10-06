"""Model-runner integration for round-level sampling tasks.

The integration runs through the real forward.

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
    execution_batch,
    finalized_report,
    record_completion,
    root_parent,
    stamp_batch,
    token_call,
)
from tests.python.fixtures.execution_worker import execution_worker
from tests.python.fixtures.simulation import expected_successor
from uniserve.sampling import SamplingParams
from uniserve_models.stub import STUB_IMG_START_TOKEN_ID
from uniserve_worker.protocol.batch import GenerationParams, NewRequest
from uniserve_worker.protocol.call import (
    Call,
    CallStatus,
    DrawLayout,
    ErrorCode,
    ForwardMode,
    Rng,
    SamplingState,
)
from uniserve_worker.protocol.identity import CallId
from uniserve_worker.protocol.output import BatchOutput

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu))
)
def test_logprob_reporting_does_not_change_sample_selection(device, request):
    worker = execution_worker(device=device)
    request.addfinalizer(worker.close)
    sampling = SamplingParams(temperature=0.8, top_k=4, top_p=0.9, seed=71)
    first = ar_params(11, block_ids=(2,), sampling=sampling)
    second = ar_params(
        12,
        block_ids=(3,),
        sampling=replace(sampling, return_logprobs=True, n_logprobs=2),
    )
    first_op = token_call(
        first.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(first),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        rng=Rng(
            seed=71,
            semantic_index_base=2,
            draw_layout=DrawLayout.TARGET_SAMPLING,
        ),
    )
    second_op = token_call(
        second.request_key,
        call_id=CallId(1, 1),
        predecessor=root_parent(second),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        logprobs=True,
        rng=Rng(
            seed=71,
            semantic_index_base=2,
            draw_layout=DrawLayout.TARGET_SAMPLING,
        ),
    )

    result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(first, second),
                calls=(first_op, second_op),
            )
        ),
    )

    assert (
        result.completions[0].committed_tokens
        == result.completions[1].committed_tokens
    )
    assert result.completions[1].sampled_logprob is not None
    assert len(result.completions[1].top_logprobs) == 2


@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu))
)
def test_batched_decode_produces_the_serial_oracle_tokens(device: str) -> None:
    with execution_worker(device=device) as worker:
        # The middle request asks for scores. Compatible sampling rows
        # therefore remain interleaved in call order, with distinct
        # tokens in every row.
        admissions = (
            ar_params(21, block_ids=(0,)),
            ar_params(
                22,
                block_ids=(1,),
                sampling=SamplingParams(return_logprobs=True),
            ),
            ar_params(23, block_ids=(2,)),
        )
        prompt_ends = (4, 5, 6)
        primed: list[tuple[Call, BatchOutput]] = []
        for index, admission in enumerate(admissions):
            extend = token_call(
                admission.request_key,
                call_id=CallId(1 + index, 0),
                predecessor=root_parent(admission),
                mode=ForwardMode.PREFILL,
                tokens=(3, prompt_ends[index]),
                logprobs=index == 1,
            )
            report = finalized_report(
                worker,
                worker.submit(
                    execution_batch(
                        batch_id=1 + index,
                        admissions=(admission,),
                        calls=(extend,),
                    )
                ),
            )
            completion = report.completions[0]
            assert (
                completion.position,
                completion.kv_visible_len,
                completion.kv_computed_len,
            ) == (
                2,
                2,
                2,
            )
            primed.append((extend, report))

        decode_ops = []
        for index, (admission, (extend, report)) in enumerate(
            zip(admissions, primed, strict=True)
        ):
            observation = record_completion(extend, report)
            call = token_call(
                admission.request_key,
                call_id=CallId(4, index),
                predecessor=observation.call_id,
                mode=ForwardMode.DECODE,
                tokens=(expected_successor(prompt_ends[index]),),
                logprobs=index == 1,
            )
            decode_ops.append(call)

        result = finalized_report(
            worker,
            worker.submit(
                execution_batch(
                    batch_id=9,
                    admissions=(),
                    calls=tuple(decode_ops),
                )
            ),
        )

        for completion, prompt_end in zip(
            result.completions, prompt_ends, strict=True
        ):
            assert completion.committed_tokens == (
                expected_successor(expected_successor(prompt_end)),
            )
            assert (
                completion.position,
                completion.kv_visible_len,
                completion.kv_computed_len,
            ) == (
                3,
                3,
                3,
            )


def test_sampling_batch_returns_serial_tokens_for_mixed_finish_policies() -> (
    None
):
    worker = execution_worker()
    first = ar_params(24, block_ids=(3,))
    second_base = ar_params(25, block_ids=(4,))
    expected = expected_successor(4)
    assert second_base.generation is not None
    second = NewRequest(
        second_base.request_key,
        request_pool_idx=second_base.request_pool_idx,
        generation=replace(
            second_base.generation, finish_token_ids=(expected,)
        ),
    )
    first_op = token_call(
        first.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(first),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    second_op = token_call(
        second.request_key,
        call_id=CallId(1, 1),
        predecessor=root_parent(second),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )

    result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(first, second),
                calls=(first_op, second_op),
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
        4,
        block_ids=(3,),
        sampling=SamplingParams(return_logprobs=True, n_logprobs=2, seed=31),
    )
    # Prime the request, then carry its selected token explicitly with the
    # draft.
    extend = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        logprobs=True,
    )
    prime = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(extend,),
                input_products=(),
            )
        ),
    )
    observation = record_completion(extend, prime)
    verify = token_call(
        admission.request_key,
        call_id=CallId(2, 0),
        predecessor=observation.call_id,
        mode=ForwardMode.VERIFY,
        tokens=(1000, 1001, STUB_IMG_START_TOKEN_ID),
        logprobs=True,
    )
    result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=2,
                admissions=(),
                calls=(verify,),
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
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is required"
                ),
            ),
        ),
    ],
)
def test_verify_selects_the_exact_target_kv_prefix_from_the_initialized_span(
    device: str,
) -> None:
    worker = execution_worker(device=device, queue_depth=2)
    admission = ar_params(5, block_ids=(4,))
    extend = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    prime = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(extend,),
            )
        ),
    )
    observation = record_completion(extend, prime)
    verify = token_call(
        admission.request_key,
        call_id=CallId(2, 0),
        predecessor=observation.call_id,
        mode=ForwardMode.VERIFY,
        tokens=(prime.completions[0].committed_tokens[0], 900, 901),
    )

    result = finalized_report(
        worker, worker.submit(execution_batch(batch_id=2, calls=(verify,)))
    )
    successor = token_call(
        admission.request_key,
        call_id=CallId(3, 0),
        predecessor=verify.call_id,
        mode=ForwardMode.DECODE,
        tokens=(0,),
        predicate=verify.token_output,
    ).replace(
        input_token_ids=(),
    )
    # Reserve the verifier's full possible prefix, as a queued scheduler would.
    # The decode must use the accepted prefix rather than the initialized tail.
    successor_batch = stamp_batch(
        worker,
        replace(
            execution_batch(batch_id=3, calls=(successor,)),
            seq_lens=(
                len(extend.input_token_ids) + len(verify.input_token_ids) + 1,
            ),
        ),
    )
    following = finalized_report(worker, worker.submit(successor_batch))

    completion = result.completions[0]
    assert completion.committed_tokens == (
        expected_successor(prime.completions[0].committed_tokens[0]),
    )
    assert completion.kv_computed_len == 5
    assert completion.kv_visible_len == 3

    assert following.completions[0].status is CallStatus.OK
    assert following.completions[0].committed_tokens == (
        expected_successor(completion.committed_tokens[0]),
    )
    assert following.completions[0].kv_visible_len == 4


def test_verify_commits_the_accepted_terminal_draft_as_its_exact_prefix() -> (
    None
):
    worker = execution_worker()
    base = ar_params(6, block_ids=(5,))
    admission = NewRequest(
        base.request_key,
        request_pool_idx=base.request_pool_idx,
        generation=replace(
            cast(GenerationParams, base.generation), finish_token_ids=(1001,)
        ),
    )
    extend = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    prime = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(extend,),
            )
        ),
    )
    observation = record_completion(extend, prime)
    verify = token_call(
        admission.request_key,
        call_id=CallId(2, 0),
        predecessor=observation.call_id,
        mode=ForwardMode.VERIFY,
        tokens=(1000, 1001),
    )

    result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=2,
                admissions=(),
                calls=(verify,),
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
        sampling=SamplingParams(
            return_prompt_logprobs=True, n_prompt_logprobs=2
        ),
    )
    first = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        logprobs=True,
    )
    first_result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(first,),
            )
        ),
    )
    observation = record_completion(first, first_result)
    second = token_call(
        admission.request_key,
        call_id=CallId(2, 0),
        predecessor=observation.call_id,
        mode=ForwardMode.PREFILL,
        tokens=(5, 6),
        logprobs=True,
    )
    second_result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=2,
                admissions=(),
                calls=(second,),
                commands=(),
            )
        ),
    )

    first_positions = first_result.completions[0].prompt_logprobs
    second_positions = second_result.completions[0].prompt_logprobs

    assert tuple(position[0][0] for position in first_positions) == (4,)
    assert tuple(position[0][0] for position in second_positions) == (5, 6)
    assert all(
        position[0][2] >= 1
        for position in (*first_positions, *second_positions)
    )
    assert all(
        position[0][1] <= 0.0
        for position in (*first_positions, *second_positions)
    )


def test_failed_prompt_chunk_preserves_the_preceding_logits() -> None:
    sampling = SamplingParams(return_prompt_logprobs=True, n_prompt_logprobs=2)
    admission = ar_params(32, block_ids=(8,), sampling=sampling)
    worker = execution_worker()
    first = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        logprobs=True,
    )
    first_result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(first,),
            )
        ),
    )
    observation = record_completion(first, first_result)
    invalid = token_call(
        admission.request_key,
        call_id=CallId(2, 0),
        predecessor=observation.call_id,
        mode=ForwardMode.PREFILL,
        tokens=(5, 6),
        logprobs=True,
    )
    invalid = invalid.replace(
        bounds=replace(invalid.bounds, max_completion_bytes=1)
    )
    failed = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=2,
                calls=(invalid,),
                commands=(),
            )
        ),
    )
    assert failed.completions[0].status is CallStatus.ERROR
    assert failed.completions[0].error_code is ErrorCode.INVALID_CALL
    # Registration failed before this chunk executed. Report the accepted
    # prefix, so the caller can continue from the preceding successful call.
    failed_output = failed.completions[0]
    assert (
        failed_output.position,
        failed_output.kv_visible_len,
        failed_output.kv_computed_len,
    ) == (2, 2, 2)
    assert failed_output.committed_tokens == ()

    continued = token_call(
        admission.request_key,
        call_id=CallId(3, 0),
        predecessor=observation.call_id,
        mode=ForwardMode.PREFILL,
        tokens=(7, 8),
        logprobs=True,
    )
    recovered = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=3,
                calls=(continued,),
            )
        ),
    )

    oracle = execution_worker()
    oracle_first = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        logprobs=True,
    )
    oracle_result = finalized_report(
        oracle,
        oracle.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(oracle_first,),
            )
        ),
    )
    oracle_observation = record_completion(oracle_first, oracle_result)
    oracle_continued = token_call(
        admission.request_key,
        call_id=CallId(3, 0),
        predecessor=oracle_observation.call_id,
        mode=ForwardMode.PREFILL,
        tokens=(7, 8),
        logprobs=True,
    )
    expected = finalized_report(
        oracle,
        oracle.submit(
            execution_batch(
                batch_id=3,
                calls=(oracle_continued,),
                commands=(),
            )
        ),
    )

    recovered_output = recovered.completions[0]
    expected_output = expected.completions[0]
    assert recovered_output.sampled_logprob == expected_output.sampled_logprob
    assert recovered_output.top_logprobs == expected_output.top_logprobs
    assert recovered_output.prompt_logprobs == expected_output.prompt_logprobs


def test_worker_samples_with_the_call_branch_state() -> None:
    worker = execution_worker()
    admission = ar_params(41, block_ids=(9,))
    call = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    call = call.replace(
        sampling_state=SamplingState(allowed_token_ids=(7,)),
    )

    result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(call,),
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
    call = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )

    result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(call,),
            )
        ),
    )

    assert result.completions[0].committed_tokens == (7,)


def test_all_masked_branch_state_produces_an_error_completion() -> None:
    worker = execution_worker()
    admission = ar_params(42, block_ids=(10,))
    call = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    call = call.replace(
        sampling_state=SamplingState(allowed_token_ids=()),
    )

    result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(call,),
            )
        ),
    )

    completion = result.completions[0]
    assert completion.status is CallStatus.ERROR
    assert completion.error_code is ErrorCode.INVALID_CALL
    assert completion.committed_tokens == ()
