from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.python.fixtures.depth_one import (
    ar_params,
    execution_batch,
    finalized_report,
    record_completion,
    root_parent,
    token_call,
)
from tests.python.fixtures.execution_worker import execution_worker
from tests.python.fixtures.simulation import expected_successor
from uniserve.sampling import SamplingParams
from uniserve_worker.config import WorkerConfig
from uniserve_worker.protocol.call import (
    CallStatus,
    DrawLayout,
    ForwardMode,
    Rng,
    SamplingState,
)
from uniserve_worker.protocol.identity import CallId

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is required"
    ),
]


@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("finish_policy", ["admission", "call", "force"])
def test_decode_terminal_policy_suppresses_only_its_own_expected_successor(
    graphs, finish_policy
) -> None:
    policy = WorkerConfig(
        graph_policy="full" if graphs else "off",
        prefill_cuda_graph=False,
        decode_graph_batch_sizes=(2,),
    )
    first = ar_params(34, block_ids=(0,))
    second = ar_params(35, block_ids=(1,))
    terminal = expected_successor(expected_successor(4))
    if finish_policy == "admission":
        assert first.generation is not None
        first = replace(
            first,
            generation=replace(first.generation, finish_token_ids=(terminal,)),
        )

    with execution_worker(
        device="cuda:0", queue_depth=3, execution=policy
    ) as worker:
        parents = tuple(
            token_call(
                admission.request_key,
                call_id=CallId(1, index),
                predecessor=root_parent(admission),
                mode=ForwardMode.PREFILL,
                tokens=(3, 4),
            )
            for index, admission in enumerate((first, second))
        )
        parent_report = worker.submit(
            execution_batch(
                batch_id=1, admissions=(first, second), calls=parents
            )
        )
        decodes = tuple(
            token_call(
                parent.request_key,
                call_id=CallId(2, index),
                predecessor=parent.call_id,
                mode=ForwardMode.DECODE,
                tokens=(0,),
                predicate=parent.token_output,
            )
            for index, parent in enumerate(parents)
        )
        if finish_policy != "admission":
            decodes = (
                replace(
                    decodes[0],
                    sampling_state=SamplingState(
                        finish_token_ids=(terminal,)
                        if finish_policy == "call"
                        else (),
                        force_finish=finish_policy == "force",
                    ),
                ),
                decodes[1],
            )
        decode_report = worker.submit(
            execution_batch(batch_id=2, calls=decodes)
        )
        successors = tuple(
            token_call(
                decode.request_key,
                call_id=CallId(3, index),
                predecessor=decode.call_id,
                mode=ForwardMode.DECODE,
                tokens=(0,),
                predicate=decode.token_output,
            )
            for index, decode in enumerate(decodes)
        )
        # Queue the dependent work before consuming either parent's host result.
        successor_report = worker.submit(
            execution_batch(batch_id=3, calls=successors)
        )
        finalized_report(worker, parent_report)
        selected = finalized_report(worker, decode_report).completions
        following = finalized_report(worker, successor_report).completions

        assert tuple(output.committed_tokens for output in selected) == (
            (terminal,),
            (terminal,),
        )
        assert following[0].status is CallStatus.PREDICATED
        assert following[0].committed_tokens == ()
        assert following[1].status is CallStatus.OK
        assert following[1].committed_tokens == (expected_successor(terminal),)


def test_same_request_continues_before_parent_report_materialization() -> None:
    device = "cuda:0"
    worker = execution_worker(device=device, queue_depth=2)
    warm_admission = ar_params(30, block_ids=(1,))
    warm_call = token_call(
        warm_admission.request_key,
        call_id=CallId(100, 0),
        predecessor=root_parent(warm_admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    worker.submit(
        execution_batch(
            batch_id=0,
            admissions=(warm_admission,),
            calls=(warm_call,),
        )
    )
    torch.cuda.synchronize()
    worker.drop_request(30)

    admission = ar_params(31, block_ids=(0,))
    predecessor = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )

    parent_report = worker.submit(
        execution_batch(
            batch_id=1,
            admissions=(admission,),
            calls=(predecessor,),
        )
    )
    device_parent = predecessor.call_id
    successor_template = token_call(
        admission.request_key,
        call_id=CallId(2, 0),
        predecessor=device_parent,
        mode=ForwardMode.DECODE,
        tokens=(0,),
        predicate=predecessor.token_output,
    )
    successor = replace(successor_template, input_token_ids=())

    successor_report = worker.submit(
        execution_batch(
            batch_id=2,
            admissions=(),
            calls=(successor,),
        )
    )

    torch.cuda.synchronize()
    parent_report = finalized_report(worker, parent_report)
    successor_report = finalized_report(worker, successor_report)

    first = expected_successor(4)
    assert parent_report.completions[0].committed_tokens == (first,)
    assert successor_report.completions[0].committed_tokens == (
        expected_successor(first),
    )


def test_device_continuation_chain_matches_serial_token_sequence() -> None:
    worker = execution_worker(device="cuda:0", queue_depth=4)
    admission = ar_params(32, block_ids=(0,))
    call = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    reports = [
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(call,),
            )
        )
    ]
    calls = [call]

    for batch_id in range(2, 5):
        predecessor = calls[-1]
        call = token_call(
            admission.request_key,
            call_id=CallId(batch_id, 0),
            predecessor=predecessor.call_id,
            mode=ForwardMode.DECODE,
            tokens=(0,),
            predicate=predecessor.token_output,
        )
        reports.append(
            worker.submit(
                execution_batch(
                    batch_id=batch_id,
                    admissions=(),
                    calls=(call,),
                )
            )
        )
        calls.append(call)

    torch.cuda.synchronize()
    completions = tuple(
        finalized_report(worker, report).completions[0] for report in reports
    )
    tokens = tuple(completion.committed_tokens[0] for completion in completions)
    for position, completion in enumerate(completions, start=2):
        assert completion.position == position
        assert completion.kv_visible_len == position
        assert completion.kv_computed_len == position
    expected = []
    current = 4
    for _ in range(4):
        current = expected_successor(current)
        expected.append(current)
    assert tokens == tuple(expected)


def test_relay_window_retains_a_consumer_fenced_predecessor() -> None:
    worker = execution_worker(device="cuda:0", queue_depth=3)
    admission = ar_params(33, block_ids=(0,))
    call = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    reports = [
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(call,),
            )
        )
    ]
    calls = [call]
    for batch_id in range(2, 5):
        predecessor = calls[-1]
        call = token_call(
            admission.request_key,
            call_id=CallId(batch_id, 0),
            predecessor=predecessor.call_id,
            mode=ForwardMode.DECODE,
            tokens=(0,),
            predicate=predecessor.token_output,
        )
        reports.append(
            worker.submit(
                execution_batch(
                    batch_id=batch_id,
                    calls=(call,),
                )
            )
        )
        calls.append(call)

    torch.cuda.synchronize()
    tokens = tuple(
        finalized_report(worker, report).completions[0].committed_tokens[0]
        for report in reports
    )
    expected = []
    current = 4
    for _ in range(4):
        current = expected_successor(current)
        expected.append(current)
    assert tokens == tuple(expected)


def test_stochastic_device_continuation_matches_depth_one_serial_execution() -> (  # noqa: E501
    None
):
    worker = execution_worker(device="cuda:0", queue_depth=2)
    sampling = SamplingParams(temperature=0.8, top_k=32, top_p=0.93, seed=917)
    pipelined = ar_params(41, block_ids=(2,), sampling=sampling)
    predecessor = token_call(
        pipelined.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(pipelined),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        rng=Rng(
            seed=917,
            semantic_index_base=2,
            draw_layout=DrawLayout.TARGET_SAMPLING,
        ),
    )
    parent_report = worker.submit(
        execution_batch(
            batch_id=1,
            admissions=(pipelined,),
            calls=(predecessor,),
        )
    )
    successor = token_call(
        pipelined.request_key,
        call_id=CallId(2, 0),
        predecessor=predecessor.call_id,
        mode=ForwardMode.DECODE,
        tokens=(0,),
        predicate=predecessor.token_output,
        rng=Rng(
            seed=917,
            semantic_index_base=3,
            draw_layout=DrawLayout.TARGET_SAMPLING,
        ),
    )
    successor_report = worker.submit(
        execution_batch(
            batch_id=2,
            admissions=(),
            calls=(successor,),
        )
    )

    torch.cuda.synchronize()
    parent_report = finalized_report(worker, parent_report)
    parent_tokens = parent_report.completions[0].committed_tokens
    successor_report = finalized_report(worker, successor_report)
    successor_tokens = successor_report.completions[0].committed_tokens

    serial_worker = execution_worker(device="cuda:0", queue_depth=1)
    serial = ar_params(41, block_ids=(2,), sampling=sampling)
    serial_parent = token_call(
        serial.request_key,
        call_id=CallId(11, 0),
        predecessor=root_parent(serial),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        rng=Rng(
            seed=917,
            semantic_index_base=2,
            draw_layout=DrawLayout.TARGET_SAMPLING,
        ),
    )
    serial_parent_report = serial_worker.submit(
        execution_batch(
            batch_id=11,
            admissions=(serial,),
            calls=(serial_parent,),
        )
    )
    torch.cuda.synchronize()
    serial_parent_report = finalized_report(serial_worker, serial_parent_report)
    serial_first = serial_parent_report.completions[0].committed_tokens[0]
    observation = record_completion(serial_parent, serial_parent_report)
    serial_successor = token_call(
        serial.request_key,
        call_id=CallId(12, 0),
        predecessor=observation.call_id,
        mode=ForwardMode.DECODE,
        tokens=(serial_first,),
        rng=Rng(
            seed=917,
            semantic_index_base=3,
            draw_layout=DrawLayout.TARGET_SAMPLING,
        ),
    )
    serial_successor_report = serial_worker.submit(
        execution_batch(
            batch_id=12,
            admissions=(),
            calls=(serial_successor,),
            commands=(),
        )
    )
    torch.cuda.synchronize()
    serial_successor_report = finalized_report(
        serial_worker, serial_successor_report
    )

    assert parent_tokens == serial_parent_report.completions[0].committed_tokens
    assert (
        successor_tokens
        == serial_successor_report.completions[0].committed_tokens
    )


def test_penalty_device_continuation_matches_depth_one_serial_execution() -> (
    None
):
    # A penalty-bearing pipelined successor must match the equivalent serial
    # lineage.
    sampling = SamplingParams(
        temperature=0.8,
        top_k=48,
        seed=613,
        repetition_penalty=1.4,
        frequency_penalty=0.6,
        presence_penalty=0.3,
    )
    worker = execution_worker(device="cuda:0", queue_depth=2)
    pipelined = ar_params(57, block_ids=(3,), sampling=sampling)
    predecessor = token_call(
        pipelined.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(pipelined),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        rng=Rng(
            seed=613,
            semantic_index_base=2,
            draw_layout=DrawLayout.TARGET_SAMPLING,
        ),
    )
    parent_report = worker.submit(
        execution_batch(
            batch_id=1,
            admissions=(pipelined,),
            calls=(predecessor,),
        )
    )
    successor = token_call(
        pipelined.request_key,
        call_id=CallId(2, 0),
        predecessor=predecessor.call_id,
        mode=ForwardMode.DECODE,
        tokens=(0,),
        predicate=predecessor.token_output,
        rng=Rng(
            seed=613,
            semantic_index_base=3,
            draw_layout=DrawLayout.TARGET_SAMPLING,
        ),
    )
    successor_report = worker.submit(
        execution_batch(
            batch_id=2,
            admissions=(),
            calls=(successor,),
        )
    )

    torch.cuda.synchronize()
    parent_report = finalized_report(worker, parent_report)
    parent_tokens = parent_report.completions[0].committed_tokens
    successor_report = finalized_report(worker, successor_report)
    successor_tokens = successor_report.completions[0].committed_tokens

    serial_worker = execution_worker(device="cuda:0", queue_depth=1)
    serial = ar_params(57, block_ids=(3,), sampling=sampling)
    serial_parent = token_call(
        serial.request_key,
        call_id=CallId(11, 0),
        predecessor=root_parent(serial),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
        rng=Rng(
            seed=613,
            semantic_index_base=2,
            draw_layout=DrawLayout.TARGET_SAMPLING,
        ),
    )
    serial_parent_report = serial_worker.submit(
        execution_batch(
            batch_id=11,
            admissions=(serial,),
            calls=(serial_parent,),
        )
    )
    torch.cuda.synchronize()
    serial_parent_report = finalized_report(serial_worker, serial_parent_report)
    serial_first = serial_parent_report.completions[0].committed_tokens[0]
    observation = record_completion(serial_parent, serial_parent_report)
    serial_successor = token_call(
        serial.request_key,
        call_id=CallId(12, 0),
        predecessor=observation.call_id,
        mode=ForwardMode.DECODE,
        tokens=(serial_first,),
        rng=Rng(
            seed=613,
            semantic_index_base=3,
            draw_layout=DrawLayout.TARGET_SAMPLING,
        ),
    )
    serial_successor_report = serial_worker.submit(
        execution_batch(
            batch_id=12,
            admissions=(),
            calls=(serial_successor,),
            commands=(),
        )
    )
    torch.cuda.synchronize()
    serial_successor_report = finalized_report(
        serial_worker, serial_successor_report
    )

    assert parent_tokens == serial_parent_report.completions[0].committed_tokens
    assert (
        successor_tokens
        == serial_successor_report.completions[0].committed_tokens
    )
