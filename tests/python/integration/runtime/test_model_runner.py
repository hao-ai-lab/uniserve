"""Serial-oracle behavior at the canonical ModelRunner boundary."""

from __future__ import annotations

import base64
import io
import time
from dataclasses import replace

import pytest
import torch
from PIL import Image

from tests.python.fixtures.depth_one import (
    ar_params,
    bind_request_allocation,
    diffusion_finalize_call,
    diffusion_prepare_call,
    diffusion_step_call,
    encode_call,
    execution_batch,
    finalized_report,
    kv_publication_call,
    record_completion,
    root_parent,
    stamp_batch,
    token_call,
    umm_params,
    visual_state_call,
)
from tests.python.fixtures.execution_worker import execution_worker
from tests.python.fixtures.simulation import expected_successor
from uniserve.model import Logits
from uniserve_models.stub import Model
from uniserve_models.stub import entry_points as entry_points
from uniserve_worker.config import LaneConfig, WorkerConfig
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.protocol.batch import (
    BlockTable,
    Free,
    GenerationParams,
    NewRequest,
    TensorPublication,
)
from uniserve_worker.protocol.call import (
    CALL_KINDS,
    Bounds,
    Call,
    CallCoordinates,
    CallStatus,
    ErrorCode,
    ForwardMode,
    ImageParams,
    MediaCall,
    SamplingState,
    TransferMode,
)
from uniserve_worker.protocol.identity import CallId
from uniserve_worker.protocol.output import RequestOutput
from uniserve_worker.protocol.tensor import (
    DType,
    ShapeBound,
    StaticDim,
    TensorRef,
)
from uniserve_worker.protocol.transfer import (
    DeviceProductTransferValue,
    EncoderTransferValue,
    TensorTransfer,
)
from uniserve_worker.transfer import segment
from uniserve_worker.transfer.tickets import _open_shared_memory

pytestmark = pytest.mark.integration

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _MisalignedOutputModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.misaligned = False
        self.failure_delay_cycles = 0

    def compute_logits(
        self, hidden: torch.Tensor, *, token_indices: torch.Tensor
    ) -> Logits:
        output = super().compute_logits(hidden, token_indices=token_indices)
        if self.misaligned:
            if self.failure_delay_cycles:
                torch.cuda._sleep(self.failure_delay_cycles)
            return Logits(output.values[:-1], output.vocab)
        return output


def _publish_conditioning(
    worker: object,
    admission: NewRequest,
    *,
    call_id: CallId,
    batch_id: int,
):
    publication, product = kv_publication_call(
        admission.request_key,
        call_id=call_id,
        predecessor=root_parent(admission),
    )
    finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=batch_id,
                admissions=(admission,),
                calls=(publication,),
            )
        ),
    )
    return product


def _prepare_media(
    worker: object,
    admission: NewRequest,
    conditioning: object,
    *,
    call_id: CallId,
    predecessor: object,
    batch_id: int,
    seed: int = 29,
    image_index: int = 1,
):
    preparation, latent = diffusion_prepare_call(
        admission.request_key,
        call_id=call_id,
        predecessor=predecessor,
        conditioning=conditioning,
        seed=seed,
        image_index=image_index,
    )
    report = worker.submit(
        execution_batch(batch_id=batch_id, calls=(preparation,))
    )
    report = finalized_report(worker, report)
    assert report.completions[0].status is CallStatus.OK
    return latent, record_completion(preparation, report)


def _prepare_decode(
    worker: object,
    admission: NewRequest,
    *,
    call_id: CallId,
    batch_id: int,
    tokens: tuple[int, ...],
):
    prefill = token_call(
        admission.request_key,
        call_id=call_id,
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=tokens,
    )
    report = worker.submit(
        execution_batch(
            batch_id=batch_id,
            admissions=(admission,),
            calls=(prefill,),
        )
    )
    report = finalized_report(worker, report)
    resolved = report
    observation = record_completion(prefill, resolved)
    decode = token_call(
        admission.request_key,
        call_id=CallId(call_id.batch_id + 1, call_id.request_index),
        predecessor=observation.call_id,
        mode=ForwardMode.DECODE,
        tokens=(resolved.completions[0].committed_tokens[0],),
    )
    return (decode, observation)


def _media_bytes(record: RequestOutput) -> bytes:
    """Claim the public shared-memory output.

    The output's encoded image bytes are consumed.
    """
    from multiprocessing.shared_memory import SharedMemory

    output = record.media_output
    assert output is not None
    memory = SharedMemory(name=output.handle.name)
    try:
        return bytes(memory.buf[: output.bytes])
    finally:
        memory.unlink()
        memory.close()


def _finalized_artifact(
    worker: object,
    admission: NewRequest,
    latent: TensorRef,
    observation: RequestOutput,
    *,
    call_id: CallId,
    batch_id: int,
) -> bytes:
    call = diffusion_finalize_call(
        admission.request_key,
        call_id=call_id,
        predecessor=observation.call_id,
        latent=latent,
    )
    report = worker.submit(
        execution_batch(batch_id=batch_id, calls=(call,), commands=())
    )
    report = finalized_report(worker, report)
    assert report.completions[0].status is CallStatus.OK
    return _media_bytes(report.completions[0])


def test_extend_then_decode_commit_the_serial_oracle_tokens():
    worker = execution_worker()
    admission = ar_params(1, block_ids=(0,))
    extend = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    extended = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(extend,),
            )
        ),
    )

    assert extended.completions[0].committed_tokens == (expected_successor(4),)
    assert extended.completions[0].kv_visible_len == 2
    assert extended.completions[0].position == 2

    first_token = extended.completions[0].committed_tokens[0]
    observation = record_completion(extend, extended)
    decode = token_call(
        admission.request_key,
        call_id=CallId(2, 0),
        predecessor=observation.call_id,
        mode=ForwardMode.DECODE,
        tokens=(first_token,),
    )
    decoded = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=2,
                admissions=(),
                calls=(decode,),
                commands=(),
            )
        ),
    )

    assert decoded.completions[0].committed_tokens == (
        expected_successor(first_token),
    )
    assert decoded.completions[0].kv_visible_len == 3
    assert decoded.completions[0].position == 3


def test_prefix_reuse_continues_from_the_admitted_logical_position():
    worker = execution_worker()
    admission = ar_params(8, block_ids=(0,), prefix_len=2)
    extend = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(4,),
    )

    report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(extend,),
            )
        ),
    )

    assert report.completions[0].kv_visible_len == 3
    assert report.completions[0].position == 3


def test_text_extension_rejects_missing_input_tokens() -> None:
    worker = execution_worker()
    admission = ar_params(9, block_ids=(0,))
    call = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    call = replace(call, input_token_ids=())
    report = finalized_report(
        worker,
        worker.submit(
            execution_batch(batch_id=1, admissions=(admission,), calls=(call,))
        ),
    )
    assert report.completions[0].status is CallStatus.ERROR
    assert report.completions[0].error_code is ErrorCode.INVALID_CALL


def test_invalid_physical_allocation_reports_error_behind_an_unobserved_parent() -> (  # noqa: E501
    None
):
    worker = execution_worker(queue_depth=2)
    admission = ar_params(9, block_ids=(0,))
    predecessor = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    worker.submit(
        execution_batch(
            batch_id=1,
            admissions=(admission,),
            calls=(predecessor,),
        )
    )
    device_parent = predecessor.call_id
    template = token_call(
        admission.request_key,
        call_id=CallId(2, 0),
        predecessor=device_parent,
        mode=ForwardMode.DECODE,
        tokens=(0,),
        predicate=predecessor.token_output,
    )
    call = replace(template, input_token_ids=())
    invalid_table = BlockTable(
        request_pool_idx=admission.request_pool_idx,
        group_id=0,
        page_ids=(),
        allocated_tokens=0,
    )

    report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=2,
                calls=(call,),
                block_tables=(invalid_table,),
            )
        ),
    )

    assert report.completions[0].status is CallStatus.ERROR
    assert report.completions[0].error_code is ErrorCode.INVALID_CALL


def test_decode_reuses_the_published_request_page_table() -> None:
    worker = execution_worker(queue_depth=2)
    admission = ar_params(10, block_ids=(0,))
    predecessor = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    worker.submit(
        execution_batch(
            batch_id=1,
            admissions=(admission,),
            calls=(predecessor,),
        )
    )
    device_parent = predecessor.call_id
    template = token_call(
        admission.request_key,
        call_id=CallId(2, 0),
        predecessor=device_parent,
        mode=ForwardMode.DECODE,
        tokens=(0,),
        predicate=predecessor.token_output,
    )
    call = replace(template, input_token_ids=())
    report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=2,
                calls=(call,),
            )
        ),
    )

    assert report.completions[0].status is CallStatus.OK
    assert report.completions[0].kv_visible_len == 3


@pytest.mark.parametrize(
    ("device", "binding", "graphs", "steps", "generation_device"),
    [
        ("cpu", "default", False, 1, None),
        ("cpu", "default", False, 2, None),
        pytest.param("cuda:0", "default", True, 2, None, marks=pytest.mark.gpu),
        pytest.param("cuda:0", "default", True, 1, None, marks=pytest.mark.gpu),
        pytest.param("cuda:0", "shared", True, 1, None, marks=pytest.mark.gpu),
        pytest.param("cuda:0", "split", True, 1, None, marks=pytest.mark.gpu),
        pytest.param("cuda:0", "split", False, 1, None, marks=pytest.mark.gpu),
        pytest.param(
            "cuda:0", "default", False, 2, "cuda:1", marks=pytest.mark.gpu
        ),
        pytest.param(
            "cuda:0", "default", True, 2, "cuda:1", marks=pytest.mark.gpu
        ),
    ],
)
def test_lane_and_graph_binding_preserve_token_and_flow_results(
    device, binding, graphs, steps, generation_device
):
    """Lanes carry a token call and a flow call without changing either.

    Each numerical domain travels in its own homogeneous batch; lanes give
    the two calls their coexistence on one rank. Binding those lanes,
    capturing graphs, or placing generation on a second device must leave
    both calls' committed results and the artifact bit-identical to a
    worker that runs them under the default execution policy.
    """
    lanes = {
        "default": (),
        "shared": (LaneConfig("compute", 152, CALL_KINDS),),
        "split": (
            LaneConfig("decode", 64, (ForwardMode.DECODE, ForwardMode.VERIFY)),
            LaneConfig(
                "compute",
                88,
                tuple(
                    kind
                    for kind in CALL_KINDS
                    if kind not in {ForwardMode.DECODE, ForwardMode.VERIFY}
                ),
            ),
        ),
    }[binding]
    policy = WorkerConfig(
        generation_device=generation_device,
        prefill_cuda_graph=graphs,
        graph_policy="full" if graphs else "off",
        decode_graph_batch_sizes=(1, 2),
        prefill_graph_token_sizes=(16, 32),
        flow_graph_batch_sizes=(1,),
        flow_graph_shapes=((16, 16),),
        lanes=lanes,
    )
    sequence_admission = ar_params(1, block_ids=(0,))
    flow_admission = umm_params(
        2, ImageParams(steps=steps, height=16, width=16, seed=29)
    )

    def run_both_domains(worker):
        """Run one token call and one flow call as separate batches."""
        conditioning = _publish_conditioning(
            worker, flow_admission, call_id=CallId(10, 0), batch_id=1
        )
        latent, preparation_observation = _prepare_media(
            worker,
            flow_admission,
            conditioning,
            call_id=CallId(11, 0),
            predecessor=root_parent(flow_admission),
            batch_id=2,
        )
        sequence, _sequence_control = _prepare_decode(
            worker,
            sequence_admission,
            call_id=CallId(12, 0),
            batch_id=3,
            tokens=(3, 4),
        )
        sequence_result = finalized_report(
            worker,
            worker.submit(execution_batch(batch_id=13, calls=(sequence,))),
        )
        flow, output_latent = diffusion_step_call(
            flow_admission.request_key,
            call_id=CallId(14, 0),
            predecessor=preparation_observation.call_id,
            conditioning=conditioning,
            latent=latent,
            steps=steps,
        )
        flow_result = finalized_report(
            worker,
            worker.submit(execution_batch(batch_id=14, calls=(flow,))),
        )
        artifact = _finalized_artifact(
            worker,
            flow_admission,
            output_latent,
            record_completion(flow, flow_result),
            call_id=CallId(15, 0),
            batch_id=15,
        )
        return sequence_result, flow_result, artifact

    with (
        execution_worker(device=device, execution=policy) as bound,
        execution_worker(device=device) as plain,
    ):
        bound_sequence, bound_flow, bound_artifact = run_both_domains(bound)
        plain_sequence, plain_flow, plain_artifact = run_both_domains(plain)

    token = bound_sequence.completions[0]
    reference_token = plain_sequence.completions[0]
    assert token.committed_tokens == reference_token.committed_tokens
    assert token.position == reference_token.position
    assert token.kv_visible_len == reference_token.kv_visible_len
    assert token.kv_computed_len == reference_token.kv_computed_len
    assert token.num_completed_steps == reference_token.num_completed_steps

    flow_completion = bound_flow.completions[0]
    reference_flow = plain_flow.completions[0]
    assert flow_completion.position == reference_flow.position
    assert flow_completion.kv_visible_len == reference_flow.kv_visible_len
    assert flow_completion.kv_computed_len == reference_flow.kv_computed_len
    assert (
        flow_completion.num_completed_steps
        == reference_flow.num_completed_steps
    )
    assert bound_artifact == plain_artifact


def test_next_image_can_start_before_the_previous_artifact_is_observed() -> (
    None
):
    worker = execution_worker(queue_depth=3)
    admission = umm_params(
        12, ImageParams(steps=1, height=16, width=16, seed=29, max_images=2)
    )
    try:
        conditioning = _publish_conditioning(
            worker, admission, call_id=CallId(1, 0), batch_id=1
        )
        initial, prepared = _prepare_media(
            worker,
            admission,
            conditioning,
            call_id=CallId(2, 0),
            predecessor=root_parent(admission),
            batch_id=2,
        )
        step, latent = diffusion_step_call(
            admission.request_key,
            call_id=CallId(3, 0),
            predecessor=prepared.call_id,
            conditioning=conditioning,
            latent=initial,
            steps=1,
        )
        stepped = worker.submit(
            execution_batch(batch_id=3, calls=(step,), commands=())
        )
        stepped = finalized_report(worker, stepped)
        accepted = record_completion(step, stepped)
        finalize = diffusion_finalize_call(
            admission.request_key,
            call_id=CallId(4, 0),
            predecessor=accepted.call_id,
            latent=latent,
        )
        first_image = worker.submit(
            execution_batch(batch_id=4, calls=(finalize,), commands=())
        )
        initial, prepared = _prepare_media(
            worker,
            admission,
            conditioning,
            call_id=CallId(5, 0),
            predecessor=accepted.call_id,
            batch_id=5,
            image_index=2,
        )
        # Reading an earlier artifact must not restore its retired trajectory.
        first_image = finalized_report(worker, first_image)
        first_report = first_image
        assert first_report.completions[0].status is CallStatus.OK
        first_png = _media_bytes(first_report.completions[0])
        with Image.open(
            io.BytesIO(base64.b64decode(first_png, validate=True))
        ) as image:
            assert image.size == (16, 16)
        step, latent = diffusion_step_call(
            admission.request_key,
            call_id=CallId(6, 0),
            predecessor=prepared.call_id,
            conditioning=conditioning,
            latent=initial,
            steps=1,
        )
        stepped = worker.submit(
            execution_batch(batch_id=6, calls=(step,), commands=())
        )
        stepped = finalized_report(worker, stepped)
        second_png = _finalized_artifact(
            worker,
            admission,
            latent,
            record_completion(step, stepped),
            call_id=CallId(7, 0),
            batch_id=7,
        )
        with Image.open(
            io.BytesIO(base64.b64decode(second_png, validate=True))
        ) as image:
            assert image.size == (16, 16)
    finally:
        worker.close()


def test_computation_identity_preserves_homogeneous_decode():
    worker = execution_worker()
    admissions = (ar_params(41, block_ids=(0,)), ar_params(42, block_ids=(1,)))
    prefill_ops = []
    last_tokens = []
    for index, admission in enumerate(admissions):
        tokens = (3 + 4 * index, 4 + 4 * index)
        call = token_call(
            admission.request_key,
            call_id=CallId(50, index),
            predecessor=root_parent(admission),
            mode=ForwardMode.PREFILL,
            tokens=tokens,
        )
        prefill_ops.append(call)

        last_tokens.append(tokens[-1])
    prefilled = worker.submit(
        execution_batch(
            batch_id=1,
            admissions=admissions,
            calls=tuple(prefill_ops),
        )
    )

    prefilled = finalized_report(worker, prefilled)
    decode_ops = []
    for index, admission in enumerate(admissions):
        observation = record_completion(prefill_ops[index], prefilled)
        call = token_call(
            admission.request_key,
            call_id=CallId(60, index),
            predecessor=observation.call_id,
            mode=ForwardMode.DECODE,
            tokens=(expected_successor(last_tokens[index]),),
        )
        decode_ops.append(call)

    decoded = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=2,
                admissions=(),
                calls=tuple(decode_ops),
                commands=(),
            )
        ),
    )

    assert tuple(
        record.committed_tokens for record in decoded.completions
    ) == tuple(
        (expected_successor(expected_successor(token)),)
        for token in last_tokens
    )


@pytest.mark.gpu
def test_failed_lane_keeps_kv_pages_until_submitted_device_work_finishes() -> (
    None
):
    model = _MisalignedOutputModel().to("cuda:0")
    worker = execution_worker(
        model,
        device="cuda:0",
        execution=WorkerConfig(
            graph_policy="off",
            prefill_cuda_graph=False,
            lanes=(LaneConfig("compute", 64, CALL_KINDS),),
        ),
    )
    admission = ar_params(4, block_ids=(4,))
    initial = token_call(
        admission.request_key,
        call_id=CallId(31, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(12, 13),
    )
    try:
        initial_report = finalized_report(
            worker,
            worker.submit(
                execution_batch(
                    batch_id=11,
                    admissions=(admission,),
                    calls=(initial,),
                )
            ),
        )
        observation = record_completion(initial, initial_report)
        retry = token_call(
            admission.request_key,
            call_id=CallId(32, 0),
            predecessor=observation.call_id,
            mode=ForwardMode.DECODE,
            tokens=(expected_successor(13),),
        )
        model.misaligned = True
        model.failure_delay_cycles = 1_000_000_000
        failed = worker.submit(
            execution_batch(
                batch_id=12,
                commands=(),
                calls=(retry,),
            )
        )
        failed = finalized_report(worker, failed)
        assert failed.completions[0].status is CallStatus.ERROR
        assert failed.completions[0].error_code is ErrorCode.COMPUTE_ERROR
        assert not worker.kv_cache.retirement_ready(
            requests=(admission.request_key,)
        )
        with pytest.raises(WorkerError, match="executing producer or consumer"):
            worker.kv_cache.zero_pages(0, (5,))

        worker.runner.synchronize()
        torch.cuda.current_stream("cuda:0").synchronize()
        worker.device_events.reap()
        assert worker.kv_cache.retirement_ready(
            requests=(admission.request_key,)
        )
        worker.kv_cache.zero_pages(0, (5,))
    finally:
        worker.close()


def test_output_validation_failure_discards_all_candidate_state():
    model = _MisalignedOutputModel()
    worker = execution_worker(model)
    admission = ar_params(4, block_ids=(4,))
    initial = token_call(
        admission.request_key,
        call_id=CallId(31, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(12, 13),
    )
    initial_report = worker.submit(
        execution_batch(
            batch_id=11,
            admissions=(admission,),
            calls=(initial,),
        )
    )
    initial_report = finalized_report(worker, initial_report)
    observation = record_completion(initial, initial_report)

    retry = token_call(
        admission.request_key,
        call_id=CallId(32, 0),
        predecessor=observation.call_id,
        mode=ForwardMode.DECODE,
        tokens=(expected_successor(13),),
    )
    retry_batch = execution_batch(
        batch_id=13, admissions=(), calls=(retry,), input_products=()
    )
    model.misaligned = True
    failed = worker.submit(retry_batch)

    failed = finalized_report(worker, failed)
    assert failed.completions[0].status is CallStatus.ERROR
    assert failed.completions[0].error_code is ErrorCode.COMPUTE_ERROR

    model.misaligned = False
    replacement = token_call(
        admission.request_key,
        call_id=CallId(33, 0),
        predecessor=observation.call_id,
        mode=ForwardMode.DECODE,
        tokens=(expected_successor(13),),
    )
    result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=14,
                calls=(replacement,),
            )
        ),
    )
    assert result.completions[0].committed_tokens == (
        expected_successor(expected_successor(13)),
    )
    assert result.completions[0].kv_visible_len == 3


def test_failed_flow_preserves_the_next_accepted_trajectory_and_final_artifact():  # noqa: E501
    worker = execution_worker(block_size=4)
    admission = umm_params(
        5, ImageParams(steps=2, height=64, width=64, seed=29)
    )
    conditioning = _publish_conditioning(
        worker, admission, call_id=CallId(40, 0), batch_id=12
    )
    latent, preparation_observation = _prepare_media(
        worker,
        admission,
        conditioning,
        call_id=CallId(41, 0),
        predecessor=root_parent(admission),
        batch_id=13,
    )
    flow, _output_latent = diffusion_step_call(
        admission.request_key,
        call_id=CallId(42, 0),
        predecessor=preparation_observation.call_id,
        conditioning=conditioning,
        latent=latent,
        steps=2,
    )

    valid_batch = execution_batch(
        batch_id=15,
        admissions=(),
        calls=(flow,),
    )
    batch = replace(
        valid_batch,
        seq_lens=tuple(
            query + 1 if slot != admission.request_pool_idx else length
            for length, query, slot in zip(
                valid_batch.seq_lens,
                valid_batch.query_lens,
                valid_batch.request_pool_indices,
                strict=True,
            )
        ),
    )

    failed = finalized_report(worker, worker.submit(batch))

    assert failed.completions[0].status is CallStatus.ERROR
    assert failed.completions[0].error_code is ErrorCode.INVALID_CALL
    assert failed.completions[0].num_completed_steps == 0

    replacement, _replacement_latent = diffusion_step_call(
        admission.request_key,
        call_id=CallId(43, 0),
        predecessor=preparation_observation.call_id,
        conditioning=conditioning,
        latent=latent,
        steps=2,
    )
    recovered = finalized_report(
        worker,
        worker.submit(execution_batch(batch_id=16, calls=(replacement,))),
    )
    assert recovered.completions[0].status is CallStatus.OK
    assert recovered.completions[0].num_completed_steps == 2
    recovered_observation = record_completion(replacement, recovered)
    recovered_artifact = _finalized_artifact(
        worker,
        admission,
        _replacement_latent,
        recovered_observation,
        call_id=CallId(44, 0),
        batch_id=17,
    )

    reference = execution_worker(block_size=4)
    reference_conditioning = _publish_conditioning(
        reference, admission, call_id=CallId(40, 0), batch_id=12
    )
    reference_latent, reference_preparation_observation = _prepare_media(
        reference,
        admission,
        reference_conditioning,
        call_id=CallId(41, 0),
        predecessor=root_parent(admission),
        batch_id=13,
    )
    reference_flow, reference_output = diffusion_step_call(
        admission.request_key,
        call_id=CallId(43, 0),
        predecessor=reference_preparation_observation.call_id,
        conditioning=reference_conditioning,
        latent=reference_latent,
        steps=2,
    )
    reference_result = reference.submit(
        execution_batch(
            batch_id=14,
            calls=(reference_flow,),
            commands=(),
        )
    )
    reference_result = finalized_report(reference, reference_result)
    assert reference_result.completions[0].status is CallStatus.OK
    reference_observation = record_completion(reference_flow, reference_result)
    reference_artifact = _finalized_artifact(
        reference,
        admission,
        reference_output,
        reference_observation,
        call_id=CallId(44, 0),
        batch_id=15,
    )
    assert recovered_artifact == reference_artifact
    reference.close()


def test_initial_flow_noise_is_stable_across_call_schedules():
    admission = umm_params(
        5, ImageParams(steps=1, height=16, width=16, seed=29)
    )
    artifacts: list[bytes] = []
    for call_id in (41, 109):
        worker = execution_worker()
        conditioning = _publish_conditioning(
            worker, admission, call_id=CallId(1, 0), batch_id=1
        )
        latent, preparation_observation = _prepare_media(
            worker,
            admission,
            conditioning,
            call_id=CallId(call_id, 0),
            predecessor=root_parent(admission),
            batch_id=2,
            seed=29,
            image_index=3,
        )
        flow, output_latent = diffusion_step_call(
            admission.request_key,
            call_id=CallId(call_id + 1, 0),
            predecessor=preparation_observation.call_id,
            conditioning=conditioning,
            latent=latent,
            steps=1,
        )
        flow_report = worker.submit(
            execution_batch(
                batch_id=3,
                admissions=(),
                calls=(flow,),
                commands=(),
            )
        )
        flow_report = finalized_report(worker, flow_report)
        flow_observation = record_completion(flow, flow_report)
        artifacts.append(
            _finalized_artifact(
                worker,
                admission,
                output_latent,
                flow_observation,
                call_id=CallId(call_id + 2, 0),
                batch_id=4,
            )
        )
        worker.close()

    assert artifacts[1] == artifacts[0]


@pytest.mark.parametrize(
    ("height", "width", "cfg_text_scale", "cfg_img_scale"),
    (
        (16, 16, 1.0, 1.0),
        (16, 32, 4.0, 1.0),
        (32, 32, 4.0, 2.0),
    ),
)
def test_multi_step_quantum_matches_the_serial_model_artifact(
    height: int,
    width: int,
    cfg_text_scale: float,
    cfg_img_scale: float,
) -> None:
    def run(step_quantum: int) -> bytes:
        worker = execution_worker()
        admission = umm_params(
            76,
            ImageParams(
                steps=4,
                height=height,
                width=width,
                seed=31,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
            ),
        )
        conditioning = _publish_conditioning(
            worker, admission, call_id=CallId(1, 0), batch_id=1
        )
        latent, observation = _prepare_media(
            worker,
            admission,
            conditioning,
            call_id=CallId(2, 0),
            predecessor=root_parent(admission),
            batch_id=2,
            seed=31,
        )
        step = 0
        call_id = 3
        while step < 4:
            count = min(step_quantum, 4 - step)
            call, successor = diffusion_step_call(
                admission.request_key,
                call_id=CallId(call_id, 0),
                predecessor=observation.call_id,
                conditioning=conditioning,
                latent=latent,
                steps=count,
            )
            report = finalized_report(
                worker,
                worker.submit(
                    execution_batch(
                        batch_id=call_id,
                        calls=(call,),
                        commands=(),
                    )
                ),
            )
            assert report.completions[0].status is CallStatus.OK
            step += count
            assert report.completions[0].num_completed_steps == step
            latent = successor
            observation = record_completion(call, report)
            call_id += 1
        artifact = _finalized_artifact(
            worker,
            admission,
            latent,
            observation,
            call_id=CallId(call_id, 0),
            batch_id=call_id,
        )
        worker.close()
        return artifact

    assert run(4) == run(1)


@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu))
)
def test_non_power_of_two_context_capacity_accepts_prefill_and_decode(
    device: str,
) -> None:
    model = Model().to(device)
    worker = execution_worker(
        model,
        block_size=4,
        device=device,
        execution=WorkerConfig(
            max_sequence_tokens=20,
            graph_policy="off",
            prefill_cuda_graph=False,
            flow_graph_batch_sizes=(1,),
            flow_graph_shapes=((16, 16),),
        ),
    )
    admission = ar_params(1, block_ids=(0, 1, 2, 3, 4))
    predecessor = root_parent(admission)
    commands = ()
    tokens = tuple(range(1, 19))
    selected = []
    try:
        for step in range(3):
            call = token_call(
                admission.request_key,
                call_id=CallId(step + 1, 0),
                predecessor=predecessor,
                mode=ForwardMode.PREFILL if step == 0 else ForwardMode.DECODE,
                tokens=tokens,
            )
            report = finalized_report(
                worker,
                worker.submit(
                    execution_batch(
                        batch_id=step + 1,
                        admissions=(admission,) if step == 0 else (),
                        commands=commands,
                        calls=(call,),
                    )
                ),
            )
            completion = report.completions[0]
            assert completion.status is CallStatus.OK
            assert completion.kv_visible_len == 18 + step
            selected.extend(completion.committed_tokens)
            tokens = completion.committed_tokens
            observation = record_completion(call, report)
            predecessor = observation.call_id
            commands = ()
        assert selected == [1000, 1001, 151670]
    finally:
        worker.close()


def test_decode_grows_logical_capacity_across_a_kv_page_boundary():
    # A small page forces the decode chain to cross a registration boundary.
    block_size = 4
    worker = execution_worker(block_size=block_size)
    admission = ar_params(1, block_ids=(0,))
    extend = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    extended = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(extend,),
            )
        ),
    )
    accepted = list(extended.completions[0].committed_tokens)
    observation = record_completion(extend, extended)
    next_block = 1
    block_count = 1
    crossed = False
    for step in range(4):
        length = 2 + step
        logical_delta: tuple[int, ...] = ()
        if length // block_size >= block_count:
            logical_delta = (next_block,)
            next_block += 1
            block_count += 1
            crossed = True
        decode = token_call(
            admission.request_key,
            call_id=CallId(2 + step, 0),
            predecessor=observation.call_id,
            mode=ForwardMode.DECODE,
            tokens=(accepted[-1],),
            block_table_delta=logical_delta,
        )
        report = finalized_report(
            worker,
            worker.submit(
                execution_batch(
                    batch_id=2 + step,
                    admissions=(),
                    calls=(decode,),
                    commands=(),
                )
            ),
        )
        assert report.completions[0].kv_visible_len == length + 1
        accepted.extend(report.completions[0].committed_tokens)
        observation = record_completion(decode, report)

    assert crossed  # the chain actually crossed a page boundary
    assert block_count == 2
    # Committed tokens follow the stub oracle unbroken across the boundary.
    chain = [expected_successor(4)]
    for _ in range(4):
        chain.append(expected_successor(chain[-1]))
    assert accepted == chain


def test_flow_run_results_cumulative_denoise_step_in_latent_len():
    # The ordered-observation validator matches latent_len against the
    # cumulative denoise step (start_step + step_count), not a constant token
    # count, so two single-step quanta must report 1 then 2.
    worker = execution_worker()
    admission = umm_params(
        2, ImageParams(steps=2, height=16, width=16, seed=29)
    )
    conditioning = _publish_conditioning(
        worker, admission, call_id=CallId(1, 0), batch_id=1
    )
    initial_latent, preparation_observation = _prepare_media(
        worker,
        admission,
        conditioning,
        call_id=CallId(2, 0),
        predecessor=root_parent(admission),
        batch_id=2,
    )
    first, first_latent = diffusion_step_call(
        admission.request_key,
        call_id=CallId(3, 0),
        predecessor=preparation_observation.call_id,
        conditioning=conditioning,
        latent=initial_latent,
        steps=1,
    )
    first_report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=3,
                admissions=(),
                calls=(first,),
                commands=(),
            )
        ),
    )
    observation = record_completion(first, first_report)
    second, _second_latent = diffusion_step_call(
        admission.request_key,
        call_id=CallId(4, 0),
        predecessor=observation.call_id,
        conditioning=conditioning,
        latent=first_latent,
        steps=1,
    )
    second_report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=4,
                admissions=(),
                calls=(second,),
                commands=(),
            )
        ),
    )

    assert first_report.completions[0].num_completed_steps == 1
    assert second_report.completions[0].num_completed_steps == 2


def test_trajectory_advances_across_many_generations_and_rejects_a_stale_reference(  # noqa: E501
):
    worker = execution_worker()
    admission = umm_params(
        71, ImageParams(steps=50, height=16, width=16, seed=29)
    )
    conditioning = _publish_conditioning(
        worker, admission, call_id=CallId(1, 0), batch_id=1
    )
    current, observation = _prepare_media(
        worker,
        admission,
        conditioning,
        call_id=CallId(2, 0),
        predecessor=root_parent(admission),
        batch_id=2,
    )
    stale = current
    releasable = None

    for index in range(50):
        call, successor = diffusion_step_call(
            admission.request_key,
            call_id=CallId(3 + index, 0),
            predecessor=observation.call_id,
            conditioning=conditioning,
            latent=current,
            steps=1,
        )
        report = finalized_report(
            worker,
            worker.submit(
                execution_batch(
                    batch_id=3 + index,
                    calls=(call,),
                    commands=(),
                )
            ),
        )
        assert report.completions[0].status is CallStatus.OK
        assert report.completions[0].num_completed_steps == index + 1
        releasable = current
        current = successor
        observation = record_completion(call, report)

    worker.submit(
        execution_batch(
            batch_id=53,
            commands=(Free(releasable.buffer_id),),
        )
    )
    stale_call, _unused = diffusion_step_call(
        admission.request_key,
        call_id=CallId(54, 0),
        predecessor=observation.call_id,
        conditioning=conditioning,
        latent=stale,
        steps=1,
    )
    stale_report = worker.submit(
        execution_batch(batch_id=54, calls=(stale_call,))
    )
    stale_report = finalized_report(worker, stale_report)
    assert stale_report.completions[0].status is CallStatus.ERROR


def test_cross_stage_feature_transfer_rebinds_exact_product_without_request_thread_wait(  # noqa: E501
) -> None:
    producer = execution_worker(transfer_backends=("shm",))
    consumer = execution_worker(transfer_backends=("shm",))
    admission = ar_params(73, block_ids=(0,))
    image = io.BytesIO()
    Image.new("RGB", (16, 16), (64, 96, 128)).save(image, format="PNG")
    encoded = base64.b64encode(image.getvalue()).decode("ascii")
    call = encode_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        image_base64=encoded,
        encoder_handle=11,
        component="vision_encoder",
    )
    try:
        produced = producer.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(call,),
            )
        )
        deadline = time.monotonic() + 5.0
        while not produced.complete and time.monotonic() < deadline:
            producer.advance()
            time.sleep(0.001)
        assert produced.complete
        produced = finalized_report(producer, produced)
        assert len(produced.products) == 1
        transferred = produced.products[0]
        assert transferred.product == call.encoder_output

        visual = visual_state_call(
            admission.request_key,
            call_id=CallId(2, 0),
            predecessor=root_parent(admission),
            feature=call.encoder_output,
            sample_continuation=False,
            max_tokens=2,
        )
        batch = execution_batch(
            batch_id=2,
            admissions=(admission,),
            calls=(visual,),
            input_products=(transferred,),
        )
        prepared = consumer.submit(batch)
        assert prepared is not None
        deadline = time.monotonic() + 5.0
        while not prepared.inputs_ready() and time.monotonic() < deadline:
            consumer.advance_inputs(prepared)
            time.sleep(0.001)
        assert prepared.inputs_ready()
        consumed = prepared
        consumed = finalized_report(consumer, consumed)
        assert consumed.completions[0].status is CallStatus.OK
        with pytest.raises(WorkerError, match="no longer owned"):
            consumer.poll(prepared)
        producer.submit(
            execution_batch(
                batch_id=3,
                commands=(Free(call.encoder_output.buffer_id),),
            )
        )
    finally:
        producer.close()
        consumer.close()


def test_free_preserves_another_requests_feature_with_the_same_generation() -> (
    None
):
    worker = execution_worker(transfer_backends=("shm",))
    admissions = (ar_params(93, block_ids=(0,)), ar_params(94, block_ids=(1,)))
    image = io.BytesIO()
    Image.new("RGB", (16, 16), (64, 96, 128)).save(image, format="PNG")
    encoded = base64.b64encode(image.getvalue()).decode("ascii")
    calls = tuple(
        encode_call(
            admission.request_key,
            call_id=CallId(1, index),
            predecessor=root_parent(admission),
            image_base64=encoded,
            encoder_handle=11,
            component="vision_encoder",
        )
        for index, admission in enumerate(admissions)
    )
    try:
        produced = finalized_report(
            worker,
            worker.submit(
                execution_batch(
                    batch_id=1,
                    admissions=admissions,
                    calls=calls,
                )
            ),
        )
        assert all(
            completion.status is CallStatus.OK
            for completion in produced.completions
        )
        worker.submit(
            execution_batch(
                batch_id=2,
                commands=(Free(calls[0].encoder_output.buffer_id),),
            )
        )
        visual = visual_state_call(
            admissions[1].request_key,
            call_id=CallId(3, 0),
            predecessor=root_parent(admissions[1]),
            feature=calls[1].encoder_output,
            sample_continuation=False,
            max_tokens=2,
        )
        consumed = finalized_report(
            worker,
            worker.submit(execution_batch(batch_id=3, calls=(visual,))),
        )
        assert consumed.completions[0].status is CallStatus.OK
    finally:
        worker.close()


def test_cross_stage_device_product_transfer_preserves_generation_and_value() -> (  # noqa: E501
    None
):
    producer = execution_worker(transfer_backends=("shm",))
    consumer = execution_worker(transfer_backends=("shm",))
    admission = ar_params(77, block_ids=(0,))
    extend = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    try:
        extended = finalized_report(
            producer,
            producer.submit(
                execution_batch(
                    batch_id=1,
                    admissions=(admission,),
                    calls=(extend,),
                )
            ),
        )
        record_completion(extend, extended)
        source = extend.token_output
        transferred = replace(
            source, producer_call_id=CallId(2, 0), generation=901
        )
        transfer = Call(
            request_key=admission.request_key,
            call_id=CallId(2, 0),
            coordinates=CallCoordinates(),
            kind=TransferMode.TENSOR,
            bounds=Bounds(max_transfer_bytes=source.max_bytes),
            token_input=source,
            token_output=transferred,
        )
        transfer_report = finalized_report(
            producer,
            producer.submit(
                execution_batch(
                    batch_id=2,
                    calls=(transfer,),
                    commands=(),
                )
            ),
        )
        assert len(transfer_report.products) == 1
        payload = transfer_report.products[0]
        assert payload.product == transferred

        descriptor = payload.value
        assert isinstance(descriptor, DeviceProductTransferValue)

        consume = token_call(
            admission.request_key,
            call_id=CallId(3, 0),
            predecessor=root_parent(admission),
            mode=ForwardMode.PREFILL,
            tokens=(9,),
            predicate=transferred,
        )

        prepared = consumer.submit(
            execution_batch(
                batch_id=3,
                admissions=(admission,),
                calls=(consume,),
                input_products=(payload,),
            )
        )
        assert prepared is not None
        deadline = time.monotonic() + 5.0
        while not prepared.inputs_ready() and time.monotonic() < deadline:
            consumer.advance_inputs(prepared)
            time.sleep(0.001)
        assert prepared.inputs_ready()
        prepared = finalized_report(consumer, prepared)
        consumed = prepared
        assert consumed.completions[0].status is CallStatus.OK
        assert consumed.completions[0].kv_visible_len == 1
    finally:
        producer.close()
        consumer.close()


@pytest.mark.parametrize("output_dtype", (DType.BF16, DType.F32))
def test_local_transfer_retains_its_value_when_the_source_buffer_is_reused(
    output_dtype: DType,
) -> None:
    worker = execution_worker()
    admission = ar_params(95, block_ids=(0,))

    def encoded_call(call_id: CallId, color: tuple[int, int, int]):
        image = io.BytesIO()
        Image.new("RGB", (16, 16), color).save(image, format="PNG")
        return encode_call(
            admission.request_key,
            call_id=call_id,
            predecessor=root_parent(admission),
            image_base64=base64.b64encode(image.getvalue()).decode("ascii"),
            encoder_handle=call_id.batch_id,
            component="vision_encoder",
        )

    original = encoded_call(CallId(1, 0), (64, 96, 128))
    initial = execution_batch(
        batch_id=1,
        admissions=(admission,),
        calls=(original,),
    )
    source = original.encoder_output
    output = replace(
        source,
        producer_call_id=CallId(2, 0),
        generation=2,
        dtype=output_dtype,
    )
    transfer = Call(
        request_key=admission.request_key,
        call_id=CallId(2, 0),
        coordinates=CallCoordinates(),
        kind=TransferMode.TENSOR,
        bounds=Bounds(
            max_transfer_bytes=source.max_bytes,
            max_latent_bytes=output.max_bytes,
        ),
        vision_input=source,
        encoder_output=output,
    )
    ticket = None
    try:
        finalized_report(worker, worker.submit(initial))
        report = finalized_report(
            worker,
            worker.submit(execution_batch(batch_id=2, calls=(transfer,))),
        )
        if output_dtype is not source.dtype:
            assert report.completions[0].status is CallStatus.ERROR
            return
        handle = report.products[0].value

        assert isinstance(handle, EncoderTransferValue)
        locator = handle.tensor.locations[0]
        ticket = worker.transports[locator.backend].fetch(
            locator, device=torch.device("cpu")
        )
        expected = ticket.result().clone()
        finalized_report(
            worker,
            worker.submit(
                execution_batch(batch_id=3, commands=(Free(source.buffer_id),))
            ),
        )

        replacement = encoded_call(CallId(3, 0), (192, 160, 32))
        reuse = execution_batch(
            batch_id=4,
            calls=(replacement,),
        )
        source_allocation = next(
            item
            for item in initial.buffer_allocations
            if item.buffer == source.buffer_id
        )
        reuse = replace(
            reuse,
            buffer_allocations=tuple(
                replace(item, offset=source_allocation.offset)
                if item.buffer == replacement.encoder_output.buffer_id
                else item
                for item in reuse.buffer_allocations
            ),
        )
        finalized_report(worker, worker.submit(reuse))
        torch.testing.assert_close(ticket.result(), expected, rtol=0, atol=0)

        released = worker.submit(
            execution_batch(batch_id=5, commands=(Free(output.buffer_id),))
        )
        assert not released.complete
        ticket.close()
        released = finalized_report(worker, released)
    finally:
        if ticket is not None:
            ticket.close()
        worker.close()


@pytest.mark.parametrize("backend", ("local", "shm"))
@pytest.mark.parametrize(
    "dtype", (DType.BF16, DType.F32, DType.I32, DType.I64, DType.I16)
)
def test_tensor_entry_input_preserves_values_through_output_release(
    backend: str,
    dtype: DType,
) -> None:
    from threading import Event

    from tests.python.fixtures.transport import make_transport
    from uniserve.runtime import EventPool
    from uniserve_worker.protocol.transfer import WorkerEndpoint

    events = EventPool()
    producer = make_transport(
        backend,
        byte_capacity=4096,
        ticket_capacity=2,
        event_pool=events,
        source=WorkerEndpoint.local("text_encoder"),
    )
    worker = execution_worker(transfer_backends=(backend,))
    admission = ar_params(97, block_ids=(0,))
    source = TensorRef(
        request_key=admission.request_key,
        producer_call_id=CallId(1, 0),
        output_index=0,
        generation=1,
        dtype=dtype,
        shape_bound=ShapeBound((StaticDim(3), StaticDim(4))),
    )
    output = replace(source, producer_call_id=CallId(2, 0), generation=2)
    call = Call(
        request_key=admission.request_key,
        call_id=CallId(2, 0),
        coordinates=CallCoordinates(),
        kind=TransferMode.TENSOR,
        bounds=Bounds(max_transfer_bytes=source.max_bytes),
        inputs=(source,),
        outputs=(output,),
    )
    storage_dtype = {
        DType.BF16: torch.bfloat16,
        DType.F32: torch.float32,
        DType.I32: torch.int32,
        DType.I64: torch.int64,
        DType.I16: torch.int16,
    }[dtype]
    expected = torch.arange(12, dtype=storage_dtype).reshape(3, 4)
    if dtype is DType.I16:
        expected[0] = torch.tensor([-32768, -1, 0, 32767], dtype=storage_dtype)
    elif dtype is DType.I32:
        expected[0] = torch.tensor(
            [-(1 << 31), -1, 0, (1 << 31) - 1], dtype=storage_dtype
        )
    elif dtype is DType.I64:
        # Preserve signed values and high bits used by device continuation data.
        expected[0] = torch.tensor(
            [-(1 << 63), -1, 1 << 40, (1 << 63) - 1], dtype=storage_dtype
        )
    location = producer.publish(expected)
    payload = TensorPublication(
        product=source,
        value=DeviceProductTransferValue(
            height=0,
            width=0,
            value_range="",
            tensor=TensorTransfer(shape=(3, 4), locations=(location,)),
        ),
    )
    reader = None
    try:
        prepared = worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(call,),
                input_products=(payload,),
            )
        )
        assert prepared is not None
        deadline = time.monotonic() + 5
        while not prepared.inputs_ready() and time.monotonic() < deadline:
            worker.advance_inputs(prepared)
            time.sleep(0.001)
        assert prepared.inputs_ready()
        prepared = finalized_report(worker, prepared)
        report = prepared
        assert report.completions[0].status is CallStatus.OK
        descriptor = report.products[0].value

        assert isinstance(descriptor, DeviceProductTransferValue)
        locator = descriptor.tensor.locations[0]
        reader = worker.transports[backend].fetch(
            locator, device=torch.device("cpu")
        )
        ready = Event()
        reader.add_done_callback(ready.set)
        assert ready.wait(5)
        torch.testing.assert_close(reader.result(), expected, rtol=0, atol=0)
        released = worker.submit(
            execution_batch(batch_id=2, commands=(Free(output.buffer_id),))
        )
        torch.testing.assert_close(reader.result(), expected, rtol=0, atol=0)
        reader.close()
        released = finalized_report(worker, released)
    finally:
        if reader is not None:
            reader.close()
        worker.close()
        producer.release(location)
        producer.close()
        events.close()


def test_cross_stage_completion_predicate_preserves_device_continuation() -> (
    None
):
    producer = execution_worker(transfer_backends=("shm",))
    consumer = execution_worker(transfer_backends=("shm",))
    generation = umm_params(
        79, ImageParams(steps=1, height=16, width=16, seed=31)
    )
    admission = NewRequest(
        generation.request_key,
        request_pool_idx=generation.request_pool_idx,
        generation=GenerationParams(),
        image=generation.image,
    )
    bind_request_allocation(
        admission.request_key,
        request_pool_idx=admission.request_pool_idx,
        page_ids=(1,),
    )
    try:
        conditioning = _publish_conditioning(
            producer, admission, call_id=CallId(1, 0), batch_id=1
        )
        preparation, _latent = diffusion_prepare_call(
            admission.request_key,
            call_id=CallId(2, 0),
            predecessor=root_parent(admission),
            conditioning=conditioning,
            seed=31,
        )
        transitioned = finalized_report(
            producer,
            producer.submit(execution_batch(batch_id=2, calls=(preparation,))),
        )
        record_completion(preparation, transitioned)
        source = preparation.completion_output
        transferred = replace(
            source, producer_call_id=CallId(3, 0), generation=903
        )
        transfer = Call(
            request_key=admission.request_key,
            call_id=CallId(3, 0),
            coordinates=CallCoordinates(),
            kind=TransferMode.TENSOR,
            bounds=Bounds(max_transfer_bytes=source.max_bytes),
            token_input=source,
            token_output=transferred,
        )
        transfer_report = finalized_report(
            producer,
            producer.submit(
                execution_batch(
                    batch_id=3,
                    calls=(transfer,),
                    commands=(),
                )
            ),
        )
        payload = next(
            product
            for product in transfer_report.products
            if product.product == transferred
        )
        consume = token_call(
            admission.request_key,
            call_id=CallId(4, 0),
            predecessor=root_parent(admission),
            mode=ForwardMode.PREFILL,
            tokens=(9,),
            predicate=transferred,
        )
        prepared = consumer.submit(
            execution_batch(
                batch_id=4,
                admissions=(admission,),
                calls=(consume,),
                input_products=(payload,),
            )
        )
        assert prepared is not None
        deadline = time.monotonic() + 5.0
        while not prepared.inputs_ready() and time.monotonic() < deadline:
            consumer.advance_inputs(prepared)
            time.sleep(0.001)
        assert prepared.inputs_ready()
        prepared = finalized_report(consumer, prepared)
        consumed = prepared
        assert consumed.completions[0].status is CallStatus.OK
        assert consumed.completions[0].kv_visible_len == 1
    finally:
        producer.close()
        consumer.close()


@pytest.mark.parametrize("height", (16, 80))
@pytest.mark.parametrize("publication", ("pages", "shards", "transfer"))
def test_cross_stage_latent_transfer_preserves_generation_step_and_artifact(
    height: int, publication: str
) -> None:
    producer = execution_worker(transfer_backends=("shm",))
    consumer = execution_worker(transfer_backends=("shm",))
    admission = umm_params(
        75, ImageParams(steps=1, height=height, width=height, seed=29)
    )
    conditioning = _publish_conditioning(
        producer, admission, call_id=CallId(1, 0), batch_id=1
    )
    initial_latent, preparation_observation = _prepare_media(
        producer,
        admission,
        conditioning,
        call_id=CallId(2, 0),
        predecessor=root_parent(admission),
        batch_id=2,
    )
    flow, final_latent = diffusion_step_call(
        admission.request_key,
        call_id=CallId(3, 0),
        predecessor=preparation_observation.call_id,
        conditioning=conditioning,
        latent=initial_latent,
        steps=1,
    )
    produced = producer.submit(
        execution_batch(
            batch_id=3,
            calls=(flow,),
            commands=(),
        )
    )
    deadline = time.monotonic() + 5.0
    while not produced.complete and time.monotonic() < deadline:
        producer.advance()
        time.sleep(0.001)
    assert produced.complete
    produced = finalized_report(producer, produced)
    flow_observation = record_completion(flow, produced)
    exported_latent = final_latent
    if publication == "transfer":
        exported_latent = replace(
            final_latent, producer_call_id=CallId(4, 0), generation=904
        )
        transfer = Call(
            request_key=admission.request_key,
            call_id=CallId(4, 0),
            coordinates=CallCoordinates(),
            kind=TransferMode.TENSOR,
            bounds=Bounds(
                max_transfer_bytes=final_latent.max_bytes,
                max_latent_bytes=exported_latent.max_bytes,
            ),
            latent_input=final_latent,
            latent_output=exported_latent,
        )
        exported = finalized_report(
            producer,
            producer.submit(
                execution_batch(batch_id=4, calls=(transfer,), commands=())
            ),
        )
        assert exported.completions[0].status is CallStatus.OK
        transferred = tuple(
            product
            for product in exported.products
            if product.product == exported_latent
        )
    else:
        transferred = tuple(
            product
            for product in produced.products
            if product.product == final_latent
        )
    assert len(transferred) == 1

    if publication == "shards":
        from uniserve_worker.transfer.layout import fetch_tensor

        descriptor = transferred[0].value
        tensor = descriptor.tensor
        source_value = torch.empty(
            tensor.shape, dtype=getattr(torch, tensor.dtype)
        )
        tickets = fetch_tensor(
            tensor,
            source_value,
            bindings={
                (location.source, location.backend): producer.transports[
                    location.backend
                ]
                for location in tensor.locations
            },
        )
        deadline = time.monotonic() + 5.0
        while (
            not all(ticket.ready() for ticket in tickets)
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        for ticket in tickets:
            ticket.result()
        axis = 0 if source_value.shape[0] > 1 else 1
        first, second = source_value.chunk(2, dim=axis)
        offset = tuple(
            first.shape[axis] if index == axis else 0 for index in range(2)
        )
        locations = (
            producer.transports["shm"].publish(first),
            producer.transports["shm"].publish(second, offset=offset),
        )
        transferred = (
            replace(
                transferred[0],
                value=replace(
                    descriptor,
                    tensor=TensorTransfer(
                        shape=tensor.shape, locations=locations
                    ),
                ),
            ),
        )

    source_artifact = _finalized_artifact(
        producer,
        admission,
        final_latent,
        flow_observation,
        call_id=CallId(5, 0),
        batch_id=5,
    )
    diffusion_finalize = diffusion_finalize_call(
        admission.request_key,
        call_id=CallId(5, 0),
        predecessor=root_parent(admission),
        latent=exported_latent,
    )
    batch = execution_batch(
        batch_id=5,
        admissions=(admission,),
        calls=(diffusion_finalize,),
        input_products=transferred,
    )
    # The consumer's physical page order is independent of the publisher's.
    batch = replace(
        batch,
        latent_params=tuple(
            replace(
                allocation,
                page_table=tuple(reversed(allocation.page_table)),
                start_step=1,
            )
            for allocation in batch.latent_params
        ),
    )
    prepared = consumer.submit(batch)
    assert prepared is not None
    deadline = time.monotonic() + 5.0
    while not prepared.inputs_ready() and time.monotonic() < deadline:
        consumer.advance_inputs(prepared)
        time.sleep(0.001)
    assert prepared.inputs_ready()
    received = prepared
    deadline = time.monotonic() + 5.0
    while not received.complete and time.monotonic() < deadline:
        consumer.advance()
        time.sleep(0.001)
    assert received.complete
    received = finalized_report(consumer, received)
    assert received.completions[0].status is CallStatus.OK
    assert received.completions[0].num_completed_steps == 0
    assert _media_bytes(received.completions[0]) == source_artifact
    producer.close()
    consumer.close()


def test_encode_publishes_an_immutable_feature_without_advancing_state():
    worker = execution_worker()
    admission = ar_params(3, block_ids=(0,))
    extend = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    extended = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(extend,),
            )
        ),
    )
    session_kv_before = 2

    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 128, 128)).save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    handle = 0xABCDEF
    observation = record_completion(extend, extended)
    encode = encode_call(
        admission.request_key,
        call_id=CallId(2, 0),
        predecessor=observation.call_id,
        image_base64=image_base64,
        encoder_handle=handle,
        component="vision_encoder",
    )
    report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=2,
                admissions=(),
                calls=(encode,),
                commands=(),
            )
        ),
    )
    completion = report.completions[0]
    assert completion.kv_visible_len == session_kv_before
    assert completion.position == 2
    assert completion.product_generations
    assert completion.product_generations[0] == handle
    assert completion.product_generations[0] != 0


@pytest.mark.parametrize(
    ("transfer_image", "encoding_fails"),
    ((False, False), (True, False), (False, True)),
)
def test_resident_image_materialization_preserves_the_decoded_artifact(
    transfer_image: bool,
    encoding_fails: bool,
    monkeypatch,
) -> None:
    worker = execution_worker(transfer_backends=("shm",))
    admission = umm_params(
        76,
        ImageParams(steps=1, height=16, width=16, seed=31, retain_images=True),
    )
    try:
        conditioning = _publish_conditioning(
            worker, admission, call_id=CallId(1, 0), batch_id=1
        )
        latent, observation = _prepare_media(
            worker,
            admission,
            conditioning,
            call_id=CallId(2, 0),
            predecessor=root_parent(admission),
            batch_id=2,
            seed=31,
        )
        step, completed = diffusion_step_call(
            admission.request_key,
            call_id=CallId(3, 0),
            predecessor=observation.call_id,
            conditioning=conditioning,
            latent=latent,
            steps=1,
        )
        stepped = finalized_report(
            worker,
            worker.submit(execution_batch(batch_id=3, calls=(step,))),
        )
        assert stepped.completions[0].status is CallStatus.OK
        decode = diffusion_finalize_call(
            admission.request_key,
            call_id=CallId(4, 0),
            predecessor=step.call_id,
            latent=completed,
            feedback_source=True,
        )
        decoded = finalized_report(
            worker,
            worker.submit(execution_batch(batch_id=4, calls=(decode,))),
        )
        assert decoded.completions[0].status is CallStatus.OK
        expected = _media_bytes(decoded.completions[0])
        image = decode.image_output
        assert image is not None
        if transfer_image:
            moved = replace(
                image, producer_call_id=CallId(5, 0), generation=500
            )
            transfer = Call(
                request_key=admission.request_key,
                call_id=moved.producer_call_id,
                coordinates=CallCoordinates(),
                kind=TransferMode.TENSOR,
                bounds=Bounds(max_transfer_bytes=image.max_bytes),
                image_input=image,
                image_output=moved,
            )
            transferred = finalized_report(
                worker,
                worker.submit(execution_batch(batch_id=5, calls=(transfer,))),
            )
            assert transferred.completions[0].status is CallStatus.OK
            descriptor = transferred.products[0].value
            assert isinstance(descriptor, DeviceProductTransferValue)
            # RGB reconstruction preserves the signed [-1, 1] numerical
            # contract.
            assert (
                descriptor.height,
                descriptor.width,
                descriptor.value_range,
            ) == (
                16,
                16,
                "signed_unit",
            )
            image = moved
        materialize = Call(
            request_key=admission.request_key,
            call_id=CallId(6, 0),
            coordinates=CallCoordinates(),
            kind=MediaCall.IMAGE_DECODING,
            bounds=Bounds(max_completion_bytes=65_536),
            image_input=image,
        )
        if encoding_fails:

            def fail_encoding(*args, **kwargs):
                raise OSError("image encoder could not write the artifact")

            # Exercise a third-party codec failure through the actual CPU task
            # and completion buffer rather than replacing the worker's owners.
            monkeypatch.setattr(Image.Image, "save", fail_encoding)
        result = finalized_report(
            worker,
            worker.submit(execution_batch(batch_id=6, calls=(materialize,))),
        )
        completion = result.completions[0]
        if encoding_fails:
            assert completion.status is CallStatus.ERROR
            assert completion.error_code is ErrorCode.COMPUTE_ERROR
            assert completion.media_output is None
            assert completion.committed_tokens == ()
            assert completion.product_generations == ()
        else:
            assert completion.status is CallStatus.OK
            assert _media_bytes(completion) == expected
    finally:
        worker.close()


def test_generated_feedback_commits_absolute_visual_token_state():
    worker = execution_worker()
    understanding = ar_params(6, block_ids=(0,))
    admission = NewRequest(
        understanding.request_key,
        request_pool_idx=understanding.request_pool_idx,
        generation=understanding.generation,
        image=ImageParams(
            steps=2, height=16, width=16, seed=29, retain_images=True
        ),
    )
    extend = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    extended = worker.submit(
        execution_batch(
            batch_id=1,
            admissions=(admission,),
            calls=(extend,),
        )
    )
    extended = finalized_report(worker, extended)
    first_observation = record_completion(extend, extended)
    publication, conditioning = kv_publication_call(
        admission.request_key,
        call_id=CallId(2, 0),
        predecessor=first_observation.call_id,
    )
    worker.submit(
        execution_batch(
            batch_id=2,
            admissions=(),
            calls=(publication,),
            commands=(),
        )
    )
    initial_latent, preparation_observation = _prepare_media(
        worker,
        admission,
        conditioning,
        call_id=CallId(3, 0),
        predecessor=first_observation.call_id,
        batch_id=3,
    )
    flow, completed_latent = diffusion_step_call(
        admission.request_key,
        call_id=CallId(4, 0),
        predecessor=preparation_observation.call_id,
        conditioning=conditioning,
        latent=initial_latent,
        steps=2,
    )
    flow_report = worker.submit(
        execution_batch(
            batch_id=4,
            admissions=(),
            calls=(flow,),
            commands=(),
        )
    )
    flow_report = finalized_report(worker, flow_report)
    observation = record_completion(flow, flow_report)
    diffusion_finalize = diffusion_finalize_call(
        admission.request_key,
        call_id=CallId(5, 0),
        predecessor=observation.call_id,
        latent=completed_latent,
        feedback_source=True,
    )
    diffusion_finalize_report = worker.submit(
        execution_batch(
            batch_id=5,
            admissions=(),
            calls=(diffusion_finalize,),
            commands=(),
        )
    )
    deadline = time.monotonic() + 5.0
    while (
        not diffusion_finalize_report.complete and time.monotonic() < deadline
    ):
        worker.advance()
        time.sleep(0.001)
    assert diffusion_finalize_report.complete
    diffusion_finalize_report = finalized_report(
        worker, diffusion_finalize_report
    )
    assert diffusion_finalize_report.completions[0].kv_visible_len == 2

    encode = encode_call(
        admission.request_key,
        call_id=CallId(6, 0),
        predecessor=observation.call_id,
        image_base64=None,
        encoder_handle=10,
        component="vision_encoder",
        source_product=diffusion_finalize.image_output,
    )
    encode_report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=6,
                admissions=(),
                calls=(encode,),
                input_products=(),
            )
        ),
    )
    assert encode_report.completions[0].kv_visible_len == 2

    state = visual_state_call(
        admission.request_key,
        call_id=CallId(7, 0),
        predecessor=observation.call_id,
        feature=encode.encoder_output,
        sample_continuation=True,
        max_tokens=2,
    )
    next_token = expected_successor(1007)
    transition_product = TensorRef(
        request_key=admission.request_key,
        producer_call_id=state.call_id,
        output_index=3,
        generation=state.call_id.batch_id * 8 + 9,
        dtype=DType.U8,
        shape_bound=ShapeBound(),
    )
    state = replace(
        state,
        sampling_state=SamplingState(
            finish_token_ids=(next_token + 1,),
            transition_token_ids=(next_token,),
        ),
        transition_output=transition_product,
    )
    report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=7,
                admissions=(),
                calls=(state,),
            )
        ),
    )

    completion = report.completions[0]
    assert completion.committed_tokens == (next_token,)
    assert completion.position == 4
    assert completion.kv_visible_len == 4

    feedback_observation = record_completion(state, report)
    publication, next_conditioning = kv_publication_call(
        admission.request_key,
        call_id=CallId(8, 0),
        predecessor=feedback_observation.call_id,
    )
    worker.submit(
        execution_batch(
            batch_id=8,
            calls=(publication,),
            commands=(),
        )
    )
    next_latent, _next_preparation_observation = _prepare_media(
        worker,
        admission,
        next_conditioning,
        call_id=CallId(9, 0),
        predecessor=feedback_observation.call_id,
        batch_id=9,
        image_index=2,
    )
    next_flow, next_completed_latent = diffusion_step_call(
        admission.request_key,
        call_id=CallId(10, 0),
        predecessor=_next_preparation_observation.call_id,
        conditioning=next_conditioning,
        latent=next_latent,
        steps=2,
    )
    next_report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=10,
                calls=(next_flow,),
                commands=(),
            )
        ),
    )
    assert next_report.completions[0].status is CallStatus.OK
    assert next_report.completions[0].num_completed_steps == 2
    next_observation = record_completion(next_flow, next_report)
    assert base64.b64decode(
        _finalized_artifact(
            worker,
            admission,
            next_completed_latent,
            next_observation,
            call_id=CallId(11, 0),
            batch_id=11,
        ).decode("ascii"),
        validate=True,
    ).startswith(_PNG_MAGIC)

    png_b64 = _media_bytes(diffusion_finalize_report.completions[0])
    png_bytes = base64.b64decode(png_b64.decode("ascii"), validate=True)
    assert png_bytes[:8] == _PNG_MAGIC
    with Image.open(io.BytesIO(png_bytes)) as image:
        assert image.size == (16, 16)


@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu))
)
def test_latent_bank_reuse_waits_for_a_reader_after_free_without_blocking_independent_work(  # noqa: E501
    device,
):
    from threading import Event

    policy = WorkerConfig(
        prefill_cuda_graph=False,
        graph_policy="off",
        lanes=(
            LaneConfig("decode", 64, (ForwardMode.DECODE, ForwardMode.VERIFY)),
            LaneConfig(
                "compute",
                88,
                tuple(
                    kind
                    for kind in CALL_KINDS
                    if kind not in {ForwardMode.DECODE, ForwardMode.VERIFY}
                ),
            ),
        )
        if device.startswith("cuda")
        else (),
    )
    worker = execution_worker(
        transfer_backends=("shm",), device=device, execution=policy
    )
    admission = umm_params(
        76, ImageParams(steps=3, height=16, width=16, seed=29)
    )
    conditioning = _publish_conditioning(
        worker, admission, call_id=CallId(1, 0), batch_id=1
    )
    initial, observation = _prepare_media(
        worker,
        admission,
        conditioning,
        call_id=CallId(2, 0),
        predecessor=root_parent(admission),
        batch_id=2,
    )
    first, first_latent = diffusion_step_call(
        admission.request_key,
        call_id=CallId(3, 0),
        predecessor=observation.call_id,
        conditioning=conditioning,
        latent=initial,
        steps=1,
    )
    # An external consumer, named on the call, holds the first latent
    # unacknowledged across its semantic Free.
    first = replace(first, consumer_slots=(1,))
    first_report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=3,
                calls=(first,),
                commands=(),
            )
        ),
    )
    payload = next(
        product.value
        for product in first_report.products
        if product.product == first_latent
    )

    locator = payload.tensor.locations[0]
    prepared = None
    held = _open_shared_memory(
        locator.transport.name, segment.HEADER_BYTES + locator.nbytes
    )
    held_header = memoryview(held)
    try:
        segment.await_ready(held_header)
        try:
            observation = record_completion(first, first_report)
            second, second_latent = diffusion_step_call(
                admission.request_key,
                call_id=CallId(4, 0),
                predecessor=observation.call_id,
                conditioning=conditioning,
                latent=first_latent,
                steps=1,
            )
            second_report = finalized_report(
                worker,
                worker.submit(
                    execution_batch(
                        batch_id=4,
                        calls=(second,),
                        commands=(Free(initial.buffer_id),),
                    )
                ),
            )
            assert second_report.completions[0].status is CallStatus.OK
            observation = record_completion(second, second_report)
            third, _final_latent = diffusion_step_call(
                admission.request_key,
                call_id=CallId(5, 0),
                predecessor=observation.call_id,
                conditioning=conditioning,
                latent=second_latent,
                steps=1,
            )
            prepared = worker.submit(
                execution_batch(
                    batch_id=5,
                    calls=(third,),
                    commands=(Free(first_latent.buffer_id),),
                )
            )
            assert not prepared.inputs_ready()
            woke = Event()
            prepared.on_dependencies_ready(woke.set)
            assert not woke.is_set()

            independent = ar_params(77, block_ids=(7,))
            call = token_call(
                independent.request_key,
                call_id=CallId(1, 0),
                predecessor=root_parent(independent),
                mode=ForwardMode.PREFILL,
                tokens=(3, 4),
            )
            report = finalized_report(
                worker,
                worker.submit(
                    execution_batch(
                        batch_id=6,
                        admissions=(independent,),
                        calls=(call,),
                    )
                ),
            )
            assert report.completions[0].status is CallStatus.OK
            assert not prepared.inputs_ready()
        finally:
            # The consumer's word lands with no wake; the rank's next pass
            # reclaims the source and the bank writer proceeds.
            segment.acknowledge(held_header, 1)
        worker.advance()
        assert woke.wait(5), "acknowledged source did not wake the bank writer"
        assert prepared.inputs_ready()
        prepared = finalized_report(worker, prepared)
        report = prepared
        assert report.completions[0].status is CallStatus.OK
        assert report.completions[0].num_completed_steps == 3
    finally:
        worker.close()


def test_later_product_release_unblocks_an_earlier_bank_writer() -> None:
    """A received Free must progress while computation waits for its storage."""
    from concurrent.futures import ThreadPoolExecutor

    from tests.python.fixtures.worker_ipc import QueuedWorkerIpc
    from uniserve_worker.protocol.output import BatchOutput

    with execution_worker(transfer_backends=("shm",), queue_depth=2) as worker:
        worker.warmup()
        admission = umm_params(
            76, ImageParams(steps=3, height=16, width=16, seed=29)
        )
        conditioning = _publish_conditioning(
            worker, admission, call_id=CallId(1, 0), batch_id=1
        )
        latent, observation = _prepare_media(
            worker,
            admission,
            conditioning,
            call_id=CallId(2, 0),
            predecessor=root_parent(admission),
            batch_id=2,
        )
        retained = latent
        for call_id in (3, 4):
            call, successor = diffusion_step_call(
                admission.request_key,
                call_id=CallId(call_id, 0),
                predecessor=observation.call_id,
                conditioning=conditioning,
                latent=latent,
                steps=1,
            )
            report = finalized_report(
                worker,
                worker.submit(
                    execution_batch(
                        batch_id=call_id,
                        calls=(call,),
                        commands=()
                        if call_id == 3
                        else (Free(retained.buffer_id),),
                    )
                ),
            )
            assert report.completions[0].status is CallStatus.OK
            retained, latent = latent, successor
            observation = record_completion(call, report)

        third, _final_latent = diffusion_step_call(
            admission.request_key,
            call_id=CallId(5, 0),
            predecessor=observation.call_id,
            conditioning=conditioning,
            latent=latent,
            steps=1,
        )
        waiting = execution_batch(batch_id=5, calls=(third,), commands=())
        release = execution_batch(
            batch_id=6, commands=(Free(retained.buffer_id),)
        )
        # This run reaches the worker through its IPC endpoint rather than
        # the submit path, so it states its own coordinates here.
        endpoint = QueuedWorkerIpc(
            tuple(
                {
                    "kind": "submit",
                    "message_id": run.batch_id,
                    "batch": stamp_batch(worker, run),
                }
                for run in (waiting, release)
            )
        )
        worker.bind(endpoint)
        with ThreadPoolExecutor(max_workers=1) as executor:
            serving = executor.submit(worker.run)
            try:
                responses = {
                    int(value["message_id"]): value
                    for value in (endpoint.receive(), endpoint.receive())
                }
                assert responses[5]["kind"] == "result", responses[5]
                result = BatchOutput.from_mapping(responses[5]["result"])
                assert result.completions[0].status is CallStatus.OK
                assert result.completions[0].num_completed_steps == 3
                assert responses[6]["kind"] == "result", responses[6]
            finally:
                # Failure cleanup sends the same valid release through IPC,
                # allowing the serving thread to leave its storage wait
                # before it is joined.
                endpoint.submit(
                    {
                        "kind": "submit",
                        "message_id": 99,
                        "batch": execution_batch(
                            batch_id=99, commands=(Free(retained.buffer_id),)
                        ),
                    }
                )
                endpoint.submit({"kind": "close", "message_id": 7})
                serving.result(timeout=10)


@pytest.mark.gpu
@pytest.mark.parametrize("warmup", [False, True])
def test_full_binding_returns_current_decode_tokens(warmup):
    policy = WorkerConfig(
        graph_policy="full",
        prefill_cuda_graph=True,
        decode_graph_batch_sizes=(1, 2),
        prefill_graph_token_sizes=(16, 32),
        flow_graph_batch_sizes=(1,),
        flow_graph_shapes=((16, 16),),
        lanes=(LaneConfig("compute", 152, CALL_KINDS),),
    )
    with execution_worker(device="cuda:0", execution=policy) as worker:
        if warmup:
            worker.warmup()
        admission = ar_params(1, block_ids=(0,))
        decode, observation = _prepare_decode(
            worker,
            admission,
            call_id=CallId(1, 0),
            batch_id=1,
            tokens=(3, 4),
        )
        result = finalized_report(
            worker,
            worker.submit(
                execution_batch(
                    batch_id=2,
                    calls=(decode,),
                    commands=(),
                )
            ),
        )
        assert result.completions[0].status is CallStatus.OK
        assert result.completions[0].committed_tokens == (
            expected_successor(expected_successor(4)),
        )


@pytest.mark.gpu
def test_failed_capture_preserves_error_through_worker_scope_and_reconstruction(
    monkeypatch,
):
    from uniserve.runtime.cuda_graph import CUDAGraphError

    policy = WorkerConfig(
        graph_policy="full",
        prefill_cuda_graph=True,
        decode_graph_batch_sizes=(1, 2),
        prefill_graph_token_sizes=(16, 32),
        flow_graph_batch_sizes=(1,),
        flow_graph_shapes=((16, 16),),
        lanes=(LaneConfig("compute", 152, CALL_KINDS),),
    )
    capture_end = torch.cuda.CUDAGraph.capture_end
    failure = RuntimeError("CUDA capture completion failed")

    def failed_capture(graph):
        capture_end(graph)
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda.CUDAGraph, "capture_end", failed_capture)
        with pytest.raises(
            CUDAGraphError, match="CUDA capture completion failed"
        ) as raised:
            with execution_worker(device="cuda:0", execution=policy) as worker:
                worker.warmup()
        assert raised.value.__cause__ is failure
    with execution_worker(device="cuda:0", execution=policy) as worker:
        worker.warmup()
        admission = ar_params(1, block_ids=(0,))
        decode, observation = _prepare_decode(
            worker,
            admission,
            call_id=CallId(1, 0),
            batch_id=1,
            tokens=(3, 4),
        )
        result = finalized_report(
            worker,
            worker.submit(
                execution_batch(
                    batch_id=2,
                    calls=(decode,),
                    commands=(),
                )
            ),
        )
        assert result.completions[0].status is CallStatus.OK
        assert result.completions[0].committed_tokens == (
            expected_successor(expected_successor(4)),
        )
