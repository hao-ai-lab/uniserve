"""Bounded tensor entry execution and loaded result declarations."""

import threading

import pytest
import torch

from tests.python.fixtures.depth_one import finalized_report
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.bootstrap.worker_info import WorkerInfo
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.model_runner import ModelRunner
from uniserve_worker.foundation.errors import ComputeError, InputError
from uniserve_worker.models.runtime import ExecutionModel, ResourceGeometry
from uniserve_worker.models.stub import StubModel
from uniserve_worker.nn.parallel import ComponentConfig
from uniserve_worker.protocol.batch import (
    Bounds,
    BufferAllocation,
    ComputationId,
    DeviceDim,
    DiffusionSamplingParams,
    DType,
    NewRequest,
    OpStatus,
    PipelineStage,
    RequestKey,
    ScheduleBatch,
    ScheduledRequest,
    ShapeBound,
    Start,
    StaticDim,
    TensorPublication,
    TensorRef,
    TensorSpec,
    TransferMode,
)
from uniserve_worker.transfer.layout import fetch_tensor

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("separate_start", (False, True))
def test_text_encoder_operation_publishes_consumable_conditioning(separate_start):
    model = StubModel()
    model.text_encoder = torch.nn.Embedding(32, 4)
    model.text_max_tokens = 16
    model.entry_outputs = {
        "text_encoder": (
            TensorSpec(
                "conditioning", DType.F32, ShapeBound((StaticDim(1), DeviceDim(16), StaticDim(4)))
            ),
        ),
    }
    model.supported_work = model.supported_work | {PipelineStage.TEXT_ENCODING}
    with torch.no_grad():
        model.text_encoder.weight.copy_(torch.arange(128).reshape(32, 4))
    worker = execution_worker(
        model,
        components=(("text_encoder", ComponentConfig((0,))), ("output", ComponentConfig((0,)))),
    )
    worker.runner.bind_module("text_encoder", model.text_encoder)
    key = RequestKey(1, 1, 1)
    prompt = (3, 8, 1)
    reference = TensorRef(
        key,
        ComputationId(1, 0),
        0,
        1,
        DType.F32,
        ShapeBound((StaticDim(1), StaticDim(3), StaticDim(4))),
    )
    operation = ScheduledRequest(
        request_key=key,
        op_id=ComputationId(1, 0),
        predecessor=None,
        kind=PipelineStage.TEXT_ENCODING,
        entry="text_encoder",
        bounds=Bounds(),
        outputs=(reference,),
    )
    run = ScheduleBatch(
        batch_id=1,
        run_id=1,
        operations=(operation,),
        commands=(
            Start(
                NewRequest(
                    key,
                    request_pool_idx=1,
                    diffusion=DiffusionSamplingParams(22, 3, 4, 1000),
                    prompt_token_ids=prompt,
                )
            ),
        ),
        buffer_allocations=(BufferAllocation(reference.buffer_id, 0, reference.max_bytes),),
    )
    try:
        if separate_start:
            from dataclasses import replace

            started = finalized_report(
                worker, worker.submit(ScheduleBatch(batch_id=2, run_id=2, commands=run.commands))
            )
            assert started.done and not started.completions
            run = replace(run, commands=())
        report = finalized_report(worker, worker.submit(run))
        (completion,) = report.completions
        assert completion.status is OpStatus.OK
        assert completion.op_id == operation.op_id
        (product,) = report.products
        assert product.product == reference

        tensor = product.value.tensor
        destination = torch.empty(1, 3, 4)
        tickets = fetch_tensor(
            tensor,
            destination,
            bindings={
                (location.source, location.backend): worker.transports[location.backend]
                for location in tensor.locations
            },
        )
        for ticket in tickets:
            ready = threading.Event()
            ticket.add_done_callback(ready.set)
            assert ready.wait(5)
            ticket.result()
            ticket.close()
        expected = torch.tensor(prompt).reshape(1, 3, 1) * 4 + torch.arange(4)
        torch.testing.assert_close(destination, expected.float(), atol=0, rtol=0)
        from dataclasses import replace

        copied = replace(reference, producer_op_id=ComputationId(2, 0))
        consumer = ScheduledRequest(
            request_key=key,
            op_id=ComputationId(2, 0),
            predecessor=None,
            kind=TransferMode.TENSOR,
            entry="output",
            bounds=Bounds(max_transfer_bytes=reference.max_bytes),
            inputs=(reference,),
            outputs=(copied,),
        )
        prepared = worker.submit(
            ScheduleBatch(
                batch_id=2,
                run_id=3,
                collective_seq=3,
                operations=(consumer,),
                input_products=(TensorPublication(reference, product.value),),
                buffer_allocations=(
                    BufferAllocation(reference.buffer_id, 0, reference.max_bytes),
                    BufferAllocation(copied.buffer_id, 256, copied.max_bytes),
                ),
            )
        )
        ready = threading.Event()
        prepared.on_dependencies_ready(ready.set)
        assert ready.wait(5)
        prepared = finalized_report(worker, prepared)
        result = prepared
        assert result.completions[0].status is OpStatus.OK
        copied_value = result.products[0].value.tensor
        tickets = fetch_tensor(
            copied_value,
            destination,
            bindings={
                (location.source, location.backend): worker.transports[location.backend]
                for location in copied_value.locations
            },
        )
        for ticket in tickets:
            ready = threading.Event()
            ticket.add_done_callback(ready.set)
            assert ready.wait(5)
            ticket.result()
            ticket.close()
        torch.testing.assert_close(destination, expected.float(), atol=0, rtol=0)
    finally:
        worker.close()


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_text_entry_stages_successive_bounded_inputs(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    model = ExecutionModel()
    model.resource_geometry = ResourceGeometry(kv=False)
    model.text_encoder = torch.nn.Embedding(32, 4, device=device)
    model.text_max_tokens = 16
    model.entry_outputs = {
        "text_encoder": (
            TensorSpec(
                "conditioning", DType.F32, ShapeBound((StaticDim(1), DeviceDim(16), StaticDim(4)))
            ),
        ),
    }
    with torch.no_grad():
        model.text_encoder.weight.copy_(torch.arange(128, device=device).reshape(32, 4))
    runner = ModelRunner(model, WorkerConfig(device=device))
    runner.bind_module("text_encoder", model.text_encoder)
    try:
        outputs = []
        prompts = ((3, 8, 1), (31,), (0, 5, 19, 7), (1, 2))
        for prompt in prompts:
            result = runner.run_entry("text_encoder", runner.stage_text_tokens(prompt))
            outputs.append(result.values[0])
            assert result.stats is not None
            assert result.stats.mode_counts == {"text_encoder": 1}
            assert result.stats.mode_tokens == {"text_encoder": 1}
        for prompt, output in zip(prompts, outputs, strict=True):
            expected = torch.tensor(prompt).reshape(1, -1, 1) * 4 + torch.arange(4)
            torch.testing.assert_close(output.cpu(), expected.float(), atol=0, rtol=0)
        for prompt in ((), (1,) * 17):
            with pytest.raises(InputError, match="capacity"):
                runner.stage_text_tokens(prompt)
    finally:
        runner.synchronize()
        runner.close()


@pytest.mark.parametrize(
    "shape,dtype,message",
    [((2, 4), torch.float64, "dtype"), ((3, 4), torch.float32, "shape")],
)
def test_entry_rejects_outputs_outside_the_loaded_contract(shape, dtype, message):
    model = ExecutionModel()
    model.resource_geometry = ResourceGeometry(kv=False)
    model.projection = torch.nn.Identity()
    model.entry_outputs = {
        "projection": (TensorSpec("values", DType.F32, ShapeBound((DeviceDim(2), StaticDim(4)))),),
    }
    runner = ModelRunner(model, WorkerConfig())
    runner.bind_module("projection", model.projection)
    try:
        with pytest.raises(ComputeError, match=message):
            runner.run_entry("projection", torch.zeros(shape, dtype=dtype))
        result = runner.run_entry("projection", torch.ones((2, 4)))
        torch.testing.assert_close(result.values[0], torch.ones((2, 4)), atol=0, rtol=0)
    finally:
        runner.close()


def test_worker_reports_entry_result_bounds_with_its_static_membership():
    model = StubModel()
    model.projection = torch.nn.Identity()
    model.entry_outputs = {
        "projection": (
            TensorSpec("features", DType.BF16, ShapeBound((DeviceDim(128), StaticDim(512)))),
        ),
    }
    worker = execution_worker(model, components=(("projection", ComponentConfig((0,))),))
    try:
        info = worker.info.to_mapping()
        (entry,) = WorkerInfo.from_mapping(info).components
        assert entry.name == "projection"
        assert entry.config.ranks == (0,)
        assert entry.outputs == model.entry_outputs["projection"]
    finally:
        worker.close()


def test_worker_binds_dense_attention_without_requesting_kv_storage():
    from uniserve_worker.nn.attention import RadixAttention
    from uniserve_worker.worker import Worker

    class DenseEntry(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = RadixAttention(2, 2, 8)

        def forward(self, value):
            return self.attention(value, value, value, None, causal=False)

    model = ExecutionModel()
    model.architecture = "DenseAttention"
    model.resource_geometry = ResourceGeometry(kv=False)
    model.supported_work = frozenset({PipelineStage.VIDEO_DECODING})
    model.decoder = DenseEntry()
    model.entry_outputs = {
        "decoder": (
            TensorSpec(
                "values",
                DType.F32,
                ShapeBound((StaticDim(1), StaticDim(2), DeviceDim(16), StaticDim(8))),
            ),
        ),
    }
    worker = Worker(
        model,
        sampling_group=None,
        worker_config=WorkerConfig(
            device="cpu",
            graph_policy="off",
            max_batch_operations=2,
            max_batch_tokens=2,
            max_request_pool_size=2,
        ),
        attention=None,
        tokenizer=None,
        allowed_work_variants=model.supported_work,
        transfer_backends=("local",),
        publication_backends=("local",),
        worker_id="decoder",
        pipeline_depth=3,
        completion_payload_bytes=1 << 16,
        components=(("decoder", ComponentConfig((0,))),),
    )
    try:
        info = WorkerInfo.from_mapping(worker.info.to_mapping())
        assert info.uses_kv is False
        assert info.kv_cache is None
        values = torch.arange(64, dtype=torch.float32).reshape(1, 2, 4, 8) / 64
        expected = torch.nn.functional.scaled_dot_product_attention(values, values, values)
        torch.testing.assert_close(model.decoder(values), expected)
    finally:
        worker.close()
