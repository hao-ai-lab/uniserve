"""Public component execution, publication and component participation."""

import threading
from types import SimpleNamespace

import pytest
import torch

from tests.python.fixtures.decoding import Config as DecoderConfig
from tests.python.fixtures.decoding import DecodedModel
from tests.python.fixtures.depth_one import finalized_report
from tests.python.fixtures.encoding import Model as EncodedModel
from tests.python.fixtures.execution_worker import execution_worker
from uniserve.distributed import Communicator, DeviceMesh
from uniserve_worker.bootstrap.config import ComponentConfig, ParallelConfig
from uniserve_worker.bootstrap.worker_info import WorkerInfo
from uniserve_worker.config import LaneConfig, WorkerConfig
from uniserve_worker.execution.component_binding import ComponentBinding
from uniserve_worker.execution.model_runner import ModelRunner
from uniserve_worker.foundation.errors import InputError
from uniserve_worker.protocol.batch import (
    Batch,
    BufferAllocation,
    DecodeRange,
    DiffusionParams,
    NewRequest,
    Start,
    TensorPublication,
)
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    CallStatus,
    MediaCall,
    TransferMode,
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.protocol.tensor import (
    DeviceDim,
    DType,
    OutputInfo,
    ShapeBound,
    StaticDim,
    TensorRef,
)
from uniserve_worker.transfer.layout import fetch_tensor

pytestmark = pytest.mark.integration


def _encoder_bindings(model, components, device="cpu"):
    group = Communicator(device=torch.device(device))
    return {
        name: ComponentBinding(
            name,
            config,
            group,
            DeviceMesh(
                ranks=config.ranks,
                rank=0,
                shape=tuple(
                    size for _, size in config.parallel_config.dimensions
                ),
                axes=tuple(
                    axis for axis, _ in config.parallel_config.dimensions
                ),
            ),
            group.device,
        )
        for name, config in components
    }


@pytest.mark.parametrize("rank", [0, 1, 3])
@pytest.mark.parametrize("units", [1, 2])
def test_temporal_output_regions_follow_declared_rank_order(rank, units):
    config = ComponentConfig((3, 1), distribution="temporal_units")
    group = Communicator((0, 1, 2, 3), rank)
    dimensions = ParallelConfig().dimensions
    binding = ComponentBinding(
        "reconstruction",
        config,
        group,
        DeviceMesh(
            ranks=(rank,),
            rank=rank,
            shape=tuple(size for _, size in dimensions),
            axes=tuple(axis for axis, _ in dimensions),
        )
        if rank in config.ranks
        else None,
        group.device,
    )
    runner = ModelRunner(
        DecodedModel(DecoderConfig(window=25, height=8, width=12)),
        WorkerConfig(rank=rank, world_size=4),
        bindings={"reconstruction": binding},
    )
    try:
        interval = DecodeRange(
            RequestKey(1, 0, 0), CallId(1, 0), cursor=2, max_units=units
        )
        result = runner.output_layout(
            "reconstruction", 0, SimpleNamespace(num_frames=50), interval, 1
        )
        if rank == 0 or (rank == 1 and units == 1):
            assert result is None
        else:
            assert result.shape == (units, 1, 3, 25, 8, 12)
            assert result.local_slice == (
                slice(0, 1) if rank == 3 else slice(1, 2),
                slice(0, 1),
                slice(0, 3),
                slice(0, 25),
                slice(0, 8),
                slice(0, 12),
            )
    finally:
        runner.close()


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_decoder_call_preserves_values_across_independent_execution_owners(
    device,
):
    model = DecodedModel().to(device)
    components = (
        (
            "reconstruction",
            ComponentConfig((0,), distribution="temporal_units"),
        ),
    )
    runners = [
        ModelRunner(
            model,
            WorkerConfig(device=device),
            bindings=_encoder_bindings(model, components, device),
        )
        for _ in range(2)
    ]
    try:
        source = (
            torch.arange(12, dtype=torch.float32, device=device).reshape(4, 3)
            / 10
        )

        def expected(value):
            normalized = value.T.unsqueeze(0) * torch.tensor(
                [0.5, 1.5, 2.5], device=device
            ).view(1, 3, 1)
            normalized += torch.tensor([0.1, 0.2, 0.3], device=device).view(
                1, 3, 1
            )
            normalized *= torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
            return normalized.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        def decode(runner):
            return runner.run_module(
                "reconstruction",
                (source,),
                method="decode",
                size=4,
                frames=(slice(0, 4),),
                num_frames=(4,),
            ).values[0]

        reference = expected(source)
        first = decode(runners[0])
        torch.testing.assert_close(first, reference, rtol=0, atol=0)
        source.add_(0.25)
        second = decode(runners[1])
        torch.testing.assert_close(second, expected(source), rtol=0, atol=0)
        torch.testing.assert_close(first, reference, rtol=0, atol=0)
        source.add_(0.5)
        torch.testing.assert_close(
            decode(runners[0]), expected(source), rtol=0, atol=0
        )
        with pytest.raises(InputError, match="unambiguous"):
            runners[0].run_module("audio", (source,), method="decode", size=4)
    finally:
        for runner in runners:
            runner.close()


@pytest.mark.parametrize("rank", [0, 1])
def test_conditioning_executes_only_on_its_declared_pipeline_stage(rank):
    model = EncodedModel()
    config = ComponentConfig((0, 1), ParallelConfig(pipeline_parallel_size=2))
    group = Communicator((0, 1), rank)
    binding = ComponentBinding(
        "conditioner",
        config,
        group,
        DeviceMesh(
            ranks=config.ranks,
            rank=rank,
            shape=tuple(size for _, size in config.parallel_config.dimensions),
            axes=tuple(axis for axis, _ in config.parallel_config.dimensions),
        ),
        group.device,
    )
    runner = ModelRunner(
        model,
        WorkerConfig(rank=rank, world_size=2),
        bindings={"conditioner": binding},
    )
    try:
        features = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        if rank == 0:
            output = runner.run_encoder("conditioning", features)
            torch.testing.assert_close(
                output.values[0], torch.tensor([[1.0, 2.0]]), rtol=0, atol=0
            )
        else:
            with pytest.raises(InputError, match="does not participate"):
                runner.run_encoder("conditioning", features)
    finally:
        runner.close()


@pytest.mark.parametrize("separate_start", (False, True))
@pytest.mark.parametrize(
    "execution_device",
    (
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("green", marks=pytest.mark.gpu),
    ),
)
def test_text_encoder_call_publishes_consumable_conditioning(
    separate_start, execution_device
):
    device = "cpu" if execution_device == "cpu" else "cuda:0"
    model = EncodedModel().to(device)
    execution = WorkerConfig(
        graph_policy="off",
        lanes=(
            LaneConfig(
                "text", 64, (MediaCall.TEXT_ENCODING, TransferMode.TENSOR)
            ),
        )
        if execution_device == "green"
        else (),
    )
    components = tuple(
        (name, ComponentConfig((0,))) for name in ("text_encoder",)
    )
    worker = execution_worker(
        model,
        device=device,
        execution=execution,
        components=components,
        bindings=_encoder_bindings(model, components, device),
    )
    key = RequestKey(1, 1, 1)
    prompt = (3, 8, 1)
    reference = TensorRef(
        key,
        CallId(1, 0),
        0,
        1,
        DType.F32,
        ShapeBound((StaticDim(3), StaticDim(4))),
    )
    call = Call(
        request_key=key,
        call_id=CallId(1, 0),
        coordinates=CallCoordinates(),
        kind=MediaCall.TEXT_ENCODING,
        component="text_encoder",
        bounds=Bounds(),
        outputs=(reference,),
    )
    run = Batch(
        batch_id=1,
        calls=(call,),
        commands=(
            Start(
                NewRequest(
                    key,
                    request_pool_idx=1,
                    diffusion=DiffusionParams(22, 3, 4, 1000),
                    prompt_token_ids=prompt,
                )
            ),
        ),
        buffer_allocations=(
            BufferAllocation(reference.buffer_id, 0, reference.max_bytes),
        ),
    )
    try:
        if separate_start:
            from dataclasses import replace

            started = finalized_report(
                worker,
                worker.submit(Batch(batch_id=0, commands=run.commands)),
            )
            assert not started.completions
            run = replace(run, commands=())
        report = finalized_report(worker, worker.submit(run))
        (completion,) = report.completions
        assert completion.status is CallStatus.OK
        assert completion.call_id == call.call_id
        (product,) = report.products
        assert product.product == reference

        tensor = product.value.tensor
        destination = torch.empty(3, 4, device=device)
        tickets = fetch_tensor(
            tensor,
            destination,
            bindings={
                (location.source, location.backend): worker.transports[
                    location.backend
                ]
                for location in tensor.locations
            },
        )
        for ticket in tickets:
            ready = threading.Event()
            ticket.add_done_callback(ready.set)
            assert ready.wait(5)
            ticket.result()
            ticket.close()
        expected = torch.tensor(prompt).reshape(3, 1) * 4 + torch.arange(4)
        torch.testing.assert_close(
            destination.cpu(), expected.float(), atol=0, rtol=0
        )
        from dataclasses import replace

        copied = replace(reference, producer_call_id=CallId(2, 0))
        consumer = Call(
            request_key=key,
            call_id=CallId(2, 0),
            coordinates=CallCoordinates(),
            kind=TransferMode.TENSOR,
            component="text_encoder",
            bounds=Bounds(max_transfer_bytes=reference.max_bytes),
            inputs=(reference,),
            outputs=(copied,),
        )
        prepared = worker.submit(
            Batch(
                batch_id=2,
                collective_seq=3,
                calls=(consumer,),
                input_products=(TensorPublication(reference, product.value),),
                buffer_allocations=(
                    BufferAllocation(
                        reference.buffer_id, 0, reference.max_bytes
                    ),
                    BufferAllocation(copied.buffer_id, 256, copied.max_bytes),
                ),
            )
        )
        ready = threading.Event()
        prepared.on_dependencies_ready(ready.set)
        assert ready.wait(5)
        prepared = finalized_report(worker, prepared)
        result = prepared
        assert result.completions[0].status is CallStatus.OK
        copied_value = result.products[0].value.tensor
        tickets = fetch_tensor(
            copied_value,
            destination,
            bindings={
                (location.source, location.backend): worker.transports[
                    location.backend
                ]
                for location in copied_value.locations
            },
        )
        for ticket in tickets:
            ready = threading.Event()
            ticket.add_done_callback(ready.set)
            assert ready.wait(5)
            ticket.result()
            ticket.close()
        torch.testing.assert_close(
            destination.cpu(), expected.float(), atol=0, rtol=0
        )
    finally:
        worker.close()


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_text_entry_stages_successive_bounded_inputs(device):
    model = EncodedModel().to(device)
    runner = ModelRunner(
        model,
        WorkerConfig(device=device, max_sequence_tokens=16),
        bindings=_encoder_bindings(
            model, (("text_encoder", ComponentConfig((0,))),), device
        ),
    )
    try:
        outputs = []
        prompts = ((3, 8, 1), (31,), (0, 5, 19, 7), (1, 2))
        for prompt in prompts:
            result = runner.run_encoder(
                "text", runner.stage_text_tokens(prompt)
            )
            outputs.append(result.values[0])
        for prompt, output in zip(prompts, outputs, strict=True):
            expected = torch.tensor(prompt).reshape(-1, 1) * 4 + torch.arange(4)
            torch.testing.assert_close(
                output.cpu(), expected.float(), atol=0, rtol=0
            )
        for prompt in ((), (1,) * 17):
            with pytest.raises(InputError, match="capacity"):
                runner.stage_text_tokens(prompt)
    finally:
        runner.synchronize()
        runner.close()


def test_worker_reports_text_bounds_and_executes_dense_attention_without_kv():
    model = EncodedModel()
    components = tuple(
        (name, ComponentConfig((0,))) for name in ("text_encoder", "dense")
    )
    with execution_worker(
        model,
        execution=WorkerConfig(
            max_sequence_tokens=16, graph_policy="off", model_dtype="float32"
        ),
        components=components,
        bindings=_encoder_bindings(model, components),
    ) as worker:
        info = WorkerInfo.from_mapping(worker.info.to_mapping())
        assert info.kv_cache is None
        component = next(
            value for value in info.components if value.name == "text_encoder"
        )
        assert component.config.ranks == (0,)
        assert component.outputs == (
            OutputInfo(
                "conditioning",
                DType.F32,
                ShapeBound((DeviceDim(16), StaticDim(4))),
            ),
        )
        values = torch.arange(64, dtype=torch.float32).reshape(4, 2, 8) / 64
        heads = values.transpose(0, 1)
        expected = torch.nn.functional.scaled_dot_product_attention(
            heads, heads, heads
        ).transpose(0, 1)
        result = worker.runner.run_encoder("conditioning", values)
        torch.testing.assert_close(result.values[0], expected)


@pytest.mark.parametrize("rows,dtype", [(2, DType.F32), (3, DType.F16)])
def test_text_encoder_rejects_incompatible_output_declaration(rows, dtype):
    model = EncodedModel()
    components = (("text_encoder", ComponentConfig((0,))),)
    with execution_worker(
        model,
        components=components,
        bindings=_encoder_bindings(model, components),
    ) as worker:
        key, op = RequestKey(1, 1, 1), CallId(1, 0)
        output = TensorRef(
            key, op, 0, 1, dtype, ShapeBound((StaticDim(rows), StaticDim(4)))
        )
        call = Call(
            request_key=key,
            call_id=op,
            coordinates=CallCoordinates(),
            kind=MediaCall.TEXT_ENCODING,
            component="text_encoder",
            bounds=Bounds(),
            outputs=(output,),
        )
        admission = NewRequest(
            key,
            request_pool_idx=1,
            diffusion=DiffusionParams(22, 3, 4, 1000),
            prompt_token_ids=(3, 8, 1),
        )
        run = Batch(
            batch_id=1,
            calls=(call,),
            commands=(Start(admission),),
            buffer_allocations=(
                BufferAllocation(output.buffer_id, 0, output.max_bytes),
            ),
        )
        result = finalized_report(worker, worker.submit(run))
        assert result.completions[0].status is CallStatus.ERROR
        assert not result.products
