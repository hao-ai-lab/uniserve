"""Bounded tensor entry execution and loaded result declarations."""

import threading

import pytest
import torch

from tests.python.fixtures.depth_one import finalized_report
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.bootstrap.worker_info import WorkerInfo
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.model_entry import ModelEntry
from uniserve_worker.execution.model_runner import ModelRunner
from uniserve_worker.foundation.errors import ComputeError, InputError
from uniserve_worker.modeling.batch import TensorOutput
from uniserve_worker.modeling.components import Call, CallSpec, ComponentSpec
from uniserve_worker.modeling.encoder import EncodeKind, EncoderMixin
from uniserve_worker.modeling.geometry import MediaShape, TensorOutputLayout, TextShape
from uniserve_worker.modeling.model import Model
from uniserve_worker.modeling.resources import TensorNeeds, TensorSchema
from uniserve_worker.models.stub import StubModel
from uniserve_worker.nn.mesh import Communicator, DeviceMesh
from uniserve_worker.nn.parallel import ComponentConfig, ParallelConfig
from uniserve_worker.protocol.batch import (
    Bounds,
    BufferAllocation,
    ComputationId,
    DecodeRange,
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
from uniserve_worker.runtime.results import resolve_outputs
from uniserve_worker.transfer.layout import fetch_tensor

pytestmark = pytest.mark.integration


class VideoSegments(Model):
    """A numerical segment result with a complete logical unit dimension."""

    def output_layout(self, entry, output_index, *, frames, units, prompt_tokens):
        return TensorOutputLayout((units, 1, 3, 25, 8, 12))


@pytest.mark.parametrize("rank", [0, 1, 3])
@pytest.mark.parametrize("units", [1, 2])
def test_temporal_output_regions_follow_declared_rank_order(rank, units):
    config = ComponentConfig((3, 1), distribution="temporal_units")
    group = Communicator((0, 1, 2, 3), rank)
    binding = ModelEntry(
        "video_decoder",
        config,
        group,
        DeviceMesh((rank,), rank, ParallelConfig()) if rank in config.ranks else None,
        group.device,
    )
    runner = ModelRunner(
        VideoSegments(),
        WorkerConfig(rank=rank, world_size=4),
        bindings={"video_decoder": binding},
    )
    try:
        interval = DecodeRange(RequestKey(1, 0, 0), ComputationId(1, 0), cursor=2, max_units=units)
        result = runner.output_layout("video_decoder", 0, None, interval, 1)
        if rank == 0 or (rank == 1 and units == 1):
            assert result is None
        else:
            assert result.shape == (units, 1, 3, 25, 8, 12)
            assert result.region.offset == ((0 if rank == 3 else 1), 0, 0, 0, 0, 0)
            assert result.region.shape == (1, 1, 3, 25, 8, 12)
    finally:
        runner.close()


class EncodedModel(StubModel):
    """Numerical embedding composition used by public encoder execution tests."""

    encoder_kinds = StubModel.encoder_kinds | frozenset({"text"})

    def __init__(self, device="cpu"):
        super().__init__()
        self.text_encoder = torch.nn.Embedding(32, 4, device=device)
        self.text_max_tokens = 16
        self.output_shapes = {
            "text_encoder": (Call.ENCODE_TEXT, TextShape(16)),
        }
        with torch.no_grad():
            self.text_encoder.weight.copy_(torch.arange(128, device=device).reshape(32, 4))

    def tensor_specs(self, call, shape):
        if call is Call.ENCODE_TEXT:
            return TensorNeeds(
                outputs={
                    "conditioning": TensorSchema(
                        (1, shape.tokens, 4), torch.float32, variable_axes=(1,)
                    )
                }
            )
        return super().tensor_specs(call, shape)

    @classmethod
    def components(cls, config):
        return (
            *super().components(config),
            ComponentSpec("text_encoder", (CallSpec(Call.ENCODE_TEXT),)),
        )

    def encode(self, kind: EncodeKind, batch, *, constants, scratch):
        if kind == "text":
            return TensorOutput(
                {"conditioning": tuple(self.text_encoder(value) for value in batch.values)}
            )
        return super().encode(kind, batch, constants=constants, scratch=scratch)


def _encoder_bindings(model, components, device="cpu"):
    specs = {component.name: component for component in model.components(model.config)}
    group = Communicator(device=torch.device(device))
    return {
        name: ModelEntry(
            name,
            config,
            group,
            DeviceMesh(config.ranks, 0, config.parallel_config, group.device),
            group.device,
            calls=specs[name].calls if name in specs else (),
            output_schema=resolve_outputs(model).get(name, ()),
        )
        for name, config in components
    }


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_decoder_call_preserves_values_across_independent_execution_owners(device):
    from tests.python.fixtures.decoding import DecodedModel
    from uniserve_worker.modeling.batch import DecodeBatch
    from uniserve_worker.modeling.geometry import MediaShape

    model = DecodedModel().to(device)
    components = (("reconstruction", ComponentConfig((0,))),)
    runners = [
        ModelRunner(
            model,
            WorkerConfig(device=device),
            bindings=_encoder_bindings(model, components, device),
        )
        for _ in range(2)
    ]
    try:
        for runner in runners:
            runner.prepare_fixed_modules()
        source = torch.arange(12, dtype=torch.float32, device=device).reshape(4, 3) / 10
        shape = MediaShape(1, 1, frames=4)

        def expected(value):
            normalized = value.T.unsqueeze(0) * torch.tensor([0.5, 1.5, 2.5], device=device).view(
                1, 3, 1
            )
            normalized += torch.tensor([0.1, 0.2, 0.3], device=device).view(1, 3, 1)
            normalized *= torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
            return normalized.unsqueeze(-1).unsqueeze(-1)

        reference = expected(source)
        first = runners[0].run_decoder(
            "video", DecodeBatch((source,), (shape,)), constants={}, scratch={}
        )
        torch.testing.assert_close(first.values[0], reference, rtol=0, atol=0)
        source.add_(0.25)
        second = runners[1].run_decoder(
            "video", DecodeBatch((source,), (shape,)), constants={}, scratch={}
        )
        torch.testing.assert_close(second.values[0], expected(source), rtol=0, atol=0)
        torch.testing.assert_close(first.values[0], reference, rtol=0, atol=0)
        source.add_(0.5)
        replay = runners[0].run_decoder(
            "video", DecodeBatch((source,), (shape,)), constants={}, scratch={}
        )
        torch.testing.assert_close(replay.values[0], expected(source), rtol=0, atol=0)
        with pytest.raises(InputError, match="does not participate"):
            runners[0].run_decoder(
                "audio", DecodeBatch((source,), (shape,)), constants={}, scratch={}
            )
    finally:
        for runner in runners:
            runner.close()


class Conditioner(EncoderMixin, Model):
    """A conditioning projection required only at a pipeline's input stage."""

    encoder_kinds = frozenset({"conditioning"})

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(4)[:2])

    @classmethod
    def components(cls, config):
        return (ComponentSpec("denoiser", (CallSpec(Call.ENCODE_CONDITIONING, stage="first"),)),)

    def encode(self, kind, batch, *, constants, scratch):
        if kind != "conditioning":
            raise ValueError("conditioner requires encoded features")
        return TensorOutput(
            {"conditioning": tuple(self.projection(value) for value in batch.values)}
        )


@pytest.mark.parametrize("rank", [0, 1])
def test_conditioning_executes_only_on_its_declared_pipeline_stage(rank):
    model = Conditioner()
    config = ComponentConfig((0, 1), ParallelConfig(pipeline_parallel_size=2))
    group = Communicator((0, 1), rank)
    binding = ModelEntry(
        "denoiser",
        config,
        group,
        DeviceMesh(config.ranks, rank, config.parallel_config),
        group.device,
        calls=model.components(model.config)[0].calls,
    )
    runner = ModelRunner(
        model, WorkerConfig(rank=rank, world_size=2), bindings={"denoiser": binding}
    )
    try:
        features = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        if rank == 0:
            output = runner.run_encoder("conditioning", features)
            torch.testing.assert_close(output.values[0], torch.tensor([[1.0, 2.0]]), rtol=0, atol=0)
        else:
            with pytest.raises(InputError, match="does not participate"):
                runner.run_encoder("conditioning", features)
    finally:
        runner.close()


@pytest.mark.parametrize("separate_start", (False, True))
def test_text_encoder_operation_publishes_consumable_conditioning(separate_start):
    model = EncodedModel()
    components = tuple(
        (name, ComponentConfig((0,))) for name in ("model", "text_encoder", "output")
    )
    worker = execution_worker(
        model,
        components=components,
        bindings=_encoder_bindings(model, components),
    )
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
    model = EncodedModel(device)
    runner = ModelRunner(
        model,
        WorkerConfig(device=device),
        bindings=_encoder_bindings(model, (("text_encoder", ComponentConfig((0,))),), device),
    )
    try:
        outputs = []
        prompts = ((3, 8, 1), (31,), (0, 5, 19, 7), (1, 2))
        for prompt in prompts:
            result = runner.run_encoder("text", runner.stage_text_tokens(prompt))
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
    model = Model()
    model.projection = torch.nn.Identity()
    runner = ModelRunner(model, WorkerConfig())
    runner.bind_module(
        "projection",
        model.projection,
        outputs=(TensorSpec("values", DType.F32, ShapeBound((DeviceDim(2), StaticDim(4)))),),
    )
    try:
        with pytest.raises(ComputeError, match=message):
            runner.run_entry("projection", torch.zeros(shape, dtype=dtype))
        result = runner.run_entry("projection", torch.ones((2, 4)))
        torch.testing.assert_close(result.values[0], torch.ones((2, 4)), atol=0, rtol=0)
    finally:
        runner.close()


def test_worker_reports_entry_result_bounds_with_its_static_membership():
    class Features(StubModel):
        output_shapes = {"projection": (Call.ENCODE_VISION, MediaShape(1, 1))}

        def tensor_specs(self, call, shape):
            return TensorNeeds(
                outputs={"features": TensorSchema((128, 512), torch.bfloat16, variable_axes=(0,))}
            )

    model = Features()
    model.projection = torch.nn.Identity()
    worker = execution_worker(model, components=(("projection", ComponentConfig((0,))),))
    try:
        info = worker.info.to_mapping()
        (entry,) = WorkerInfo.from_mapping(info).components
        assert entry.name == "projection"
        assert entry.config.ranks == (0,)
        assert entry.outputs == (
            TensorSpec("features", DType.BF16, ShapeBound((DeviceDim(128), StaticDim(512)))),
        )
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

    class DenseModel(Model):
        output_shapes = {"decoder": (Call.DECODE_VIDEO, MediaShape(1, 1))}

        def tensor_specs(self, call, shape):
            return TensorNeeds(
                outputs={"values": TensorSchema((1, 2, 16, 8), torch.float32, variable_axes=(2,))}
            )

        @classmethod
        def components(cls, config):
            return (ComponentSpec("decoder", (CallSpec(Call.DECODE_VIDEO),)),)

    model = DenseModel()
    model.architecture = "DenseAttention"
    model.decoder = DenseEntry()
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
        allowed_work_variants=frozenset({PipelineStage.VIDEO_DECODING}),
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
