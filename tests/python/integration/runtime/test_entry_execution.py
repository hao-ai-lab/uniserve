"""Public component execution, export and component participation."""

import threading
from types import SimpleNamespace

import pytest
import torch

from tests.python.fixtures.audio_decoding import CONFIG as AUDIO_CONFIG
from tests.python.fixtures.audio_decoding import AudioModel
from tests.python.fixtures.audio_decoding import (
    entry_points as audio_entry_points,
)
from tests.python.fixtures.decoding import Config as DecoderConfig
from tests.python.fixtures.decoding import DecodedModel
from tests.python.fixtures.depth_one import (
    ar_params,
    execution_batch,
    finalized_report,
)
from tests.python.fixtures.encoding import Model as EncodedModel
from tests.python.fixtures.execution_worker import execution_worker
from tests.python.fixtures.transport import make_transport
from uniserve.distributed import Communicator, DeviceMesh
from uniserve.media import image, video
from uniserve.runtime import EventPool
from uniserve_worker.config.deployment import ComponentConfig, ParallelConfig
from uniserve_worker.config.execution import LaneConfig, WorkerConfig
from uniserve_worker.errors import InputError
from uniserve_worker.execution import media
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.model_executor.component_binding import ComponentBinding
from uniserve_worker.protocol.batch import (
    Batch,
    BufferAllocation,
    DecodeRange,
    DiffusionParams,
    NewRequest,
    Start,
    TensorExport,
)
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    CallStatus,
    ErrorCode,
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
from uniserve_worker.protocol.transfer import (
    DeviceProductTransferValue,
    TensorTransfer,
    WorkerEndpoint,
)
from uniserve_worker.protocol.video import VideoAdmission, VideoTask
from uniserve_worker.protocol.worker_info import WorkerInfo
from uniserve_worker.transport.fetch import fetch_tensor

pytestmark = pytest.mark.integration

# A four-frame single-pixel clip: one window of the fixture decoder.
PIXEL = video.Config(4, image.Config(1, 1))


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
    runner = ModelExecutor(
        DecodedModel(DecoderConfig(window=25, height=8, width=12)),
        WorkerConfig(rank=rank, world_size=4),
        bindings={"reconstruction": binding},
    )
    try:
        interval = DecodeRange(
            RequestKey(1, 0, 0), CallId(1, 0), cursor=2, max_units=units
        )
        result = runner.output_layout(
            "reconstruction",
            0,
            SimpleNamespace(num_frames=50, canvas=image.Config(8, 12)),
            interval,
            1,
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
    ("ranks", "units_per_rank"),
    [((0,), 2), ((0, 1), 1), ((0, 1), 2), ((1, 0), 2), ((0, 1, 2), 3)],
)
def test_audio_decoding_ranks_reconstruct_contiguous_unit_runs(
    ranks, units_per_rank
):
    model = AudioModel()
    decoder = model.audio_decoder
    units = len(ranks) * units_per_rank
    frames = 4 * units
    # A trimmed final latent frame keeps the last unit's crop in play.
    num_samples = frames * decoder.latent_rate - 3
    latent = torch.randn(2 * frames, AUDIO_CONFIG.latent_channels)
    whole = decoder.decode(
        (latent,),
        frames=(slice(0, frames),),
        num_samples=(num_samples,),
        workspace={
            "audio_latents": torch.zeros(
                2, AUDIO_CONFIG.latent_channels, frames
            )
        },
    )[0]
    spans = decoder.unit_samples(num_samples, units)

    config = ComponentConfig(
        ranks, distribution="temporal_units", units_per_rank=units_per_rank
    )
    dimensions = ParallelConfig().dimensions
    decoded = {}
    for rank in ranks:
        group = Communicator(tuple(sorted(ranks)), rank)
        binding = ComponentBinding(
            "audio_decoder",
            config,
            group,
            DeviceMesh(
                ranks=(rank,),
                rank=rank,
                shape=tuple(size for _, size in dimensions),
                axes=tuple(axis for axis, _ in dimensions),
            ),
            group.device,
        )
        runner = ModelExecutor(
            model,
            WorkerConfig(rank=rank, world_size=len(ranks)),
            bindings={"audio_decoder": binding},
            entry_points=audio_entry_points(AUDIO_CONFIG),
        )
        try:
            decoded[rank] = media.decode_audio(
                runner,
                "audio_decoder",
                latent,
                num_samples,
                cursor=0,
                count=units,
            ).values[0]
        finally:
            runner.close()

        # Each rank reconstructs the contiguous run of ``units_per_rank``
        # media units its position in ``ranks`` names, which is the sample
        # region the rank reserves for the track.
        first = ranks.index(rank) * units_per_rank
        run = slice(spans[first].start, spans[first + units_per_rank - 1].stop)
        assert torch.equal(decoded[rank], whole[run])

    # Together the ranks, in their declared order, reproduce the whole track.
    assert torch.equal(torch.cat([decoded[rank] for rank in ranks]), whole)


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
        ModelExecutor(
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
            return media.decode_video_unit(
                runner, "reconstruction", source, slice(0, 4), PIXEL
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


def test_module_call_statistics_count_the_call_without_tokens():
    """A standalone module call has no query tokens to count.

    It reports one call of its mode and no token count for it.
    """
    model = DecodedModel()
    components = (
        (
            "reconstruction",
            ComponentConfig((0,), distribution="temporal_units"),
        ),
    )
    runner = ModelExecutor(
        model,
        WorkerConfig(device="cpu"),
        bindings=_encoder_bindings(model, components),
    )
    try:
        output = media.decode_video_unit(
            runner, "reconstruction", torch.zeros(4, 3), slice(0, 4), PIXEL
        )
    finally:
        runner.close()

    assert output.stats is not None
    assert sum(output.stats.mode_counts.values()) == 1
    assert output.stats.mode_tokens == {}


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
    runner = ModelExecutor(
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


def test_a_batch_reading_more_regions_than_read_tickets_completes():
    """A rank's read tickets bound the reads in flight, not one batch's.

    Each of four requests' transfer calls imports a product another rank
    wrote in two regions: eight reads against the rank's four read tickets.
    While other reads hold three of the tickets, no import can start all of
    its reads, so the batch waits; as tickets return, it resumes and every
    call delivers its whole product.
    """
    worker = execution_worker(max_batch_calls=4)
    events = EventPool()
    producer = make_transport(
        "local",
        byte_capacity=1 << 16,
        ticket_capacity=1,
        event_pool=events,
        source=WorkerEndpoint.local("producer"),
    )
    local = worker.transports["local"]
    assert local.capacity.ticket_capacity == 4
    published = []
    held = []
    try:
        admissions, calls, inputs, expected = [], [], [], {}
        for index in range(4):
            admission = ar_params(index + 1)
            admissions.append(admission)
            value = torch.arange(24, dtype=torch.float32).reshape(6, 4) + (
                100 * index
            )
            # The producer wrote rows [0, 3) and [3, 6) as separate regions.
            locations = tuple(
                producer.export(value[start : start + 3], offset=(start, 0))
                for start in (0, 3)
            )
            published.extend(locations)
            reference = TensorRef(
                admission.request_key,
                CallId(1, index),
                0,
                1,
                DType.F32,
                ShapeBound((StaticDim(6), StaticDim(4))),
            )
            copied = TensorRef(
                admission.request_key,
                CallId(2, index),
                0,
                1,
                DType.F32,
                ShapeBound((StaticDim(6), StaticDim(4))),
            )
            calls.append(
                Call(
                    request_key=admission.request_key,
                    call_id=CallId(2, index),
                    coordinates=CallCoordinates(),
                    kind=TransferMode.TENSOR,
                    bounds=Bounds(max_transfer_bytes=reference.max_bytes),
                    inputs=(reference,),
                    outputs=(copied,),
                )
            )
            inputs.append(
                TensorExport(
                    reference,
                    DeviceProductTransferValue(
                        0, 0, "", TensorTransfer((6, 4), locations)
                    ),
                )
            )
            expected[copied] = value

        # Reads elsewhere on the rank hold three of its four tickets: a
        # borrowed view does until its consumers finish.
        source = producer.export(torch.zeros(4))
        published.append(source)
        held = [
            local.fetch(source, device=torch.device("cpu")) for _ in range(3)
        ]
        submission = worker.submit(
            execution_batch(
                batch_id=2,
                admissions=admissions,
                calls=calls,
                input_products=inputs,
            )
        )
        for _ in range(100):
            worker.advance()
        assert worker.poll(submission) is None

        for ticket in held:
            ticket.close()
        held = []
        report = finalized_report(worker, submission)
        assert [completion.status for completion in report.completions] == [
            CallStatus.OK
        ] * 4
        assert len(report.products) == 4
        for product in report.products:
            destination = torch.empty(6, 4)
            tickets = fetch_tensor(
                product.value.tensor,
                destination,
                bindings={
                    (location.source, location.backend): worker.transports[
                        location.backend
                    ]
                    for location in product.value.tensor.locations
                },
            )
            for ticket in tickets:
                ready = threading.Event()
                ticket.add_done_callback(ready.set)
                assert ready.wait(5)
                ticket.result()
                ticket.close()
            torch.testing.assert_close(
                destination, expected[product.product], rtol=0, atol=0
            )
    finally:
        for ticket in held:
            ticket.close()
        worker.close()
        for location in published:
            producer.release(location)
        producer.close()
        events.close()


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
                    diffusion=DiffusionParams(
                        22, 3, 4, 1000, width=1344, height=768
                    ),
                    video=VideoAdmission(
                        VideoTask.T2VA, text_tags=(1,) * len(prompt)
                    ),
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
            run = run.replace(commands=())
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
        transfer = Batch(
            batch_id=2,
            collective_seq=3,
            calls=(consumer,),
            input_products=(TensorExport(reference, product.value),),
            buffer_allocations=(
                BufferAllocation(reference.buffer_id, 0, reference.max_bytes),
                BufferAllocation(copied.buffer_id, 256, copied.max_bytes),
            ),
        )

        # Refuse the destination export after consuming the source. Its
        # read lease and output reservation must retire so a retry can use
        # the same source and physical destination on every execution lane.
        capacity = worker.transports["local"].capacity
        reserved = capacity.capacity - capacity.used
        capacity.acquire(reserved)
        try:
            refused = finalized_report(worker, worker.submit(transfer))
        finally:
            capacity.release(reserved)
        assert refused.completions[0].status is CallStatus.ERROR
        assert refused.completions[0].error_code is ErrorCode.RESOURCE_EXHAUSTED
        assert not refused.products

        copied = replace(copied, producer_call_id=CallId(3, 0))
        consumer = consumer.replace(call_id=CallId(3, 0), outputs=(copied,))
        transfer = transfer.replace(
            batch_id=3,
            collective_seq=4,
            calls=(consumer,),
            buffer_allocations=(
                transfer.buffer_allocations[0],
                BufferAllocation(copied.buffer_id, 256, copied.max_bytes),
            ),
        )
        result = finalized_report(worker, worker.submit(transfer))
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
    runner = ModelExecutor(
        model,
        WorkerConfig(device=device, max_sequence_tokens=16),
        bindings=_encoder_bindings(
            model, (("text_encoder", ComponentConfig((0,))),), device
        ),
    )
    try:
        # Startup creates reusable buffer tensors under inference mode;
        # serving callers need not enter that scope themselves.
        with torch.inference_mode():
            runner.run_encoder("text", runner.prepare_text_tokens((2, 4, 6)))
        outputs = []
        prompts = ((3, 8, 1), (31,), (0, 5, 19, 7), (1, 2))
        for prompt in prompts:
            result = runner.run_encoder(
                "text", runner.prepare_text_tokens(prompt)
            )
            outputs.append(result.values[0])
        for prompt, output in zip(prompts, outputs, strict=True):
            expected = torch.tensor(prompt).reshape(-1, 1) * 4 + torch.arange(4)
            torch.testing.assert_close(
                output.cpu(), expected.float(), atol=0, rtol=0
            )
        for prompt in ((), (1,) * 17):
            with pytest.raises(InputError, match="capacity"):
                runner.prepare_text_tokens(prompt)
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
            diffusion=DiffusionParams(22, 3, 4, 1000, width=1344, height=768),
            video=VideoAdmission(VideoTask.T2VA, text_tags=(1, 1, 1)),
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
