"""Independently encoded media units assemble into one decodable artifact.

Each media unit is encoded where it was reconstructed and the audio track is
encoded on the muxer rank, so assembly concatenates already-encoded tracks
without re-encoding and must still preserve frame order, dimensions and clocks.
"""

import io

import av
import numpy as np
import pytest

from uniserve_worker.media.container import (
    AvMuxConfig,
    AvMuxSession,
    encode_audio_track,
    encode_video_unit,
)
from uniserve_worker.media.mux import (
    encoded_unit_bytes,
    frame_encoded_unit,
    read_encoded_unit,
)

pytestmark = pytest.mark.integration


def _config() -> AvMuxConfig:
    return AvMuxConfig(
        width=32,
        height=16,
        frame_count=6,
        frame_rate=24,
        audio_rate=32000,
        video_unit_frames=(4, 2),
    )


def test_encoded_units_and_audio_assemble_into_a_decodable_mp4():
    config = _config()
    red = np.zeros((4, 16, 32, 3), dtype=np.uint8)
    red[..., 0] = 255
    blue = np.zeros((2, 16, 32, 3), dtype=np.uint8)
    blue[..., 2] = 255
    units = (
        encode_video_unit(config, red),
        encode_video_unit(config, blue),
    )
    audio = encode_audio_track(config, np.zeros((8000, 2), dtype=np.int16))
    # Rounds arrive one at a time; the artifact is assembled once the audio
    # track follows the last of them.
    session = AvMuxSession(config)
    session.append(units[:1])
    session.append(units[1:])
    encoded = session.finalize(audio)

    with av.open(io.BytesIO(encoded)) as container:
        video, track = container.streams.video[0], container.streams.audio[0]
        assert (video.width, video.height, video.average_rate) == (32, 16, 24)
        assert (track.sample_rate, track.layout.name) == (32000, "stereo")
        assert (
            abs(float(track.duration * track.time_base) - 6 / 24)
            <= 1024 / 32000
        )
        frames = tuple(container.decode(video))
        assert len(frames) == 6
        for index, frame in enumerate(frames):
            assert float(frame.pts * frame.time_base) == index / 24
            pixels = frame.to_ndarray(format="rgb24").mean((0, 1))
            assert pixels.argmax() == (0 if index < 4 else 2)
            assert pixels.max() > 240
    with av.open(io.BytesIO(encoded)) as container:
        decoded = tuple(container.decode(audio=0))
        assert sum(frame.samples for frame in decoded) >= 8000


def test_assembly_requires_every_media_unit_of_the_request():
    config = _config()
    red = np.zeros((4, 16, 32, 3), dtype=np.uint8)
    audio = encode_audio_track(config, np.zeros((8000, 2), dtype=np.int16))
    session = AvMuxSession(config)
    session.append((encode_video_unit(config, red),))
    with pytest.raises(Exception, match="every media unit"):
        session.finalize(audio)
    blue = np.zeros((2, 16, 32, 3), dtype=np.uint8)
    session.append((encode_video_unit(config, blue),))
    with pytest.raises(Exception, match="more media units"):
        session.append((encode_video_unit(config, blue),))


@pytest.mark.parametrize(
    "frames,height,width", ((4, 16, 32), (8, 256, 256), (3, 34, 50))
)
def test_a_product_row_carries_one_encoded_unit_and_its_length(
    frames, height, width
):
    from dataclasses import replace

    import torch

    config = replace(
        _config(),
        frame_count=frames,
        video_unit_frames=(frames,),
        height=height,
        width=width,
    )
    payload = encode_video_unit(
        config,
        np.random.default_rng(0).integers(
            0,
            256,
            (frames, height, width, 3),
            dtype=np.uint8,
        ),
    )
    row = torch.zeros(
        encoded_unit_bytes(frames, height, width), dtype=torch.uint8
    )
    frame_encoded_unit(payload, row)
    assert read_encoded_unit(row) == payload
    with av.open(io.BytesIO(read_encoded_unit(row))) as container:
        decoded = tuple(container.decode(video=0))
    assert len(decoded) == frames
    assert all(
        (frame.height, frame.width) == (height, width) for frame in decoded
    )


def test_a_unit_that_exceeds_its_reserved_row_fails_by_name():
    import torch

    config = _config()
    payload = encode_video_unit(
        config, np.zeros((4, 16, 32, 3), dtype=np.uint8)
    )
    with pytest.raises(Exception, match="exceeds the"):
        frame_encoded_unit(payload, torch.zeros(16, dtype=torch.uint8))


@pytest.fixture
def mux_worker():
    from dataclasses import replace

    import torch

    from tests.python.fixtures.execution_worker import execution_worker
    from tests.python.fixtures.h3 import fasth3_config
    from uniserve.distributed import Communicator, DeviceMesh
    from uniserve.media import image
    from uniserve_models.minimax_h3.model import Model
    from uniserve_worker.config.deployment import ComponentConfig
    from uniserve_worker.config.execution import WorkerConfig
    from uniserve_worker.model_executor.component_binding import (
        ComponentBinding,
    )

    # Host ranks keep the ordinary model description without loading weights.
    # A small, tile-aligned canvas bounds the codec inputs used by this test.
    config = fasth3_config()
    config = replace(
        config,
        denoisers={
            name: replace(denoiser, canvases=(image.Config(256, 256),))
            for name, denoiser in config.denoisers.items()
        },
    )
    with torch.device("meta"):
        model = Model(config)
    placement = ComponentConfig((0,))
    dimensions = placement.parallel_config.dimensions
    binding = ComponentBinding(
        "muxer",
        placement,
        Communicator(device=torch.device("cpu")),
        DeviceMesh(
            ranks=(0,),
            rank=0,
            shape=tuple(size for _, size in dimensions),
            axes=tuple(axis for axis, _ in dimensions),
        ),
        torch.device("cpu"),
    )
    with execution_worker(
        model,
        components=(("muxer", placement),),
        bindings={"muxer": binding},
        transfer_backends=("shm", "channel", "local"),
        max_request_pool_size=1,
        queue_depth=3,
        execution=WorkerConfig(
            graph_policy="off",
            prefill_cuda_graph=False,
            max_video_seconds=2,
            max_sequence_tokens=8,
            video_text_capacities=(8,),
        ),
    ) as worker:
        yield worker


def _mux_request(key):
    from uniserve_worker.protocol.batch import DiffusionParams, NewRequest
    from uniserve_worker.protocol.video import VideoAdmission, VideoTask

    return NewRequest(
        key,
        request_pool_idx=1,
        diffusion=DiffusionParams(39, 2, 4, 0, width=256, height=256),
        video=VideoAdmission(VideoTask.T2VA, text_tags=(1,)),
        prompt_token_ids=(1,),
    )


def _mux_call(worker, key, batch_id, kind, *, tensor=None, dtype=None):
    from tests.python.fixtures.depth_one import finalized_report
    from uniserve_worker.protocol.batch import (
        Batch,
        BufferAllocation,
        DecodeRange,
        Start,
        TensorExport,
    )
    from uniserve_worker.protocol.call import (
        Bounds,
        Call,
        CallCoordinates,
        MediaCall,
    )
    from uniserve_worker.protocol.identity import CallId
    from uniserve_worker.protocol.tensor import ShapeBound, StaticDim, TensorRef
    from uniserve_worker.protocol.transfer import DeviceProductTransferValue

    inputs = ()
    products = ()
    allocations = ()
    if tensor is not None:
        reference = TensorRef(
            key,
            CallId(2 * batch_id - 1, 0),
            0,
            1,
            dtype,
            ShapeBound(tuple(StaticDim(size) for size in tensor.shape)),
        )
        inputs = (reference,)
        products = (
            TensorExport(
                reference, DeviceProductTransferValue(0, 0, "", tensor)
            ),
        )
        allocations = (
            BufferAllocation(reference.buffer_id, 0, reference.max_bytes),
        )
    call = Call(
        request_key=key,
        call_id=CallId(2 * batch_id, 0),
        coordinates=CallCoordinates(),
        kind=kind,
        component="muxer",
        bounds=Bounds(max_completion_bytes=1 << 16),
        inputs=inputs,
    )
    return finalized_report(
        worker,
        worker.submit(
            Batch(
                batch_id=2 * batch_id,
                calls=(call,),
                decode_ranges=(DecodeRange(key, call.call_id, 0, 1),)
                if kind is MediaCall.AUDIO_ENCODING
                else (),
                input_products=products,
                buffer_allocations=allocations,
                commands=(Start(_mux_request(key)),) if batch_id == 1 else (),
            )
        ),
    )


@pytest.mark.parametrize(
    "backends",
    [
        ("channel", "channel"),
        ("shm", "channel"),
        ("shm", "shm"),
        ("local", "channel"),
        ("local", "local"),
    ],
)
def test_mux_worker_reads_initialized_units_and_delivers_the_artifact(
    mux_worker, backends
):
    from multiprocessing.shared_memory import SharedMemory

    import torch

    from uniserve_worker.protocol.call import CallStatus, MediaCall
    from uniserve_worker.protocol.identity import RequestKey
    from uniserve_worker.protocol.tensor import DType
    from uniserve_worker.protocol.transfer import TensorTransfer

    config = AvMuxConfig(
        width=256,
        height=256,
        frame_count=39,
        frame_rate=24,
        audio_rate=32000,
        video_unit_frames=(17, 22),
    )
    units = (
        encode_video_unit(config, np.zeros((17, 256, 256, 3), dtype=np.uint8)),
        encode_video_unit(
            config, np.full((22, 256, 256, 3), 255, dtype=np.uint8)
        ),
    )
    capacity = encoded_unit_bytes(22, 256, 256)
    storage = torch.empty((2, capacity), dtype=torch.uint8)
    locations = []
    key = RequestKey(1, 9, 0)
    try:
        for index, (payload, backend) in enumerate(
            zip(units, backends, strict=True)
        ):
            framed = frame_encoded_unit(payload, storage[index])
            locations.append(
                mux_worker.transports[backend].export(
                    framed.unsqueeze(0),
                    offset=(index, 0),
                    consumers=(0,),
                )
            )
        tensor = TensorTransfer(
            shape=tuple(storage.shape), locations=tuple(locations)
        )
        assert sum(location.nbytes for location in locations) == sum(
            len(unit) + 8 for unit in units
        )
        appended = _mux_call(
            mux_worker, key, 1, MediaCall.MUXING, tensor=tensor, dtype=DType.U8
        )
        assert appended.completions[0].status is CallStatus.OK

        pcm = torch.zeros((52000, 2), dtype=torch.int16)
        audio = mux_worker.transports["local"].export(pcm, consumers=(0,))
        locations.append(audio)
        encoded = _mux_call(
            mux_worker,
            key,
            2,
            MediaCall.AUDIO_ENCODING,
            tensor=TensorTransfer(shape=tuple(pcm.shape), locations=(audio,)),
            dtype=DType.I16,
        )
        assert encoded.completions[0].status is CallStatus.OK
        report = _mux_call(mux_worker, key, 3, MediaCall.MUXING)
        completion = report.completions[0]
        assert completion.status is CallStatus.OK
        output = completion.media_output
        assert output is not None
        shared = SharedMemory(name=output.handle.name)
        try:
            artifact = bytes(shared.buf[: output.bytes])
        finally:
            shared.unlink()
            shared.close()
    finally:
        for location in locations:
            mux_worker.transports[location.backend].release(location)

    with av.open(io.BytesIO(artifact)) as container:
        video = container.streams.video[0]
        assert (video.width, video.height, video.average_rate) == (256, 256, 24)
        assert container.streams.audio[0].sample_rate == 32000
        frames = tuple(container.decode(video))
        assert len(frames) == 39
        for index, frame in enumerate(frames):
            assert float(frame.pts * frame.time_base) == index / 24
            pixels = frame.to_ndarray(format="rgb24")
            assert pixels.mean() == (0 if index < 17 else 255)


def test_mux_worker_rejects_a_length_beyond_the_transferred_row(mux_worker):
    import torch

    from uniserve_worker.protocol.call import CallStatus, ErrorCode, MediaCall
    from uniserve_worker.protocol.identity import RequestKey
    from uniserve_worker.protocol.tensor import DType
    from uniserve_worker.protocol.transfer import TensorTransfer

    # The logical row is larger than its physical input. A length prefix must
    # be checked against received bytes, before it reaches a codec.
    row = torch.zeros((1, 9), dtype=torch.uint8)
    row[0, :8] = torch.from_numpy(
        np.frombuffer(np.uint64(9).tobytes(), dtype=np.uint8).copy()
    )
    location = mux_worker.transports["channel"].export(row, offset=(0, 0))
    tensor = TensorTransfer(shape=(1, 128), locations=(location,))
    report = _mux_call(
        mux_worker,
        RequestKey(1, 10, 0),
        1,
        MediaCall.MUXING,
        tensor=tensor,
        dtype=DType.U8,
    )
    assert report.completions[0].status is CallStatus.ERROR
    assert report.completions[0].error_code is ErrorCode.INVALID_CALL
    assert report.completions[0].media_output is None
