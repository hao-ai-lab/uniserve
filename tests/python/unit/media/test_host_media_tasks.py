"""A host rank encodes and assembles media as tasks on its own lane."""

import io

import av
import numpy as np
import pytest

from uniserve_worker._uniserve_ipc import (
    SHM_HEADER_BYTES,
    SharedBuffer,
    SharedRead,
)
from uniserve_worker.errors import WorkerError
from uniserve_worker.execution.host import HostLane
from uniserve_worker.media.container import AvMuxConfig
from uniserve_worker.media.mux import MediaEncoder, MediaMux
from uniserve_worker.protocol.identity import CallId, RequestKey

pytestmark = pytest.mark.unit

CALL = CallId(1, 0)


def _config() -> AvMuxConfig:
    return AvMuxConfig(
        width=32,
        height=16,
        frame_count=6,
        frame_rate=24,
        audio_rate=32000,
        video_unit_frames=(4, 2),
    )


@pytest.fixture
def lane():
    # A host rank is one codec slot: one task at a time on one thread.
    lane = HostLane(max_inflight=1, workers=1)
    try:
        yield lane
    finally:
        lane.close()


def _run(lane: HostLane, schedule):
    task = schedule(lane.reserve())
    task.submit_if_ready()
    return task.result(timeout=30)


def test_borrowed_and_imported_units_assemble_with_audio_into_an_artifact(
    lane,
):
    config = _config()
    red = np.zeros((4, 16, 32, 3), dtype=np.uint8)
    red[..., 0] = 255
    blue = np.zeros((2, 16, 32, 3), dtype=np.uint8)
    blue[..., 2] = 255
    storage = SharedBuffer(red.nbytes, (0,))
    memoryview(storage)[SHM_HEADER_BYTES:] = red.tobytes()
    storage.mark_ready()
    borrow = SharedRead(storage.name, red.nbytes, 0)
    assert not storage.settled()
    request = RequestKey(1, 7, 0)
    encoder = MediaEncoder(rank=0)
    mux = MediaMux(rank=0)
    try:
        # A unit decoded on this host is read in place and its borrow is
        # released once encoded; an imported unit arrives as an array.
        units = tuple(
            _run(
                lane,
                lambda reservation, index=index, source=source: encoder.unit(
                    request,
                    config=config,
                    unit_index=index,
                    source=source,
                    reservation=reservation,
                    call_id=CALL,
                ),
            )
            for index, source in enumerate((borrow, blue.reshape(-1)))
        )
        assert storage.settled()

        mux.open(request, config=config)
        pcm = np.zeros((8000, 2), dtype=np.int16)
        _run(
            lane, lambda reservation: mux.audio(request, pcm, reservation, CALL)
        )
        for unit in units:
            _run(
                lane,
                lambda reservation, unit=unit: mux.append_units(
                    request, (unit,), reservation, CALL
                ),
            )
        artifact = _run(
            lane,
            lambda reservation: mux.finalize_artifact(
                request, reservation, CALL
            ),
        )
    finally:
        borrow.release()
        storage.close()

    with av.open(io.BytesIO(artifact)) as container:
        video = container.streams.video[0]
        assert (video.width, video.height, video.average_rate) == (32, 16, 24)
        assert container.streams.audio[0].sample_rate == 32000
        frames = tuple(container.decode(video))
        assert len(frames) == 6
        for index, frame in enumerate(frames):
            pixels = frame.to_ndarray(format="rgb24").mean((0, 1))
            assert pixels.argmax() == (0 if index < 4 else 2)


def test_a_unit_whose_bytes_do_not_form_frames_fails_its_task(lane):
    config = _config()
    encoder = MediaEncoder(rank=0)
    with pytest.raises(WorkerError, match="RGB24 dimensions"):
        _run(
            lane,
            lambda reservation: encoder.unit(
                RequestKey(1, 8, 0),
                config=config,
                unit_index=0,
                source=np.zeros(16 * 32 * 3 * 3 + 3, dtype=np.uint8),
                reservation=reservation,
                call_id=CALL,
            ),
        )


def test_a_dropped_request_refuses_further_assembly(lane):
    config = _config()
    request = RequestKey(1, 9, 0)
    mux = MediaMux(rank=0)
    mux.open(request, config=config)
    audio = mux.audio(
        request, np.zeros((8000, 2), dtype=np.int16), lane.reserve(), CALL
    )

    # The session is discarded after its task was scheduled: the task finds
    # the container closed, and the request has no session left.
    mux.drop(request.request_id)
    audio.submit_if_ready()
    with pytest.raises(WorkerError, match="discarded"):
        audio.result(timeout=30)
    with pytest.raises(WorkerError, match="no active session"):
        mux.finalize_artifact(request, lane.reserve(), CALL)
