"""Host results retain their output row until delivery or abandonment."""

from dataclasses import replace
from multiprocessing import shared_memory

import pytest

from uniserve.runtime import EventPool
from uniserve_worker.execution.host import HostLane
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.execution.request import RequestPool
from uniserve_worker.media.storage import publish_media_bytes
from uniserve_worker.protocol.batch import NewRequest
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    CallStatus,
    ErrorCode,
    ImageParams,
    MediaCall,
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.protocol.output import MediaOutput, PosixShmArtifact
from uniserve_worker.storage.output import OutputPool

pytestmark = pytest.mark.unit


@pytest.fixture
def output():
    events = EventPool()
    buffers = OutputPool(capacity=1, max_words=4, event_pool=events)
    requests = RequestPool(1)
    lane = HostLane(max_inflight=1, workers=1)
    key = RequestKey(1, 7, 1)
    requests.start(NewRequest(key, 1, image=ImageParams(height=16, width=16)))
    call = Call(
        request_key=key,
        call_id=CallId(1, 0),
        coordinates=CallCoordinates(),
        kind=MediaCall.IMAGE_DECODING,
        bounds=Bounds(),
    )
    buffer = buffers.acquire(1, token_capacity=1)
    pending = PendingOutput(call, requests.get(key.request_id), buffer, 0)
    try:
        yield pending, buffer, buffers, lane
    finally:
        pending.abandon()
        lane.close()
        buffers.close()
        requests.close()
        events.close()


@pytest.mark.parametrize("finish_callback", [False, True])
def test_media_delivery_releases_the_output_row(output, finish_callback):
    pending, buffer, buffers, lane = output
    payload = b"encoded image bytes"
    task = lane.reserve().configure(lambda: payload)
    finish = None
    if finish_callback:

        def finish(results):
            # The numerical callback may update its own result object.
            data = b"".join(results)
            pending.set_media(
                MediaOutput(
                    PosixShmArtifact(publish_media_bytes(data)), len(data)
                )
            )

    pending.set_host_tasks((task,), finish=finish)

    task.submit_if_ready()
    task.result(timeout=5)
    assert not pending.ready()
    buffer.seal()

    result = pending.materialize()
    assert result.status is CallStatus.OK
    assert result.media_output is not None
    media = result.media_output
    segment = shared_memory.SharedMemory(name=media.handle.name)
    try:
        assert bytes(segment.buf[: media.bytes]) == payload
        assert pending.materialize() == result
    finally:
        segment.close()
        segment.unlink()

    replacement = buffers.acquire(1, token_capacity=1)
    replacement.abandon()


def test_failed_host_work_reports_compute_error_and_releases_the_row(output):
    pending, buffer, buffers, lane = output
    pending.progress = replace(
        pending.progress,
        logical_position=8,
        flow_step=3,
        rng_counter=2,
        kv_visible_len=8,
        kv_computed_len=10,
    )

    def encode():
        raise ValueError("codec failed")

    task = lane.reserve().configure(encode)
    pending.set_host_tasks((task,))
    task.submit_if_ready()
    with pytest.raises(ValueError, match="codec failed"):
        task.result(timeout=5)
    buffer.seal()

    result = pending.materialize()
    assert result.status is CallStatus.ERROR
    assert result.error_code is ErrorCode.COMPUTE_ERROR
    assert pending.status is result.status
    assert pending.error_code is result.error_code
    assert result.committed_tokens == ()
    assert result.media_output is None
    assert result.num_completed_steps == 0
    assert (
        result.position == result.kv_visible_len == result.kv_computed_len == 0
    )
    assert pending.progress.rng_counter == 0
    replacement = buffers.acquire(1, token_capacity=1)
    replacement.abandon()
