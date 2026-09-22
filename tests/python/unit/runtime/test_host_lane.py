"""CPU admission and pinned-input lifetime at the public executor boundary."""

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from threading import Event

import pytest
import torch

from uniserve.runtime import EventPool
from uniserve_worker.errors import WorkerError
from uniserve_worker.execution.host import HostLane
from uniserve_worker.storage.output import OutputPool


def test_submitted_task_retains_capacity_until_actual_completion() -> None:
    pool = HostLane(max_inflight=1, workers=1)
    entered, finish = Event(), Event()

    def work() -> int:
        entered.set()
        assert finish.wait(5)
        return 17

    task = pool.reserve()
    result = task.submit(work)
    try:
        assert entered.wait(5)
        task.abandon()
        with pytest.raises(WorkerError, match="capacity is exhausted"):
            pool.reserve()
        assert not task.ready()
        finish.set()
        assert result.result(timeout=5) == 17
        replacement = pool.reserve()
        replacement.abandon()
        assert replacement.promise.cancelled()
    finally:
        finish.set()
        pool.close()


def test_close_cancels_unsubmitted_dependency_and_drains_submitted_work() -> (
    None
):
    pool = HostLane(max_inflight=2, workers=1)
    predecessor = pool.reserve()
    successor = pool.reserve().configure(
        lambda: 2, dependencies=(predecessor.promise,)
    )
    successor.submit_if_ready()
    # Closing must cancel pending promises outside the admission lock: a running
    # dependent can then fail and release its capacity while shutdown waits.
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(pool.close).result(timeout=5)
    assert predecessor.promise.cancelled()
    with pytest.raises(CancelledError):
        successor.result()
    with pytest.raises(WorkerError, match="closed"):
        pool.reserve()
    assert pool.reserved == 0


def test_abort_returns_before_running_work_and_cancels_queued_work() -> None:
    pool = HostLane(max_inflight=2, workers=1)
    entered, finish, queued_ran = Event(), Event(), Event()

    def work() -> int:
        entered.set()
        assert finish.wait(5)
        return 17

    running = pool.reserve().submit(work)
    try:
        assert entered.wait(5)
        queued = pool.reserve().submit(queued_ran.set)
        pool.abort()
        assert not running.done()
        with pytest.raises(WorkerError, match="closed"):
            pool.reserve()
        finish.set()
        assert running.result(timeout=5) == 17
        with pytest.raises(CancelledError):
            queued.result(timeout=5)
        assert not queued_ran.is_set()
    finally:
        finish.set()
        pool.close()


def test_ready_does_not_submit_and_failure_releases_capacity() -> None:
    pool = HostLane(max_inflight=1, workers=1)
    called = Event()

    def fail() -> None:
        called.set()
        raise ValueError("encoding failed")

    task = pool.reserve().configure(fail)
    try:
        assert not task.ready()
        assert not called.is_set()
        task.submit_if_ready()
        with pytest.raises(ValueError, match="encoding failed"):
            task.promise.result(timeout=5)
        assert pool.reserved == 0
    finally:
        pool.close()


def test_cancel_preserves_input_until_its_producer_completes() -> None:
    pool = HostLane(max_inflight=1, workers=1)
    copied: Future[None] = Future()
    released = Event()
    task = pool.reserve().configure(
        lambda: 1,
        input_ready=copied.done,
        input_completion=lambda: copied,
        release=released.set,
    )
    try:
        task.submit_if_ready()
        task.abandon()
        assert task.promise.cancelled()
        assert not released.is_set()
        copied.set_result(None)
        assert released.is_set()
    finally:
        pool.close()


def test_abandoned_output_remains_readable_until_cpu_reader_finishes() -> None:
    events = EventPool()
    outputs = OutputPool(capacity=1, max_words=8, event_pool=events)
    pool = HostLane(max_inflight=1, workers=1)
    entered, finish = Event(), Event()
    buffer = outputs.acquire(1, token_capacity=8)
    capture = buffer.capture_bytes(torch.tensor([3, 5, 7], dtype=torch.uint8))
    buffer.seal()

    def read() -> list[int]:
        entered.set()
        assert finish.wait(5)
        return capture.tolist()

    task = pool.reserve().configure(
        read,
        input_ready=buffer.ready,
        input_completion=buffer.completion_future,
        release=buffer.retain_cpu_reader(),
    )
    try:
        task.submit_if_ready()
        assert entered.wait(5)
        task.abandon()
        buffer.abandon()
        with pytest.raises(WorkerError, match="leases are active"):
            outputs.acquire(1, token_capacity=8)
        finish.set()
        assert task.promise.result(timeout=5) == [3, 5, 7]
        replacement = outputs.acquire(1, token_capacity=8)
        replacement.abandon()
    finally:
        finish.set()
        pool.close()
        outputs.close()
        events.close()


def test_session_jobs_share_one_codec_process_and_discard_ends_a_session() -> (
    None
):
    """A mux session's jobs reach the process that holds it, on any worker."""
    import io
    from multiprocessing import shared_memory

    import av
    import numpy as np

    from uniserve_worker.media.codec_process import (
        AvMuxConfig,
        EncodeAudioTrack,
        EncodeVideoUnit,
        MuxAppend,
        MuxFinalize,
        SharedSlice,
    )

    config = AvMuxConfig(
        width=32,
        height=16,
        frame_count=6,
        frame_rate=24,
        audio_rate=32000,
        video_unit_frames=(4, 2),
    )
    red = np.zeros((4, 16, 32, 3), dtype=np.uint8)
    red[..., 0] = 255
    blue = np.zeros((2, 16, 32, 3), dtype=np.uint8)
    blue[..., 2] = 255
    pcm = np.zeros((8000, 2), dtype=np.int16)
    segment = shared_memory.SharedMemory(
        create=True, size=red.nbytes + blue.nbytes + pcm.nbytes
    )
    payload = red.tobytes() + blue.tobytes() + pcm.tobytes()
    segment.buf[: len(payload)] = payload
    first = SharedSlice(segment.name, 0, red.nbytes)
    second = SharedSlice(segment.name, red.nbytes, blue.nbytes)
    audio = SharedSlice(segment.name, red.nbytes + blue.nbytes, pcm.nbytes)

    pool = HostLane(max_inflight=8, workers=2, codec=True)
    try:
        pool.probe()
        encodes = []
        for source in (first, second):
            task = pool.reserve().configure(EncodeVideoUnit(config, source))
            task.submit_if_ready()
            encodes.append(task.promise)
        units = tuple(promise.result(timeout=30) for promise in encodes)

        session = (1, 5, 0)
        track = pool.reserve().configure(
            EncodeAudioTrack(session, config, audio), session=session
        )
        track.submit_if_ready()
        # Each append follows the previous one; finalization follows the last
        # append and the audio track, and ends the session.
        appends = []
        for unit in units:
            task = pool.reserve().configure(
                MuxAppend(session, config, (unit,)),
                dependencies=tuple(appends[-1:]),
                session=session,
            )
            task.submit_if_ready()
            appends.append(task.promise)
        final = pool.reserve().configure(
            MuxFinalize(session),
            dependencies=(appends[-1], track.promise),
            session=session,
            ends_session=True,
            transform=lambda value: value[0],
        )
        final.submit_if_ready()
        name = final.promise.result(timeout=30)

        artifact = shared_memory.SharedMemory(name=name)
        try:
            encoded = bytes(artifact.buf)
        finally:
            artifact.close()
            artifact.unlink()
        with av.open(io.BytesIO(encoded)) as container:
            assert len(tuple(container.decode(container.streams.video[0]))) == 6

        # A discarded session is gone from the process that held it.
        other = (1, 6, 0)
        started = pool.reserve().configure(
            EncodeAudioTrack(other, config, audio), session=other
        )
        started.submit_if_ready()
        started.promise.result(timeout=30)
        pool.discard_session(other)
        late = pool.reserve().configure(MuxFinalize(other), session=other)
        late.submit_if_ready()
        with pytest.raises(ValueError, match="no assembly session"):
            late.promise.result(timeout=30)
    finally:
        pool.close()
        segment.close()
        segment.unlink()
