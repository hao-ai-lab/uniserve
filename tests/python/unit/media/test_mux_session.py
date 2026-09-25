"""A discarded assembly session closes its container once no task holds it.

A request's container is touched by lane tasks on host worker threads, while
a dropped or cancelled request discards its session on the executor thread.
The discard must not close a container a task is still using, and the
container must be closed once neither side holds it. A closed container
refuses to finalize its artifact, which is how these tests observe it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from uniserve_worker.media.container import (
    AvMuxConfig,
    AvMuxSession,
    encode_audio_track,
    encode_video_unit,
)
from uniserve_worker.media.mux import MuxSession

pytestmark = pytest.mark.unit


class _ReleaseInterleaving:
    """A lock that lets another thread run to completion just before release.

    It fixes the scheduling of the executor thread's discard at the last
    instant a lane task still holds the container, which the operating
    system otherwise reaches only by chance.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.before_release: Callable[[], None] | None = None

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        action, self.before_release = self.before_release, None
        if action is not None:
            other = threading.Thread(target=action)
            other.start()
            other.join()
        self._lock.release()

    def __enter__(self) -> _ReleaseInterleaving:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


@dataclass(frozen=True)
class _Media:
    """One request's settings with its encoded units and audio track."""

    config: AvMuxConfig
    units: tuple[bytes, ...]
    audio: bytes


@pytest.fixture(scope="module")
def media() -> _Media:
    config = AvMuxConfig(
        width=32,
        height=16,
        frame_count=6,
        frame_rate=24,
        audio_rate=32000,
        video_unit_frames=(4, 2),
    )
    units = tuple(
        encode_video_unit(config, np.zeros((frames, 16, 32, 3), np.uint8))
        for frames in config.video_unit_frames
    )
    audio = encode_audio_track(config, np.zeros((8000, 2), np.int16))
    return _Media(config, units, audio)


def _discard_on_executor_thread(session: MuxSession) -> None:
    discard = threading.Thread(target=session.discard)
    discard.start()
    discard.join()


def test_a_discard_during_a_task_leaves_the_container_to_the_task(
    media: _Media,
) -> None:
    session = MuxSession(media.config, AvMuxSession(media.config))

    with session.held() as container:
        container.append(media.units)
        _discard_on_executor_thread(session)
        assert container.finalize(media.audio), (
            "a task's container closed under it"
        )


@pytest.mark.parametrize("landing", ("during_task", "at_release"))
def test_a_discarded_container_is_closed_once_the_task_releases_it(
    media: _Media, landing: str
) -> None:
    lock = _ReleaseInterleaving()
    session = MuxSession(media.config, AvMuxSession(media.config), lock)

    with session.held() as container:
        container.append(media.units)
        if landing == "during_task":
            _discard_on_executor_thread(session)
        else:
            # The discard runs after the task's work is done but while it
            # still holds the lock, so it cannot close the container itself.
            lock.before_release = session.discard

    assert session.discarded
    with pytest.raises(ValueError, match="requires every media unit"):
        session.container.finalize(media.audio)
