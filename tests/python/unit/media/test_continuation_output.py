"""A continuation's artifact omits the frames that regenerate its clip.

The generated video of a continuation starts by regenerating the clip it
continues, in place; the artifact holds only the new frames and the audio
from the same instant. Every decoded media unit is still encoded where it
was reconstructed: a unit wholly inside the clip encodes to no bytes, one
that straddles its end encodes its remaining frames, and the audio track
skips the clip's duration.
"""

import io
from multiprocessing import shared_memory
from types import SimpleNamespace

import av
import numpy as np
import pytest

from uniserve.model import ConditionRole
from uniserve_models.minimax_h3.output import video_clips
from uniserve_worker.errors import WorkerError
from uniserve_worker.execution.host import HostLane
from uniserve_worker.execution.media import continuation_frames, mux_config
from uniserve_worker.media.container import (
    AvMuxConfig,
    AvMuxSession,
    encode_audio_track,
    encode_video_unit,
)
from uniserve_worker.media.mux import MediaEncoder, MediaMux
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.transport.shm import HostBorrow

pytestmark = pytest.mark.unit

CALL = CallId(1, 0)
# Red, green and blue frames, then a gray one.
COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128))


def _config(skip: int) -> AvMuxConfig:
    return AvMuxConfig(
        width=32,
        height=16,
        frame_count=8,
        frame_rate=24,
        audio_rate=32000,
        video_unit_frames=(4, 4),
        skip_frames=skip,
    )


def _frames(colors) -> np.ndarray:
    frames = np.zeros((len(colors), 16, 32, 3), dtype=np.uint8)
    for index, color in enumerate(colors):
        frames[index] = color
    return frames


def test_units_omit_the_frames_before_the_skip():
    config = AvMuxConfig(
        width=32,
        height=32,
        frame_count=158,
        frame_rate=24,
        audio_rate=32000,
        video_unit_frames=tuple(c.stop - c.start for c in video_clips(158)),
        skip_frames=22,
    )
    # Eight 17-frame bodies and a final 22-frame clip.
    assert config.video_unit_frames == (17,) * 8 + (22,)
    assert [config.unit_skip(unit) for unit in range(9)] == [17, 5] + [0] * 7
    assert config.output_frames == 136
    # 22 frames at 24 fps are 29,333 samples at 32 kHz.
    assert config.audio_samples(config.skip_frames) == 29_333
    assert config.audio_samples(config.output_frames) == 181_333
    assert _config(0).unit_skip(0) == 0
    for skip in (-1, 8, 9):
        with pytest.raises(ValueError, match="keep at least one"):
            _config(skip)


def test_the_audio_track_starts_at_the_skip():
    # Loud through the five skipped frames, silent after them.
    skipped = _config(5).audio_samples(5)
    pcm = np.zeros((_config(0).audio_samples(8), 2), dtype=np.int16)
    pcm[:skipped] = 20_000
    for skip, loud in ((0, True), (5, False)):
        config = _config(skip)
        encoded = encode_audio_track(config, pcm)
        with av.open(io.BytesIO(encoded)) as container:
            track = container.streams.audio[0]
            assert (
                abs(
                    float(track.duration * track.time_base)
                    - config.output_frames / 24
                )
                <= 1024 / 32000
            )
            samples = np.concatenate(
                [frame.to_ndarray() for frame in container.decode(audio=0)],
                axis=1,
            )
        level = float(np.abs(samples).max())
        assert (level > 0.3) == loud, (skip, level)


@pytest.fixture
def lane():
    lane = HostLane(max_inflight=1, workers=1)
    try:
        yield lane
    finally:
        lane.close()


def _run(lane: HostLane, schedule):
    task = schedule(lane.reserve())
    task.submit_if_ready()
    return task.promise.result(timeout=30)


def test_an_artifact_holds_only_the_new_frames(lane):
    """Five skipped frames leave the second unit's last three.

    The first unit encodes nothing, the second its last three frames, and
    the artifact holds those three.
    """
    config = _config(5)
    first = _frames([COLORS[0]] * 4)
    second = _frames([COLORS[0], COLORS[1], COLORS[2], COLORS[3]])
    segment = shared_memory.SharedMemory(create=True, size=first.nbytes)
    segment.buf[: first.nbytes] = first.tobytes()
    released = []
    borrow = HostBorrow(
        segment=segment.name,
        offset=0,
        nbytes=first.nbytes,
        _release=lambda: released.append(True),
    )
    request = RequestKey(1, 9, 0)
    encoder, mux = MediaEncoder(rank=0), MediaMux(rank=0)
    try:
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
            for index, source in enumerate((borrow, second.reshape(-1)))
        )
        # An omitted unit still releases its borrow.
        assert released == [True]
        assert units[0] == b""
        assert units[1]

        mux.open(request, config=config)
        pcm = np.zeros((config.audio_samples(8), 2), dtype=np.int16)
        _run(
            lane, lambda reservation: mux.audio(request, pcm, reservation, CALL)
        )
        _run(
            lane,
            lambda reservation: mux.append_units(
                request, units, reservation, CALL
            ),
        )
        artifact = _run(
            lane,
            lambda reservation: mux.finalize_artifact(
                request, reservation, CALL
            ),
        )
    finally:
        segment.close()
        segment.unlink()

    published = shared_memory.SharedMemory(name=artifact.handle.name)
    try:
        encoded = bytes(published.buf[: artifact.bytes])
    finally:
        published.close()
        published.unlink()
    with av.open(io.BytesIO(encoded)) as container:
        frames = tuple(container.decode(container.streams.video[0]))
        assert len(frames) == 3
        for index, (frame, color) in enumerate(
            zip(frames, COLORS[1:], strict=True)
        ):
            assert float(frame.pts * frame.time_base) == index / 24
            pixels = frame.to_ndarray(format="rgb24").mean((0, 1))
            assert np.abs(pixels - np.array(color)).max() < 24, (index, pixels)
        track = container.streams.audio[0]
        assert abs(float(track.duration * track.time_base) - 3 / 24) <= (
            1024 / 32000
        )


def test_a_session_counts_an_omitted_unit():
    config = _config(4)
    unit = encode_video_unit(config, _frames([COLORS[1]] * 4))
    session = AvMuxSession(config)
    session.append((b"",))
    session.append((unit,))
    encoded = session.finalize(
        encode_audio_track(config, np.zeros((1000, 2), dtype=np.int16))
    )
    with av.open(io.BytesIO(encoded)) as container:
        assert len(tuple(container.decode(video=0))) == 4
    with pytest.raises(ValueError, match="more media units"):
        AvMuxSession(config).append((b"", b"", b""))


def _admission(num_frames: int, clip: int | None):
    conditions = [
        SimpleNamespace(
            role=ConditionRole.REFERENCE, video=SimpleNamespace(vae_frames=22)
        )
    ]
    if clip is not None:
        conditions.append(
            SimpleNamespace(
                role=ConditionRole.CONTINUATION,
                video=SimpleNamespace(vae_frames=clip),
            )
        )
    return SimpleNamespace(
        diffusion=SimpleNamespace(num_frames=num_frames, width=64, height=32),
        video=SimpleNamespace(conditions=tuple(conditions)),
    )


def _runner():
    return SimpleNamespace(
        video_decoder=SimpleNamespace(clips=video_clips),
        video_postprocessor=SimpleNamespace(frame_rate=24),
        audio_decoder=SimpleNamespace(sample_rate=32000),
    )


def test_a_request_skips_its_continuation_clip():
    """Only a continuation clip's frames are skipped, not a reference's."""
    for clip, skip in ((None, 0), (5, 5), (22, 22), (39, 39)):
        admission = _admission(158, clip)
        assert continuation_frames(admission) == skip
        config = mux_config(_runner(), admission)
        assert (config.frame_count, config.skip_frames) == (158, skip)
        assert config.output_frames == 158 - skip
    assert continuation_frames(SimpleNamespace(video=None)) == 0
    with pytest.raises(WorkerError, match="keep at least one"):
        mux_config(_runner(), _admission(39, 39))
