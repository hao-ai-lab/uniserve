"""Codec jobs run in a codec process against media units in shared memory."""

import io
from multiprocessing import shared_memory

import av
import numpy as np
import pytest

from uniserve_worker.media.codec_process import (
    AvMuxConfig,
    CodecProcess,
    EncodeAudioTrack,
    EncodeVideoUnit,
    MuxAppend,
    MuxClose,
    MuxFinalize,
    Probe,
    SharedSlice,
)


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
def codec():
    process = CodecProcess()
    try:
        yield process
    finally:
        process.close()


@pytest.fixture
def segment():
    """One shared-memory segment holding every media unit of a test."""
    storage = shared_memory.SharedMemory(create=True, size=1 << 20)
    try:
        yield storage
    finally:
        storage.close()
        storage.unlink()


def _write(
    segment: shared_memory.SharedMemory, offset: int, values: np.ndarray
) -> SharedSlice:
    payload = values.tobytes()
    segment.buf[offset : offset + len(payload)] = payload
    return SharedSlice(segment.name, offset, len(payload))


def test_the_probe_confirms_the_process_serves_its_codecs(codec):
    assert codec.execute(Probe()) is True


def test_units_and_audio_from_shared_memory_assemble_into_an_artifact(
    codec, segment
):
    config = _config()
    red = np.zeros((4, 16, 32, 3), dtype=np.uint8)
    red[..., 0] = 255
    blue = np.zeros((2, 16, 32, 3), dtype=np.uint8)
    blue[..., 2] = 255
    pcm = np.zeros((8000, 2), dtype=np.int16)

    first = _write(segment, 0, red)
    second = _write(segment, red.nbytes, blue)
    audio = _write(segment, red.nbytes + blue.nbytes, pcm)

    units = (
        codec.execute(EncodeVideoUnit(config, first)),
        codec.execute(EncodeVideoUnit(config, second)),
    )
    session = (1, 7, 0)
    assert codec.execute(EncodeAudioTrack(session, config, audio)) is None
    # Rounds are appended as they complete; the artifact follows the last.
    codec.execute(MuxAppend(session, config, units[:1]))
    codec.execute(MuxAppend(session, config, units[1:]))
    name, nbytes = codec.execute(MuxFinalize(session))

    artifact = shared_memory.SharedMemory(name=name)
    try:
        encoded = bytes(artifact.buf[:nbytes])
    finally:
        artifact.close()
        artifact.unlink()

    with av.open(io.BytesIO(encoded)) as container:
        video, track = container.streams.video[0], container.streams.audio[0]
        assert (video.width, video.height, video.average_rate) == (32, 16, 24)
        assert track.sample_rate == 32000
        frames = tuple(container.decode(video))
        assert len(frames) == 6
        for index, frame in enumerate(frames):
            pixels = frame.to_ndarray(format="rgb24").mean((0, 1))
            assert pixels.argmax() == (0 if index < 4 else 2)


def test_a_failed_job_answers_its_task_and_the_process_serves_on(
    codec, segment
):
    config = _config()
    pixels = np.zeros((4, 16, 32, 3), dtype=np.uint8)
    source = _write(segment, 0, pixels)
    with pytest.raises(ValueError, match="RGB24 dimensions"):
        codec.execute(
            EncodeVideoUnit(
                config, SharedSlice(segment.name, 0, 16 * 32 * 3 * 3 + 3)
            )
        )
    with pytest.raises(ValueError, match="outside its shared-memory"):
        codec.execute(
            EncodeVideoUnit(
                config,
                SharedSlice(segment.name, segment.size - 8, pixels.nbytes),
            )
        )
    with pytest.raises(FileNotFoundError):
        codec.execute(
            EncodeVideoUnit(
                config,
                SharedSlice("uniserve-no-such-segment", 0, pixels.nbytes),
            )
        )
    assert len(codec.execute(EncodeVideoUnit(config, source))) > 0


def test_a_closed_session_cannot_be_finalized(codec, segment):
    config = _config()
    session = (1, 9, 0)
    codec.execute(MuxClose(session))
    with pytest.raises(ValueError, match="no assembly session"):
        codec.execute(MuxFinalize(session))
    # A finalize without audio leaves no half-built session behind either.
    pcm = np.zeros((8000, 2), dtype=np.int16)
    codec.execute(EncodeAudioTrack(session, config, _write(segment, 0, pcm)))
    with pytest.raises(ValueError, match="every media unit"):
        codec.execute(MuxFinalize(session))
    with pytest.raises(ValueError, match="no assembly session"):
        codec.execute(MuxFinalize(session))
