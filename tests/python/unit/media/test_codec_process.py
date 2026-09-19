"""Codec jobs run in a codec process against media units in a shared mapping."""

import io

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
    SharedMapping,
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


def _write(
    mapping: SharedMapping, offset: int, values: np.ndarray
) -> SharedSlice:
    payload = values.tobytes()
    mapping.buffer[offset : offset + len(payload)] = payload
    return SharedSlice(mapping.name, offset, len(payload))


def test_units_and_audio_from_a_shared_mapping_assemble_into_an_artifact(
    codec,
):
    from multiprocessing import shared_memory

    config = _config()
    red = np.zeros((4, 16, 32, 3), dtype=np.uint8)
    red[..., 0] = 255
    blue = np.zeros((2, 16, 32, 3), dtype=np.uint8)
    blue[..., 2] = 255
    pcm = np.zeros((8000, 2), dtype=np.int16)

    mapping = SharedMapping(
        "test-media-ring", red.nbytes + blue.nbytes + pcm.nbytes
    )
    try:
        codec.attach(mapping)
        first = _write(mapping, 0, red)
        second = _write(mapping, red.nbytes, blue)
        audio = _write(mapping, red.nbytes + blue.nbytes, pcm)

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
    finally:
        mapping.close()

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


def test_a_failed_job_answers_its_task_and_the_process_serves_on(codec):
    config = _config()
    pixels = np.zeros((4, 16, 32, 3), dtype=np.uint8)
    mapping = SharedMapping("test-media-ring", pixels.nbytes)
    try:
        codec.attach(mapping)
        source = _write(mapping, 0, pixels)
        with pytest.raises(ValueError, match="RGB24 dimensions"):
            codec.execute(
                EncodeVideoUnit(
                    config, SharedSlice(mapping.name, 0, 16 * 32 * 3 * 3 + 3)
                )
            )
        with pytest.raises(ValueError, match="outside its shared mapping"):
            codec.execute(
                EncodeVideoUnit(
                    config, SharedSlice(mapping.name, 1, pixels.nbytes)
                )
            )
        assert len(codec.execute(EncodeVideoUnit(config, source))) > 0
    finally:
        mapping.close()


def test_a_closed_session_cannot_be_finalized(codec):
    config = _config()
    session = (1, 9, 0)
    codec.execute(MuxClose(session))
    with pytest.raises(ValueError, match="no assembly session"):
        codec.execute(MuxFinalize(session))
    # A finalize without audio leaves no half-built session behind either.
    pcm = np.zeros((8000, 2), dtype=np.int16)
    mapping = SharedMapping("test-media-ring", pcm.nbytes)
    try:
        codec.attach(mapping)
        codec.execute(
            EncodeAudioTrack(session, config, _write(mapping, 0, pcm))
        )
        with pytest.raises(ValueError, match="every media unit"):
            codec.execute(MuxFinalize(session))
        with pytest.raises(ValueError, match="no assembly session"):
            codec.execute(MuxFinalize(session))
    finally:
        mapping.close()
