"""Independently encoded media units assemble into one decodable artifact.

Each media unit is encoded where it was reconstructed and the audio track is
encoded on the muxer rank, so assembly concatenates already-encoded tracks
without re-encoding and must still preserve frame order, dimensions and clocks.
"""

import io

import av
import numpy as np
import pytest

from uniserve_worker.media.codec_process import (
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


def test_a_product_row_carries_one_encoded_unit_and_its_length():
    import torch

    config = _config()
    payload = encode_video_unit(
        config, np.zeros((4, 16, 32, 3), dtype=np.uint8)
    )
    row = torch.zeros(
        encoded_unit_bytes(4, config.height, config.width), dtype=torch.uint8
    )
    frame_encoded_unit(payload, row)
    assert read_encoded_unit(row) == payload


def test_a_unit_that_exceeds_its_reserved_row_fails_by_name():
    import torch

    config = _config()
    payload = encode_video_unit(
        config, np.zeros((4, 16, 32, 3), dtype=np.uint8)
    )
    with pytest.raises(Exception, match="exceeds the"):
        frame_encoded_unit(payload, torch.zeros(16, dtype=torch.uint8))
