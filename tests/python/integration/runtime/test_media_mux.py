"""Encoded media preserves declared frame order and dimensions.

It also preserves output clocks.
"""

import io

import av
import numpy as np
import pytest

from uniserve_worker.media.mux import AvMuxConfig, AvMuxSession

pytestmark = pytest.mark.integration


def test_audio_and_ordered_video_units_form_a_decodable_mp4():
    config = AvMuxConfig(
        width=32,
        height=16,
        frame_count=6,
        frame_rate=24,
        audio_rate=32000,
        video_unit_frames=(4, 2),
    )
    session = AvMuxSession(config)
    pcm = np.zeros((8000, 2), dtype=np.int16)
    session.write_audio(pcm)
    red = np.zeros((4, 16, 32, 3), dtype=np.uint8)
    red[..., 0] = 255
    blue = np.zeros((2, 16, 32, 3), dtype=np.uint8)
    blue[..., 2] = 255
    session.write_video(0, 1, red)
    session.write_video(1, 1, blue)
    encoded = session.close()

    with av.open(io.BytesIO(encoded)) as container:
        video, audio = container.streams.video[0], container.streams.audio[0]
        assert (video.width, video.height, video.average_rate) == (32, 16, 24)
        assert (audio.sample_rate, audio.layout.name) == (32000, "stereo")
        assert (
            abs(float(audio.duration * audio.time_base) - 6 / 24)
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
        audio = tuple(container.decode(audio=0))
        assert sum(frame.samples for frame in audio) >= 8000
