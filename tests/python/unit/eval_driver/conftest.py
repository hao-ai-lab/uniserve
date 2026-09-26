"""Fixtures shared by the evaluator driver tests."""

from __future__ import annotations

import io
from fractions import Fraction

import av
import numpy as np
import pytest


@pytest.fixture(scope="session")
def media():
    """Encoded fixture with the real H3 shape, audio and frame contract."""
    result = io.BytesIO()
    frames = 107  # Four requested seconds align upward to 107 frames.
    with av.open(result, "w", format="mp4") as container:
        video = container.add_stream("libx264", rate=24)
        video.width, video.height, video.pix_fmt = 1344, 768, "yuv420p"
        video.options = {"preset": "ultrafast", "crf": "35"}
        audio = container.add_stream("aac", rate=32000)
        audio.layout = "stereo"
        pixels = np.zeros((768, 1344, 3), dtype=np.uint8)
        pixels[:, :672] = 180
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        samples = round(frames / 24 * 32000)
        signal = (
            0.1 * np.sin(np.arange(samples) * 2 * np.pi * 440 / 32000)
        ).astype("float32")
        frame = av.AudioFrame.from_ndarray(
            np.stack([signal, signal]), format="fltp", layout="stereo"
        )
        frame.sample_rate = 32000
        frame.pts = 0
        frame.time_base = Fraction(1, 32000)
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    return result.getvalue()
