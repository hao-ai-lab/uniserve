"""Observable CPU reference decoding and frame-selection contracts."""

import io
from fractions import Fraction

import av
import numpy as np
import pytest
from PIL import Image

from uniserve_worker.models.minimax_h3.reference_media import (
    MAX_SOURCE_BYTES,
    prepare_reference_image,
    prepare_reference_video,
    reference_canvas,
    sample_reference_video_frames,
    trim_reference_num_frames,
)


def video_source(*, count=39, fps=24, layout="stereo", audio_rate=32000):
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width = stream.height = 32
        stream.pix_fmt = "bgr0"
        audio = None
        if layout is not None:
            audio = container.add_stream("pcm_f32le", rate=audio_rate)
            audio.layout = layout
        for index in range(count):
            pixels = np.full((32, 32, 3), index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        if audio is not None:
            channels = len(av.AudioLayout(layout).channels)
            waveform = np.full((channels, round(count / fps * audio_rate)), 0.25, np.float32)
            if channels > 2:
                # Only center/surround channels carry signal. Dropping all but
                # the front pair would incorrectly turn this soundtrack silent.
                waveform[:2] = 0
            frame = av.AudioFrame.from_ndarray(waveform, format="fltp", layout=layout)
            frame.sample_rate = audio_rate
            frame.pts = 0
            frame.time_base = Fraction(1, audio_rate)
            for packet in audio.encode(frame):
                container.mux(packet)
            for packet in audio.encode(None):
                container.mux(packet)
    return buffer.getvalue()


@pytest.mark.parametrize("layout", ["mono", "stereo", "5.1", None])
def test_video_canvas_window_and_embedded_audio(layout):
    prepared = prepare_reference_video(video_source(layout=layout), num_frames=30)
    assert prepared.frames.shape == (30, 768, 768, 3)
    assert prepared.frames.dtype == np.uint8
    np.testing.assert_array_equal(prepared.frames[:, 0, 0, 0], np.arange(30))
    assert prepared.vae_frames.shape == (22, 768, 768, 3)
    sampled, timestamps = sample_reference_video_frames(prepared.frames)
    np.testing.assert_array_equal(sampled[:, 0, 0, 0], [0, 12, 24])
    assert timestamps == (0.25, 1.0)
    if layout is None:
        assert prepared.waveform is None
    else:
        assert prepared.waveform.shape == (2, 40000)
        assert prepared.waveform.dtype == np.float32
        assert np.isfinite(prepared.waveform).all()
        assert np.count_nonzero(prepared.waveform) == prepared.waveform.size
        if layout != "5.1":
            np.testing.assert_array_equal(prepared.waveform, 0.25)


@pytest.mark.parametrize(
    "fps,count,expected", [(12, 11, np.repeat(np.arange(11), 2)), (48, 44, np.arange(0, 44, 2))]
)
def test_nearest_24fps_timeline(fps, count, expected):
    prepared = prepare_reference_video(
        video_source(fps=fps, count=count, layout=None), num_frames=22
    )
    np.testing.assert_array_equal(prepared.frames[:, 0, 0, 0], expected)


def test_non_native_audio_rate():
    prepared = prepare_reference_video(video_source(count=22, audio_rate=48000), num_frames=22)
    assert prepared.waveform.shape == (2, 29334)
    # Constant stereo channels stay identical through the released CPU filter.
    np.testing.assert_array_equal(prepared.waveform[0], prepared.waveform[1])
    assert abs(float(prepared.waveform[:, 100:-100].mean()) - 0.25) < 0.001


@pytest.mark.parametrize("frames", [0, 21, 346])
def test_frame_budget_rejects_before_decode(frames):
    with pytest.raises(ValueError, match="frame budget"):
        prepare_reference_video(b"not decoded", num_frames=frames)


def test_short_video_is_not_padded_into_a_vae_chunk():
    with pytest.raises(ValueError, match="at least 22"):
        prepare_reference_video(video_source(count=21, layout=None), num_frames=22)


def test_source_budget():
    with pytest.raises(ValueError, match="encoded media"):
        prepare_reference_image(b"x" * (MAX_SOURCE_BYTES + 1))


def test_image_canvas_and_color():
    buffer = io.BytesIO()
    Image.new("RGB", (96, 64), (10, 20, 30)).save(buffer, format="PNG")
    image = prepare_reference_image(buffer.getvalue())
    assert image.shape == (2048, 3072, 3)
    np.testing.assert_array_equal(image[0, 0], [10, 20, 30])
    # The 768x1344 area ceiling applies before rounding each edge to 32.
    assert reference_canvas(16, 9, video=True) == (768, 1344)
    assert reference_canvas(21, 9, video=True) == (672, 1536)
    with pytest.raises(ValueError, match="aspect"):
        reference_canvas(5, 1, video=False)


@pytest.mark.parametrize("source,trimmed", [(22, 22), (38, 22), (39, 39), (124, 124), (361, 345)])
def test_complete_reference_chunks(source, trimmed):
    assert trim_reference_num_frames(source) == trimmed
