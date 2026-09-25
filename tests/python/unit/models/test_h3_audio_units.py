"""An audio media unit decoded with its halo equals the whole-track decode.

Section 5.5 of the serving architecture plan distributes audio reconstruction
by media unit. The claim it rests on is that the decoder is convolutional, so a
unit decoded with the decoder's receptive field of context on each side and
trimmed is identical to decoding the whole track and taking the same samples.
These tests hold the decoder to that, and hold the derived halo to being a
context that actually covers the receptive field.
"""

import pytest
import torch

from tests.python.fixtures.audio_decoding import CONFIG
from tests.python.fixtures.audio_decoding import decoder as _decoder
from uniserve_models.minimax_h3.decoding import AudioDecoder

pytestmark = pytest.mark.unit


def _track(decoder: AudioDecoder, frames: int, trim: int):
    """Return a latent timeline and the sample count it decodes to."""
    num_samples = frames * decoder.latent_rate - trim
    latent = torch.randn(
        2 * decoder.latent_frames(num_samples), CONFIG.latent_channels
    )
    workspace = {
        "audio_latents": torch.zeros(2, CONFIG.latent_channels, frames)
    }
    return latent, num_samples, workspace


@pytest.mark.parametrize("units", [2, 3, 5])
@pytest.mark.parametrize("trim", [0, 3])
def test_media_units_reproduce_the_whole_track_decode(units, trim):
    decoder = _decoder()
    frames = 6 * units
    latent, num_samples, workspace = _track(decoder, frames, trim)

    whole = decoder.decode(
        (latent,),
        frames=(slice(0, frames),),
        num_samples=(num_samples,),
        workspace=workspace,
    )[0]
    windows = decoder.unit_frames(num_samples, units)
    pieces = [
        decoder.decode(
            (latent,),
            frames=(window,),
            num_samples=(num_samples,),
            workspace=workspace,
        )[0]
        for window in windows
    ]

    assert sum(piece.shape[0] for piece in pieces) == num_samples
    assert torch.equal(torch.cat(pieces, dim=0), whole)


@pytest.mark.parametrize("units", [2, 3, 5])
@pytest.mark.parametrize("trim", [0, 3])
def test_media_units_cover_the_sample_timeline_without_overlap(units, trim):
    decoder = _decoder()
    frames = 6 * units
    _latent, num_samples, _workspace = _track(decoder, frames, trim)

    spans = decoder.unit_samples(num_samples, units)
    assert spans[0].start == 0
    assert spans[-1].stop == num_samples
    for earlier, later in zip(spans, spans[1:], strict=False):
        assert earlier.stop == later.start


def test_media_units_must_lie_within_the_latent_timeline():
    decoder = _decoder()
    frames = 12
    latent, num_samples, workspace = _track(decoder, frames, 0)

    with pytest.raises(ValueError, match="within its latent timeline"):
        decoder.decode(
            (latent,),
            frames=(slice(frames - 2, frames + 2),),
            num_samples=(num_samples,),
            workspace=workspace,
        )
    with pytest.raises(ValueError, match="divide the latent timeline"):
        decoder.unit_frames(num_samples, frames + 1)
