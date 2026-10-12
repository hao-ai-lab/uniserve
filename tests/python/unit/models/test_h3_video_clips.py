"""An H3 video clip unpacks to its native latent frames at every duration.

The video decoder reconstructs each clip from seven latent frames. Every
clip of every duration decodes to the same clip output at a canvas, so
decoding resources and graphs are shared across durations, and only
unpacking a clip from the packed latent depends on the frame count.
"""

import pytest
import torch

from tests.python.fixtures.h3 import WIDE, video_vae_config
from uniserve.media import video
from uniserve_models.minimax_h3.config import DenseAttention, SparseAttention
from uniserve_models.minimax_h3.decoding import VideoDecoder
from uniserve_models.minimax_h3.packing import video_latent_frames, video_order

pytestmark = pytest.mark.unit

# Dense packing keeps raster order; tile packing (the FastH3 students) and
# region packing (reference-conditioned students) order rows tile-major.
ATTENTION = {
    "dense": DenseAttention(),
    "tiles": SparseAttention(tile=64, sparsity=0.9),
    "regions": SparseAttention(tile=128, sparsity=0.9, reference_keep=0.5),
}


def _decoder(attention) -> VideoDecoder:
    # Unpacking reads no parameter, so the VAE stays on the meta device.
    with torch.device("meta"):
        return VideoDecoder(video_vae_config(), attention=attention)


def test_every_duration_decodes_one_clip_output_per_canvas():
    decoder = _decoder(ATTENTION["dense"])
    outputs = {
        decoder.clip_output(video.Config(num_frames, WIDE), clip)
        for num_frames in (22, 39, 124, 362)
        for clip in decoder.clips(num_frames)
    }
    assert outputs == {video.Config(25, WIDE)}

    with pytest.raises(ValueError, match="one of the video's clips"):
        decoder.clip_output(video.Config(39, WIDE), slice(0, 22))


@pytest.mark.parametrize("packing", tuple(ATTENTION))
def test_a_clip_unpacks_its_native_latent_frames(packing):
    attention = ATTENTION[packing]
    decoder = _decoder(attention)
    num_frames = 73
    size = video.Config(num_frames, WIDE)
    count = video_latent_frames(num_frames)
    native = torch.randn(1, 24, count, 48, 84)
    # Raster patch rows of 2 x 2 latent pixels, then the denoiser's packed
    # order: packed row i holds raster row video_order[i].
    raster = (
        native.reshape(1, 24, count, 24, 2, 42, 2)
        .permute(0, 2, 3, 5, 1, 4, 6)
        .reshape(-1, 96)
    )
    order = video_order(attention, num_frames=num_frames, canvas=WIDE)
    packed = raster[order]

    for index, clip in enumerate(decoder.clips(num_frames)):
        config = decoder.clip_input(decoder.clip_output(size, clip))
        values = torch.full(config.shape, float("nan"), dtype=config.dtype)
        decoder.unpack_latents(packed, clip, size, out=values)
        # Clip k decodes latent frames [5k, 5k + 7).
        assert torch.equal(values, native[:, :, 5 * index : 5 * index + 7])
