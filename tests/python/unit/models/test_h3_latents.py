"""Native CPU random draws map to the same H3 samples.

The mapping holds under sequence sharding, for both packings and for every
canvas a denoiser generates.
"""

import pytest
import torch

from tests.python.fixtures.h3 import WIDE, base_denoiser, dmd_denoiser
from uniserve.diffusion import normal_noise
from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.media import image
from uniserve.nn.attention import AttentionParallelConfig, Ulysses
from uniserve.runtime import TensorBuffers
from uniserve_models.minimax_h3.denoiser import Denoiser
from uniserve_models.minimax_h3.inputs import DenoiserSize

pytestmark = pytest.mark.unit

TALL = image.Config(1344, 768)
# The 480p 21:9 training bucket: 13x31 patches per frame, so tiles at its
# bottom and right edges are partial.
NARROW = image.Config(416, 992)


def _tile_order(video: torch.Tensor) -> torch.Tensor:
    """Canonical tile rows of a draw.

    Rows visit 4x4x4 tiles and valid positions within each tile in temporal,
    height, width order; a tile at an edge holds only its valid positions.
    Patch channels retain native channel, patch-height, patch-width order.
    """
    _, frames, height, width = video.shape
    rows, columns = height // 2, width // 2
    patches = video.reshape(24, frames, rows, 2, columns, 2).permute(
        1, 2, 4, 0, 3, 5
    )
    return torch.cat(
        [
            patches[t : t + 4, h : h + 4, w : w + 4].reshape(-1, 96)
            for t in range(0, frames, 4)
            for h in range(0, rows, 4)
            for w in range(0, columns, 4)
        ]
    )


def _raster_order(video: torch.Tensor) -> torch.Tensor:
    """Canonical raster rows: latent frame, patch row, patch column."""
    _, frames, height, width = video.shape
    patches = video.reshape(24, frames, height // 2, 2, width // 2, 2)
    return patches.permute(1, 2, 4, 0, 3, 5).reshape(-1, 96)


@pytest.mark.parametrize(
    "config,canvas,order",
    [
        (dmd_denoiser(), WIDE, _tile_order),
        (dmd_denoiser(), TALL, _tile_order),
        (dmd_denoiser(), NARROW, _tile_order),
        (base_denoiser(), WIDE, _raster_order),
        (base_denoiser(), TALL, _raster_order),
    ],
    ids=("tile-wide", "tile-tall", "tile-narrow", "dense-wide", "dense-tall"),
)
@pytest.mark.parametrize(
    "frames,seed,tokens",
    [
        (22, 0, 63),
        (22, 0, 64),
        (22, 0, 65),
        (39, 1000, 128),
        (124, 19, 10000),
        (362, 23, 1000),
    ],
)
def test_native_draws_and_canonical_shards(
    config, canvas, order, frames, seed, tokens
):
    size = DenoiserSize(frames, canvas, tokens, 0)
    ranks = (7, 3, 5, 1, 6, 2, 4, 0)
    outputs = {"video": [], "audio": []}
    noise = None
    for rank in ranks:
        with torch.device("meta"):
            model = Denoiser(config)
        mesh = DeviceMesh(ranks=ranks, shape=(8,), axes=("tokens",), rank=rank)
        parallelize_(
            model.transformer,
            mesh,
            attention=AttentionParallelConfig(heads=Ulysses("tokens")),
        )
        declarations = model.constant_buffers(size)
        with TensorBuffers.allocate(declarations, device="cpu") as owner:
            constants = owner.view(declarations)
            model.prepare_constants(size, out=constants)
            if noise is None:
                noise = {
                    name: torch.empty(
                        (1, *model.noise_shape(name, size)), dtype=torch.float32
                    )
                    for name in model.modalities
                }
                normal_noise((seed,), out=tuple(noise.values()))
                generator = torch.Generator().manual_seed(seed)
                video = torch.randn(
                    model.noise_shape("video", size), generator=generator
                )
                audio = torch.randn(
                    model.noise_shape("audio", size), generator=generator
                )
                torch.testing.assert_close(
                    noise["video"][0], video, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    noise["audio"][0], audio, rtol=0, atol=0
                )
                expected_video = order(video[0])
            buffers = model.state_buffers(size)
            with TensorBuffers.allocate(buffers, device="cpu") as state_owner:
                state = {
                    name: value.unsqueeze(0)
                    for name, value in state_owner.view(buffers).items()
                    if name in model.modalities
                }
                model.prepare_latents(
                    (size,),
                    noise=noise,
                    state=state,
                    constants=constants,
                    workspace={},
                )
                for name, value in state.items():
                    layout = model.output_layout(size)[name]
                    assert tuple(value[0].shape) == tuple(
                        part.stop - part.start for part in layout.local_slice
                    )
                    expected = expected_video if name == "video" else audio
                    torch.testing.assert_close(
                        value[0], expected[layout.local_slice], rtol=0, atol=0
                    )
                    outputs[name].append(value[0].clone())
    torch.testing.assert_close(
        torch.cat(outputs["video"]), expected_video, rtol=0, atol=0
    )
    torch.testing.assert_close(
        torch.cat(outputs["audio"]), audio, rtol=0, atol=0
    )


def test_dmd_export_generates_only_its_training_buckets():
    with torch.device("meta"):
        model = Denoiser(dmd_denoiser())
    for canvas in (WIDE, TALL, NARROW):
        model.make_size(124, 64, canvas=canvas)
    # 3:2 at the 768 short edge follows the canvas rule, but no bucket has it.
    with pytest.raises(ValueError, match="generates only"):
        model.make_size(124, 64, canvas=image.Config(768, 1152))
