"""Native CPU random draws map to the same H3 samples under sequence sharding."""

import pytest
import torch

from uniserve.diffusion import normal_noise
from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.nn.attention import AttentionParallelConfig, Ulysses
from uniserve.runtime import TensorBuffers
from uniserve_models.minimax_h3.config import DiffusionConfig, TransformerConfig
from uniserve_models.minimax_h3.denoiser import Denoiser
from uniserve_models.minimax_h3.inputs import DenoiserSize

pytestmark = pytest.mark.unit


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
def test_native_draws_and_canonical_shards(frames, seed, tokens):
    size = DenoiserSize(frames, tokens)
    ranks = (7, 3, 5, 1, 6, 2, 4, 0)
    outputs = {"video": [], "audio": []}
    noise = None
    for rank in ranks:
        with torch.device("meta"):
            model = Denoiser(TransformerConfig(), DiffusionConfig())
        mesh = DeviceMesh(ranks=ranks, shape=(8,), axes=("tokens",), rank=rank)
        parallelize_(
            model.transformer, mesh, attention=AttentionParallelConfig(heads=Ulysses("tokens"))
        )
        declarations = model.constant_buffers(size)
        with TensorBuffers.allocate(declarations, device="cpu") as owner:
            constants = owner.view(declarations)
            model.prepare_constants(size, out=constants)
            if noise is None:
                noise = {
                    name: torch.empty((1, *model.noise_shape(name, size)), dtype=torch.float32)
                    for name in model.modalities
                }
                normal_noise((seed,), out=tuple(noise.values()))
                generator = torch.Generator().manual_seed(seed)
                video = torch.randn(model.noise_shape("video", size), generator=generator)
                audio = torch.randn(model.noise_shape("audio", size), generator=generator)
                torch.testing.assert_close(noise["video"][0], video, rtol=0, atol=0)
                torch.testing.assert_close(noise["audio"][0], audio, rtol=0, atol=0)
                # Canonical rows visit 4x4x4 tiles and valid positions within
                # each tile in temporal, height, width order. Patch channels
                # retain native channel, patch-height, patch-width order.
                patches = video.reshape(24, video.shape[2], 24, 2, 42, 2).permute(1, 2, 4, 0, 3, 5)
                expected_video = torch.cat(
                    [
                        patches[t : t + 4, h : h + 4, w : w + 4].reshape(-1, 96)
                        for t in range(0, video.shape[2], 4)
                        for h in range(0, 24, 4)
                        for w in range(0, 42, 4)
                    ]
                )
            buffers = model.state_buffers(size)
            with TensorBuffers.allocate(buffers, device="cpu") as state_owner:
                state = {
                    name: value.unsqueeze(0) for name, value in state_owner.view(buffers).items()
                }
                model.prepare_latents(
                    (size,), noise=noise, state=state, constants=constants, workspace={}
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
    torch.testing.assert_close(torch.cat(outputs["video"]), expected_video, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(outputs["audio"]), audio, rtol=0, atol=0)
    torch.testing.assert_close(noise["video"][0], video, rtol=0, atol=0)
    torch.testing.assert_close(noise["audio"][0], audio, rtol=0, atol=0)
