"""Loaded H3 conditioning encoders reproduce the native diffusers recipe.

The released video and audio VAE encoders load through the public loader and
encode a keyframe, a portrait reference image, a multi-clip reference video
and mono and stereo soundtracks; the native diffusers modules, driven by the
MiniMax-H3 pipeline's own conditioning helpers, give the reference values.
Video units are encoded in separate calls, as separate ranks encode them.
"""

import json
import os
from dataclasses import fields
from pathlib import Path

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
    AutoencoderKLMiniMaxH3,
)
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
    AutoencoderKLMiniMaxH3Audio,
)
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.encoders import (
    encode_vae_condition,
)
from safetensors import safe_open
from torch.nn import functional as F

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve.media import image, video
from uniserve_models.minimax_h3 import audio_vae, video_vae
from uniserve_models.minimax_h3.encoding import AudioEncoder, VideoEncoder

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
    pytest.mark.slow,
]

# The video encoder rounds each posterior sample to FP16 before normalizing
# it, so FP32 differences below that resolution can move a latent by one FP16
# ulp, at most 2**-10 of its magnitude. Normalization maps that to at most
# 2**-10 * (|normalized| + |mean| / std), and the released channel statistics
# keep |mean| < std.
FP16_ULP = 2**-10

_VIDEO_ENCODER = ("encoder.", "quant_conv.")
_AUDIO_ENCODER = ("encoder.", "pre_block.", "mean_proj.")


def _directory(name: str) -> Path:
    root = os.environ.get("UNISERVE_H3_MODEL", "")
    if not root or not (Path(root) / name).is_dir():
        pytest.fail(
            "UNISERVE_H3_MODEL must name a MiniMax-H3 checkpoint directory "
            "with vae/ and audio_vae/"
        )
    return Path(root) / name


def _metadata(directory: Path) -> dict:
    values = json.loads((directory / "config.json").read_text())
    return {key: value for key, value in values.items() if key[0] != "_"}


def _tuples(value):
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    return value


def _tensors(directory: Path) -> dict[str, Path]:
    """Map every checkpoint tensor name to the file that stores it."""
    index = directory / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        return {
            name: directory / shard
            for name, shard in json.loads(index.read_text())[
                "weight_map"
            ].items()
        }
    path = directory / "diffusion_pytorch_model.safetensors"
    with safe_open(path, "pt") as handle:
        return dict.fromkeys(handle.keys(), path)


def _load(factory, config_class, directory: Path, assignments, resident):
    """Load an encoder capability, accounting for every other tensor."""
    metadata = _metadata(directory)
    config = config_class(
        **{
            field.name: _tuples(metadata[field.name])
            for field in fields(config_class)
        }
    )
    names = _tensors(directory)

    def mapping(model):
        return (
            weights.ModuleMapping(
                model,
                "primary",
                lambda reader: assignments(model.encoder, reader),
                frozenset(name for name, _ in model.named_parameters()),
                nonresident=frozenset(
                    name for name in names if not name.startswith(resident)
                ),
            ),
        )

    return loading.load_model(
        factory,
        config,
        checkpoint=(
            checkpoint.Config().resolve(directory, io=loading.Config()),
        ),
        mapping=mapping,
        device="cuda",
        weights=weights.Config(dtype=torch.float32),
    ).model


def _native(model_class, directory: Path, resident):
    """Materialize only the native encoder fields, on the GPU."""
    with torch.device("meta"):
        model = model_class(**_metadata(directory))
    state = {}
    for name, path in _tensors(directory).items():
        if name.startswith(resident):
            with safe_open(path, "pt", device="cuda") as handle:
                state[name] = handle.get_tensor(name)
    model.load_state_dict(state, strict=False, assign=True)
    return model.eval()


@pytest.fixture(scope="module")
def video_encoders():
    directory = _directory("vae")
    return (
        _load(
            VideoEncoder,
            video_vae.Config,
            directory,
            video_vae.encoder_assignments,
            _VIDEO_ENCODER,
        ),
        _native(AutoencoderKLMiniMaxH3, directory, _VIDEO_ENCODER),
    )


@pytest.fixture(scope="module")
def audio_encoders():
    directory = _directory("audio_vae")
    return (
        _load(
            lambda config: AudioEncoder(config, sample_rate=32000),
            audio_vae.Config,
            directory,
            audio_vae.encoder_assignments,
            _AUDIO_ENCODER,
        ),
        # The native encode also evaluates the log-scale head.
        _native(
            AutoencoderKLMiniMaxH3Audio,
            directory,
            (*_AUDIO_ENCODER, "logs_proj."),
        ),
    )


def _frames(num_frames: int, height: int, width: int, seed: int):
    """Return ``[frames, height, width, 3]`` uint8 frames on the GPU.

    Upsampled coarse noise gives the dominant low frequencies of natural
    images, and a little per-pixel noise adds fine texture.
    """
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(
        (num_frames, 3, height // 16, width // 16), generator=generator
    )
    fine = torch.rand((num_frames, 3, height, width), generator=generator)
    values = (
        F.interpolate(coarse, size=(height, width), mode="bilinear") * 0.9
        + fine * 0.1
    )
    pixels = values.mul(255).round().to(torch.uint8)
    return pixels.permute(0, 2, 3, 1).contiguous().cuda()


@pytest.mark.parametrize(
    ("num_frames", "height", "width"),
    (
        # A keyframe on the 16:9 canvas and a portrait reference image take
        # the single-frame path; 39 frames take two clips and a padded third.
        (1, 768, 1344),
        (1, 1344, 768),
        (39, 768, 1344),
    ),
)
def test_video_conditioning_matches_native_encoding(
    video_encoders, num_frames, height, width
):
    encoder, native = video_encoders
    pixels = _frames(num_frames, height, width, seed=num_frames + height)
    with torch.inference_mode():
        expected = patchify_video_latents(
            encode_vae_condition(
                native,
                pixels.permute(3, 0, 1, 2).unsqueeze(0),
                video_vae.PIXEL_MEAN,
                video_vae.PIXEL_STD,
            ),
            (1, 2, 2),
        )

    # Every unit is encoded in its own call, last first, and placed where its
    # layout says its rows belong.
    layout = encoder.output_layout(
        video.Config(num_frames, image.Config(height, width))
    )["video"]
    assembled = torch.full(layout.shape, torch.nan, device="cuda")
    for unit in reversed(encoder.frame_slices(num_frames)):
        (result,) = encoder.encode(
            (pixels[unit],), frames=(unit,), num_frames=(num_frames,)
        )
        assembled[result.layout.local_slice] = result.tensor
    torch.testing.assert_close(
        assembled.cpu(), expected, rtol=FP16_ULP, atol=FP16_ULP
    )


def _track(num_samples: int, channels: int, seed: int):
    """Return ``[samples, channels]`` float32 PCM on the GPU.

    Each channel mixes a few tones of random pitch and phase with a little
    noise, within [-0.8, 0.8].
    """
    generator = torch.Generator().manual_seed(seed)
    time = torch.arange(num_samples, dtype=torch.float64) / 32000
    pitch = 80 + 2000 * torch.rand((channels, 4, 1), generator=generator)
    phase = 2 * torch.pi * torch.rand((channels, 4, 1), generator=generator)
    tones = torch.sin(2 * torch.pi * pitch * time + phase).mean(dim=1)
    noise = torch.rand((channels, num_samples), generator=generator) - 0.5
    track = (0.7 * tones + 0.2 * noise).to(torch.float32)
    return track.T.contiguous().cuda()


@pytest.mark.parametrize("channels", (1, 2))
def test_audio_conditioning_matches_native_posterior_mean(
    audio_encoders, channels
):
    encoder, native = audio_encoders
    # Five seconds and a partial latent frame of 123 samples.
    track = _track(5 * 32000 + 123, channels, seed=channels)
    waveform = track.T.expand(2, -1).contiguous()
    with torch.inference_mode():
        posterior = native.encode(waveform[:, None]).latent_dist
        latents = posterior.mode().float().cpu().transpose(1, 2)
    config = encoder.config
    expected = (
        (latents - torch.tensor(config.latents_mean).view(1, 1, -1))
        / torch.tensor(config.latents_std).view(1, 1, -1)
    ).reshape(-1, config.latent_channels)

    (actual,) = encoder.encode((track,))
    assert actual.shape == (2 * 201, 32)
    # The audio latent stays FP32 end to end; this is the FP32 tolerance of
    # the VAE loading parity tests.
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
