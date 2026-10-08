"""H3 conditioning latents from host-decoded frames and PCM.

``VideoEncoder`` turns keyframes, reference images and reference videos into
the normalized latent rows the denoiser conditions on; every 17-frame clip of
a video is a temporal unit that one rank encodes on its own. ``AudioEncoder``
turns reference soundtracks into channel-major stereo latent rows. Both
reproduce the conditioning recipe of the diffusers MiniMax-H3 pipeline.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping

import torch
from torch.nn import functional as F

from uniserve.diffusion import normal_noise
from uniserve.media import image, video
from uniserve.model import AudioEncoder as BaseAudioEncoder
from uniserve.model import VideoEncoder as BaseVideoEncoder
from uniserve.nn.vae import ChannelStatistics, DiagonalGaussian, LatentEncoder
from uniserve.tensors import OutputLayout

from . import audio_vae, video_vae
from .packing import patchify_video

# The reference samples every visual condition's posterior under this seed,
# independently of the request's seed, so a condition always encodes to the
# same latent.
POSTERIOR_SEED = 42

# Latent rows are the denoiser's (1, 2, 2) patches of the latent raster.
_PATCH = 2


@functools.lru_cache(maxsize=16)
def _posterior_draw(
    channels: int, frames: int, height: int, width: int
) -> torch.Tensor:
    """Return the host FP32 posterior draw of one latent shape.

    One standard normal draw of ``[1, channels, frames, height, width]``
    from a fresh generator seeded with ``POSTERIOR_SEED``: the draw is a
    function of the shape alone, so every encoding round and request of a
    condition size shares it instead of drawing it again on the host. The
    shared tensor is read-only; callers copy the slices they use. Sixteen
    shapes bound the cache at about 230 MB for five-second 16:9 videos.
    """
    draw = torch.empty(
        (1, 1, channels, frames, height, width),
        dtype=torch.float32,
        device="cpu",
    )
    normal_noise((POSTERIOR_SEED,), out=(draw,))
    return draw[0]


class VideoEncoder(BaseVideoEncoder):
    """Encode H3 visual conditions into normalized latent patch rows.

    A video of two or more frames is encoded in clips of ``clip_length``
    frames, a shorter final clip repeating its last frame to the full
    length. A clip yields ``ceil`` of its length through each temporal
    stride, five latent frames for H3, and the video drops the final clip's
    trailing ``token_drop`` of them: ``17 * n + 5`` frames encode to
    ``5 * n + 2`` latent frames. A single frame (an image or a keyframe) runs
    through the same network alone and yields one latent frame. Clips share
    no state, because causal padding restarts at every clip and
    normalization never mixes frames, so every clip is one unit. A still
    frame's row groups are its rows of 2x2 patches, so its bands are runs of
    whole patch rows, which its tiled encoder encodes on their own.

    The posterior is sampled with a host FP32 draw seeded with
    ``POSTERIOR_SEED`` over the condition's complete ``[1, channels, frames,
    height / 16, width / 16]`` latent, rounded to FP16 and normalized by the
    latent channel statistics. Rows follow the denoiser's raster patch order:
    frame-major, then row-major 2x2 patches with channel-major features.
    """

    pixel_mean: torch.Tensor
    pixel_std: torch.Tensor

    # A row of 2x2 patches spans two latent rows.
    row_group = _PATCH

    def __init__(self, config: video_vae.Config):
        # Each causal stride maps T frames to ceil(T / stride).
        span = config.clip_length
        for stride in config.temporal_downsample_factors:
            span = math.ceil(span / stride)
        if config.token_drop >= span:
            raise ValueError(
                "H3 video token_drop must leave latent frames in a clip"
            )
        channels = config.latent_channels
        super().__init__(
            LatentEncoder(
                video_vae.Encoder(config),
                normalization=ChannelStatistics(
                    mean=torch.tensor(
                        config.latents_mean, dtype=torch.float32, device="cpu"
                    ).view(1, channels, 1, 1, 1),
                    std=torch.tensor(
                        config.latents_std, dtype=torch.float32, device="cpu"
                    ).view(1, channels, 1, 1, 1),
                    latent_dtype=torch.float16,
                ),
                posterior=DiagonalGaussian(log_variance_range=(-30.0, 20.0)),
            )
        )
        # Latent frames one complete clip yields.
        self.config, self.clip_latents = config, span
        for name, values in (
            ("pixel_mean", video_vae.PIXEL_MEAN),
            ("pixel_std", video_vae.PIXEL_STD),
        ):
            self.register_buffer(
                name,
                torch.tensor(values, dtype=torch.float32, device="cpu").view(
                    1, 3, 1, 1, 1
                ),
                persistent=False,
            )

    def _latent_frames(self, num_frames: int) -> int:
        if type(num_frames) is not int or num_frames < 1:
            raise ValueError("H3 conditions must have a positive frame count")
        if num_frames == 1:
            return 1
        clips = math.ceil(num_frames / self.config.clip_length)
        return clips * self.clip_latents - self.config.token_drop

    def latent_size(self, frame: image.Config) -> tuple[int, int]:
        # Latent rows are whole 2x2 patches of the latent raster.
        alignment = self.config.spatial_compression * _PATCH
        if frame.height % alignment or frame.width % alignment:
            raise ValueError(
                f"H3 condition frames must align with {alignment}-pixel "
                "latent patches"
            )
        return (
            frame.height // self.config.spatial_compression,
            frame.width // self.config.spatial_compression,
        )

    def frame_slices(self, num_frames: int) -> tuple[slice, ...]:
        self._latent_frames(num_frames)
        clip = self.config.clip_length
        return tuple(
            slice(start, min(start + clip, num_frames))
            for start in range(0, num_frames, clip)
        )

    def latent_slices(self, num_frames: int) -> tuple[slice, ...]:
        total = self._latent_frames(num_frames)
        span = self.clip_latents
        return tuple(
            slice(index * span, min((index + 1) * span, total))
            for index in range(len(self.frame_slices(num_frames)))
        )

    def output_layout(self, size: video.Config) -> Mapping[str, OutputLayout]:
        height, width = self.latent_size(size.frame)
        rows = (
            self._latent_frames(size.num_frames)
            * (height // _PATCH)
            * (width // _PATCH)
        )
        shape = (rows, self.config.latent_channels * _PATCH * _PATCH)
        return {
            "video": OutputLayout(
                shape,
                torch.float32,
                tuple(slice(0, extent) for extent in shape),
                variable_axes=(0,),
            )
        }

    def posterior_noise(self, size: video.Config) -> torch.Tensor:
        # One standard normal FP32 draw of the complete latent from a fresh
        # host generator, which is the reference's draw; the shape fixes how
        # the generator's stream maps onto latent positions.
        height, width = self.latent_size(size.frame)
        return _posterior_draw(
            self.config.latent_channels,
            self._latent_frames(size.num_frames),
            height,
            width,
        )

    def unpack_pixels(
        self, pixels: torch.Tensor, frames: slice, num_frames: int
    ) -> torch.Tensor:
        # [frames, height, width, 3] uint8 -> [1, 3, frames, height, width],
        # ImageNet-normalized over the [0, 1] range.
        values = pixels.permute(3, 0, 1, 2).unsqueeze(0)
        values = (
            values.to(torch.float32).div(255.0) - self.pixel_mean
        ) / self.pixel_std
        # A shorter final clip repeats its last frame to the clip length; a
        # single frame stays alone.
        missing = self.config.clip_length - values.shape[2]
        if num_frames > 1 and missing > 0:
            values = torch.cat(
                (values, values[:, :, -1:].expand(-1, -1, missing, -1, -1)),
                dim=2,
            )
        return values

    def pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return patchify_video(latents, (1, _PATCH, _PATCH))[0]


class AudioEncoder(BaseAudioEncoder):
    """Encode H3 reference soundtracks into channel-major stereo latent rows.

    A mono track is duplicated to stereo. Both channels are encoded as a
    batch of two mono waveforms right-padded with zeros to whole latent
    frames; the posterior mean is normalized by the latent channel statistics
    and laid out as every frame of the left channel, then every frame of the
    right: ``[2 * frames, channels]``, the order of the denoiser's audio
    rows.
    """

    def __init__(self, config: audio_vae.Config, *, sample_rate: int):
        channels = config.latent_channels
        super().__init__(
            LatentEncoder(
                audio_vae.Encoder(config),
                normalization=ChannelStatistics(
                    mean=torch.tensor(
                        config.latents_mean, dtype=torch.float32, device="cpu"
                    ).view(1, channels, 1),
                    std=torch.tensor(
                        config.latents_std, dtype=torch.float32, device="cpu"
                    ).view(1, channels, 1),
                ),
            ),
            sample_rate=sample_rate,
        )
        self.config = config

    @property
    def latent_rate(self) -> int:
        return math.prod(self.config.encoder_rates)

    def latent_frames(self, num_samples: int) -> int:
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError(
                "audio duration must contain a positive sample count"
            )
        return math.ceil(num_samples / self.latent_rate)

    def output_layout(self, num_samples: int) -> Mapping[str, OutputLayout]:
        shape = (
            2 * self.latent_frames(num_samples),
            self.config.latent_channels,
        )
        return {
            "audio": OutputLayout(
                shape,
                torch.float32,
                tuple(slice(0, extent) for extent in shape),
                variable_axes=(0,),
            )
        }

    def unpack_samples(self, samples: torch.Tensor) -> torch.Tensor:
        if samples.shape[1] not in (1, 2):
            raise ValueError("H3 reference audio is mono or stereo")
        # [samples, channels] -> [stereo, 1, samples] mono batch rows.
        waveform = samples.to(torch.float32).transpose(0, 1)
        waveform = waveform.expand(2, -1).unsqueeze(1)
        padding = (
            self.latent_frames(int(samples.shape[0])) * self.latent_rate
            - samples.shape[0]
        )
        return F.pad(waveform, (0, padding))

    def pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # [stereo, channels, frames] -> [stereo * frames, channels].
        return latents.transpose(1, 2).reshape(-1, latents.shape[1])
