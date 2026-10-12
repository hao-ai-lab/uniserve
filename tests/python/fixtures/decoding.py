"""Numerical temporal projection through the public video decoder contract."""

from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn

from uniserve.media import video
from uniserve.model import ComponentEntry, EntryPoint, VideoDecoder
from uniserve.nn.vae import ChannelStatistics, LatentDecoder
from uniserve.tensors import BufferConfig, OutputLayout


@dataclass(frozen=True)
class Config:
    clip_length: int = 4


class TemporalProjection(nn.Module):
    def __init__(self, clip_length):
        super().__init__()
        self.projection = nn.Linear(clip_length, clip_length, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.diag(torch.arange(1.0, clip_length + 1))
            )

    def forward(self, values):
        return self.projection(values.movedim(2, -1)).movedim(-1, 2)


class Decoder(VideoDecoder):
    def __init__(self, config):
        super().__init__(
            LatentDecoder(
                TemporalProjection(config.clip_length),
                # The raster is the decoded video's, so one decoder serves
                # every canvas.
                latent_shape=(1, 3, config.clip_length, None, None),
                normalization=ChannelStatistics(
                    mean=torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1, 1, 1),
                    std=torch.tensor([0.5, 1.5, 2.5]).view(1, 3, 1, 1, 1),
                ),
            ),
        )
        self.clip_length = config.clip_length

    def clips(self, num_frames):
        if num_frames < 1 or num_frames % self.clip_length:
            raise ValueError("the temporal projection requires complete clips")
        return tuple(
            slice(start, start + self.clip_length)
            for start in range(0, num_frames, self.clip_length)
        )

    def output_layout(self, size):
        shape = (
            len(self.clips(size.num_frames)),
            1,
            3,
            self.clip_length,
            size.frame.height,
            size.frame.width,
        )
        return {
            "video": OutputLayout(
                shape, torch.float32, tuple(slice(0, n) for n in shape)
            )
        }

    def clip_output(self, size, clip):
        if clip not in self.clips(size.num_frames):
            raise ValueError("the temporal projection decodes whole clips")
        return video.Config(self.clip_length, size.frame)

    def clip_input(self, output):
        return BufferConfig(
            (
                1,
                3,
                output.num_frames,
                output.frame.height,
                output.frame.width,
            ),
            torch.float32,
        )

    def unpack_latents(self, latent, clip, size, *, out):
        self.clip_output(size, clip)
        out.copy_(
            latent.T.reshape(
                1, 3, size.num_frames, size.frame.height, size.frame.width
            )[:, :, clip]
        )


class DecodedModel(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.reconstruction = Decoder(config)


def entry_points(config):
    return MappingProxyType(
        {
            "reconstruction": ComponentEntry(
                "reconstruction", (EntryPoint("decode"),)
            )
        }
    )
