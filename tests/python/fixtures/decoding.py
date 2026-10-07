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
    window: int = 4


class TemporalProjection(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.projection = nn.Linear(window, window, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.diag(torch.arange(1.0, window + 1))
            )

    def forward(self, values):
        return self.projection(values.movedim(2, -1)).movedim(-1, 2)


class Decoder(VideoDecoder):
    def __init__(self, config):
        super().__init__(
            LatentDecoder(
                TemporalProjection(config.window),
                # The raster is the decoded video's, so one decoder serves
                # every canvas.
                latent_shape=(1, 3, config.window, None, None),
                normalization=ChannelStatistics(
                    mean=torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1, 1, 1),
                    std=torch.tensor([0.5, 1.5, 2.5]).view(1, 3, 1, 1, 1),
                ),
            ),
        )
        self.window = config.window

    def frame_slices(self, num_frames):
        if num_frames < 1 or num_frames % self.window:
            raise ValueError(
                "the temporal projection requires complete windows"
            )
        return tuple(
            slice(start, start + self.window)
            for start in range(0, num_frames, self.window)
        )

    def output_layout(self, size):
        shape = (
            len(self.frame_slices(size.num_frames)),
            1,
            3,
            self.window,
            size.frame.height,
            size.frame.width,
        )
        return {
            "video": OutputLayout(
                shape, torch.float32, tuple(slice(0, n) for n in shape)
            )
        }

    def segment(self, size, frames):
        if frames not in self.frame_slices(size.num_frames):
            raise ValueError("the temporal projection decodes whole windows")
        return video.Config(self.window, size.frame)

    def window_input(self, segment):
        return BufferConfig(
            (
                1,
                3,
                segment.num_frames,
                segment.frame.height,
                segment.frame.width,
            ),
            torch.float32,
        )

    def unpack_latents(self, latent, frames, size, *, out):
        self.segment(size, frames)
        out.copy_(
            latent.T.reshape(
                1, 3, size.num_frames, size.frame.height, size.frame.width
            )[:, :, frames]
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
