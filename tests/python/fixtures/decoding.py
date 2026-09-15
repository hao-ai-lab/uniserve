"""Numerical temporal projection through the public video decoder contract."""

from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn

from uniserve.media import image
from uniserve.model import EntryPoint, VideoDecoder
from uniserve.nn.vae import LatentDecoder
from uniserve.tensors import OutputLayout


@dataclass(frozen=True)
class Config:
    window: int = 4
    height: int = 1
    width: int = 1


class TemporalProjection(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.projection = nn.Linear(window, window, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.diag(torch.arange(1.0, window + 1)))

    def forward(self, values):
        return self.projection(values.movedim(2, -1)).movedim(-1, 2)


class Decoder(VideoDecoder):
    def __init__(self, config):
        super().__init__(
            LatentDecoder(
                TemporalProjection(config.window),
                latent_shape=(1, 3, config.window, config.height, config.width),
                mean=torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1, 1, 1),
                std=torch.tensor([0.5, 1.5, 2.5]).view(1, 3, 1, 1, 1),
            ),
            frame_size=image.Config(config.height, config.width),
        )
        self.window = config.window

    def frame_slices(self, num_frames):
        if num_frames < 1 or num_frames % self.window:
            raise ValueError("the temporal projection requires complete windows")
        return tuple(
            slice(start, start + self.window) for start in range(0, num_frames, self.window)
        )

    def output_layout(self, num_frames):
        shape = (
            len(self.frame_slices(num_frames)),
            1,
            3,
            self.window,
            self.frame_size.height,
            self.frame_size.width,
        )
        return {"video": OutputLayout(shape, torch.float32, tuple(slice(0, n) for n in shape))}

    def unpack_latents(self, latent, frames, num_frames, *, constants, workspace):
        return latent.T.reshape(1, 3, num_frames, self.frame_size.height, self.frame_size.width)[
            :, :, frames
        ]


class DecodedModel(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.reconstruction = Decoder(config)


def entry_points(config):
    return MappingProxyType({"reconstruction": (EntryPoint("decode"),)})


entry_paths = MappingProxyType({"reconstruction": "reconstruction.decode"})
