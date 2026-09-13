"""Small numerical video reconstruction through the public decoder contracts."""

import torch

from uniserve_worker.modeling.batch import TensorOutput
from uniserve_worker.modeling.components import Call, CallSpec, ComponentSpec
from uniserve_worker.modeling.decoder import DecoderMixin
from uniserve_worker.modeling.geometry import TensorOutputLayout
from uniserve_worker.modeling.model import Model
from uniserve_worker.nn.vae.decoder import LatentDecoder


class ChannelDecoder(LatentDecoder):
    """Apply channel statistics and a learned temporal projection to one RGB pixel."""

    latent_shape = (1, 3, 4)

    def __init__(self):
        super().__init__()
        self.vae = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.vae.weight.copy_(torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])))
        self.register_buffer("latents_mean", torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1))
        self.register_buffer("latents_std", torch.tensor([0.5, 1.5, 2.5]).view(1, 3, 1))

    def _reconstruct(self, latents):
        return self.vae(latents).unsqueeze(-1).unsqueeze(-1)


class DecodedModel(DecoderMixin, Model):
    """Decode four RGB latent rows without text or diffusion capability."""

    decoder_kinds = frozenset({"video"})

    def __init__(self):
        super().__init__()
        self.video_decoder = ChannelDecoder()
        self.audio_decoder = None

    @classmethod
    def components(cls, config):
        return (ComponentSpec("reconstruction", (CallSpec(Call.DECODE_VIDEO),)),)

    def decode(self, kind, batch, *, constants, scratch):
        if kind != "video":
            raise ValueError("this decoder reconstructs video")
        values = tuple(self.video_decoder(value.T.unsqueeze(0).float()) for value in batch.latents)
        return TensorOutput(
            {"video": values},
            {"video": tuple(TensorOutputLayout(tuple(value.shape)) for value in values)},
        )
