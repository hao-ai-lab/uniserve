"""Small numerical video reconstruction through the public decoder contracts."""

import torch

from uniserve.model.batch import TensorOutput
from uniserve.model.components import ComponentCall
from uniserve.model.model import Model
from uniserve.nn.vae.decoder import LatentDecoder
from uniserve.tensors import OutputLayout


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


class VideoDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.native = ChannelDecoder()

    def decode(self, latents, size, windows, *, constants, scratch):
        values = tuple(self.native(value.T.unsqueeze(0).float()) for value in latents)
        return TensorOutput(
            {"video": values},
            {"video": tuple(OutputLayout(tuple(value.shape), value.dtype) for value in values)},
        )


class DecodedModel(Model):
    """Decode four RGB latent rows without text or diffusion capability."""

    decoder_kinds = frozenset({"video"})

    def __init__(self):
        super().__init__()
        self.reconstruction = VideoDecoder()
        self.audio_decoder = None

    @classmethod
    def component_calls(cls, config):
        return (ComponentCall("reconstruction", "decode:video"),)
