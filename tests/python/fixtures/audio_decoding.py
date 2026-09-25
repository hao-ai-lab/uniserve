"""A CPU-sized MiniMax H3 audio decoder through the public contract."""

from types import MappingProxyType

import torch
from torch import nn

from uniserve.model import ComponentEntry, EntryPoint
from uniserve_models.minimax_h3 import audio_vae
from uniserve_models.minimax_h3.decoding import AudioDecoder

# A decoder small enough to run on CPU whose halo is a few latent frames, so a
# short track still exercises a partial context window rather than the whole
# timeline.
CONFIG = audio_vae.Config(
    encoder_dim=4,
    encoder_rates=(2, 2),
    latent_dim=8,
    latent_channels=2,
    decoder_dim=8,
    decoder_rates=(2, 2),
    decoder_kernel_sizes=(4, 4),
    num_attention_heads=1,
    resblock_kernel_sizes=(3,),
    resblock_dilation_sizes=((1,),),
    latents_mean=(0.0, 0.0),
    latents_std=(1.0, 1.0),
)


def decoder() -> AudioDecoder:
    """Build the decoder with seeded random weights.

    The weight scale keeps the int16 PCM well inside its range yet varied
    along the track, so samples decoded for the wrong media unit, or in the
    wrong order, differ from the whole-track decode.
    """
    torch.manual_seed(20260917)
    module = AudioDecoder(CONFIG, sample_rate=32000).eval()
    for parameter in module.parameters():
        parameter.data = torch.randn_like(parameter) * 0.3
    return module


class AudioModel(nn.Module):
    """A model whose only component is the audio decoder."""

    def __init__(self, config=CONFIG):
        super().__init__()
        self.config = config
        self.audio_decoder = decoder()


def entry_points(config):
    return MappingProxyType(
        {
            "audio_decoder": ComponentEntry(
                "audio_decoder", (EntryPoint("decode"),)
            )
        }
    )
