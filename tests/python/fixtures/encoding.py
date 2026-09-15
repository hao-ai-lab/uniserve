"""Standalone numerical encoders with explicit worker component exports."""

from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn

from uniserve.model import Encoder, EntryPoint, TextEncoder, TransformerDecoder
from uniserve.nn.attention import Attention, SequenceLengths, VarlenInput


@dataclass(frozen=True)
class Config:
    vocab_size: int = 32
    hidden_size: int = 4


class Residual(nn.Module):
    def forward(self, hidden, residual, positions, attention):
        return hidden, torch.zeros_like(hidden) if residual is None else residual


class DenseAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention(2, 2, 8)

    def forward(self, values):
        batch, tokens, heads, width = values.shape
        lengths = SequenceLengths.from_lengths((tokens,) * batch, device=values.device)
        inputs = VarlenInput(lengths, lengths, (False,) * batch)
        packed = values.reshape(-1, heads, width)
        return self.attention(packed, packed, packed, inputs).reshape_as(values)


class Model(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        with torch.no_grad():
            embedding.weight.copy_(
                torch.arange(config.vocab_size * config.hidden_size).reshape_as(embedding.weight)
            )
        self.text_encoder = TextEncoder(
            TransformerDecoder(embedding, nn.ModuleDict({"0": Residual()}), nn.Identity()), (0,)
        )
        self.conditioner = Encoder(nn.Linear(4, 2, bias=False))
        with torch.no_grad():
            self.conditioner.network.weight.copy_(torch.eye(4)[:2])
        self.dense = Encoder(DenseAttention())


def entry_points(config):
    return MappingProxyType(
        {
            "text_encoder": (EntryPoint("encode"),),
            "conditioner": (EntryPoint("encode", stage="first"),),
            "dense": (EntryPoint("encode"),),
        }
    )


entry_paths = MappingProxyType(
    {
        "text_encoder": "text_encoder.encode",
        "conditioner": "conditioner.encode",
        "dense": "dense.encode",
    }
)
