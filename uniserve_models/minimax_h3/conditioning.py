"""Dense text refinement for H3's multimodal token stream."""

from __future__ import annotations

import torch
from torch import nn

from uniserve.loading import weights
from uniserve.model import Encoder
from uniserve.nn import GatedMLP, Linear, QKVParallelLinear, RMSNorm, RowParallelLinear
from uniserve.nn.attention import Attention, DenseInput

from .config import TransformerConfig


class RefinerBlock(nn.Module):
    """Refine each document with dense attention and a pre-normalized SwiGLU."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.norms = nn.ModuleList(
            (
                RMSNorm(config.hidden_size, config.norm_eps),
                RMSNorm(config.hidden_size, config.norm_eps),
                RMSNorm(config.head_dim, config.qk_norm_eps),
                RMSNorm(config.head_dim, config.qk_norm_eps),
            )
        )
        self.qkv = QKVParallelLinear(
            config.hidden_size,
            config.num_attention_heads,
            config.num_attention_heads,
            config.head_dim,
            bias=False,
        )
        self.attention = Attention(
            config.num_attention_heads, config.num_attention_heads, config.head_dim
        )
        self.output = RowParallelLinear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=False
        )
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden is [batch, tokens, hidden_size]; one document per batch row.
        batch, tokens, _ = hidden.shape
        projections = self.qkv(self.norms[0](hidden))
        q, k, v = (
            projections[name].reshape(batch, tokens, -1, self.head_dim) for name in ("q", "k", "v")
        )
        q, k = self.norms[2](q), self.norms[3](k)

        attended = self.attention(
            q.transpose(1, 2),  # [batch, heads, tokens, head_dim]
            k.transpose(1, 2),
            v.transpose(1, 2),
            DenseInput(causal=False, mask=None),
        )
        hidden = hidden + self.output(attended.transpose(1, 2).reshape(batch, tokens, -1))
        return hidden + self.mlp(self.norms[1](hidden))


class TokenRefiner(nn.Module):
    """Project Qwen features, refine their document, and normalize the result."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.input = Linear(config.text_dim, config.hidden_size)
        self.blocks = nn.ModuleList(RefinerBlock(config) for _ in range(config.num_refiner_layers))
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.input(hidden.to(self.input.weight.dtype))
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm(hidden)


class Conditioner(Encoder[tuple[torch.Tensor, ...]]):
    """Batch equal-length documents through one learned conditioning projection.

    The public refiner is the numerical module in the projection sequence;
    neither property registers a second copy of its parameters. Encoder owns
    homogeneous batching and restores the caller's sample order.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(nn.Sequential(TokenRefiner(config)))

    @property
    def projection(self) -> nn.Sequential:
        return self.network

    @property
    def refiner(self) -> TokenRefiner:
        return self.network[0]


def assignments(model: Conditioner, reader):
    """Map the checkpoint's dense refiner and value-first SwiGLU matrices."""

    refiner = model.refiner
    for name, parameter in refiner.input.named_parameters():
        yield weights.Assignment(parameter, reader.get(f"context_embedder.{name}"))
    yield weights.Assignment(refiner.norm.weight, reader.get("token_refiner.final_norm.weight"))

    for index, block in enumerate(refiner.blocks):
        prefix = f"token_refiner.refiner_blocks.{index}"
        for name, norm in zip(
            ("norm1", "norm2", "attn.norm_q", "attn.norm_k"), block.norms, strict=True
        ):
            yield weights.Assignment(norm.weight, reader.get(f"{prefix}.{name}.weight"))
        for name, projection in block.qkv.projections.items():
            yield weights.Assignment(
                projection.weight, reader.get(f"{prefix}.attn.to_{name}.weight")
            )
        yield weights.Assignment(block.output.weight, reader.get(f"{prefix}.attn.to_out.0.weight"))

        # The fused checkpoint rows store the value branch first, gate second.
        source = reader.get(f"{prefix}.ff.net.0.proj.weight")
        width = source.shape[0] // 2
        for name, begin in (("up", 0), ("gate", width)):
            yield weights.Assignment(
                block.mlp.gate_up.projections[name].weight,
                source,
                source_slice=(slice(begin, begin + width), slice(0, source.shape[1])),
            )
        yield weights.Assignment(block.mlp.down.weight, reader.get(f"{prefix}.ff.net.2.weight"))
