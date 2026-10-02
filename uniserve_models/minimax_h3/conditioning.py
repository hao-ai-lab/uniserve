"""Dense text refinement for H3's multimodal token stream.

The ``Conditioner`` projects the text encoder's Qwen hidden states to the
denoiser width and refines each prompt with dense bidirectional attention.
It runs once per request through the denoiser's ``conditioner.encode``
entry; the worker retains the result and supplies it to every denoising
step as ``DenoiserInput.text_features``. A prompt may be padded to a row
capacity (``TextConditioner``): its padding rows are masked out of every
attention, so they never reach the prompt's rows.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from uniserve.loading import weights
from uniserve.model import TextConditioner
from uniserve.nn import (
    ColumnParallelLinear,
    GatedMLP,
    Linear,
    QKVParallelLinear,
    RMSNorm,
    RowParallelLinear,
)
from uniserve.nn.attention import (
    Attention,
    DenseInput,
    SequenceLengths,
    VisibleInput,
)

from .config import TransformerConfig


class RefinerBlock(nn.Module):
    """Refine each document with dense attention and a pre-normalized SwiGLU."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.head_dim = config.head_dim
        # Pre-attention, pre-MLP, query and key norms (checkpoint norm1,
        # norm2, attn.norm_q and attn.norm_k).
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
            config.num_attention_heads,
            config.num_attention_heads,
            config.head_dim,
        )
        self.output = RowParallelLinear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.mlp = GatedMLP(
            config.hidden_size,
            config.intermediate_size,
            rounding=config.rounding,
        )

    def forward(
        self, hidden: torch.Tensor, visible: VisibleInput | None = None
    ) -> torch.Tensor:
        # hidden is [batch, tokens, hidden_size]; one document per batch row.
        # ``visible`` bounds the keys of padded documents, packed one after
        # another; without it every row of a document is text.
        batch, tokens, _ = hidden.shape
        projections = self.qkv(self.norms[0](hidden))
        q, k, v = (
            projections[name].reshape(batch, tokens, -1, self.head_dim)
            for name in ("q", "k", "v")
        )
        q, k = self.norms[2](q), self.norms[3](k)

        if visible is None:
            attended = self.attention(
                q.transpose(1, 2),  # [batch, heads, tokens, head_dim]
                k.transpose(1, 2),
                v.transpose(1, 2),
                DenseInput(causal=False, mask=None),
            ).transpose(1, 2)
        else:
            # Packed [batch * tokens, heads, head_dim] rows, each document's
            # queries seeing only its text keys.
            attended = self.attention(
                q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), visible
            )
        hidden = hidden + self.output(attended.reshape(batch, tokens, -1))
        return hidden + self.mlp(self.norms[1](hidden))


class TokenRefiner(nn.Module):
    """Project Qwen features, refine their document, and normalize the result."""  # noqa: E501

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.input = Linear(config.text_dim, config.hidden_size)
        self.blocks = nn.ModuleList(
            RefinerBlock(config) for _ in range(config.num_refiner_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self, hidden: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Refine ``[documents, rows, text_dim]`` features.

        ``lengths`` holds each document's text rows as int32. Every query of
        a document then sees only the keys before its length, so every text
        row's output ignores the padding rows. The endpoints are device data
        and the packed row counts are the fixed capacity, so one captured
        call serves every length.
        """
        visible = None
        if lengths is not None:
            documents, rows = hidden.shape[:2]
            device = hidden.device
            counts = SequenceLengths(
                torch.full(
                    (documents,), rows, dtype=torch.int32, device=device
                ),
                torch.arange(
                    0,
                    (documents + 1) * rows,
                    rows,
                    dtype=torch.int32,
                    device=device,
                ),
                (rows,) * documents,
            )
            visible = VisibleInput(
                counts,
                counts,
                lengths.reshape(documents, 1),
                None,
                prefix_bounds=True,
                fully_visible=False,
            )
        hidden = self.input(hidden.to(self.input.weight.dtype))
        for block in self.blocks:
            hidden = block(hidden, visible)
        return self.norm(hidden)


class Conditioner(TextConditioner):
    """Refine documents, padded to one row count, through the text refiner.

    ``TextConditioner`` owns homogeneous batching, the padding contract and
    the caller's sample order.
    """

    network: TokenRefiner

    def __init__(self, config: TransformerConfig):
        super().__init__(TokenRefiner(config))

    @property
    def refiner(self) -> TokenRefiner:
        return self.network


def assignments(model: Conditioner, reader):
    """Map the checkpoint's dense refiner and value-first SwiGLU matrices."""
    refiner = model.refiner
    for name, parameter in refiner.input.named_parameters():
        yield weights.Assignment(
            parameter, reader.get(f"context_embedder.{name}")
        )
    yield weights.Assignment(
        refiner.norm.weight, reader.get("token_refiner.final_norm.weight")
    )

    # Module containers hold one class each: RefinerBlock, RMSNorm, and the
    # column-parallel branches of the merged projections.
    for index, block in enumerate(refiner.blocks):
        block = cast(RefinerBlock, block)
        prefix = f"token_refiner.refiner_blocks.{index}"
        for name, norm in zip(
            ("norm1", "norm2", "attn.norm_q", "attn.norm_k"),
            block.norms,
            strict=True,
        ):
            yield weights.Assignment(
                cast(RMSNorm, norm).weight,
                reader.get(f"{prefix}.{name}.weight"),
            )
        for name, projection in block.qkv.projections.items():
            yield weights.Assignment(
                cast(ColumnParallelLinear, projection).weight,
                reader.get(f"{prefix}.attn.to_{name}.weight"),
            )
        yield weights.Assignment(
            block.output.weight, reader.get(f"{prefix}.attn.to_out.0.weight")
        )

        # The fused checkpoint rows store the value branch first, gate second.
        source = reader.get(f"{prefix}.ff.net.0.proj.weight")
        width = source.shape[0] // 2
        for name, begin in (("up", 0), ("gate", width)):
            branch = cast(
                ColumnParallelLinear, block.mlp.gate_up.projections[name]
            )
            yield weights.Assignment(
                branch.weight,
                source,
                source_slice=(
                    slice(begin, begin + width),
                    slice(0, source.shape[1]),
                ),
            )
        yield weights.Assignment(
            block.mlp.down.weight, reader.get(f"{prefix}.ff.net.2.weight")
        )
