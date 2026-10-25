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

from dataclasses import replace
from typing import cast

import torch
from torch import nn

from uniserve.distributed.chunks import payload_rows
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
    AttentionBatch,
    DenseInput,
    SequenceLengths,
    VisibleInput,
)
from uniserve.nn.functional import Rounding

from .config import TransformerConfig


def _feed_forward_rows(hidden_size: int, itemsize: int) -> int:
    """Rows of one refiner range.

    The feed-forward is row-local, so a block runs it over successive row
    ranges and its intermediates (the gate and up projections and their
    gated product, each ``intermediate_size`` wide) stay the size of one
    range; the attention's projections and its query rows take the same
    ranges. The 84,529-token prompt of nine 2048x3648 reference images, a
    15 s reference video and a 10,000-character prompt holds about 13.5 GiB
    of whole-prompt intermediates in BF16, against about 1 GiB per 6144-row
    range; on eight RTX PRO 6000 those whole-prompt intermediates ran a
    resident Ulysses8 deployment out of memory between 54,577 prompt tokens
    (served) and 58,609 (failed). A range is one payload of the hidden width,
    as for the denoiser's feed-forward ranges; ranges this long keep the
    projections compute-bound.
    """
    return payload_rows(hidden_size, itemsize)


def _query_rows(visible: VisibleInput, rows: int) -> VisibleInput:
    """``visible`` for ``rows`` query rows of each document, all its keys.

    Every query row of a document shares its key endpoint, so a range of the
    document's query rows sees what those rows saw among all of them.
    """
    documents = visible.queries.batch_size
    device = visible.queries.values.device
    return replace(
        visible,
        queries=SequenceLengths(
            torch.full((documents,), rows, dtype=torch.int32, device=device),
            torch.arange(
                0,
                (documents + 1) * rows,
                rows,
                dtype=torch.int32,
                device=device,
            ),
            (rows,) * documents,
        ),
    )


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
            rounding=Rounding.STEPWISE,
        )

    def forward(
        self, hidden: torch.Tensor, visible: VisibleInput | None = None
    ) -> torch.Tensor:
        # hidden is [batch, tokens, hidden_size]; one document per batch row.
        # ``visible`` bounds the keys of padded documents, packed one after
        # another; without it every row of a document is text.
        batch, tokens, _ = hidden.shape
        rows = hidden.view(-1, hidden.shape[-1])
        step = _feed_forward_rows(rows.shape[-1], rows.element_size())

        # Every row's normalized queries, keys and values, projected over row
        # ranges so that one range's fused projection is live at a time.
        qkv: list[torch.Tensor] = []
        for start in range(0, len(rows), step):
            projections = self.qkv(self.norms[0](rows[start : start + step]))
            parts = [
                projections[name].unflatten(-1, (-1, self.head_dim))
                for name in ("q", "k", "v")
            ]
            parts[0], parts[1] = (
                self.norms[2](parts[0]),
                self.norms[3](parts[1]),
            )
            if not qkv:
                qkv = [
                    part.new_empty((len(rows), *part.shape[1:]))
                    for part in parts
                ]
            for whole, part in zip(qkv, parts, strict=True):
                whole[start : start + len(part)] = part
        q, k, v = (whole.unflatten(0, (batch, tokens)) for whole in qkv)

        # Each range of query rows attends to all of its document's keys. No
        # query row reads another, so the ranges leave every row's output
        # unchanged, and the residual stream takes them in place: every key
        # and value above was read from it first.
        ranges = {}
        for start in range(0, tokens, step):
            count = min(step, tokens - start)
            if visible is None:
                attended = self.attention(
                    # [batch, heads, rows, head_dim]
                    q[:, start : start + count].transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    AttentionBatch.single(DenseInput(causal=False, mask=None)),
                ).transpose(1, 2)
            else:
                # Packed [batch * rows, heads, head_dim] rows, each document's
                # queries seeing only its text keys.
                if count not in ranges:
                    ranges[count] = _query_rows(visible, count)
                attended = self.attention(
                    q[:, start : start + count].flatten(0, 1),
                    k.flatten(0, 1),
                    v.flatten(0, 1),
                    AttentionBatch.single(ranges[count]),
                )
            hidden[:, start : start + count] += self.output(
                attended.reshape(batch, count, -1)
            )
        del q, k, v, qkv

        for start in range(0, len(rows), step):
            part = rows[start : start + step]
            part += self.mlp(self.norms[1](part))
        return hidden


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
