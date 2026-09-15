"""H3's learned Q/K normalization, partial RoPE and sparse tile compression."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import math

import torch
from torch import nn

from uniserve.distributed import DeviceMesh
from uniserve.nn import MergedColumnParallelLinear, RMSNorm, RowParallelLinear
from uniserve.nn.attention import vsa
from uniserve.nn.functional import qk_norm_rope
from uniserve.tensors import BufferConfig

from .config import TransformerConfig
from .inputs import AttentionInput


class Attention(nn.Module):
    """Compose head projections and VSA over the complete visible key domain."""

    def __init__(self, config: TransformerConfig, *, sparsity: float = 0.9):
        super().__init__()
        if not 0 <= sparsity < 1:
            raise ValueError("VSA sparsity must lie in [0, 1)")
        self.head_dim, self.sparsity = config.head_dim, sparsity
        inner = config.num_attention_heads * config.head_dim
        self.projection = MergedColumnParallelLinear(
            config.hidden_size,
            {name: inner for name in ("q", "k", "v", "gate")},
            branch_width=config.head_dim,
            bias=False,
        )
        self.output = RowParallelLinear(inner, config.hidden_size, bias=False)
        self.query_norm = RMSNorm(config.head_dim, config.qk_norm_eps)
        self.key_norm = RMSNorm(config.head_dim, config.qk_norm_eps)
        self.vsa = vsa.Attention(vsa.BlockAttention(config.head_dim**-0.5))

    @property
    def mesh(self) -> DeviceMesh:
        return self.vsa.mesh

    def workspace_buffers(
        self, num_tokens: int, num_query_tokens: int, *, dtype: torch.dtype
    ) -> Mapping[str, BufferConfig]:
        if (
            num_tokens < 128
            or num_query_tokens < 64
            or num_tokens % 64
            or num_query_tokens % 64
            or num_query_tokens > num_tokens
        ):
            raise ValueError("H3 attention requires complete query and key tiles")
        heads = self.projection.projections["q"].weight.shape[0] // self.head_dim
        queries, keys = num_query_tokens // 64, num_tokens // 64
        selected = max(1, math.ceil((1 - self.sparsity) * keys))
        return {
            "attention_output": BufferConfig((num_query_tokens, heads, self.head_dim), dtype),
            "tile_scores": BufferConfig((heads, queries, keys), torch.float32),
            "block_counts": BufferConfig((heads, queries), torch.int32),
            "block_indices": BufferConfig((heads, queries, keys), torch.int32),
            "pooled_query": BufferConfig((queries, heads, self.head_dim), torch.float32),
            "pooled_key": BufferConfig((keys, heads, self.head_dim), torch.float32),
            "pooled_value": BufferConfig((keys, heads, self.head_dim), torch.float32),
            "compressed_tiles": BufferConfig((heads, queries, self.head_dim), torch.float32),
            "topk_indices": BufferConfig((heads, queries, selected), torch.int32),
        }

    @torch.inference_mode()
    def forward_chunks(
        self,
        hidden: torch.Tensor | Iterator[tuple[slice, torch.Tensor]],
        cos: torch.Tensor,
        sin: torch.Tensor,
        inputs: AttentionInput,
        *,
        workspace: Mapping[str, torch.Tensor],
    ) -> Iterator[tuple[slice, torch.Tensor]]:
        def project():
            for interval, values in self.projection.forward_chunks(
                hidden,
                token_slice=inputs.token_slice,
                num_tokens=inputs.packing.padded_tokens,
            ):
                q, k, v, gate = (
                    values[name].view(-1, values[name].shape[-1] // self.head_dim, self.head_dim)
                    for name in ("q", "k", "v", "gate")
                )
                q, k = qk_norm_rope(
                    q,
                    k,
                    self.query_norm.weight,
                    self.key_norm.weight,
                    (cos[interval],),
                    (sin[interval],),
                    eps=self.query_norm.eps,
                    axis_dims=(self.head_dim,),
                    out=(q, k),
                )
                yield interval, (q, k, v, gate)

        batch = inputs.vsa
        distribution = self.projection.projections["q"].output_distribution
        context = distribution.mesh.get_group(distribution.shard_axes(0))
        query_tiles = batch.padded_tokens // (64 * context.size)
        start = context.rank * query_tiles
        video_queries = max(
            0, min(start + query_tiles, batch.valid_tiles) - max(start, batch.prefix_tiles)
        )
        selected = max(1, math.ceil((1 - self.sparsity) * batch.video_tiles))
        heads = workspace["attention_output"].shape[1]
        shape = (heads, video_queries, selected)
        topk = workspace["topk_indices"].view(-1)[: math.prod(shape)].view(shape)
        buffers = vsa.Workspace(
            **{
                name: workspace[name]
                for name in (
                    "attention_output",
                    "tile_scores",
                    "block_counts",
                    "block_indices",
                    "pooled_query",
                    "pooled_key",
                    "pooled_value",
                    "compressed_tiles",
                )
            },
            topk_indices=topk,
        )
        attended = self.vsa.forward_chunks(
            project(), batch, selected_tiles=selected, workspace=buffers
        )
        flattened = ((interval, value.flatten(1)) for interval, value in attended)
        yield from self.output.forward_chunks(flattened)

    def forward(self, hidden, cos, sin, inputs, *, workspace):
        outputs = tuple(
            value for _, value in self.forward_chunks(hidden, cos, sin, inputs, workspace=workspace)
        )
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
