"""H3's learned Q/K normalization, partial RoPE and sparse tile compression.

``Attention`` owns the per-layer head projections and output projection and
delegates the attention itself to ``uniserve.nn.attention.vsa``: text and
audio query tiles attend every valid key tile, video query tiles attend the
dense text/audio prefix plus a selected subset of video key tiles, and VSA
adds a gated attention over mean-pooled tiles to that fine result. All token
counts here are rows of the complete packing built by
``packing.build_packing``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import cast

import torch
from torch import nn

from uniserve.distributed import DeviceMesh
from uniserve.nn import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RMSNorm,
    RowParallelLinear,
)
from uniserve.nn.attention import vsa
from uniserve.tensors import BufferConfig

from .config import TransformerConfig
from .inputs import AttentionInput


class Attention(nn.Module):
    """Compose head projections and VSA over the complete visible key domain.

    The merged projection produces four branches: query, key, value, and
    ``gate``, which weights VSA's compressed-tile attention elementwise before
    it is added to the selected fine attention.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.sparsity = config.vsa_sparsity
        inner = config.num_attention_heads * config.head_dim
        self.projection = MergedColumnParallelLinear(
            config.hidden_size,
            dict.fromkeys(("q", "k", "v", "gate"), inner),
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
        """Describe the VSA scratch for one attention call.

        Args:
            num_tokens: Packed key rows, the complete padded token domain.
            num_query_tokens: Query rows this rank attends for.
            dtype: Dtype of the attention output; pooled statistics, tile
                scores and compressed tiles are always FP32.

        Returns:
            Buffer declarations for this rank's local heads, keyed by the
            ``vsa.Workspace`` field names ``forward_chunks`` reads.

        Raises:
            ValueError: Either count is not a positive multiple of the 64-row
                tile, there are fewer than two key tiles, or the queries
                exceed the keys.
        """
        if (
            num_tokens < 128
            or num_query_tokens < 64
            or num_tokens % 64
            or num_query_tokens % 64
            or num_query_tokens > num_tokens
        ):
            raise ValueError(
                "H3 attention requires complete query and key tiles"
            )

        # VSA addresses keys and queries in 64-token tiles. Every merged
        # branch is a column-parallel linear holding this rank's heads.
        query = cast(ColumnParallelLinear, self.projection.projections["q"])
        heads = query.weight.shape[0] // self.head_dim
        queries, keys = num_query_tokens // 64, num_tokens // 64
        return {
            "attention_output": BufferConfig(
                (num_query_tokens, heads, self.head_dim), dtype
            ),
            "tile_scores": BufferConfig((heads, queries, keys), torch.float32),
            "block_counts": BufferConfig((heads, queries), torch.int32),
            "block_indices": BufferConfig((heads, queries, keys), torch.int32),
            "pooled_query": BufferConfig(
                (queries, heads, self.head_dim), torch.float32
            ),
            "pooled_key": BufferConfig(
                (keys, heads, self.head_dim), torch.float32
            ),
            "pooled_value": BufferConfig(
                (keys, heads, self.head_dim), torch.float32
            ),
            "compressed_tiles": BufferConfig(
                (heads, queries, self.head_dim), torch.float32
            ),
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
        """Attend this rank's token shard and yield projected output chunks.

        ``hidden`` is the normalized shard, whole or as ``(global row slice,
        rows)`` chunks. ``cos`` and ``sin`` hold the rotary factors of every
        packed row, and ``workspace`` supplies the buffers
        ``workspace_buffers`` declares. Yielded slices are global packed rows.
        """

        def project():
            for interval, values in self.projection.forward_chunks(
                hidden,
                token_slice=inputs.token_slice,
                num_tokens=inputs.packing.padded_tokens,
            ):
                # Each branch is [tokens, heads, head_dim].
                q, k, v, gate = (
                    values[name].view(
                        -1,
                        values[name].shape[-1] // self.head_dim,
                        self.head_dim,
                    )
                    for name in ("q", "k", "v", "gate")
                )
                yield interval, (q, k, v, gate)

        batch = inputs.vsa
        # Video key tiles each video query tile keeps beyond the dense prefix;
        # VSA requires at least one.
        selected = max(1, math.ceil((1 - self.sparsity) * batch.video_tiles))

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
        )
        # Q/K normalization and partial RoPE are applied by VSA while it
        # prepares each projected chunk, in the same pass that pools and
        # packs the rows.
        attended = self.vsa.forward_chunks(
            project(),
            batch,
            selected_tiles=selected,
            workspace=buffers,
            norm_rope=vsa.NormRope(
                self.query_norm.weight,
                self.key_norm.weight,
                self.query_norm.eps,
                cos,
                sin,
            ),
        )
        flattened = (
            (interval, value.flatten(1)) for interval, value in attended
        )
        yield from self.output.forward_chunks(flattened)

    def forward(self, hidden, cos, sin, inputs, *, workspace):
        outputs = tuple(
            value
            for _, value in self.forward_chunks(
                hidden, cos, sin, inputs, workspace=workspace
            )
        )
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
