"""Projected sparse video attention over logical tensor and sequence partitions."""

from __future__ import annotations

import torch
from torch import nn

from uniserve.distributed.mesh import DeviceMesh
from uniserve.model.tensors import TensorViews
from uniserve.nn.layer import LayerConfig
from uniserve.nn.linear import InterleavedMergedColumnParallelLinear, RowParallelLinear
from uniserve.nn.norm import RMSNorm
from uniserve.nn.parallel_attention import AttentionRowExchange, ParallelAttention
from uniserve.nn.quant.config import LinearPrecision, create_linear_method
from uniserve.nn.row_pipeline import ProjectedRows
from uniserve.nn.sparse_attention import (
    PreparedVideoSparseInputs,
    VideoSparseAttention,
    VideoSparseAttentionWorkspace,
)
from uniserve.ops import qk_norm_rope
from uniserve.tensors import BufferConfig


class VideoAttention(nn.Module):
    """Project packed rows for sparse attention with learned tile compression.

    Q/K/V and compression gates share an interleaved projection. Query and
    key heads use learned RMS normalization followed by supplied rotary tables.
    Logical TP, Ulysses, and context partitions preserve the global attention
    domain. Callers supply borrowed scratch and bind public attention scopes;
    the layer retains only numerical modules and non-owning communicators.
    """

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        head_dim: int,
        *,
        mesh: DeviceMesh,
        norm_eps: float,
        linear_precision: LinearPrecision,
        layer_config: LayerConfig,
        device: torch.device | str,
    ) -> None:
        """Bind sharded projections to the sparse-video attention workspace contract."""

        super().__init__()
        if min(hidden_size, heads, head_dim) < 1 or norm_eps <= 0:
            raise ValueError(
                "video attention requires positive dimensions and normalization epsilon"
            )
        if heads % (mesh.size("tp") * mesh.size("ulysses")):
            raise ValueError("attention heads must divide tensor and sequence partitions")
        inner = heads * head_dim
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.linear_precision = linear_precision
        self.sequence_size = mesh.size("ulysses")
        self.tensor_heads = heads // mesh.size("tp")
        self.local_heads = self.tensor_heads // self.sequence_size
        self.context_size = mesh.size("cp")
        self.context_rank = mesh.coord("cp")
        self.projected_head = mesh.size("tp") == 1 and self.context_size == 1
        projection_group = (
            mesh.get_group("ulysses") if self.projected_head else mesh.get_group("tp")
        )
        projection_config = LayerConfig(projection_group, None, layer_config.prefix)
        self.parallel_attention = ParallelAttention(mesh=mesh)
        with torch.device(device):
            self.to_qkvg = InterleavedMergedColumnParallelLinear(
                hidden_size,
                inner,
                4,
                head_dim,
                layer_config=projection_config,
                prefix="to_qkvg",
                sequence_group=mesh.get_group("sp"),
                input_scale_group=mesh.get_group("sp"),
                quant_method=create_linear_method(linear_precision, tensorwise=True),
                bias=False,
            )
            self.to_out = nn.Sequential(
                RowParallelLinear(
                    inner,
                    hidden_size,
                    layer_config=layer_config,
                    prefix="to_out.0",
                    bias=False,
                    quant_method=create_linear_method(linear_precision),
                )
            )
        self.norm_q = RMSNorm(head_dim, norm_eps, device=device)
        self.norm_k = RMSNorm(head_dim, norm_eps, device=device)

    def workspace_buffers(
        self, rows: int, query_rows: int, *, dtype: torch.dtype
    ) -> dict[str, BufferConfig]:
        """Declare scratch for projected inputs and restored attention rows.

        ``rows`` is the complete packed sequence; ``query_rows`` is this
        context partition's query extent after Ulysses head exchange. Input
        preparation can encode quantized values and scales in byte storage,
        while attention results retain their activation dtype. These sequential
        uses share backing sized for the larger complete numerical payload.
        """

        if min(rows, query_rows) < 1 or query_rows > rows:
            raise ValueError("video attention requires positive bounded query rows")
        method = self.to_qkvg.quant_method
        storage_dtype = (
            torch.uint8 if method.is_quantized and method.input_scale_domain == "tensor" else dtype
        )
        # Tensor-scaled FP8 uses one byte per input element. Packed NVFP4
        # values plus their block scales fit within the same byte bound.
        elements = rows * self.hidden_size
        output_shape = (query_rows, self.local_heads, self.head_dim)
        if self.context_size == 1 and self.sequence_size > 1:
            output_bytes = query_rows * self.local_heads * self.head_dim * dtype.itemsize
            elements = max(
                elements, (output_bytes + storage_dtype.itemsize - 1) // storage_dtype.itemsize
            )
        return {
            "attention_workspace": BufferConfig((elements,), storage_dtype),
            "attention_output": BufferConfig(output_shape, dtype),
        }

    def stream_projection(
        self,
        rows: int,
        workspace: torch.Tensor,
        *,
        rotary: tuple[torch.Tensor, torch.Tensor],
        valid_sizes: torch.Tensor,
        scratch: TensorViews,
        backend: VideoSparseAttention,
    ) -> ProjectedRows[PreparedVideoSparseInputs]:
        """Prepare each completed QKVG interval while later peer inputs arrive."""

        global_rows = rows * self.sequence_size
        inputs = backend.prepare_input_rows(
            (global_rows, self.local_heads, self.head_dim),
            valid_sizes,
            dtype=torch.bfloat16,
            owners=self.sequence_size,
            chunk_rows=AttentionRowExchange.chunk_rows(scratch["attention_output"]),
            pooled_query=scratch["pooled_query"],
            pooled_key=scratch["pooled_key"],
            pooled_value=scratch["pooled_value"],
        )
        cosine, sine = rotary

        def consume(interval: slice, projected: torch.Tensor) -> None:
            query, key, value, _ = projected.view(-1, self.local_heads, 4, self.head_dim).unbind(2)
            qk_norm_rope(
                query,
                key,
                self.norm_q.weight,
                self.norm_k.weight,
                cosine[interval],
                sine[interval],
                self.norm_eps,
                in_place=True,
            )
            inputs.append(interval, query, key, value)

        projection = self.to_qkvg.stream_sequence_parallel(rows, workspace, row_consumer=consume)
        return ProjectedRows[PreparedVideoSparseInputs](projection, inputs)

    def forward(
        self,
        hidden: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        tile_valid_sizes: torch.Tensor,
        prefix_key_indices: torch.Tensor,
        dense_key_indices: torch.Tensor,
        prefix_count: torch.Tensor,
        attention_workspace: torch.Tensor,
        attention_output: torch.Tensor,
        tile_scores: torch.Tensor,
        block_counts: torch.Tensor,
        block_indices: torch.Tensor,
        pooled_query: torch.Tensor,
        pooled_key: torch.Tensor,
        pooled_value: torch.Tensor,
        compressed_tiles: torch.Tensor,
        topk_indices_i32: torch.Tensor,
        backend: VideoSparseAttention,
        consume_row_intervals: bool = False,
        prepared_projection: ProjectedRows[PreparedVideoSparseInputs] | None = None,
    ) -> torch.Tensor | AttentionRowExchange:
        """Compute sparse global attention and publish its row-exchange dependency."""

        local = hidden[0]
        head_dim = self.head_dim
        local_rows = local.shape[0]
        global_rows = local_rows * self.sequence_size
        prepared_inputs = None
        if prepared_projection is not None:
            projected, prepared_inputs = prepared_projection.finish()
            exchanged = projected.view(global_rows, self.local_heads, 4, head_dim)
        elif self.projected_head:
            exchanged = self.to_qkvg.forward_sequence_parallel(local, attention_workspace).view(
                global_rows,
                self.local_heads,
                4,
                head_dim,
            )
        else:
            projected = self.to_qkvg(local).view(local_rows, self.tensor_heads, 4, head_dim)
            exchanged = self.parallel_attention.exchange_heads(projected)
        query, key, value, gate = exchanged.unbind(2)
        cosine, sine = rotary
        start = self.context_rank * global_rows
        cosine, sine = cosine[start : start + global_rows], sine[start : start + global_rows]

        # Query/key normalization and rotary application mutate their views of
        # the shared projection buffer before sparse block selection.
        if prepared_projection is None:
            qk_norm_rope(
                query,
                key,
                self.norm_q.weight,
                self.norm_k.weight,
                cosine,
                sine,
                self.norm_eps,
                in_place=True,
            )
        workspace = VideoSparseAttentionWorkspace(
            attention_output=attention_output,
            tile_scores=tile_scores,
            block_counts=block_counts,
            block_indices=block_indices,
            pooled_query=pooled_query,
            pooled_key=pooled_key,
            pooled_value=pooled_value,
            compressed_tiles=compressed_tiles,
            topk_indices_i32=topk_indices_i32,
        )

        return backend.forward_parallel(
            self.parallel_attention,
            query,
            key,
            value,
            gate,
            tile_valid_sizes,
            prefix_key_indices,
            dense_key_indices,
            prefix_count,
            workspace,
            consume_row_intervals=consume_row_intervals,
            prepared_inputs=prepared_inputs,
        )
