"""MiniMax H3 transformer math over rank-local packed rows."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from ...backends.attention.video_sparse import (
    PreparedVideoSparseInputs,
    VideoSparseAttentionBackend,
    VideoSparseAttentionWorkspace,
    build_video_sparse_metadata,
)
from ...nn.attention import RadixAttention
from ...nn.diffusion.modulation import ModulationPlan
from ...nn.layer import LayerConfig
from ...nn.linear import (
    InterleavedMergedColumnParallelLinear,
    RowParallelLinear,
)
from ...nn.mesh import DeviceMesh
from ...nn.mlp import GatedMLP
from ...nn.norm import RMSNorm
from ...nn.parallel_attention import (
    AttentionContextWorkspace,
    AttentionRowExchange,
    ParallelAttention,
)
from ...nn.parallel_pipeline import LayerPipeline
from ...nn.quant.base import PreparedLinearInput
from ...nn.quant.config import LinearPrecision, create_linear_method
from ...nn.row_pipeline import (
    ProjectedRows,
    RowStage,
    independent_linear_rows,
    map_attention_rows,
    run_row_pipeline,
)
from ...ops import (
    gated_residual,
    gated_residual_rms_norm,
    gated_residual_rms_norm_fp8,
    modulated_rms_norm,
    qk_norm_rope,
)
from .config import H3TransformerConfig
from .layout import H3Layout, H3Scratch, H3Tensors
from .packing import AUDIO_TAG, dense_key_mask

__all__ = [
    "H3TransformerMetadata",
    "MiniMaxH3Transformer",
]

MODALITIES = 3


@dataclass(frozen=True, slots=True)
class H3TransformerMetadata:
    """Holds rank-local row indices, routing masks, positions, and sparse-attention metadata for one H3 forward."""

    layout: H3Layout
    vsa: VideoSparseAttentionBackend | None
    local_text_indices: torch.Tensor
    global_text_indices: torch.Tensor
    local_video_indices: torch.Tensor
    local_audio_indices: torch.Tensor
    timestep_indices: torch.Tensor
    adaln_indices: torch.Tensor
    positions: torch.Tensor
    non_text_mask: torch.Tensor


def build_transformer_metadata(layout: H3Layout, device: torch.device) -> H3TransformerMetadata:
    """Build device indices and sparse-attention metadata for one packed page layout."""

    vsa = None
    if layout.attention == "vsa":
        metadata = build_video_sparse_metadata(
            padded_rows=layout.packed.padded_rows,
            prefix_tiles=layout.packed.prefix_tiles,
            video_tiles=layout.packed.video_tiles,
            valid_sizes=layout.packed.tile_valid_sizes,
            device=device,
        )
        vsa = VideoSparseAttentionBackend(metadata)
    local_tags = layout.packed.token_tags[layout.local_start : layout.local_end]
    timestep_indices = (local_tags == AUDIO_TAG).to(torch.long)
    global_text = layout.packed.text_indices[
        (layout.packed.text_indices >= layout.local_start)
        & (layout.packed.text_indices < layout.local_end)
    ]
    return H3TransformerMetadata(
        layout=layout,
        vsa=vsa,
        local_text_indices=layout.local_indices(layout.packed.text_indices).to(device),
        global_text_indices=global_text.to(device),
        local_video_indices=layout.local_indices(layout.packed.video_indices).to(device),
        local_audio_indices=layout.local_indices(layout.packed.audio_indices).to(device),
        timestep_indices=timestep_indices.to(device),
        adaln_indices=(timestep_indices * MODALITIES + local_tags).to(device),
        positions=layout.packed.position_ids.to(device=device, dtype=torch.float32),
        non_text_mask=(layout.packed.token_tags != 1).to(device),
    )


class _RotaryEmbedding(nn.Module):
    """Builds multimodal rotary frequencies and applies them to query and key heads."""

    inv_freq: torch.Tensor

    def __init__(self, config: H3TransformerConfig, *, device: torch.device | str) -> None:
        """Precompute three-axis rotary frequencies for packed multimodal positions."""

        super().__init__()
        inv = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(
                    0, config.rope_frequency_dim * 2, 2, dtype=torch.float32, device=device
                )
                / (config.rope_frequency_dim * 2)
            )
        )
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build cosine and sine tables for ``[token, temporal/height/width]`` positions."""

        frequencies = positions.float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        time, height, width = frequencies.unbind(1)
        frequencies = torch.cat((time, height, width), dim=-1)
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        return frequencies.cos(), frequencies.sin()

    def forward_into(
        self,
        positions: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> None:
        """Write fixed-layout rotary values into caller-owned storage."""

        torch.mul(
            positions.unsqueeze(-1),
            self.inv_freq.view(1, 1, -1),
            out=frequencies,
        )
        width = int(self.inv_freq.numel())
        for axis in range(3):
            start = axis * width
            stop = start + width
            torch.cos(frequencies[:, axis], out=cosine[:, start:stop])
            torch.sin(frequencies[:, axis], out=sine[:, start:stop])
            cosine[:, start + 3 * width : stop + 3 * width].copy_(cosine[:, start:stop])
            sine[:, start + 3 * width : stop + 3 * width].copy_(sine[:, start:stop])


class H3TimestepEmbedding(nn.Module):
    """Embeds diffusion timesteps into conditioning vectors for adaptive modulation."""

    frequencies: torch.Tensor

    def __init__(
        self,
        config: H3TransformerConfig,
        *,
        device: torch.device | str,
        buffer_device: torch.device | str,
    ) -> None:
        """Build sinusoidal timestep features and their learned conditioning projection."""

        super().__init__()
        half = config.frequency_dim // 2
        self.register_buffer(
            "frequencies",
            torch.exp(
                -math.log(10_000.0)
                * torch.arange(half, dtype=torch.float32, device=buffer_device)
                / half
            ),
            persistent=False,
        )
        self.linear_1 = nn.Linear(config.frequency_dim, config.time_hidden_dim, device=device)
        self.linear_2 = nn.Linear(config.time_hidden_dim, config.time_dim, device=device)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Map one scalar timestep per branch into adaptive-conditioning vectors."""

        angles = timesteps.float()[:, None] * self.frequencies[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.linear_2(F.silu(self.linear_1(embedding)))


class _DenseAttention(nn.Module):
    """Computes dense attention for token-refinement layers before multimodal packing."""

    def __init__(self, config: H3TransformerConfig, *, device: torch.device | str) -> None:
        """Build dense query, key, value, and output projections for token refinement."""

        super().__init__()
        inner = config.heads * config.head_dim
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.attention = RadixAttention(self.heads, self.heads, self.head_dim)
        self.to_q = nn.Linear(config.hidden_size, inner, bias=False, device=device)
        self.to_k = nn.Linear(config.hidden_size, inner, bias=False, device=device)
        self.to_v = nn.Linear(config.hidden_size, inner, bias=False, device=device)
        self.to_out = nn.Sequential(nn.Linear(inner, config.hidden_size, bias=False, device=device))
        self.norm_q = RMSNorm(config.head_dim, config.qk_norm_eps, device=device)
        self.norm_k = RMSNorm(config.head_dim, config.qk_norm_eps, device=device)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Attend over dense ``[batch, rows, hidden]`` refinement sequences."""

        batch, rows, _ = hidden.shape
        query = self.norm_q(self.to_q(hidden).view(batch, rows, self.heads, self.head_dim))
        key = self.norm_k(self.to_k(hidden).view(batch, rows, self.heads, self.head_dim))
        value = self.to_v(hidden).view(batch, rows, self.heads, self.head_dim)
        result = self.attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), None, causal=False
        )
        return self.to_out(result.transpose(1, 2).reshape(batch, rows, -1))


class _TokenRefinerBlock(nn.Module):
    """Refines conditioning tokens with time-modulated attention and feed-forward residuals."""

    def __init__(
        self,
        config: H3TransformerConfig,
        *,
        linear_precision: LinearPrecision,
        layer_config: LayerConfig,
        device: torch.device | str,
    ) -> None:
        """Assemble one normalized dense-attention and feed-forward refinement block."""

        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, config.norm_eps, device=device)
        self.attn = _DenseAttention(config, device=device)
        self.norm2 = RMSNorm(config.hidden_size, config.norm_eps, device=device)
        with torch.device(device):
            self.ff = GatedMLP(
                config.hidden_size,
                config.ffn_dim,
                quant_method=create_linear_method(linear_precision),
                layer_config=layer_config.child("ff"),
                order="value_gate",
            )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply pre-normalized attention and feed-forward residuals to refinement rows."""

        hidden = hidden + self.attn(self.norm1(hidden))
        return hidden + self.ff(self.norm2(hidden))


class _TokenRefiner(nn.Module):
    """Projects and refines text conditioning before it enters the multimodal transformer."""

    def __init__(
        self,
        config: H3TransformerConfig,
        *,
        linear_precision: LinearPrecision,
        layer_config: LayerConfig,
        device: torch.device | str,
    ) -> None:
        """Build the configured stack of dense text-conditioning refinement blocks."""

        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            _TokenRefinerBlock(
                config,
                linear_precision=linear_precision,
                layer_config=layer_config.child(f"refiner_blocks.{index}"),
                device=device,
            )
            for index in range(config.refiner_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps, device=device)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run every refinement block and normalize the resulting text condition."""

        for block in self.refiner_blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)


class _H3Attention(nn.Module):
    """Routes packed multimodal Q/K/V through dense or sparse attention and output projection."""

    def __init__(
        self,
        config: H3TransformerConfig,
        mesh: DeviceMesh,
        *,
        linear_precision: LinearPrecision,
        layer_config: LayerConfig,
        device: torch.device | str,
        attention: str = "vsa",
    ) -> None:
        """Bind sharded projections to the selected attention workspace contract."""

        super().__init__()
        inner = config.heads * config.head_dim
        self.config = config
        self.projection_count = 3 if attention == "dense" else 4
        self.sequence_size = mesh.size("ulysses")
        self.tensor_heads = config.heads // mesh.size("tp")
        self.local_heads = self.tensor_heads // self.sequence_size
        self.context_size = mesh.size("cp")
        self.context_rank = mesh.coord("cp")
        self.projected_head = mesh.size("tp") == 1 and self.context_size == 1
        projection_group = (
            mesh.get_group("ulysses") if self.projected_head else mesh.get_group("tp")
        )
        projection_config = LayerConfig(projection_group, None, layer_config.prefix)
        self.parallel_attention = ParallelAttention(mesh=mesh)
        self.dense_attention = RadixAttention(self.local_heads, self.local_heads, config.head_dim)
        with torch.device(device):
            self.to_qkvg = InterleavedMergedColumnParallelLinear(
                config.hidden_size,
                inner,
                self.projection_count,
                config.head_dim,
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
                    config.hidden_size,
                    layer_config=layer_config,
                    prefix="to_out.0",
                    bias=False,
                    quant_method=create_linear_method(linear_precision),
                )
            )
        self.norm_q = RMSNorm(config.head_dim, config.qk_norm_eps, device=device)
        self.norm_k = RMSNorm(config.head_dim, config.qk_norm_eps, device=device)

    def stream_projection(
        self,
        rows: int,
        workspace: torch.Tensor,
        *,
        rotary: tuple[torch.Tensor, torch.Tensor],
        valid_sizes: torch.Tensor,
        scratch: H3Scratch,
        backend: VideoSparseAttentionBackend,
    ) -> ProjectedRows[PreparedVideoSparseInputs]:
        """Prepare each completed QKVG interval while later peer inputs arrive."""

        global_rows = rows * self.sequence_size
        inputs = backend.prepare_input_rows(
            (global_rows, self.local_heads, self.config.head_dim),
            valid_sizes,
            dtype=torch.bfloat16,
            owners=self.sequence_size,
            chunk_rows=AttentionRowExchange.chunk_rows(
                scratch.projection_peers[self.parallel_attention.ulysses_group.rank_in_group].view(
                    global_rows, self.local_heads, self.config.head_dim
                )
            ),
            pooled_query=scratch.pooled_query,
            pooled_key=scratch.pooled_key,
            pooled_value=scratch.pooled_value,
        )
        cosine, sine = rotary

        def consume(interval: slice, projected: torch.Tensor) -> None:
            query, key, value, _ = projected.view(
                -1, self.local_heads, 4, self.config.head_dim
            ).unbind(2)
            qk_norm_rope(
                query,
                key,
                self.norm_q.weight,
                self.norm_k.weight,
                cosine[interval],
                sine[interval],
                self.config.qk_norm_eps,
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
        projection_peers: tuple[torch.Tensor, ...],
        projection_sync_input: torch.Tensor,
        projection_sync_output: torch.Tensor,
        attention_workspace: torch.Tensor,
        attention_output: torch.Tensor,
        context_workspace: AttentionContextWorkspace | None,
        tile_scores: torch.Tensor,
        block_counts: torch.Tensor,
        block_indices: torch.Tensor,
        pooled_query: torch.Tensor,
        pooled_key: torch.Tensor,
        pooled_value: torch.Tensor,
        compressed_tiles: torch.Tensor,
        topk_indices_i32: torch.Tensor,
        backend: VideoSparseAttentionBackend | None,
        consume_row_intervals: bool = False,
        prepared_projection: ProjectedRows[PreparedVideoSparseInputs] | None = None,
    ) -> torch.Tensor | AttentionRowExchange:
        """Compute sparse global attention and publish its row-exchange dependency."""

        local = hidden[0]
        head_dim = self.config.head_dim
        local_rows = local.shape[0]
        global_rows = local_rows * self.sequence_size
        prepared_inputs = None
        if prepared_projection is not None:
            projected, prepared_inputs = prepared_projection.finish()
            exchanged = projected.view(
                global_rows, self.local_heads, self.projection_count, head_dim
            )
        elif self.projected_head:
            exchanged = self.to_qkvg.forward_sequence_parallel(local, attention_workspace).view(
                global_rows,
                self.local_heads,
                self.projection_count,
                head_dim,
            )
        else:
            projected = self.to_qkvg(local).view(
                local_rows, self.tensor_heads, self.projection_count, head_dim
            )
            exchanged = self.parallel_attention.exchange_heads(projected)
        query, key, value = exchanged[:, :, :3].unbind(2)
        gate = exchanged[:, :, 3] if self.projection_count == 4 else None
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
                self.config.qk_norm_eps,
                in_place=True,
            )
        if backend is None:
            # Base H3 is ordinary all-to-all attention: the checkpoint's VSA
            # compression gate has no role in its dense numerical recipe.
            parallel = self.parallel_attention
            key, value = parallel.distribute_key_value(key, value, context_workspace)
            mask = dense_key_mask(tile_valid_sizes)
            if parallel.mapped:
                assert context_workspace is not None
                owners = parallel.key_group.world_size
                capacity = key.shape[0] // owners
                logical = mask.numel() // owners
                mask = F.pad(mask.reshape(owners, logical), (0, capacity - logical))
                mask = mask.reshape(1, 1, 1, -1)
            result = self.dense_attention(
                query.transpose(0, 1).unsqueeze(0),
                key.transpose(0, 1).unsqueeze(0),
                value.transpose(0, 1).unsqueeze(0),
                None,
                causal=False,
                attn_mask=mask,
            )
            result = result.squeeze(0).transpose(0, 1).contiguous()
            parallel.finish_context(context_workspace)
            return parallel.restore_rows(result)

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
            outputs=projection_peers,
            sync_input=projection_sync_input,
            sync_output=projection_sync_output,
            context_workspace=context_workspace,
            consume_row_intervals=consume_row_intervals,
            prepared_inputs=prepared_inputs,
        )


class _TransformerBlock(nn.Module):
    """Applies adaptively modulated attention and expert feed-forward residual updates."""

    def __init__(
        self,
        config: H3TransformerConfig,
        mesh: DeviceMesh,
        *,
        attention_linear_precision: LinearPrecision,
        mlp_linear_precision: LinearPrecision,
        layer_config: LayerConfig,
        device: torch.device | str,
        attention: str = "vsa",
    ) -> None:
        """Assemble one adaptive attention and gated feed-forward block."""

        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, config.norm_eps, device=device)
        self.attn = _H3Attention(
            config,
            mesh,
            linear_precision=attention_linear_precision,
            layer_config=layer_config.child("attn"),
            device=device,
            attention=attention,
        )
        self.norm2 = RMSNorm(config.hidden_size, config.norm_eps, device=device)
        with torch.device(device):
            self.ff = GatedMLP(
                config.hidden_size,
                config.ffn_dim,
                quant_method=create_linear_method(mlp_linear_precision),
                layer_config=layer_config.child("ff"),
                order="value_gate",
            )
        self.hidden_size = config.hidden_size
        # Projected-head execution provides three registered transport buffers;
        # dense projections keep each row independent of tensor-wide scales.
        self.overlap_output_exchange = self.attn.projected_head and independent_linear_rows(
            self.attn.to_out, self.ff
        )

    def forward(
        self,
        hidden: torch.Tensor,
        adaln_values: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        tile_valid_sizes: torch.Tensor,
        prefix_key_indices: torch.Tensor,
        dense_key_indices: torch.Tensor,
        prefix_count: torch.Tensor,
        projection_peers: tuple[torch.Tensor, ...],
        projection_sync_input: torch.Tensor,
        projection_sync_output: torch.Tensor,
        attention_workspace: torch.Tensor,
        attention_output: torch.Tensor,
        context_workspace: AttentionContextWorkspace | None,
        tile_scores: torch.Tensor,
        block_counts: torch.Tensor,
        block_indices: torch.Tensor,
        pooled_query: torch.Tensor,
        pooled_key: torch.Tensor,
        pooled_value: torch.Tensor,
        compressed_tiles: torch.Tensor,
        topk_indices_i32: torch.Tensor,
        backend: VideoSparseAttentionBackend | None,
        prepared_projection: ProjectedRows[PreparedVideoSparseInputs] | None = None,
        row_consumer: Callable[[slice, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Apply one time-modulated sparse-attention and feed-forward residual block."""

        # Parameters are stored by modality; row indices select the appropriate
        # six-vector modulation tuple for each packed hidden row.
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
            tensor.to(hidden.dtype)
            for tensor in adaln_values.reshape(-1, self.hidden_size * 6).chunk(6, dim=-1)
        )
        normalized = (
            hidden
            if prepared_projection is not None
            else modulated_rms_norm(
                hidden,
                self.norm1.weight,
                shift_attn,
                scale_attn,
                adaln_indices,
                eps=self.norm1.eps,
            )
        )
        attention = self.attn(
            normalized,
            rotary,
            tile_valid_sizes,
            prefix_key_indices,
            dense_key_indices,
            prefix_count,
            projection_peers,
            projection_sync_input,
            projection_sync_output,
            attention_workspace,
            attention_output,
            context_workspace,
            tile_scores,
            block_counts,
            block_indices,
            pooled_query,
            pooled_key,
            pooled_value,
            compressed_tiles,
            topk_indices_i32,
            backend,
            consume_row_intervals=self.overlap_output_exchange,
            prepared_projection=prepared_projection,
        )
        del normalized

        def finish(interval: slice, rows: torch.Tensor) -> torch.Tensor:
            projected = self.attn.to_out(rows.reshape(1, rows.shape[0], -1))
            return self._finish_attention(
                hidden[:, interval],
                projected,
                gate_attn,
                shift_ffn,
                scale_ffn,
                gate_ffn,
                adaln_indices[interval],
            )

        return map_attention_rows(
            attention,
            hidden,
            attention_workspace,
            finish,
            row_axis=1,
            row_independent=self.overlap_output_exchange,
            consumer=row_consumer,
        )

    def _finish_attention(
        self,
        hidden: torch.Tensor,
        attention: torch.Tensor,
        gate_attn: torch.Tensor,
        shift_ffn: torch.Tensor,
        scale_ffn: torch.Tensor,
        gate_ffn: torch.Tensor,
        adaln_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Consume complete head vectors through the residual and feed-forward edges."""

        # Dynamic FP8 keeps the normalized feed-forward input quantized across
        # the expansion boundary; other precision modes consume the BF16 view.
        if self.ff.accepts_prequantized_fp8:
            hidden, normalized, normalized_scale = gated_residual_rms_norm_fp8(
                hidden,
                attention,
                gate_attn,
                self.norm2.weight,
                shift_ffn,
                scale_ffn,
                adaln_indices,
                eps=self.norm2.eps,
            )
            feed_forward = self.ff.forward_prepared(
                PreparedLinearInput(normalized, row_scales=normalized_scale)
            )
        else:
            hidden, normalized = gated_residual_rms_norm(
                hidden,
                attention,
                gate_attn,
                self.norm2.weight,
                shift_ffn,
                scale_ffn,
                adaln_indices,
                eps=self.norm2.eps,
            )
            feed_forward = self.ff(normalized)
        return gated_residual(
            hidden,
            feed_forward,
            gate_ffn,
            adaln_indices,
        )


class _OutputNorm(nn.Module):
    """Applies final adaptive normalization and projects transformer states into latent velocities."""

    def __init__(self, config: H3TransformerConfig, *, device: torch.device | str) -> None:
        """Build final adaptive normalization for latent-velocity prediction."""

        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.norm_eps, device=device)

    def forward(
        self,
        hidden: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize packed rows and apply precomputed per-timestep affine parameters."""

        normalized = self.norm(hidden)
        return normalized * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


def build_conditioner(mesh: DeviceMesh, device: torch.device | str) -> nn.Sequential:
    """Compose the checkpoint's conditioning projection and dense text refiner."""

    config = H3TransformerConfig()
    layers = LayerConfig(mesh.get_group("tp"), None)
    return nn.Sequential(
        OrderedDict(
            context_embedder=nn.Linear(config.text_dim, config.hidden_size, device=device),
            token_refiner=_TokenRefiner(
                config,
                linear_precision="bf16",
                layer_config=layers.child("token_refiner"),
                device=device,
            ),
        )
    )


class MiniMaxH3Transformer(nn.Module):
    """DiT weights with sequence-parallel rank-local activations."""

    architecture = "MiniMaxH3Transformer3DModel"

    def __init__(
        self,
        mesh: DeviceMesh,
        *,
        parameter_device: torch.device | str = "meta",
        attention_linear_precision: LinearPrecision,
        mlp_linear_precision: LinearPrecision,
        attention: str = "vsa",
    ) -> None:
        """Construct rank-sharded H3 projections, sparse blocks, and layout-index buffers."""

        super().__init__()
        config = H3TransformerConfig()
        if config.heads % (mesh.size("tp") * mesh.size("ulysses")):
            raise ValueError("H3 attention heads must divide the sequence-parallel size")
        self.config = config
        self.mesh = mesh
        self.pipeline = LayerPipeline(mesh.get_group("pp"), config.layers)
        self.attention_linear_precision = attention_linear_precision
        self.mlp_linear_precision = mlp_linear_precision

        layer_config = LayerConfig(
            communicator=mesh.get_group("tp"),
            quantization=None,
        )

        # Input and output projections retain modality-specific widths around a
        # common hidden stream shared by text, video, and audio rows.
        video_patch_width = config.video_channels * 4
        self.proj_in = (
            nn.Linear(video_patch_width, config.hidden_size, device=parameter_device)
            if self.pipeline.first
            else None
        )
        self.audio_proj_in = (
            nn.Linear(config.audio_channels, config.hidden_size, device=parameter_device)
            if self.pipeline.first
            else None
        )
        self.modulation_plan: ModulationPlan
        self.rope = _RotaryEmbedding(config, device=mesh.local_device)
        # Global layer names remain checkpoint identities on every stage.
        self.transformer_blocks = nn.ModuleDict(
            {
                str(layer): _TransformerBlock(
                    config,
                    mesh,
                    attention=attention,
                    attention_linear_precision=attention_linear_precision,
                    mlp_linear_precision=mlp_linear_precision,
                    layer_config=layer_config.child(f"transformer_blocks.{layer}"),
                    device=parameter_device,
                )
                for layer in self.pipeline.layers
            }
        )
        self.norm_out = _OutputNorm(config, device=parameter_device) if self.pipeline.last else None
        self.proj_out = (
            nn.Linear(config.hidden_size, video_patch_width, device=parameter_device)
            if self.pipeline.last
            else None
        )
        self.audio_proj_out = (
            nn.Linear(config.hidden_size, config.audio_channels, device=parameter_device)
            if self.pipeline.last
            else None
        )

    def select_adaln_step(self, scratch: H3Scratch, step: int) -> None:
        """Bind the fixed solver ladder's modulation products to execution scratch."""

        self.modulation_plan.copy_step(step, scratch.block_adaln_params, scratch.final_adaln_params)

    @torch.inference_mode()
    def forward(
        self,
        slot: H3Tensors,
        scratch: H3Scratch,
        metadata: H3TransformerMetadata,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return rank-local video/audio predictions on the final pipeline stage.

        Inputs and scratch are caller-owned. Predictions borrow the velocity
        buffers until the next forward; request latents are read-only here.
        The caller selects a trained modulation step and owns solver progression.
        """

        self.select_adaln_step(scratch, step)
        hidden = scratch.packed_hidden
        if self.pipeline.first:
            assert self.proj_in is not None and self.audio_proj_in is not None
            hidden.zero_()

            # Assemble rank-local packed rows from persistent text, video, and audio state.
            if metadata.local_text_indices.numel():
                torch.index_select(
                    slot.text_condition,
                    1,
                    metadata.global_text_indices,
                    out=scratch.local_text_hidden,
                )
                hidden.index_copy_(1, metadata.local_text_indices, scratch.local_text_hidden)
            for rows, projection, indices in (
                (slot.video_rows, self.proj_in, metadata.local_video_indices),
                (slot.audio_rows, self.audio_proj_in, metadata.local_audio_indices),
            ):
                projected = scratch.projected_input[: rows.shape[0]]
                torch.addmm(
                    projection.bias,
                    rows,
                    projection.weight.t(),
                    out=projected,
                )
                projected_bf16 = scratch.projected_input_bf16[: rows.shape[0]]
                projected_bf16.copy_(projected)
                hidden[0].index_copy_(0, indices, projected_bf16)

        else:
            self.pipeline.receive_activation(hidden)

        # Every block consumes the same layout metadata and caller-owned collective buffers.
        rotary = (slot.rotary_cosine, slot.rotary_sine)

        def bind_stage(layer: int, block: _TransformerBlock) -> RowStage:
            adaln = scratch.block_adaln_params[layer]
            operation = partial(
                block,
                adaln_values=adaln,
                adaln_indices=metadata.adaln_indices,
                rotary=rotary,
                tile_valid_sizes=slot.tile_valid_sizes,
                prefix_key_indices=slot.prefix_key_indices,
                dense_key_indices=slot.dense_key_indices,
                prefix_count=slot.prefix_count,
                projection_peers=scratch.projection_peers,
                projection_sync_input=scratch.projection_sync_input,
                projection_sync_output=scratch.projection_sync_output,
                attention_workspace=scratch.attention_workspace,
                attention_output=scratch.attention_output,
                context_workspace=scratch.context_workspace,
                tile_scores=scratch.tile_scores,
                block_counts=scratch.block_counts,
                block_indices=scratch.block_indices,
                pooled_query=scratch.pooled_query,
                pooled_key=scratch.pooled_key,
                pooled_value=scratch.pooled_value,
                compressed_tiles=scratch.compressed_tiles,
                topk_indices_i32=scratch.topk_indices_i32,
                backend=metadata.vsa,
            )

            def prepare(value: torch.Tensor) -> ProjectedRows[PreparedVideoSparseInputs]:
                _, spare = AttentionRowExchange.partition_workspace(
                    scratch.attention_workspace,
                    scratch.projection_peers[
                        block.attn.parallel_attention.ulysses_group.rank_in_group
                    ],
                )
                # The current attention exchange owns only its receive prefix.
                # Its suffix and the consumed provider output are both free for
                # next-layer gathering; use the larger legal row capacity.
                workspace = (
                    spare
                    if spare.nbytes > scratch.attention_output.nbytes
                    else scratch.attention_output
                )
                projection = block.attn.stream_projection(
                    value.shape[1],
                    workspace,
                    rotary=rotary,
                    valid_sizes=slot.tile_valid_sizes,
                    scratch=scratch,
                    backend=metadata.vsa,
                )
                shift, scale = (
                    tensor.to(value.dtype)
                    for tensor in adaln.reshape(-1, block.hidden_size * 6).chunk(6, dim=-1)[:2]
                )

                def normalize(interval: slice, rows: torch.Tensor) -> torch.Tensor:
                    return modulated_rms_norm(
                        rows,
                        block.norm1.weight,
                        shift,
                        scale,
                        metadata.adaln_indices[interval],
                        eps=block.norm1.eps,
                    )[0]

                projection.transform = normalize
                return projection

            accepts_rows = (
                metadata.vsa is not None
                and block.attn.projected_head
                and block.attn.sequence_size > 1
                and independent_linear_rows(block.attn.to_qkvg)
            )
            return RowStage(
                operation, block.overlap_output_exchange, prepare if accepts_rows else None
            )

        hidden = run_row_pipeline(
            hidden,
            tuple(
                bind_stage(layer, cast(_TransformerBlock, block))
                for layer, block in enumerate(self.transformer_blocks.values())
            ),
        )
        self.pipeline.send_activation(hidden)
        if not self.pipeline.last:
            return None
        assert self.norm_out is not None and self.proj_out is not None
        assert self.audio_proj_out is not None
        # Final adaptive normalization precedes separate rank-local media heads.
        final_shift, final_scale = scratch.final_adaln_params.chunk(2, dim=-1)
        hidden = self.norm_out(
            hidden,
            final_shift,
            final_scale,
            metadata.timestep_indices,
        )
        for indices, projection, velocity in (
            (metadata.local_video_indices, self.proj_out, scratch.video_velocity),
            (metadata.local_audio_indices, self.audio_proj_out, scratch.audio_velocity),
        ):
            selected_bf16 = scratch.projected_input_bf16[: indices.numel()]
            torch.index_select(hidden[0], 0, indices, out=selected_bf16)
            selected = scratch.projected_input[: indices.numel()]
            selected.copy_(selected_bf16)
            torch.addmm(
                projection.bias,
                selected,
                projection.weight.t(),
                out=velocity,
            )
        return scratch.video_velocity, scratch.audio_velocity
