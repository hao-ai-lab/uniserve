"""MiniMax H3 transformer math over rank-local packed rows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from ...nn.mesh import DeviceMesh
from .fusions import qk_rmsnorm_rope, row_modulated_rmsnorm, value_first_swiglu
from .packing import AUDIO_TAG
from .state import H3Layout, H3Scratch, H3StateSlot
from .vsa import H3VsaAttention, build_vsa_metadata

__all__ = ["H3TransformerConfig", "MiniMaxH3Transformer"]

MODALITIES = 3


@dataclass(frozen=True, slots=True)
class H3TransformerConfig:
    hidden_size: int = 5376
    heads: int = 56
    head_dim: int = 128
    layers: int = 50
    refiner_layers: int = 2
    ffn_dim: int = 14336
    video_channels: int = 24
    audio_channels: int = 32
    text_dim: int = 5120
    frequency_dim: int = 256
    time_hidden_dim: int = 5376
    time_dim: int = 2688
    rope_frequency_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5


class _RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float, *, device: torch.device | str) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, device=device))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(value.dtype)


class _SwiGLUProjection(nn.Module):
    def __init__(self, width: int, expanded: int, *, device: torch.device | str) -> None:
        super().__init__()
        self.proj = nn.Linear(width, expanded * 2, bias=False, device=device)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value_first_swiglu(self.proj(value))


class _FeedForward(nn.Module):
    def __init__(self, config: H3TransformerConfig, *, device: torch.device | str) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            (
                _SwiGLUProjection(config.hidden_size, config.ffn_dim, device=device),
                nn.Identity(),
                nn.Linear(config.ffn_dim, config.hidden_size, bias=False, device=device),
            )
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net[2](self.net[0](value))


class _RotaryEmbedding(nn.Module):
    def __init__(self, config: H3TransformerConfig, *, device: torch.device | str) -> None:
        super().__init__()
        inv = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.rope_frequency_dim * 2, 2, dtype=torch.float32, device=device)
                / (config.rope_frequency_dim * 2)
            )
        )
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
            cosine[:, start + 3 * width : stop + 3 * width].copy_(
                cosine[:, start:stop]
            )
            sine[:, start + 3 * width : stop + 3 * width].copy_(
                sine[:, start:stop]
            )


class _TimeEmbedding(nn.Module):
    def __init__(
        self,
        config: H3TransformerConfig,
        *,
        device: torch.device | str,
        buffer_device: torch.device | str,
    ) -> None:
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
        angles = timesteps.float()[:, None] * self.frequencies[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.linear_2(F.silu(self.linear_1(embedding)))


class _DenseAttention(nn.Module):
    def __init__(self, config: H3TransformerConfig, *, device: torch.device | str) -> None:
        super().__init__()
        inner = config.heads * config.head_dim
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.to_q = nn.Linear(config.hidden_size, inner, bias=False, device=device)
        self.to_k = nn.Linear(config.hidden_size, inner, bias=False, device=device)
        self.to_v = nn.Linear(config.hidden_size, inner, bias=False, device=device)
        self.to_out = nn.Sequential(nn.Linear(inner, config.hidden_size, bias=False, device=device))
        self.norm_q = _RMSNorm(config.head_dim, config.qk_norm_eps, device=device)
        self.norm_k = _RMSNorm(config.head_dim, config.qk_norm_eps, device=device)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, rows, _ = hidden.shape
        query = self.norm_q(self.to_q(hidden).view(batch, rows, self.heads, self.head_dim))
        key = self.norm_k(self.to_k(hidden).view(batch, rows, self.heads, self.head_dim))
        value = self.to_v(hidden).view(batch, rows, self.heads, self.head_dim)
        result = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        return self.to_out(result.transpose(1, 2).reshape(batch, rows, -1))


class _TokenRefinerBlock(nn.Module):
    def __init__(self, config: H3TransformerConfig, *, device: torch.device | str) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(config.hidden_size, config.norm_eps, device=device)
        self.attn = _DenseAttention(config, device=device)
        self.norm2 = _RMSNorm(config.hidden_size, config.norm_eps, device=device)
        self.ff = _FeedForward(config, device=device)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(self.norm1(hidden))
        return hidden + self.ff(self.norm2(hidden))


class _TokenRefiner(nn.Module):
    def __init__(self, config: H3TransformerConfig, *, device: torch.device | str) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            _TokenRefinerBlock(config, device=device) for _ in range(config.refiner_layers)
        )
        self.final_norm = _RMSNorm(config.hidden_size, config.norm_eps, device=device)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.refiner_blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)


class _H3Attention(nn.Module):
    def __init__(
        self,
        config: H3TransformerConfig,
        mesh: DeviceMesh,
        vsa: H3VsaAttention,
        *,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        inner = config.heads * config.head_dim
        self.config = config
        self.mesh = mesh
        self.vsa = vsa
        self.to_q = nn.Linear(config.hidden_size, inner, bias=False, device=device)
        self.to_k = nn.Linear(config.hidden_size, inner, bias=False, device=device)
        self.to_v = nn.Linear(config.hidden_size, inner, bias=False, device=device)
        self.to_gate_compress = nn.Linear(config.hidden_size, inner, bias=False, device=device)
        self.to_out = nn.Sequential(nn.Linear(inner, config.hidden_size, bias=False, device=device))
        self.norm_q = _RMSNorm(config.head_dim, config.qk_norm_eps, device=device)
        self.norm_k = _RMSNorm(config.head_dim, config.qk_norm_eps, device=device)

    def _exchange(
        self,
        send: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        size = self.mesh.size("sp")
        if size == 1:
            output.copy_(send[0])
            return output
        _, local_rows, local_heads, projections, width = send.shape
        count = local_rows * local_heads * projections * width
        self.mesh.all_to_all_single_into(
            output.reshape(-1),
            send.reshape(-1),
            (count,) * size,
            (count,) * size,
        )
        return output

    def _reverse_exchange(
        self,
        value: torch.Tensor,
        receive: torch.Tensor,
        local_output: torch.Tensor,
    ) -> torch.Tensor:
        size = self.mesh.size("sp")
        if size == 1:
            local_output.copy_(value)
            return local_output
        global_rows, local_heads, width = value.shape
        local_rows = global_rows // size
        count = local_rows * local_heads * width
        self.mesh.all_to_all_single_into(
            receive.reshape(-1),
            value.reshape(-1),
            (count,) * size,
            (count,) * size,
        )
        local_output.copy_(
            receive.permute(1, 0, 2, 3).reshape(
                local_rows, local_heads * size, width
            )
        )
        return local_output

    def forward(
        self,
        hidden: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        tile_valid_sizes: torch.Tensor,
        row_valid_mask: torch.Tensor,
        prefix_key_indices: torch.Tensor,
        dense_key_indices: torch.Tensor,
        prefix_count: torch.Tensor,
        projection_buffer: torch.Tensor,
        qkvg_send: torch.Tensor,
        qkvg_exchange: torch.Tensor,
        attention_output: torch.Tensor,
        tile_scores: torch.Tensor,
        block_counts: torch.Tensor,
        block_indices: torch.Tensor,
        pooled_query: torch.Tensor,
        pooled_key: torch.Tensor,
        pooled_value: torch.Tensor,
        compressed_tiles: torch.Tensor,
        topk_values: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_indices_i32: torch.Tensor,
    ) -> torch.Tensor:
        local = hidden[0]
        heads, head_dim = self.config.heads, self.config.head_dim
        local_rows = local.shape[0]
        sp_size = self.mesh.size("sp")
        local_heads = heads // sp_size
        projection_staging = qkvg_exchange.view(4, local_rows, heads, head_dim)
        for projection_index, projection in enumerate(
            (self.to_q, self.to_k, self.to_v, self.to_gate_compress)
        ):
            torch.mm(
                local,
                projection.weight.t(),
                out=projection_staging[projection_index].view(
                    local_rows, heads * head_dim
                ),
            )
        qkvg_send.copy_(
            projection_staging.view(
                4, local_rows, sp_size, local_heads, head_dim
            ).permute(2, 1, 3, 0, 4)
        )
        exchanged = self._exchange(qkvg_send, qkvg_exchange)
        query, key, value, gate = exchanged.unbind(2)
        cosine, sine = rotary
        normalized_query, normalized_key = qk_rmsnorm_rope(
            query,
            key,
            self.norm_q.weight,
            self.norm_k.weight,
            cosine[:, None],
            sine[:, None],
            eps=self.config.qk_norm_eps,
        )
        query.copy_(normalized_query)
        key.copy_(normalized_key)
        exchanged.masked_fill_(
            ~row_valid_mask.view(-1, 1, 1, 1),
            0,
        )
        self.vsa(
            query,
            key,
            value,
            gate,
            tile_valid_sizes,
            prefix_key_indices,
            dense_key_indices,
            prefix_count,
            output=attention_output,
            tile_scores=tile_scores,
            block_counts=block_counts,
            block_indices=block_indices,
            pooled_query=pooled_query,
            pooled_key=pooled_key,
            pooled_value=pooled_value,
            compressed_tiles=compressed_tiles,
            topk_values=topk_values,
            topk_indices=topk_indices,
            topk_indices_i32=topk_indices_i32,
        )
        receive = qkvg_send.reshape(-1)[: attention_output.numel()].view(
            self.mesh.size("sp"), local_rows, local_heads, head_dim
        )
        local_output = self._reverse_exchange(
            attention_output,
            receive,
            projection_buffer,
        )
        return self.to_out(local_output.reshape(1, local_output.shape[0], -1))


class _AdaModulation(nn.Module):
    def __init__(self, config: H3TransformerConfig, *, device: torch.device | str) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.linear = nn.Linear(config.time_dim, config.hidden_size * 6 * MODALITIES, device=device)

    def forward(self, time: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = self.linear(F.silu(time).to(self.linear.weight.dtype)).reshape(
            -1, self.hidden_size * 6
        )
        return values.chunk(6, dim=-1)


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        config: H3TransformerConfig,
        mesh: DeviceMesh,
        vsa: H3VsaAttention,
        *,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(config.hidden_size, config.norm_eps, device=device)
        self.attn = _H3Attention(config, mesh, vsa, device=device)
        self.norm2 = _RMSNorm(config.hidden_size, config.norm_eps, device=device)
        self.ff = _FeedForward(config, device=device)
        self.adaln_proj = _AdaModulation(config, device=device)

    def forward(
        self,
        hidden: torch.Tensor,
        time: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        tile_valid_sizes: torch.Tensor,
        row_valid_mask: torch.Tensor,
        prefix_key_indices: torch.Tensor,
        dense_key_indices: torch.Tensor,
        prefix_count: torch.Tensor,
        projection_buffer: torch.Tensor,
        qkvg_send: torch.Tensor,
        qkvg_exchange: torch.Tensor,
        attention_output: torch.Tensor,
        tile_scores: torch.Tensor,
        block_counts: torch.Tensor,
        block_indices: torch.Tensor,
        pooled_query: torch.Tensor,
        pooled_key: torch.Tensor,
        pooled_value: torch.Tensor,
        compressed_tiles: torch.Tensor,
        topk_values: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_indices_i32: torch.Tensor,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
            tensor.to(hidden.dtype) for tensor in self.adaln_proj(time)
        )
        normalized = row_modulated_rmsnorm(
            hidden,
            self.norm1.weight,
            shift_attn,
            scale_attn,
            adaln_indices,
            eps=self.norm1.eps,
        )
        hidden = hidden + gate_attn.index_select(0, adaln_indices) * self.attn(
            normalized,
            rotary,
            tile_valid_sizes,
            row_valid_mask,
            prefix_key_indices,
            dense_key_indices,
            prefix_count,
            projection_buffer,
            qkvg_send,
            qkvg_exchange,
            attention_output,
            tile_scores,
            block_counts,
            block_indices,
            pooled_query,
            pooled_key,
            pooled_value,
            compressed_tiles,
            topk_values,
            topk_indices,
            topk_indices_i32,
        )
        normalized = row_modulated_rmsnorm(
            hidden,
            self.norm2.weight,
            shift_ffn,
            scale_ffn,
            adaln_indices,
            eps=self.norm2.eps,
        )
        return hidden + gate_ffn.index_select(0, adaln_indices) * self.ff(normalized)


class _OutputNorm(nn.Module):
    def __init__(self, config: H3TransformerConfig, *, device: torch.device | str) -> None:
        super().__init__()
        self.norm = _RMSNorm(config.hidden_size, config.norm_eps, device=device)
        self.linear = nn.Linear(config.time_dim, config.hidden_size * 2, device=device)

    def forward(self, hidden: torch.Tensor, time: torch.Tensor, timestep_indices: torch.Tensor) -> torch.Tensor:
        shift, scale = self.linear(F.silu(time).to(self.linear.weight.dtype)).chunk(2, dim=-1)
        normalized = self.norm(hidden)
        return normalized * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


class MiniMaxH3Transformer(nn.Module):
    """Replicated DiT weights with SP4 rank-local activations."""

    architecture = "MiniMaxH3Transformer3DModel"

    def __init__(
        self,
        mesh: DeviceMesh,
        layout: H3Layout,
        *,
        parameter_device: torch.device | str = "meta",
        attention_mode: Literal[
            "sparse_kernel", "sparse_oracle", "dense_oracle"
        ] = "sparse_kernel",
    ) -> None:
        super().__init__()
        config = H3TransformerConfig()
        if config.heads % mesh.size("sp"):
            raise ValueError("H3 attention heads must divide the sequence-parallel size")
        self.config = config
        self.mesh = mesh
        self.layout = layout
        metadata = build_vsa_metadata(
            padded_rows=layout.packed.padded_rows,
            prefix_tiles=layout.packed.prefix_tiles,
            video_tiles=layout.packed.video_tiles,
            valid_sizes=layout.packed.tile_valid_sizes,
            device=mesh.local_device,
        )
        vsa = H3VsaAttention(mesh, metadata, mode=attention_mode)
        video_patch_width = config.video_channels * 4
        self.proj_in = nn.Linear(video_patch_width, config.hidden_size, device=parameter_device)
        self.audio_proj_in = nn.Linear(config.audio_channels, config.hidden_size, device=parameter_device)
        self.context_embedder = nn.Linear(config.text_dim, config.hidden_size, device=parameter_device)
        self.time_embedder = _TimeEmbedding(
            config,
            device=parameter_device,
            buffer_device=mesh.local_device,
        )
        self.rope = _RotaryEmbedding(config, device=mesh.local_device)
        self.token_refiner = _TokenRefiner(config, device=parameter_device)
        self.transformer_blocks = nn.ModuleList(
            _TransformerBlock(config, mesh, vsa, device=parameter_device)
            for _ in range(config.layers)
        )
        self.norm_out = _OutputNorm(config, device=parameter_device)
        self.proj_out = nn.Linear(config.hidden_size, video_patch_width, device=parameter_device)
        self.audio_proj_out = nn.Linear(config.hidden_size, config.audio_channels, device=parameter_device)
        local_text = layout.local_indices(layout.packed.text_indices)
        global_text = layout.packed.text_indices[
            (layout.packed.text_indices >= layout.local_start)
            & (layout.packed.text_indices < layout.local_end)
        ]
        local_video = layout.local_indices(layout.packed.video_indices)
        local_audio = layout.local_indices(layout.packed.audio_indices)
        local_tags = layout.packed.token_tags[layout.local_start : layout.local_end]
        timestep_indices = (local_tags == AUDIO_TAG).to(torch.long)
        for name, value in (
            ("local_text_indices", local_text),
            ("global_text_indices", global_text),
            ("local_video_indices", local_video),
            ("local_audio_indices", local_audio),
            ("timestep_indices", timestep_indices),
            ("adaln_indices", timestep_indices * MODALITIES + local_tags),
            ("positions", layout.packed.position_ids.to(torch.float32)),
            ("non_text_mask", layout.packed.token_tags != 1),
        ):
            self.register_buffer(
                name,
                value.to(mesh.local_device),
                persistent=False,
            )

    def refine_text(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        return self.token_refiner(
            self.context_embedder(encoder_hidden.to(self.context_embedder.weight.dtype))
        )

    def forward_local(
        self,
        slot: H3StateSlot,
        scratch: H3Scratch,
        *,
        video_timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = scratch.packed_hidden
        hidden.zero_()
        if self.local_text_indices.numel():
            torch.index_select(
                slot.text_condition,
                1,
                self.global_text_indices,
                out=scratch.local_text_hidden,
            )
            hidden.index_copy_(1, self.local_text_indices, scratch.local_text_hidden)
        for rows, projection, indices in (
            (slot.video_rows, self.proj_in, self.local_video_indices),
            (slot.audio_rows, self.audio_proj_in, self.local_audio_indices),
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

        scratch.time_values[0].copy_(video_timestep)
        scratch.time_values[1].copy_(audio_timestep)
        time = self.time_embedder(scratch.time_values)
        rotary = (slot.rotary_cosine, slot.rotary_sine)
        for block in self.transformer_blocks:
            hidden = block(
                hidden,
                time,
                self.adaln_indices,
                rotary,
                slot.tile_valid_sizes,
                slot.row_valid_mask,
                slot.prefix_key_indices,
                slot.dense_key_indices,
                slot.prefix_count,
                scratch.projection_buffer,
                scratch.qkvg_send,
                scratch.qkvg_exchange,
                scratch.attention_output,
                scratch.tile_scores,
                scratch.block_counts,
                scratch.block_indices,
                scratch.pooled_query,
                scratch.pooled_key,
                scratch.pooled_value,
                scratch.compressed_tiles,
                scratch.topk_values,
                scratch.topk_indices,
                scratch.topk_indices_i32,
            )
        hidden = self.norm_out(hidden, time, self.timestep_indices)
        for indices, projection, velocity in (
            (self.local_video_indices, self.proj_out, scratch.video_velocity),
            (self.local_audio_indices, self.audio_proj_out, scratch.audio_velocity),
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

    def compile_blocks(self) -> None:
        for index, block in enumerate(self.transformer_blocks):
            self.transformer_blocks[index] = torch.compile(block, fullgraph=True)
