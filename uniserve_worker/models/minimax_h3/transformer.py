"""MiniMax H3 transformer math over rank-local packed rows."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from ...backends.attention.torch_sdpa import TorchSDPAAttentionBackend
from ...backends.attention.video_sparse import (
    VideoSparseAttentionBackend,
    VideoSparseAttentionWorkspace,
    build_video_sparse_metadata,
)
from ...nn.attention import RadixAttention
from ...nn.diffusion.integrator import clean_sample_euler_step_
from ...nn.diffusion.modulation import ModulationPlan
from ...nn.diffusion.schedule import DiffusionSchedule
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
from ...nn.quant.base import PreparedLinearInput, UnquantizedLinearMethod
from ...nn.quant.config import LinearPrecision, create_linear_method
from ...ops import (
    gated_residual,
    gated_residual_rms_norm,
    gated_residual_rms_norm_fp8,
    modulated_rms_norm,
    qk_norm_rope,
)
from ...profiling import profile_range
from .packing import AUDIO_TAG
from .state import H3Layout, H3Scratch, H3Tensors

__all__ = [
    "H3TransformerConfig",
    "H3TransformerMetadata",
    "MiniMaxH3Transformer",
]

MODALITIES = 3

if TYPE_CHECKING:
    from .model import H3ComputeInputs


@dataclass(frozen=True, slots=True)
class H3TransformerConfig:
    """Defines H3 multimodal width, layer, attention, expert, modulation, and sparse-video geometry."""

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


@dataclass(frozen=True, slots=True)
class H3TransformerMetadata:
    """Holds rank-local row indices, routing masks, positions, and sparse-attention metadata for one H3 forward."""

    layout: H3Layout
    vsa: VideoSparseAttentionBackend
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
        self.attention = RadixAttention(
            self.heads, self.heads, self.head_dim, dense_provider=TorchSDPAAttentionBackend()
        )
        self.to_q = nn.Linear(config.hidden_size, inner, bias=False, device=device)
        self.to_k = nn.Linear(config.hidden_size, inner, bias=False, device=device)
        self.to_v = nn.Linear(config.hidden_size, inner, bias=False, device=device)
        self.to_out = nn.Sequential(nn.Linear(inner, config.hidden_size, bias=False, device=device))
        self.norm_q = RMSNorm(
            config.head_dim, config.qk_norm_eps, affine_in_fp32=True, device=device
        )
        self.norm_k = RMSNorm(
            config.head_dim, config.qk_norm_eps, affine_in_fp32=True, device=device
        )

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
        self.norm1 = RMSNorm(
            config.hidden_size, config.norm_eps, affine_in_fp32=True, device=device
        )
        self.attn = _DenseAttention(config, device=device)
        self.norm2 = RMSNorm(
            config.hidden_size, config.norm_eps, affine_in_fp32=True, device=device
        )
        with torch.device(device):
            self.ff = GatedMLP(
                config.hidden_size,
                config.ffn_dim,
                quant_method=create_linear_method(linear_precision),
                layer_config=layer_config.child("ff"),
                order="value_gate",
                activation_dtype=torch.bfloat16,
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
        self.final_norm = RMSNorm(
            config.hidden_size, config.norm_eps, affine_in_fp32=True, device=device
        )

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
    ) -> None:
        """Bind sharded projections to the sparse-video attention workspace contract."""

        super().__init__()
        inner = config.heads * config.head_dim
        self.config = config
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
        with torch.device(device):
            self.to_qkvg = InterleavedMergedColumnParallelLinear(
                config.hidden_size,
                inner,
                4,
                config.head_dim,
                layer_config=projection_config,
                prefix="to_qkvg",
                sequence_group=mesh.get_group("sp"),
                input_scale_group=mesh.get_group("sp"),
                weight_scale_partition_size=14 * 4 * config.head_dim,
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
                    logical_input_row_partitions=4 // mesh.size("sp"),
                )
            )
        self.norm_q = RMSNorm(
            config.head_dim, config.qk_norm_eps, affine_in_fp32=True, device=device
        )
        self.norm_k = RMSNorm(
            config.head_dim, config.qk_norm_eps, affine_in_fp32=True, device=device
        )

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
        backend: VideoSparseAttentionBackend,
    ) -> torch.Tensor | AttentionRowExchange:
        """Compute sparse global attention and publish its row-exchange dependency."""

        local = hidden[0]
        head_dim = self.config.head_dim
        local_rows = local.shape[0]
        global_rows = local_rows * self.sequence_size
        if self.projected_head:
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
    ) -> None:
        """Assemble one adaptive sparse-attention and gated feed-forward block."""

        super().__init__()
        self.norm1 = RMSNorm(
            config.hidden_size, config.norm_eps, affine_in_fp32=True, device=device
        )
        self.attn = _H3Attention(
            config,
            mesh,
            linear_precision=attention_linear_precision,
            layer_config=layer_config.child("attn"),
            device=device,
        )
        self.norm2 = RMSNorm(
            config.hidden_size, config.norm_eps, affine_in_fp32=True, device=device
        )
        with torch.device(device):
            self.ff = GatedMLP(
                config.hidden_size,
                config.ffn_dim,
                logical_input_row_partitions=4 // mesh.size("sp"),
                # Preserve complete FP32 dot products across physical row intervals.
                quant_method=(
                    UnquantizedLinearMethod(accumulation_dtype=torch.float32)
                    if mlp_linear_precision == "bf16"
                    else create_linear_method(mlp_linear_precision)
                ),
                layer_config=layer_config.child("ff"),
                order="value_gate",
                activation_dtype=torch.bfloat16,
            )
        self.hidden_size = config.hidden_size
        # Projected-head execution provides three registered transport buffers;
        # dense projections keep each row independent of tensor-wide scales.
        self.overlap_output_exchange = self.attn.projected_head and all(
            not projection.quant_method.is_quantized
            for projection in (self.attn.to_out[0], self.ff.gate_up_proj, self.ff.down_proj)
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
        backend: VideoSparseAttentionBackend,
    ) -> torch.Tensor:
        """Apply one time-modulated sparse-attention and feed-forward residual block."""

        # Parameters are stored by modality; row indices select the appropriate
        # six-vector modulation tuple for each packed hidden row.
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
            tensor.to(hidden.dtype)
            for tensor in adaln_values.reshape(-1, self.hidden_size * 6).chunk(6, dim=-1)
        )
        normalized = modulated_rms_norm(
            hidden,
            self.norm1.weight,
            shift_attn,
            scale_attn,
            adaln_indices,
            eps=self.norm1.eps,
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
        )
        del normalized

        if isinstance(attention, AttentionRowExchange):
            if self.overlap_output_exchange:
                output = torch.empty_like(hidden)
                # QKVG projection has consumed this registered gather buffer.
                # Reuse it for receives while keeping composed input read-only.
                for interval, rows in attention.chunks(attention_workspace):
                    projected = self.attn.to_out(rows.reshape(1, rows.shape[0], -1))
                    output[:, interval] = self._finish_attention(
                        hidden[:, interval],
                        projected,
                        gate_attn,
                        shift_ffn,
                        scale_ffn,
                        gate_ffn,
                        adaln_indices[interval],
                    )
                return output
            attention = attention.materialize()
        projected = self.attn.to_out(attention.reshape(1, attention.shape[0], -1))
        return self._finish_attention(
            hidden, projected, gate_attn, shift_ffn, scale_ffn, gate_ffn, adaln_indices
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
        self.norm = RMSNorm(config.hidden_size, config.norm_eps, affine_in_fp32=True, device=device)

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
        execution: H3ComputeInputs,
        start_step: int,
        step_count: int,
        schedule: DiffusionSchedule,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance explicit media Tensor views through one scheduled solver step."""

        if int(step_count) != 1:
            raise ValueError("H3 denoise calls evaluate exactly one scheduled step")
        if not 0 <= start_step < 4:
            raise ValueError("H3 denoise step is outside the four-evaluation ladder")
        scratch = execution.scratch
        assert scratch is not None and execution.transformer_metadata is not None
        self.select_adaln_step(scratch, start_step)
        with profile_range("uniserve.h3.denoise"):
            self._forward_layers(slot, scratch, execution.transformer_metadata)
            if self.pipeline.last:
                clean_sample_euler_step_(
                    slot.video_rows,
                    scratch.video_velocity,
                    schedule.timesteps[0][start_step],
                    schedule.sigmas[0][start_step],
                    schedule.sigmas[0][start_step + 1],
                )
                clean_sample_euler_step_(
                    slot.audio_rows,
                    scratch.audio_velocity,
                    schedule.timesteps[1][start_step],
                    schedule.sigmas[1][start_step],
                    schedule.sigmas[1][start_step + 1],
                )
            self.pipeline.feedback((slot.video_rows, slot.audio_rows))
        return slot.video_rows, slot.audio_rows

    def _forward_layers(
        self,
        slot: H3Tensors,
        scratch: H3Scratch,
        metadata: H3TransformerMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Run assigned layers and publish media predictions on the final stage."""

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
        for layer, block in enumerate(self.transformer_blocks.values()):
            hidden = block(
                hidden,
                scratch.block_adaln_params[layer],
                metadata.adaln_indices,
                rotary,
                slot.tile_valid_sizes,
                slot.prefix_key_indices,
                slot.dense_key_indices,
                slot.prefix_count,
                scratch.projection_peers,
                scratch.projection_sync_input,
                scratch.projection_sync_output,
                scratch.attention_workspace,
                scratch.attention_output,
                scratch.context_workspace,
                scratch.tile_scores,
                scratch.block_counts,
                scratch.block_indices,
                scratch.pooled_query,
                scratch.pooled_key,
                scratch.pooled_value,
                scratch.compressed_tiles,
                scratch.topk_indices_i32,
                metadata.vsa,
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
