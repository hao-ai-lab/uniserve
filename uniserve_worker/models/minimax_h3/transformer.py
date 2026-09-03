"""MiniMax H3 transformer math over rank-local packed rows."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ...backends.attention.video_sparse import (
    VideoSparseAttentionBackend,
    VideoSparseAttentionWorkspace,
    build_video_sparse_metadata,
)
from ...nn.diffusion.modulation import prepare_modulation_plan, select_modulation_step
from ...nn.layer import LayerConfig
from ...nn.linear import InterleavedMergedColumnParallelLinear, LinearBase
from ...nn.mesh import DeviceMesh, SymmetricMemoryWorkspace, TensorParallel
from ...nn.quant import (
    DynamicW4A4NvFp4LinearMethod,
    DynamicW8A8Fp8LinearMethod,
    DynamicW8A8MxFp8LinearMethod,
    QuantizeMethodBase,
    UnquantizedLinearMethod,
)
from ...ops import qk_norm_rope
from .fusions import (
    attention_residual_modulated_rmsnorm,
    attention_residual_modulated_rmsnorm_fp8,
    gated_residual,
    row_modulated_rmsnorm,
    value_first_swiglu,
    value_first_swiglu_fp8,
)
from .packing import AUDIO_TAG
from .precision import LinearPrecision
from .state import H3Layout, H3Scratch, H3StateSlot

__all__ = [
    "H3TransformerConfig",
    "H3TransformerExecution",
    "LinearPrecision",
    "MiniMaxH3Transformer",
]

MODALITIES = 3


def _dynamic_quant_method(
    precision: LinearPrecision,
    *,
    tensorwise: bool = False,
) -> QuantizeMethodBase:
    if precision == "bf16":
        return UnquantizedLinearMethod()
    if precision == "fp8":
        return DynamicW8A8Fp8LinearMethod(tensorwise=tensorwise)
    if precision == "mxfp8":
        if tensorwise:
            raise ValueError("MXFP8 is not supported at the sequence-parallel attention boundary")
        return DynamicW8A8MxFp8LinearMethod()
    if precision == "nvfp4":
        return DynamicW4A4NvFp4LinearMethod()
    raise ValueError(f"unsupported dynamic linear precision {precision!r}")


def _dynamic_quantized_linear(
    input_size: int,
    output_size: int,
    *,
    linear_precision: LinearPrecision,
    layer_config: LayerConfig,
    device: torch.device | str,
    bias: bool = True,
    tensorwise: bool = False,
) -> LinearBase:
    with torch.device(device):
        linear = LinearBase(
            input_size,
            output_size,
            layer_config=layer_config,
            quant_method=_dynamic_quant_method(linear_precision, tensorwise=tensorwise),
            bias=bias,
        )
    return linear


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


@dataclass(frozen=True, slots=True)
class H3TransformerExecution:
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


class _RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float, *, device: torch.device | str) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, device=device))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(value.dtype)


class _SwiGLUProjection(nn.Module):
    def __init__(
        self,
        width: int,
        expanded: int,
        *,
        linear_precision: LinearPrecision,
        layer_config: LayerConfig,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.proj = _dynamic_quantized_linear(
            width,
            expanded * 2,
            linear_precision=linear_precision,
            bias=False,
            layer_config=layer_config,
            device=device,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value_first_swiglu(self.proj(value))


class _FeedForward(nn.Module):
    def __init__(
        self,
        config: H3TransformerConfig,
        *,
        linear_precision: LinearPrecision,
        layer_config: LayerConfig,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            (
                _SwiGLUProjection(
                    config.hidden_size,
                    config.ffn_dim,
                    linear_precision=linear_precision,
                    layer_config=layer_config,
                    device=device,
                ),
                nn.Identity(),
                _dynamic_quantized_linear(
                    config.ffn_dim,
                    config.hidden_size,
                    linear_precision=linear_precision,
                    bias=False,
                    layer_config=layer_config,
                    device=device,
                ),
            )
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.accepts_prequantized_fp8:
            value_gate = self.net[0].proj(value)
            activated, activated_scale = value_first_swiglu_fp8(value_gate)
            return self.net[2].forward_prequantized(activated, activated_scale)
        return self.net[2](self.net[0](value))

    @property
    def accepts_prequantized_fp8(self) -> bool:
        return all(
            isinstance(linear.quant_method, DynamicW8A8Fp8LinearMethod)
            and not linear.quant_method.tensorwise
            for linear in (self.net[0].proj, self.net[2])
        )

    def forward_prequantized_fp8(
        self,
        value: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if not self.accepts_prequantized_fp8:
            raise RuntimeError("feed-forward precision cannot consume prequantized FP8 input")
        value_gate = self.net[0].proj.forward_prequantized(value, scale)
        activated, activated_scale = value_first_swiglu_fp8(value_gate)
        return self.net[2].forward_prequantized(activated, activated_scale)


class _RotaryEmbedding(nn.Module):
    def __init__(self, config: H3TransformerConfig, *, device: torch.device | str) -> None:
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
    def __init__(
        self,
        config: H3TransformerConfig,
        *,
        linear_precision: LinearPrecision,
        layer_config: LayerConfig,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(config.hidden_size, config.norm_eps, device=device)
        self.attn = _DenseAttention(config, device=device)
        self.norm2 = _RMSNorm(config.hidden_size, config.norm_eps, device=device)
        self.ff = _FeedForward(
            config,
            linear_precision=linear_precision,
            layer_config=layer_config,
            device=device,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(self.norm1(hidden))
        return hidden + self.ff(self.norm2(hidden))


class _TokenRefiner(nn.Module):
    def __init__(
        self,
        config: H3TransformerConfig,
        *,
        linear_precision: LinearPrecision,
        layer_config: LayerConfig,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            _TokenRefinerBlock(
                config,
                linear_precision=linear_precision,
                layer_config=layer_config,
                device=device,
            )
            for _ in range(config.refiner_layers)
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
        vsa: VideoSparseAttentionBackend,
        *,
        linear_precision: LinearPrecision,
        layer_config: LayerConfig,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        inner = config.heads * config.head_dim
        self.config = config
        self.mesh = mesh
        self.vsa = vsa
        with torch.device(device):
            self.to_qkvg = InterleavedMergedColumnParallelLinear(
                config.hidden_size,
                inner,
                4,
                config.head_dim,
                layer_config=layer_config,
                quant_method=_dynamic_quant_method(linear_precision, tensorwise=True),
                bias=False,
            )
        self.to_out = nn.Sequential(
            _dynamic_quantized_linear(
                inner,
                config.hidden_size,
                linear_precision=linear_precision,
                bias=False,
                layer_config=layer_config,
                device=device,
            )
        )
        self.norm_q = _RMSNorm(config.head_dim, config.qk_norm_eps, device=device)
        self.norm_k = _RMSNorm(config.head_dim, config.qk_norm_eps, device=device)

    def forward(
        self,
        hidden: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        tile_valid_sizes: torch.Tensor,
        prefix_key_indices: torch.Tensor,
        dense_key_indices: torch.Tensor,
        prefix_count: torch.Tensor,
        projection_peers: tuple[torch.Tensor, ...],
        projection_exchange: SymmetricMemoryWorkspace,
        projection_sync_input: torch.Tensor,
        projection_sync_output: torch.Tensor,
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
    ) -> torch.Tensor:
        local = hidden[0]
        heads, head_dim = self.config.heads, self.config.head_dim
        local_rows = local.shape[0]
        sp_size = self.mesh.size("sp")
        local_heads = heads // sp_size
        global_rows = local_rows * sp_size
        exchanged = self.to_qkvg.forward_sequence_parallel(
            local,
            self.mesh,
            attention_workspace,
            group="sp",
        ).view(
            global_rows,
            local_heads,
            4,
            head_dim,
        )
        query, key, value, gate = exchanged.unbind(2)
        cosine, sine = rotary
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
            exchange=projection_exchange,
            exchange_outputs=projection_peers,
            exchange_sync_input=projection_sync_input,
            exchange_sync_output=projection_sync_output,
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
        local_output = self.vsa.forward_local(
            query,
            key,
            value,
            gate,
            tile_valid_sizes,
            prefix_key_indices,
            dense_key_indices,
            prefix_count,
            workspace,
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
        vsa: VideoSparseAttentionBackend,
        *,
        attention_linear_precision: LinearPrecision,
        mlp_linear_precision: LinearPrecision,
        layer_config: LayerConfig,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(config.hidden_size, config.norm_eps, device=device)
        self.attn = _H3Attention(
            config,
            mesh,
            vsa,
            linear_precision=attention_linear_precision,
            layer_config=layer_config,
            device=device,
        )
        self.norm2 = _RMSNorm(config.hidden_size, config.norm_eps, device=device)
        self.ff = _FeedForward(
            config,
            linear_precision=mlp_linear_precision,
            layer_config=layer_config,
            device=device,
        )
        self.adaln_proj = _AdaModulation(config, device=device)

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
        projection_exchange: SymmetricMemoryWorkspace,
        projection_sync_input: torch.Tensor,
        projection_sync_output: torch.Tensor,
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
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
            tensor.to(hidden.dtype)
            for tensor in adaln_values.reshape(-1, self.adaln_proj.hidden_size * 6).chunk(6, dim=-1)
        )
        normalized = row_modulated_rmsnorm(
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
            projection_exchange,
            projection_sync_input,
            projection_sync_output,
            attention_workspace,
            attention_output,
            tile_scores,
            block_counts,
            block_indices,
            pooled_query,
            pooled_key,
            pooled_value,
            compressed_tiles,
            topk_indices_i32,
        )
        if self.ff.accepts_prequantized_fp8:
            hidden, normalized, normalized_scale = attention_residual_modulated_rmsnorm_fp8(
                hidden,
                attention,
                gate_attn,
                self.norm2.weight,
                shift_ffn,
                scale_ffn,
                adaln_indices,
                eps=self.norm2.eps,
            )
            feed_forward = self.ff.forward_prequantized_fp8(
                normalized,
                normalized_scale,
            )
        else:
            hidden, normalized = attention_residual_modulated_rmsnorm(
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
    def __init__(self, config: H3TransformerConfig, *, device: torch.device | str) -> None:
        super().__init__()
        self.norm = _RMSNorm(config.hidden_size, config.norm_eps, device=device)
        self.linear = nn.Linear(config.time_dim, config.hidden_size * 2, device=device)

    def forward(
        self, hidden: torch.Tensor, time: torch.Tensor, timestep_indices: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.linear(F.silu(time).to(self.linear.weight.dtype)).chunk(2, dim=-1)
        return self.forward_precomputed(hidden, shift, scale, timestep_indices)

    def forward_precomputed(
        self,
        hidden: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
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
        attention_linear_precision: LinearPrecision,
        mlp_linear_precision: LinearPrecision,
    ) -> None:
        super().__init__()
        config = H3TransformerConfig()
        if config.heads % mesh.size("sp"):
            raise ValueError("H3 attention heads must divide the sequence-parallel size")
        self.config = config
        self.mesh = mesh
        self.layout = layout
        self.attention_linear_precision = attention_linear_precision
        self.mlp_linear_precision = mlp_linear_precision
        layer_config = LayerConfig(
            parallel=TensorParallel(
                rank=mesh.coord("sp"),
                size=mesh.size("sp"),
            ),
            quantization=None,
        )
        execution = self.build_execution(layout)
        video_patch_width = config.video_channels * 4
        self.proj_in = nn.Linear(video_patch_width, config.hidden_size, device=parameter_device)
        self.audio_proj_in = nn.Linear(
            config.audio_channels, config.hidden_size, device=parameter_device
        )
        self.context_embedder = nn.Linear(
            config.text_dim, config.hidden_size, device=parameter_device
        )
        self.time_embedder = _TimeEmbedding(
            config,
            device=parameter_device,
            buffer_device=mesh.local_device,
        )
        self.rope = _RotaryEmbedding(config, device=mesh.local_device)
        self.token_refiner = _TokenRefiner(
            config,
            linear_precision="bf16",
            layer_config=layer_config,
            device=parameter_device,
        )
        self.transformer_blocks = nn.ModuleList(
            _TransformerBlock(
                config,
                mesh,
                execution.vsa,
                attention_linear_precision=attention_linear_precision,
                mlp_linear_precision=mlp_linear_precision,
                layer_config=layer_config,
                device=parameter_device,
            )
            for _ in range(config.layers)
        )
        self.norm_out = _OutputNorm(config, device=parameter_device)
        self.proj_out = nn.Linear(config.hidden_size, video_patch_width, device=parameter_device)
        self.audio_proj_out = nn.Linear(
            config.hidden_size, config.audio_channels, device=parameter_device
        )
        for name in (
            "local_text_indices",
            "global_text_indices",
            "local_video_indices",
            "local_audio_indices",
            "timestep_indices",
            "adaln_indices",
            "positions",
            "non_text_mask",
        ):
            self.register_buffer(
                name,
                getattr(execution, name),
                persistent=False,
            )
        self.execution = execution

    def build_execution(self, layout: H3Layout) -> H3TransformerExecution:
        metadata = build_video_sparse_metadata(
            padded_rows=layout.packed.padded_rows,
            prefix_tiles=layout.packed.prefix_tiles,
            video_tiles=layout.packed.video_tiles,
            valid_sizes=layout.packed.tile_valid_sizes,
            device=self.mesh.local_device,
        )
        vsa = getattr(self, "vsa", None)
        if vsa is None:
            vsa = VideoSparseAttentionBackend(metadata)
            self.vsa = vsa
        local_tags = layout.packed.token_tags[layout.local_start : layout.local_end]
        timestep_indices = (local_tags == AUDIO_TAG).to(torch.long)
        global_text = layout.packed.text_indices[
            (layout.packed.text_indices >= layout.local_start)
            & (layout.packed.text_indices < layout.local_end)
        ]
        device = self.mesh.local_device
        return H3TransformerExecution(
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

    def bind_execution(self, execution: H3TransformerExecution) -> None:
        self.layout = execution.layout
        self.execution = execution
        for name in (
            "local_text_indices",
            "global_text_indices",
            "local_video_indices",
            "local_audio_indices",
            "timestep_indices",
            "adaln_indices",
            "positions",
            "non_text_mask",
        ):
            setattr(self, name, getattr(execution, name))
        for block in self.transformer_blocks:
            block.attn.vsa = execution.vsa

    def refine_text(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        return self.token_refiner(
            self.context_embedder(encoder_hidden.to(self.context_embedder.weight.dtype))
        )

    @torch.inference_mode()
    def prepare_adaln_plan(
        self,
        slot: H3StateSlot,
        video_timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
    ) -> None:
        activated_time = torch.stack(
            tuple(
                F.silu(
                    self.time_embedder(torch.stack((video_timesteps[step], audio_timesteps[step])))
                )
                for step in range(int(video_timesteps.numel()))
            )
        ).flatten(0, 1)
        prepare_modulation_plan(
            activated_time,
            tuple(
                block.adaln_proj.linear for block in self.transformer_blocks
            ),
            self.norm_out.linear,
            slot.block_adaln_plan,
            slot.final_adaln_plan,
        )

    @staticmethod
    def select_adaln_step(slot: H3StateSlot, scratch: H3Scratch, step: int) -> None:
        select_modulation_step(
            slot.block_adaln_plan,
            slot.final_adaln_plan,
            step,
            scratch.block_adaln_params,
            scratch.final_adaln_params,
        )

    def forward_local_prepared(
        self,
        slot: H3StateSlot,
        scratch: H3Scratch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the fixed-shape local forward from caller-populated timestep storage."""

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

        rotary = (slot.rotary_cosine, slot.rotary_sine)
        for layer, block in enumerate(self.transformer_blocks):
            hidden = block(
                hidden,
                scratch.block_adaln_params[layer],
                self.adaln_indices,
                rotary,
                slot.tile_valid_sizes,
                slot.prefix_key_indices,
                slot.dense_key_indices,
                slot.prefix_count,
                scratch.projection_peers,
                scratch.projection_exchange,
                scratch.projection_sync_input,
                scratch.projection_sync_output,
                scratch.attention_workspace,
                scratch.attention_output,
                scratch.tile_scores,
                scratch.block_counts,
                scratch.block_indices,
                scratch.pooled_query,
                scratch.pooled_key,
                scratch.pooled_value,
                scratch.compressed_tiles,
                scratch.topk_indices_i32,
            )
        final_shift, final_scale = scratch.final_adaln_params.chunk(2, dim=-1)
        hidden = self.norm_out.forward_precomputed(
            hidden,
            final_shift,
            final_scale,
            self.timestep_indices,
        )
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
