"""Tensor-parallel Qwen3-VL text conditioner for MiniMax H3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ... import ops
from ...nn.layer import LayerConfig
from ...nn.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from ...nn.mesh import DeviceMesh, TensorParallel
from ...nn.norm import RMSNorm
from ...nn.placement import WeightMode
from ...nn.quant.nvfp4 import DynamicW4A4NvFp4LinearMethod
from ...nn.vocab_parallel_embedding import VocabParallelEmbedding

__all__ = ["H3TextEncoderConfig", "MiniMaxH3TextEncoder"]


@dataclass(frozen=True, slots=True)
class H3TextEncoderConfig:
    """Defines the H3 text encoder's vocabulary and tensor geometry.

    The configuration fixes hidden width, attention heads, layer count, and rotary
    settings.
    """

    vocab_size: int = 151_936
    hidden_size: int = 5_120
    intermediate_size: int = 25_600
    checkpoint_layers: int = 64
    retained_layers: int = 50
    heads: int = 64
    kv_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 5_000_000.0
    norm_eps: float = 1e-6
    max_text_rows: int = 1_024


def _enable_nvfp4(module: nn.Module) -> None:
    """Finalize eligible text projections for dynamic NVFP4 execution."""

    if not isinstance(module, (MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)):
        return
    if not hasattr(module, "weight_scale"):
        module.register_buffer("weight_scale", None, persistent=False)
    if not hasattr(module, "weight_scale_2"):
        module.register_buffer("weight_scale_2", None, persistent=False)
    method = DynamicW4A4NvFp4LinearMethod()
    module.quant_method = method
    method.process_weights_after_loading(module)


class _TextRotaryEmbedding(nn.Module):
    """Qwen3-VL mRoPE reduced to the text-only position path."""

    def __init__(self, config: H3TextEncoderConfig, *, device: torch.device | str) -> None:
        """Precompute text rotary frequencies on the execution device."""

        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=device)
                / config.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build dtype-matched rotary tables for the supplied text positions."""

        frequencies = positions.float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(hidden.dtype), embedding.sin().to(hidden.dtype)


class _Attention(nn.Module):
    """Applies grouped-query self-attention with rotary query and key coordinates."""

    def __init__(self, config: H3TextEncoderConfig, layer_config: LayerConfig) -> None:
        """Build rank-sharded grouped-query projections and per-head normalization."""

        super().__init__()
        parallel = layer_config.parallel
        self.heads = config.heads // parallel.size
        self.kv_heads = config.kv_heads // parallel.size
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            config.head_dim,
            config.heads,
            config.kv_heads,
            layer_config=layer_config,
            bias=False,
        )
        self.q_size, self.kv_size, _ = self.qkv_proj.output_sizes
        self.o_proj = RowParallelLinear(
            config.heads * config.head_dim,
            config.hidden_size,
            layer_config=layer_config,
            bias=False,
        )
        self.q_norm = RMSNorm(config.head_dim, config.norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        mesh: DeviceMesh,
    ) -> torch.Tensor:
        """Run grouped-query causal attention over one encoded prompt sequence."""

        batch, rows, _ = hidden.shape
        qkv = self.qkv_proj(hidden)
        query, key, value = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        query = query.view(batch, rows, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, rows, self.kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, rows, self.kv_heads, self.head_dim).transpose(1, 2)
        query, key = ops.qk_norm_rope(
            query,
            key,
            self.q_norm.weight,
            self.k_norm.weight,
            rotary[0],
            rotary[1],
            self.q_norm.eps,
            unsqueeze_dim=1,
        )
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=rows > 1,
            scale=self.scaling,
            enable_gqa=self.heads != self.kv_heads,
        )
        return self.o_proj(output.transpose(1, 2).reshape(batch, rows, -1), mesh)


class _MLP(nn.Module):
    """Applies the gated feed-forward projection used by each H3 text layer."""

    def __init__(self, config: H3TextEncoderConfig, layer_config: LayerConfig) -> None:
        """Build the tensor-parallel gated expansion and reduction projections."""

        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            (config.intermediate_size, config.intermediate_size),
            layer_config=layer_config,
            bias=False,
            weight_mode=WeightMode.FUSED_GATE_UP_LINEAR,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            layer_config=layer_config,
            bias=False,
        )

    def forward(self, hidden: torch.Tensor, mesh: DeviceMesh) -> torch.Tensor:
        """Apply the sharded gated expansion and tensor-parallel output projection."""

        return self.down_proj(ops.silu_and_mul(self.gate_up_proj(hidden)), mesh)


class _DecoderLayer(nn.Module):
    """Composes pre-normalized attention and gated MLP residual updates for the H3 text encoder."""

    def __init__(self, config: H3TextEncoderConfig, layer_config: LayerConfig) -> None:
        """Assemble one pre-normalized attention and feed-forward residual layer."""

        super().__init__()
        self.self_attn = _Attention(config, layer_config)
        self.mlp = _MLP(config, layer_config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        rotary: tuple[torch.Tensor, torch.Tensor],
        mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the fused residual stream through attention and gated MLP updates."""

        if residual is None:
            residual = hidden
            attention_input = self.input_layernorm(hidden)
        else:
            attention_input, residual = self.input_layernorm.forward_with_residual(
                hidden,
                residual,
                in_place=True,
            )
        attention_output = self.self_attn(attention_input, rotary, mesh)
        mlp_input, residual = self.post_attention_layernorm.forward_with_residual(
            attention_output,
            residual,
            in_place=True,
        )
        return self.mlp(mlp_input, mesh), residual


class _LanguageModel(nn.Module):
    """Owns token embeddings, decoder layers, and final normalization for H3 text conditioning."""

    def __init__(
        self,
        config: H3TextEncoderConfig,
        mesh: DeviceMesh,
        *,
        parameter_device: torch.device | str,
    ) -> None:
        """Allocate retained text layers and bind them to the encoder device mesh."""

        super().__init__()
        layer_config = LayerConfig(
            parallel=TensorParallel.from_mesh(mesh),
            quantization=None,
        )
        with torch.device(parameter_device):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                layer_config=layer_config,
                init_weights=False,
            )
            self.layers = nn.ModuleList(
                _DecoderLayer(config, layer_config) for _ in range(config.retained_layers)
            )
        self.rotary_emb = _TextRotaryEmbedding(config, device=mesh.local_device)
        self.mesh = mesh

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode one bounded token sequence into checkpoint-layer-50 conditioning states."""

        hidden = self.embed_tokens(token_ids, self.mesh)
        positions = torch.arange(
            token_ids.shape[1], dtype=torch.long, device=token_ids.device
        ).view(1, -1)
        rotary = self.rotary_emb(hidden, positions)
        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, rotary, self.mesh)
        if residual is not None:
            hidden = hidden + residual
        return hidden


class MiniMaxH3TextEncoder(nn.Module):
    """The Qwen3-VL language path through checkpoint hidden state 50."""

    architecture = "Qwen3VLForConditionalGeneration"

    def __init__(
        self,
        mesh: DeviceMesh,
        *,
        max_text_rows: int,
        parameter_device: torch.device | str = "meta",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Configure a bounded BF16 text-conditioning path on the supplied mesh."""

        super().__init__()
        if dtype != torch.bfloat16:
            raise ValueError("the H3 text encoder uses bfloat16 activations")
        self.config = H3TextEncoderConfig(max_text_rows=int(max_text_rows))
        if mesh.size("tp") != 4:
            raise ValueError("the H3 text encoder requires TP4")
        self.mesh = mesh
        self.language_model = _LanguageModel(
            self.config,
            mesh,
            parameter_device=parameter_device,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Validate a single bounded prompt and produce its text-conditioning states."""

        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("H3 text conditioning requires one token sequence")
        if token_ids.shape[1] < 1 or token_ids.shape[1] > self.config.max_text_rows:
            raise ValueError(
                f"H3 prompt token count must be between 1 and {self.config.max_text_rows}"
            )
        return self.language_model(token_ids)

    def enable_nvfp4(self) -> None:
        """Quantize eligible resident text projections to dynamic NVFP4 execution."""

        self.apply(_enable_nvfp4)
