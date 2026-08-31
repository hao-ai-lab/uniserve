"""Resident TP4 Qwen3-VL text conditioner for the fixed H3 T2VA profile."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ...nn.mesh import DeviceMesh, divide
from ...nn.quant.nvfp4 import DynamicW4A4NvFp4LinearMethod

__all__ = ["H3TextEncoderConfig", "MiniMaxH3TextEncoder"]


@dataclass(frozen=True, slots=True)
class H3TextEncoderConfig:
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


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class _RMSNorm(nn.Module):
    def __init__(
        self,
        width: int,
        eps: float,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, device=device, dtype=dtype))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(value.dtype)


class _ColumnLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        mesh: DeviceMesh,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.global_output_size = int(output_size)
        self.local_output_size = divide(output_size, mesh.size("tp"))
        self.input_size = int(input_size)
        self.output_size = self.local_output_size
        self.bias = None
        self.quant_method: DynamicW4A4NvFp4LinearMethod | None = None
        self.weight = nn.Parameter(
            torch.empty(
                (self.local_output_size, input_size),
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(self, value)
        return F.linear(value, self.weight)

    def enable_nvfp4(self) -> None:
        self.register_buffer("weight_scale", None, persistent=False)
        self.register_buffer("weight_scale_2", None, persistent=False)
        self.quant_method = DynamicW4A4NvFp4LinearMethod()
        self.quant_method.process_weights_after_loading(self)


class _RowLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        mesh: DeviceMesh,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.mesh = mesh
        self.global_input_size = int(input_size)
        self.local_input_size = divide(input_size, mesh.size("tp"))
        self.input_size = self.local_input_size
        self.output_size = int(output_size)
        self.bias = None
        self.quant_method: DynamicW4A4NvFp4LinearMethod | None = None
        self.weight = nn.Parameter(
            torch.empty(
                (output_size, self.local_input_size),
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = (
            self.quant_method.apply(self, value)
            if self.quant_method is not None
            else F.linear(value, self.weight)
        )
        return self.mesh.all_reduce(output, "tp")

    def enable_nvfp4(self) -> None:
        self.register_buffer("weight_scale", None, persistent=False)
        self.register_buffer("weight_scale_2", None, persistent=False)
        self.quant_method = DynamicW4A4NvFp4LinearMethod()
        self.quant_method.process_weights_after_loading(self)


class _VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        config: H3TextEncoderConfig,
        mesh: DeviceMesh,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.mesh = mesh
        self.rows = divide(config.vocab_size, mesh.size("tp"))
        self.start = mesh.coord("tp") * self.rows
        self.end = self.start + self.rows
        self.weight = nn.Parameter(
            torch.empty((self.rows, config.hidden_size), device=device, dtype=dtype)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        owned = (token_ids >= self.start) & (token_ids < self.end)
        local_ids = (token_ids - self.start).masked_fill(~owned, 0)
        output = F.embedding(local_ids, self.weight)
        output.masked_fill_(~owned.unsqueeze(-1), 0)
        return self.mesh.all_reduce(output, "tp")


class _TextRotaryEmbedding(nn.Module):
    """Qwen3-VL interleaved mRoPE reduced to its text-only position path."""

    def __init__(
        self,
        config: H3TextEncoderConfig,
        *,
        device: torch.device | str,
    ) -> None:
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
        frequencies = positions.float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(hidden.dtype), embedding.sin().to(hidden.dtype)


class _Attention(nn.Module):
    def __init__(
        self,
        config: H3TextEncoderConfig,
        mesh: DeviceMesh,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        tp_size = mesh.size("tp")
        self.heads = divide(config.heads, tp_size)
        self.kv_heads = divide(config.kv_heads, tp_size)
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5
        self.q_proj = _ColumnLinear(
            config.hidden_size,
            config.heads * config.head_dim,
            mesh,
            device=device,
            dtype=dtype,
        )
        self.k_proj = _ColumnLinear(
            config.hidden_size,
            config.kv_heads * config.head_dim,
            mesh,
            device=device,
            dtype=dtype,
        )
        self.v_proj = _ColumnLinear(
            config.hidden_size,
            config.kv_heads * config.head_dim,
            mesh,
            device=device,
            dtype=dtype,
        )
        self.o_proj = _RowLinear(
            config.heads * config.head_dim,
            config.hidden_size,
            mesh,
            device=device,
            dtype=dtype,
        )
        self.q_norm = _RMSNorm(
            config.head_dim, config.norm_eps, device=device, dtype=dtype
        )
        self.k_norm = _RMSNorm(
            config.head_dim, config.norm_eps, device=device, dtype=dtype
        )

    def forward(
        self,
        hidden: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, rows, _ = hidden.shape
        query = self.q_norm(
            self.q_proj(hidden).view(batch, rows, self.heads, self.head_dim)
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden).view(batch, rows, self.kv_heads, self.head_dim)
        ).transpose(1, 2)
        value = self.v_proj(hidden).view(
            batch, rows, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        cosine, sine = rotary
        cosine = cosine.unsqueeze(1)
        sine = sine.unsqueeze(1)
        query = query * cosine + _rotate_half(query) * sine
        key = key * cosine + _rotate_half(key) * sine
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=rows > 1,
            scale=self.scaling,
            enable_gqa=self.heads != self.kv_heads,
        )
        return self.o_proj(output.transpose(1, 2).reshape(batch, rows, -1))


class _MLP(nn.Module):
    def __init__(
        self,
        config: H3TextEncoderConfig,
        mesh: DeviceMesh,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.gate_proj = _ColumnLinear(
            config.hidden_size,
            config.intermediate_size,
            mesh,
            device=device,
            dtype=dtype,
        )
        self.up_proj = _ColumnLinear(
            config.hidden_size,
            config.intermediate_size,
            mesh,
            device=device,
            dtype=dtype,
        )
        self.down_proj = _RowLinear(
            config.intermediate_size,
            config.hidden_size,
            mesh,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class _DecoderLayer(nn.Module):
    def __init__(
        self,
        config: H3TextEncoderConfig,
        mesh: DeviceMesh,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.self_attn = _Attention(config, mesh, device=device, dtype=dtype)
        self.mlp = _MLP(config, mesh, device=device, dtype=dtype)
        self.input_layernorm = _RMSNorm(
            config.hidden_size, config.norm_eps, device=device, dtype=dtype
        )
        self.post_attention_layernorm = _RMSNorm(
            config.hidden_size, config.norm_eps, device=device, dtype=dtype
        )

    def forward(
        self,
        hidden: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), rotary)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class _LanguageModel(nn.Module):
    def __init__(
        self,
        config: H3TextEncoderConfig,
        mesh: DeviceMesh,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.embed_tokens = _VocabParallelEmbedding(
            config, mesh, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            _DecoderLayer(config, mesh, device=device, dtype=dtype)
            for _ in range(config.retained_layers)
        )
        # The requested hidden_states[50] is captured before layer 50 and is
        # therefore not passed through the checkpoint's final language norm.
        self.rotary_emb = _TextRotaryEmbedding(config, device=mesh.local_device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(token_ids)
        positions = torch.arange(
            token_ids.shape[1], dtype=torch.long, device=token_ids.device
        ).view(1, -1)
        rotary = self.rotary_emb(hidden, positions)
        for layer in self.layers:
            hidden = layer(hidden, rotary)
        return hidden


class MiniMaxH3TextEncoder(nn.Module):
    """The T2VA-only Qwen3-VL language path through hidden state index 50."""

    architecture = "Qwen3VLForConditionalGeneration"

    def __init__(
        self,
        mesh: DeviceMesh,
        *,
        max_text_rows: int,
        parameter_device: torch.device | str = "meta",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.config = H3TextEncoderConfig(max_text_rows=int(max_text_rows))
        if mesh.size("tp") != 4:
            raise ValueError("the H3 text encoder requires TP4")
        self.mesh = mesh
        self.language_model = _LanguageModel(
            self.config,
            mesh,
            device=parameter_device,
            dtype=dtype,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("H3 text conditioning requires one token sequence")
        if token_ids.shape[1] < 1 or token_ids.shape[1] > self.config.max_text_rows:
            raise ValueError(
                f"H3 prompt token count must be between 1 and {self.config.max_text_rows}"
            )
        return self.language_model(token_ids)

    def enable_nvfp4(self) -> None:
        for module in self.modules():
            if isinstance(module, (_ColumnLinear, _RowLinear)):
                module.enable_nvfp4()
