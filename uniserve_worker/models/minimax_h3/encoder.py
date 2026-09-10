"""Tensor-parallel Qwen3-VL text conditioner for MiniMax H3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ...nn.decoder.qwen import Qwen3Config, Qwen3Model
from ...nn.layer import LayerConfig
from ...nn.mesh import DeviceMesh
from ...nn.quant.config import LinearPrecision, QuantizationConfig

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
        linear_precision: LinearPrecision = "bf16",
    ) -> None:
        """Configure a bounded BF16 text-conditioning path on the supplied mesh."""

        super().__init__()
        if dtype != torch.bfloat16:
            raise ValueError("the H3 text encoder uses bfloat16 activations")
        if linear_precision not in ("bf16", "fp8", "nvfp4"):
            raise ValueError(f"unsupported H3 text encoder linear precision {linear_precision!r}")
        self.config = H3TextEncoderConfig(max_text_rows=int(max_text_rows))
        if any(
            width % mesh.size("tp")
            for width in (self.config.heads, self.config.kv_heads, self.config.intermediate_size)
        ):
            raise ValueError("H3 text encoder TP must divide query/KV heads and MLP width")
        self.mesh = mesh
        decoder_config = Qwen3Config(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=self.config.retained_layers,
            num_attention_heads=self.config.heads,
            num_key_value_heads=self.config.kv_heads,
            head_dim=self.config.head_dim,
            hidden_act="silu",
            rms_norm_eps=self.config.norm_eps,
            rope_theta=self.config.rope_theta,
            max_position_embeddings=self.config.max_text_rows,
            attention_bias=False,
            tie_word_embeddings=False,
            num_experts=0,
            num_experts_per_tok=1,
            moe_intermediate_size=self.config.intermediate_size,
        )
        quantization = QuantizationConfig(
            method="unquantized" if linear_precision == "bf16" else linear_precision
        )
        attention_quantization = QuantizationConfig(
            method="nvfp4" if linear_precision == "nvfp4" else "unquantized"
        )
        with torch.device(parameter_device):
            self.language_model = Qwen3Model(
                decoder_config,
                layer_config=LayerConfig(mesh.get_group("tp"), quantization, "language_model"),
                attention_quantization=attention_quantization,
                mlp_input_dtype=torch.float8_e4m3fn if linear_precision == "fp8" else None,
                normalize_output=False,
            )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Validate a single bounded prompt and produce its text-conditioning states."""

        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("H3 text conditioning requires one token sequence")
        if token_ids.shape[1] < 1 or token_ids.shape[1] > self.config.max_text_rows:
            raise ValueError(
                f"H3 prompt token count must be between 1 and {self.config.max_text_rows}"
            )
        positions = torch.arange(token_ids.shape[1], dtype=torch.long, device=token_ids.device)
        return self.language_model(token_ids, positions)
