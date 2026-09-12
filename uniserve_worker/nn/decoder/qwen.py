"""Shared Qwen-style decoder components."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import torch
import torch.nn as nn

from ...execution.forward_batch import ForwardBatch
from ..attention import RadixAttention
from ..layer import LayerConfig
from ..linear import LinearBase, QKVParallelLinear, RowParallelLinear
from ..mlp import GatedMLP
from ..moe import FusedMoE
from ..norm import RMSNorm
from ..parallel_pipeline import LayerPipeline
from ..parallel_sequence import SequencePartition
from ..quant.base import PreparedLinearInput
from ..quant.config import QuantizationConfig
from ..quant.fp8 import quantize_fp8_rowwise
from ..rope import get_rope, qk_norm_rope
from ..row_pipeline import (
    RowStage,
    RowTensors,
    RowTensorSegments,
    independent_linear_rows,
    packed_row_stage,
    run_row_pipeline,
)
from ..vocab_parallel_embedding import VocabParallelEmbedding

__all__ = [
    "Qwen3Config",
    "Qwen3Attention",
    "Qwen3MoE",
    "Qwen3DecoderLayer",
    "Qwen3Model",
]


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    """Geometry and numerical parameters shared by Qwen language decoders and conditioners."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    attention_bias: bool
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int


class Qwen3Attention(nn.Module):
    """Multi-head self-attention with QK-norm, RoPE, and paged KV via ``RadixAttention``."""

    def __init__(
        self,
        cfg: Qwen3Config,
        layer_id: int,
        *,
        layer_config: LayerConfig,
    ) -> None:
        """Build rank-local QKV projections, rotary normalization, and paged attention."""

        super().__init__()
        self.total_num_heads = cfg.num_attention_heads
        self.total_num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.total_q_size = self.total_num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            cfg.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            layer_config=layer_config,
            prefix="qkv_proj",
            bias=cfg.attention_bias,
        )
        self.q_size = int(self.qkv_proj.output_sizes[0])
        self.kv_size = int(self.qkv_proj.output_sizes[1])
        self.num_heads = self.q_size // self.head_dim
        self.num_kv_heads = self.kv_size // self.head_dim
        if self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Qwen3 local attention heads must be positive")
        self.o_proj = RowParallelLinear(
            self.total_q_size,
            cfg.hidden_size,
            layer_config=layer_config,
            prefix="o_proj",
            bias=cfg.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.rope_theta = float(getattr(cfg, "rope_theta", 1000000.0))
        self.attn = RadixAttention(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            layer_id=layer_id,
            sequence=layer_config.sequence,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: ForwardBatch | None,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor | None = None,
        partition: SequencePartition | None = None,
    ) -> torch.Tensor:
        """Run QK-normalized rotary attention over packed prefill or batched decode rows."""

        state_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        batched = len(state_shape) == 2
        batched_decode = context is not None and batched and int(state_shape[1]) == 1
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._prepare_qk(q, k, v, cos, sin)
        q_attn, k_attn, v_attn = self._attention_inputs(
            q, k, v, state_shape, batched_decode, batched
        )
        out = self.attn(
            q_attn,
            k_attn,
            v_attn,
            None if context is None else context.attention,
            causal=True,
            scale=self.scale,
            partition=partition,
        )
        return self.o_proj(
            self._restore_attention_output(out, state_shape, batched_decode, batched),
        )

    def project_rows(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare packed QKV rows before their global attention dependency."""

        qkv = self.qkv_proj(hidden)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._prepare_qk(q, k, v, cos, sin)
        return q, k, v.reshape(-1, self.num_kv_heads, self.head_dim)

    def _prepare_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape, normalize, and rotate query and key heads for attention."""

        q_heads = q.reshape(-1, self.num_heads, self.head_dim)
        k_heads = k.reshape(-1, self.num_kv_heads, self.head_dim)
        q, k = qk_norm_rope(
            q_heads,
            k_heads,
            self.q_norm.weight,
            self.k_norm.weight,
            cos,
            sin,
            self.q_norm.eps,
        )
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        return q.to(dtype=v.dtype), k.to(dtype=v.dtype)

    def _attention_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state_shape: torch.Size,
        batched_decode: bool,
        batched: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert flattened QKV heads to the layout required by the active attention mode."""

        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        if batched_decode:
            batch = int(state_shape[0])
            return (
                q.reshape(batch, self.num_heads, self.head_dim).contiguous(),
                k.reshape(batch, self.num_kv_heads, self.head_dim),
                v.reshape(batch, self.num_kv_heads, self.head_dim),
            )
        if batched:
            batch, seq = int(state_shape[0]), int(state_shape[1])
            return (
                q.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2).contiguous(),
                k.reshape(batch, seq, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous(),
                v.reshape(batch, seq, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous(),
            )
        return q, k, v

    def _restore_attention_output(
        self,
        out: torch.Tensor,
        state_shape: torch.Size,
        batched_decode: bool,
        batched: bool,
    ) -> torch.Tensor:
        """Restore backend attention output to the caller's batched or flattened state shape."""

        if batched_decode:
            return out.reshape(int(state_shape[0]), 1, self.q_size)
        if batched:
            return out.transpose(1, 2).reshape(*state_shape, self.q_size)
        return out.reshape(*state_shape, self.q_size)


class Qwen3MoE(nn.Module):
    """Mixture-of-experts feed-forward routed by a learned gate."""

    def __init__(self, cfg: Qwen3Config, *, layer_config: LayerConfig) -> None:
        """Build the token router and dense collection of rank-sharded experts."""

        super().__init__()
        num_experts = cfg.num_experts
        top_k = cfg.num_experts_per_tok
        self.gate = LinearBase(
            cfg.hidden_size, num_experts, layer_config=layer_config, prefix="gate", bias=False
        )
        expert_intermediate = int(cfg.moe_intermediate_size or cfg.intermediate_size)
        self.experts = FusedMoE(
            [
                GatedMLP(
                    cfg.hidden_size,
                    expert_intermediate,
                    layer_config=layer_config.child(f"experts.experts.{index}"),
                )
                for index in range(num_experts)
            ],
            top_k=top_k,
            norm_topk_prob=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Route packed hidden rows through the selected experts and combine their outputs."""

        return self.experts(hidden_states, self.gate(hidden_states))


class Qwen3DecoderLayer(nn.Module):
    """One transformer decoder layer (attention + MLP or MoE)."""

    def __init__(
        self,
        cfg: Qwen3Config,
        layer_id: int,
        *,
        layer_config: LayerConfig,
        attention_quantization: QuantizationConfig | None = None,
        mlp_input_dtype: torch.dtype | None = None,
    ) -> None:
        """Assemble one normalized attention layer with dense or expert feed-forward work."""

        super().__init__()
        attention_config = layer_config.child("self_attn")
        if attention_quantization is not None:
            attention_config = replace(attention_config, quantization=attention_quantization)
        self.self_attn = Qwen3Attention(cfg, layer_id, layer_config=attention_config)
        self.mlp_input_dtype = mlp_input_dtype
        self.mlp = (
            Qwen3MoE(cfg, layer_config=layer_config.child("mlp"))
            if cfg.num_experts > 0
            else GatedMLP(
                cfg.hidden_size,
                cfg.intermediate_size,
                hidden_act=cfg.hidden_act,
                layer_config=layer_config.child("mlp"),
            )
        )
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        context: ForwardBatch | None,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor | None = None,
        partition: SequencePartition | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the fused residual stream through attention and dense or expert MLP work."""

        if residual is None:
            residual = hidden_states
            attn_in = self.input_layernorm(hidden_states)
        else:
            attn_in, residual = self.input_layernorm.forward_with_residual(
                hidden_states,
                residual,
                in_place=True,
            )
        attn_out = self.self_attn(
            attn_in, context, cos=cos, sin=sin, positions=positions, partition=partition
        )
        mlp_in, residual = self.post_attention_layernorm.forward_with_residual(
            attn_out,
            residual,
            in_place=True,
        )
        return self.feed_forward(mlp_in), residual

    def feed_forward(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the declared activation format to dense or expert feed-forward rows."""

        if self.mlp_input_dtype is not None:
            if self.mlp_input_dtype != torch.float8_e4m3fn or not isinstance(self.mlp, GatedMLP):
                raise ValueError("prepared decoder MLP input requires a row-scaled FP8 gated MLP")
            quantized, scales = quantize_fp8_rowwise(values.reshape(-1, values.shape[-1]))
            prepared = PreparedLinearInput(quantized.reshape(values.shape), row_scales=scales)
            return self.mlp.forward_prepared(prepared)
        return self.mlp(values)

    def row_stage(
        self,
        context: ForwardBatch,
        cos: torch.Tensor,
        sin: torch.Tensor,
        partition: SequencePartition | None,
    ) -> RowStage[RowTensorSegments]:
        """Declare packed decoder equations and their numerical row dependencies."""

        independent_output = independent_linear_rows(self.self_attn.o_proj, self.mlp)

        def project(interval: slice, values: RowTensors) -> RowTensors:
            hidden = values[0]
            if len(values) == 1:
                residual = hidden
                normalized = self.input_layernorm(hidden)
            else:
                normalized, residual = self.input_layernorm.forward_with_residual(
                    hidden, values[1], in_place=True
                )
            return (
                *self.self_attn.project_rows(normalized, cos[interval], sin[interval]),
                residual,
            )

        def finish(interval: slice, attended: torch.Tensor, state: RowTensors) -> RowTensors:
            output = self.self_attn.o_proj(
                attended.reshape(attended.shape[0], self.self_attn.q_size)
            )
            normalized, residual = self.post_attention_layernorm.forward_with_residual(
                output, state[0], in_place=True
            )
            return self.feed_forward(normalized), residual

        return packed_row_stage(
            project,
            self.self_attn.attn,
            finish,
            context=context.attention,
            partition=partition,
            causal=True,
            scale=self.self_attn.scale,
            independent_input=independent_linear_rows(self.self_attn.qkv_proj),
            independent_output=independent_output,
        )


class Qwen3Model(nn.Module):
    """Stack of Qwen3 decoder layers with token embeddings and final RMSNorm."""

    def __init__(
        self,
        cfg: Qwen3Config,
        *,
        layer_config: LayerConfig,
        attention_quantization: QuantizationConfig | None = None,
        mlp_input_dtype: torch.dtype | None = None,
        normalize_output: bool = True,
    ) -> None:
        """Build sharded token embeddings, decoder layers, rotary tables, and final norm."""

        super().__init__()
        self.pipeline = LayerPipeline(layer_config.pipeline, cfg.num_hidden_layers)
        self.sequence = layer_config.sequence
        self.hidden_size = cfg.hidden_size
        self.embed_tokens = (
            VocabParallelEmbedding(
                cfg.vocab_size,
                cfg.hidden_size,
                layer_config=layer_config,
                init_weights=False,
            )
            if self.pipeline.first
            else None
        )
        self.layers = nn.ModuleDict(
            {
                str(idx): Qwen3DecoderLayer(
                    cfg,
                    idx - self.pipeline.layers.start,
                    layer_config=layer_config.child(f"layers.{idx}"),
                    attention_quantization=attention_quantization,
                    mlp_input_dtype=mlp_input_dtype,
                )
                for idx in self.pipeline.layers
            }
        )
        self.norm = (
            RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            if normalize_output and self.pipeline.last
            else None
        )
        self.rotary = get_rope(
            cfg.head_dim,
            theta=cfg.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context: ForwardBatch | None = None,
        *,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode token or supplied embedding rows and return final normalized hidden states."""

        partition = None
        if self.sequence.world_size > 1:
            if context is None or input_ids.ndim != 1:
                raise ValueError("sequence decoding requires packed token inputs")
            partition = SequencePartition(input_ids.numel(), self.sequence)
            input_ids = partition.local(input_ids)
            positions = partition.local(positions)
            if input_embeds is not None:
                input_embeds = partition.local(input_embeds)
        residual = None
        if self.pipeline.first:
            assert self.embed_tokens is not None
            hidden_states = (
                input_embeds if input_embeds is not None else self.embed_tokens(input_ids)
            )
        else:
            first = cast(Qwen3DecoderLayer, next(iter(self.layers.values())))
            hidden_states = first.input_layernorm.weight.new_empty(
                (*input_ids.shape, self.hidden_size)
            )
            residual = torch.empty_like(hidden_states)
            self.pipeline.receive_activation(hidden_states, residual)
        cos, sin = self.rotary.cos_sin_1d(positions.reshape(-1))
        if context is not None and hidden_states.ndim == 2:
            values: RowTensors = (hidden_states,) if residual is None else (hidden_states, residual)
            hidden_states, residual = run_row_pipeline(
                RowTensorSegments.complete(values),
                tuple(
                    cast(Qwen3DecoderLayer, layer).row_stage(context, cos, sin, partition)
                    for layer in self.layers.values()
                ),
            ).materialize()
        else:
            for layer_module in self.layers.values():
                layer = cast(Qwen3DecoderLayer, layer_module)
                hidden_states, residual = layer(
                    hidden_states,
                    residual,
                    context,
                    cos=cos,
                    sin=sin,
                    positions=positions,
                    partition=partition,
                )
        assert residual is not None
        self.pipeline.send_activation(hidden_states, residual)
        if not self.pipeline.last:
            return hidden_states
        if self.norm is None:
            hidden_states = hidden_states + residual
        else:
            hidden_states, _ = self.norm.forward_with_residual(
                hidden_states, residual, in_place=True
            )
        return hidden_states if partition is None else partition.gather(hidden_states)
