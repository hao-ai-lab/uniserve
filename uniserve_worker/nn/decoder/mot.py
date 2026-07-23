"""Stateless Mixture-of-Transformers decoder composition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import torch
import torch.nn as nn

from ...forward import ForwardContext, PackedAttentionPlan
from ..attention import RadixAttention
from ..layer import LayerSpec
from ..linear import (
    QKVParallelLinear,
    RowParallelLinear,
    local_attention_head_count,
    local_kv_head_count,
)
from ..norm import RMSNorm
from ..placement import set_tower_coord
from ..rope import apply_rotary_emb, get_rope
from ..vocab_parallel_embedding import VocabParallelEmbedding
from .qwen import Qwen3MLP

__all__ = ["MoTDecoderLayer", "MoTModel"]

_TEXT_COORDINATE = 0
_FLOW_COORDINATE = 1


class _MoTConfig(Protocol):
    @property
    def hidden_size(self) -> int: ...

    @property
    def intermediate_size(self) -> int: ...

    @property
    def num_hidden_layers(self) -> int: ...

    @property
    def num_attention_heads(self) -> int: ...

    @property
    def num_key_value_heads(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def rms_norm_eps(self) -> float: ...

    @property
    def rope_theta(self) -> float: ...

    @property
    def head_dim(self) -> int: ...


@dataclass(frozen=True, slots=True)
class _MlpConfig:
    hidden_size: int
    intermediate_size: int
    hidden_act: str = "silu"


@dataclass(frozen=True, slots=True)
class _Expert:
    input_norm: nn.Module
    qkv: nn.Module
    output: nn.Module
    query_norm: nn.Module
    key_norm: nn.Module
    post_norm: nn.Module
    mlp: nn.Module
    coordinate: int


def _apply(
    module: nn.Module,
    value: torch.Tensor,
    *,
    context: ForwardContext,
    coordinate: int,
    target: torch.device,
    call: Callable[[nn.Module, torch.Tensor, ForwardContext], torch.Tensor],
) -> torch.Tensor:
    staged = context.mesh.dispatch(value, "tower", coordinate)
    result = call(module, staged, context)
    if not isinstance(result, torch.Tensor):
        raise TypeError("MoT sublayer must return a tensor")
    return context.mesh.combine(result, "tower", coordinate, target)


def _route(
    value: torch.Tensor,
    *,
    text_indices: torch.Tensor,
    has_text: bool,
    has_flow: bool,
    text_module: nn.Module,
    flow_module: nn.Module,
    context: ForwardContext,
    call: Callable[[nn.Module, torch.Tensor, ForwardContext], torch.Tensor],
) -> torch.Tensor:
    target = value.device
    if has_text and has_flow:
        result = _apply(
            flow_module,
            value,
            context=context,
            coordinate=_FLOW_COORDINATE,
            target=target,
            call=call,
        )
        text = _apply(
            text_module,
            value.index_select(0, text_indices),
            context=context,
            coordinate=_TEXT_COORDINATE,
            target=target,
            call=call,
        )
        result.index_copy_(0, text_indices, text)
        return result
    if has_flow:
        return _apply(
            flow_module,
            value,
            context=context,
            coordinate=_FLOW_COORDINATE,
            target=target,
            call=call,
        )
    if has_text:
        return _apply(
            text_module,
            value,
            context=context,
            coordinate=_TEXT_COORDINATE,
            target=target,
            call=call,
        )
    raise ValueError("MoT routing requires at least one modality")


def _plain_call(
    module: nn.Module,
    value: torch.Tensor,
    context: ForwardContext,
) -> torch.Tensor:
    del context
    return cast(torch.Tensor, module(value))


def _parallel_call(
    module: nn.Module,
    value: torch.Tensor,
    context: ForwardContext,
) -> torch.Tensor:
    return cast(torch.Tensor, module(value, context.mesh))


class MoTDecoderLayer(nn.Module):
    """One decoder layer with text and flow experts over shared attention."""

    def __init__(self, config: _MoTConfig, *, spec: LayerSpec) -> None:
        super().__init__()
        hidden = int(config.hidden_size)
        head_dim = int(config.head_dim)
        total_heads = int(config.num_attention_heads)
        total_kv_heads = int(config.num_key_value_heads)
        self.num_heads = local_attention_head_count(total_heads, parallel=spec.parallel)
        self.num_kv_heads = local_kv_head_count(total_kv_heads, parallel=spec.parallel)
        self.head_dim = head_dim
        self.query_size = self.num_heads * head_dim
        self.total_query_size = total_heads * head_dim
        self.scale = head_dim**-0.5

        self.input_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.qkv_proj = QKVParallelLinear(
            hidden,
            head_dim,
            total_heads,
            total_kv_heads,
            spec=spec,
            bias=True,
        )
        self.o_proj = RowParallelLinear(
            self.total_query_size,
            hidden,
            spec=spec,
            bias=False,
        )
        self.q_norm = RMSNorm(head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.mlp = Qwen3MLP(
            _MlpConfig(
                hidden_size=hidden,
                intermediate_size=int(config.intermediate_size),
            ),
            spec=spec,
        )

        self.input_layernorm_moe_gen = RMSNorm(hidden, config.rms_norm_eps)
        self.qkv_proj_moe_gen = QKVParallelLinear(
            hidden,
            head_dim,
            total_heads,
            total_kv_heads,
            spec=spec,
            bias=True,
        )
        self.o_proj_moe_gen = RowParallelLinear(
            self.total_query_size,
            hidden,
            spec=spec,
            bias=False,
        )
        self.q_norm_moe_gen = RMSNorm(head_dim, config.rms_norm_eps)
        self.k_norm_moe_gen = RMSNorm(head_dim, config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = RMSNorm(hidden, config.rms_norm_eps)
        self.mlp_moe_gen = Qwen3MLP(
            _MlpConfig(
                hidden_size=hidden,
                intermediate_size=int(config.intermediate_size),
            ),
            spec=spec,
        )
        self.attention = RadixAttention(self.num_heads, self.num_kv_heads, head_dim)

        for module in (
            self.input_layernorm_moe_gen,
            self.qkv_proj_moe_gen,
            self.o_proj_moe_gen,
            self.q_norm_moe_gen,
            self.k_norm_moe_gen,
            self.post_attention_layernorm_moe_gen,
            self.mlp_moe_gen,
        ):
            set_tower_coord(module, _FLOW_COORDINATE)

        self._text = _Expert(
            input_norm=self.input_layernorm,
            qkv=self.qkv_proj,
            output=self.o_proj,
            query_norm=self.q_norm,
            key_norm=self.k_norm,
            post_norm=self.post_attention_layernorm,
            mlp=self.mlp,
            coordinate=_TEXT_COORDINATE,
        )
        self._flow = _Expert(
            input_norm=self.input_layernorm_moe_gen,
            qkv=self.qkv_proj_moe_gen,
            output=self.o_proj_moe_gen,
            query_norm=self.q_norm_moe_gen,
            key_norm=self.k_norm_moe_gen,
            post_norm=self.post_attention_layernorm_moe_gen,
            mlp=self.mlp_moe_gen,
            coordinate=_FLOW_COORDINATE,
        )

    def _project(
        self,
        expert: _Expert,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        context: ForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = hidden.device
        staged = context.mesh.dispatch(hidden, "tower", expert.coordinate)
        staged_cos = context.mesh.dispatch(cos, "tower", expert.coordinate)
        staged_sin = context.mesh.dispatch(sin, "tower", expert.coordinate)
        qkv = expert.qkv(staged)
        if not isinstance(qkv, torch.Tensor):
            raise TypeError("MoT QKV projection must return a tensor")
        query_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split((query_size, kv_size, kv_size), dim=-1)
        query = query.view(-1, self.num_heads, self.head_dim)
        key = key.view(-1, self.num_kv_heads, self.head_dim)
        value = value.view(-1, self.num_kv_heads, self.head_dim)
        query = expert.query_norm(query.float())
        key = expert.key_norm(key.float())
        if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
            raise TypeError("MoT QK normalization must return tensors")
        query = apply_rotary_emb(query, staged_cos, staged_sin).to(torch.bfloat16)
        key = apply_rotary_emb(key, staged_cos, staged_sin).to(torch.bfloat16)
        return (
            context.mesh.combine(query, "tower", expert.coordinate, target),
            context.mesh.combine(key, "tower", expert.coordinate, target),
            context.mesh.combine(value.to(torch.bfloat16), "tower", expert.coordinate, target),
        )

    def forward(
        self,
        layer: int,
        hidden: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        context: ForwardContext,
        plan: PackedAttentionPlan,
    ) -> torch.Tensor:
        """Apply both experts and one shared packed attention operation."""

        text_indices = plan.text_indices
        normalized = _route(
            hidden,
            text_indices=text_indices,
            has_text=plan.has_text,
            has_flow=plan.has_flow,
            text_module=self._text.input_norm,
            flow_module=self._flow.input_norm,
            context=context,
            call=_plain_call,
        )
        if plan.has_text and plan.has_flow:
            query, key, value = self._project(self._flow, normalized, cos, sin, context)
            text_query, text_key, text_value = self._project(
                self._text,
                normalized.index_select(0, text_indices),
                cos.index_select(0, text_indices),
                sin.index_select(0, text_indices),
                context,
            )
            query.index_copy_(0, text_indices, text_query)
            key.index_copy_(0, text_indices, text_key)
            value.index_copy_(0, text_indices, text_value)
        else:
            expert = self._flow if plan.has_flow else self._text
            query, key, value = self._project(expert, normalized, cos, sin, context)

        self.attention.layer_id = int(layer)
        attended = self.attention(
            query,
            key,
            value,
            context,
            causal=False,
            scale=self.scale,
        ).reshape(hidden.shape[0], self.query_size)
        projected = _route(
            attended,
            text_indices=text_indices,
            has_text=plan.has_text,
            has_flow=plan.has_flow,
            text_module=self._text.output,
            flow_module=self._flow.output,
            context=context,
            call=_parallel_call,
        )
        residual = hidden + projected
        normalized = _route(
            residual,
            text_indices=text_indices,
            has_text=plan.has_text,
            has_flow=plan.has_flow,
            text_module=self._text.post_norm,
            flow_module=self._flow.post_norm,
            context=context,
            call=_plain_call,
        ).to(torch.bfloat16)
        feed_forward = _route(
            normalized,
            text_indices=text_indices,
            has_text=plan.has_text,
            has_flow=plan.has_flow,
            text_module=self._text.mlp,
            flow_module=self._flow.mlp,
            context=context,
            call=_parallel_call,
        )
        return residual + feed_forward


class MoTModel(nn.Module):
    """Packed text/flow decoder with no request or runtime state."""

    def __init__(self, config: _MoTConfig, *, spec: LayerSpec) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            spec=spec,
        )
        self.layers = nn.ModuleList(
            MoTDecoderLayer(config, spec=spec) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm_moe_gen = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary = get_rope(config.head_dim, theta=config.rope_theta)
        set_tower_coord(self.norm_moe_gen, _FLOW_COORDINATE)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        context: ForwardContext,
    ) -> torch.Tensor:
        """Run one packed decoder sweep described by the explicit attention plan."""

        plan = context.attention
        if not isinstance(plan, PackedAttentionPlan):
            raise ValueError("MoT forward requires a packed attention plan")
        if inputs_embeds.ndim != 2:
            raise ValueError("MoT inputs must have shape [tokens, hidden]")
        token_count = int(inputs_embeds.shape[0])
        if tuple(plan.route_indicators.shape) != (token_count,):
            raise ValueError("MoT route indicators must align with input tokens")
        if plan.indexes.ndim != 2 or int(plan.indexes.shape[1]) != token_count:
            raise ValueError("MoT positions must align with input tokens")
        if plan.text_indices.ndim != 1:
            raise ValueError("MoT text indices must be one-dimensional")
        if not plan.has_text and int(plan.text_indices.numel()) != 0:
            raise ValueError("MoT plan without text cannot contain text indices")
        if not plan.has_text and not plan.has_flow:
            raise ValueError("MoT plan must contain text or flow tokens")

        cos, sin = self.rotary.cos_sin_1d(plan.indexes[0].reshape(-1))
        hidden = inputs_embeds
        for layer_index, layer_module in enumerate(self.layers):
            layer = cast(MoTDecoderLayer, layer_module)
            hidden = layer(
                layer_index,
                hidden,
                cos=cos,
                sin=sin,
                context=context,
                plan=plan,
            )
        return _route(
            hidden,
            text_indices=plan.text_indices,
            has_text=plan.has_text,
            has_flow=plan.has_flow,
            text_module=self.norm,
            flow_module=self.norm_moe_gen,
            context=context,
            call=_plain_call,
        )
