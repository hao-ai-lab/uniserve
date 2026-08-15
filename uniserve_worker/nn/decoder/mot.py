"""Stateless Mixture-of-Transformers decoder composition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import torch
import torch.nn as nn

from ...execution.forward_batch import (
    ExpertRoute,
    ForwardBatch,
    PackedAttentionPlan,
    PagedDecodePlan,
    RouteSpan,
)
from ..attention import RadixAttention
from ..expert_routing import RoutedTensor
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
    context: ForwardBatch,
    coordinate: int,
    target: torch.device,
    call: Callable[[nn.Module, torch.Tensor, ForwardBatch], torch.Tensor],
) -> torch.Tensor:
    staged = context.mesh.dispatch(value, "tower", coordinate)
    result = call(module, staged, context)
    if not isinstance(result, torch.Tensor):
        raise TypeError("MoT sublayer must return a tensor")
    return context.mesh.combine(result, "tower", coordinate, target)


def _route_modules(
    value: RoutedTensor,
    *,
    text_module: nn.Module,
    flow_module: nn.Module,
    context: ForwardBatch,
    call: Callable[[nn.Module, torch.Tensor, ForwardBatch], torch.Tensor],
) -> RoutedTensor:
    def apply_text(item: torch.Tensor) -> torch.Tensor:
        return _apply(
            text_module,
            item,
            context=context,
            coordinate=_TEXT_COORDINATE,
            target=item.device,
            call=call,
        )

    def apply_flow(item: torch.Tensor) -> torch.Tensor:
        return _apply(
            flow_module,
            item,
            context=context,
            coordinate=_FLOW_COORDINATE,
            target=item.device,
            call=call,
        )

    return value.map(apply_text, apply_flow)


def _plain_call(
    module: nn.Module,
    value: torch.Tensor,
    context: ForwardBatch,
) -> torch.Tensor:
    del context
    return cast(torch.Tensor, module(value))


def _parallel_call(
    module: nn.Module,
    value: torch.Tensor,
    context: ForwardBatch,
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
        context: ForwardBatch,
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
        hidden: RoutedTensor,
        *,
        cos: RoutedTensor,
        sin: RoutedTensor,
        context: ForwardBatch,
        spans: tuple[RouteSpan, ...],
        causal: bool,
    ) -> RoutedTensor:
        """Apply the selected experts and one shared attention operation."""

        normalized = _route_modules(
            hidden,
            text_module=self._text.input_norm,
            flow_module=self._flow.input_norm,
            context=context,
            call=_plain_call,
        )
        text_projection = (
            None
            if normalized.text is None or cos.text is None or sin.text is None
            else self._project(self._text, normalized.text, cos.text, sin.text, context)
        )
        flow_projection = (
            None
            if normalized.flow is None or cos.flow is None or sin.flow is None
            else self._project(self._flow, normalized.flow, cos.flow, sin.flow, context)
        )
        query = RoutedTensor(
            None if text_projection is None else text_projection[0],
            None if flow_projection is None else flow_projection[0],
        ).packed(spans)
        key = RoutedTensor(
            None if text_projection is None else text_projection[1],
            None if flow_projection is None else flow_projection[1],
        ).packed(spans)
        value = RoutedTensor(
            None if text_projection is None else text_projection[2],
            None if flow_projection is None else flow_projection[2],
        ).packed(spans)

        self.attention.layer_id = int(layer)
        attended = self.attention(
            query,
            key,
            value,
            context,
            causal=causal,
            scale=self.scale,
        ).reshape(query.shape[0], self.query_size)
        projected = _route_modules(
            RoutedTensor.from_packed(attended, spans),
            text_module=self._text.output,
            flow_module=self._flow.output,
            context=context,
            call=_parallel_call,
        )
        residual = hidden.add(projected)
        normalized = _route_modules(
            residual,
            text_module=self._text.post_norm,
            flow_module=self._flow.post_norm,
            context=context,
            call=_plain_call,
        ).map(
            lambda item: item.to(torch.bfloat16),
            lambda item: item.to(torch.bfloat16),
        )
        feed_forward = _route_modules(
            normalized,
            text_module=self._text.mlp,
            flow_module=self._flow.mlp,
            context=context,
            call=_parallel_call,
        )
        return residual.add(feed_forward)


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
        context: ForwardBatch,
        *,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one decoder sweep described by the explicit attention plan."""

        if inputs_embeds.ndim != 2:
            raise ValueError("MoT inputs must have shape [tokens, hidden]")
        token_count = int(inputs_embeds.shape[0])
        attention = context.attention
        spans: tuple[RouteSpan, ...]
        temporal_positions: torch.Tensor
        causal: bool
        if isinstance(attention, PackedAttentionPlan):
            if attention.indexes.ndim != 2 or int(attention.indexes.shape[1]) != token_count:
                raise ValueError("MoT positions must align with input tokens")
            spans = attention.route_spans
            temporal_positions = attention.indexes[0].reshape(-1)
            causal = False
        elif isinstance(attention, PagedDecodePlan):
            if positions is None or tuple(positions.shape) != (token_count,):
                raise ValueError("MoT paged decode positions must align with text tokens")
            spans = (RouteSpan(ExpertRoute.TEXT, 0, token_count),)
            temporal_positions = positions
            causal = True
        else:
            raise ValueError("MoT forward requires packed attention or paged decode")

        cos, sin = self.rotary.cos_sin_1d(temporal_positions)
        routed_cos = RoutedTensor.from_packed(cos, spans)
        routed_sin = RoutedTensor.from_packed(sin, spans)
        hidden = RoutedTensor.from_packed(inputs_embeds, spans)
        for layer_index, layer_module in enumerate(self.layers):
            layer = cast(MoTDecoderLayer, layer_module)
            hidden = layer(
                layer_index,
                hidden,
                cos=routed_cos,
                sin=routed_sin,
                context=context,
                spans=spans,
                causal=causal,
            )
        return _route_modules(
            hidden,
            text_module=self.norm,
            flow_module=self.norm_moe_gen,
            context=context,
            call=_plain_call,
        ).packed(spans)
