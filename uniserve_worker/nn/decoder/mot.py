"""Stateless Mixture-of-Transformers decoder composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from uniserve_worker.modeling.tensors import AttentionMode, ExpertRoute, RouteSpan

from ...modeling.tensors import AttentionMetadata
from ..attention import RadixAttention
from ..branch import branch
from ..expert_routing import RoutedTensor, slice_route_spans
from ..layer import LayerConfig
from ..linear import (
    QKVParallelLinear,
    RowParallelLinear,
    local_attention_head_count,
    local_kv_head_count,
)
from ..mlp import GatedMLP
from ..norm import RMSNorm
from ..parallel_sequence import SequencePartition
from ..qkv import RotaryQKV
from ..rope import get_rope
from ..row_pipeline import (
    RowStage,
    RowTensors,
    RowTensorSegments,
    independent_linear_rows,
    packed_row_stage,
)
from .base import Decoder

__all__ = ["MoTConfig", "MoTDecoderLayer", "MoTModel"]


@dataclass(frozen=True, slots=True)
class MoTConfig:
    """Defines a Mixture-of-Transformers decoder's tensor geometry.

    The configuration fixes hidden width, attention heads, experts, rotary settings,
    and mesh params.
    """

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    head_dim: int


class MoTDecoderLayer(nn.Module):
    """One decoder layer with text and flow experts over shared attention."""

    def __init__(
        self,
        config: MoTConfig,
        *,
        layer_config: LayerConfig,
    ) -> None:
        """Build text and flow projections around shared paged attention geometry."""

        super().__init__()
        hidden = int(config.hidden_size)
        head_dim = int(config.head_dim)
        total_heads = int(config.num_attention_heads)
        total_kv_heads = int(config.num_key_value_heads)
        self.num_heads = local_attention_head_count(total_heads, parallel=layer_config.communicator)
        self.num_kv_heads = local_kv_head_count(total_kv_heads, parallel=layer_config.communicator)
        self.head_dim = head_dim
        self.query_size = self.num_heads * head_dim
        self.total_query_size = total_heads * head_dim
        self.scale = head_dim**-0.5

        self.input_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        text_projection = QKVParallelLinear(
            hidden,
            head_dim,
            total_heads,
            total_kv_heads,
            layer_config=layer_config.child("self_attn"),
            prefix="qkv_proj",
            bias=True,
        )
        self.o_proj = RowParallelLinear(
            self.total_query_size,
            hidden,
            layer_config=layer_config.child("self_attn"),
            prefix="o_proj",
            bias=False,
        )
        self.text_qkv = RotaryQKV(
            text_projection,
            RMSNorm(head_dim, config.rms_norm_eps),
            RMSNorm(head_dim, config.rms_norm_eps),
        )
        self.post_attention_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.mlp = GatedMLP(
            hidden,
            int(config.intermediate_size),
            layer_config=layer_config.child("mlp"),
        )

        self.input_layernorm_moe_gen = branch(
            RMSNorm(hidden, config.rms_norm_eps), ExpertRoute.FLOW
        )

        flow_projection = QKVParallelLinear(
            hidden,
            head_dim,
            total_heads,
            total_kv_heads,
            layer_config=layer_config.child("self_attn"),
            prefix="qkv_proj_moe_gen",
            packed_names=("q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen"),
            bias=True,
        )
        self.o_proj_moe_gen = branch(
            RowParallelLinear(
                self.total_query_size,
                hidden,
                layer_config=layer_config.child("self_attn"),
                prefix="o_proj_moe_gen",
                bias=False,
            ),
            ExpertRoute.FLOW,
        )

        self.flow_qkv = branch(
            RotaryQKV(
                flow_projection,
                RMSNorm(head_dim, config.rms_norm_eps),
                RMSNorm(head_dim, config.rms_norm_eps),
            ),
            ExpertRoute.FLOW,
        )
        self.post_attention_layernorm_moe_gen = branch(
            RMSNorm(hidden, config.rms_norm_eps), ExpertRoute.FLOW
        )

        self.mlp_moe_gen = branch(
            GatedMLP(
                hidden,
                int(config.intermediate_size),
                layer_config=layer_config.child("mlp_moe_gen"),
            ),
            ExpertRoute.FLOW,
        )

        self.attention = RadixAttention(
            self.num_heads, self.num_kv_heads, head_dim, sequence=layer_config.sequence
        )

    def row_stage(
        self,
        layer: int,
        *,
        cos: RoutedTensor,
        sin: RoutedTensor,
        context: AttentionMetadata,
        spans: tuple[RouteSpan, ...],
        routes: frozenset[ExpertRoute],
        causal: bool,
        partition: SequencePartition,
    ) -> RowStage[RowTensorSegments]:
        """Declare routed equations around the shared packed attention dependency."""

        self.attention.layer_id = int(layer)
        independent_output = independent_linear_rows(
            self.o_proj, self.o_proj_moe_gen, self.mlp, self.mlp_moe_gen
        )
        complete_cos, complete_sin = cos, sin

        def project(interval: slice, values: RowTensors) -> RowTensors:
            local_spans = slice_route_spans(spans, interval)
            hidden = RoutedTensor.from_packed(values[0], local_spans, routes=routes)
            cos = complete_cos.narrow(interval, spans)
            sin = complete_sin.narrow(interval, spans)
            normalized = hidden.apply(
                text=self.input_layernorm,
                flow=self.input_layernorm_moe_gen,
            )
            text_projection = (
                None
                if normalized.text is None or cos.text is None or sin.text is None
                else self.text_qkv(normalized.text, (cos.text,), (sin.text,))
            )
            flow_projection = (
                None
                if normalized.flow is None or cos.flow is None or sin.flow is None
                else self.flow_qkv(normalized.flow, (cos.flow,), (sin.flow,))
            )
            query = RoutedTensor(
                None if text_projection is None else text_projection[0],
                None if flow_projection is None else flow_projection[0],
            ).packed(local_spans)
            key = RoutedTensor(
                None if text_projection is None else text_projection[1],
                None if flow_projection is None else flow_projection[1],
            ).packed(local_spans)
            value = RoutedTensor(
                None if text_projection is None else text_projection[2],
                None if flow_projection is None else flow_projection[2],
            ).packed(local_spans)

            return query, key, value, values[0]

        def finish(interval: slice, attended: torch.Tensor, state: RowTensors) -> RowTensors:
            local_spans = slice_route_spans(spans, interval)
            hidden = RoutedTensor.from_packed(state[0], local_spans, routes=routes)
            attended = attended.reshape(attended.shape[0], self.query_size)
            projected = RoutedTensor.from_packed(attended, local_spans, routes=hidden.routes).apply(
                text=self.o_proj,
                flow=self.o_proj_moe_gen,
            )
            residual = hidden.add(projected)
            normalized = residual.apply(
                text=self.post_attention_layernorm,
                flow=self.post_attention_layernorm_moe_gen,
            ).map(
                lambda item: item.to(torch.bfloat16),
                lambda item: item.to(torch.bfloat16),
            )
            feed_forward = normalized.apply(text=self.mlp, flow=self.mlp_moe_gen)
            return (residual.add(feed_forward).packed(local_spans),)

        return packed_row_stage(
            project,
            self.attention,
            finish,
            context=context,
            partition=partition,
            causal=causal,
            scale=self.scale,
            independent_input=independent_linear_rows(
                self.text_qkv.projection, self.flow_qkv.projection
            ),
            independent_output=independent_output,
        )


class MoTModel(Decoder):
    """Packed text/flow decoder with no request or runtime state."""

    def __init__(
        self,
        config: MoTConfig,
        *,
        layer_config: LayerConfig,
    ) -> None:
        """Build the routed decoder and bind flow normalization to its tower coordinate."""

        super().__init__(
            config.hidden_size,
            config.vocab_size,
            config.num_hidden_layers,
            layer_config=layer_config,
            init_embeddings=True,
        )
        self.layers = nn.ModuleDict(
            {
                str(index): MoTDecoderLayer(
                    config,
                    layer_config=layer_config.child(f"layers.{index}"),
                )
                for index in self.pipeline.layers
            }
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps) if self.pipeline.last else None
        self.norm_moe_gen = (
            branch(RMSNorm(config.hidden_size, config.rms_norm_eps), ExpertRoute.FLOW)
            if self.pipeline.last
            else None
        )
        self.rotary = get_rope(config.head_dim, theta=config.rope_theta)
        parameters = tuple(dict(next(iter(self.layers.values())).named_parameters()))
        nonresident = self.pipeline.nonresident_layer_names("layers", parameters)
        if not self.pipeline.first:
            nonresident |= {"embed_tokens.weight"}
        if not self.pipeline.last:
            nonresident |= {"norm.weight", "norm_moe_gen.weight"}
        self.nonresident_parameters = nonresident

    def forward(
        self,
        inputs_embeds: torch.Tensor | None,
        context: AttentionMetadata,
        *,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one decoder sweep described by the explicit attention plan."""

        if inputs_embeds is not None:
            token_count = int(inputs_embeds.shape[0])
        elif (
            context.attention_mode is AttentionMode.PACKED and context.attention_indexes is not None
        ):
            token_count = int(context.attention_indexes.shape[1])
        elif context.attention_mode is AttentionMode.PAGED_DECODE and positions is not None:
            token_count = positions.numel()
        else:
            raise ValueError("pipeline input requires packed or decode row geometry")
        partition = SequencePartition(token_count, self.sequence)
        if inputs_embeds is not None and inputs_embeds.ndim != 2:
            raise ValueError("decoder inputs must have shape [tokens, hidden]")
        values = self.receive(
            inputs_embeds,
            token_count,
            reference=cast(
                MoTDecoderLayer, next(iter(self.layers.values()))
            ).input_layernorm.weight,
            partition=partition,
        )
        spans: tuple[RouteSpan, ...]
        temporal_positions: torch.Tensor
        causal: bool
        if context.attention_mode is AttentionMode.PACKED:
            indexes = context.attention_indexes
            if indexes is None or indexes.ndim != 2 or int(indexes.shape[1]) != token_count:
                raise ValueError("MoT positions must align with input tokens")
            spans = context.route_spans
            temporal_positions = indexes[0].reshape(-1)
            causal = False
        elif context.attention_mode is AttentionMode.PAGED_DECODE:
            if positions is None or tuple(positions.shape) != (token_count,):
                raise ValueError("MoT paged decode positions must align with text tokens")
            spans = (RouteSpan(ExpertRoute.TEXT, 0, token_count),)
            temporal_positions = positions
            causal = True
        else:
            raise ValueError("MoT forward requires packed attention or paged decode")

        routes = frozenset(span.route for span in spans)
        spans = partition.routes(spans)
        cos, sin = self.rotary.cos_sin_1d(partition.local(temporal_positions))
        routed_cos = RoutedTensor.from_packed(cos, spans, routes=routes)
        routed_sin = RoutedTensor.from_packed(sin, spans, routes=routes)
        (packed,) = self.run_layers(
            values,
            tuple(
                cast(MoTDecoderLayer, layer).row_stage(
                    int(index) - self.pipeline.layers.start,
                    cos=routed_cos,
                    sin=routed_sin,
                    context=context,
                    spans=spans,
                    routes=routes,
                    causal=causal,
                    partition=partition,
                )
                for index, layer in self.layers.items()
            ),
        )
        if not self.pipeline.last:
            return packed
        hidden = RoutedTensor.from_packed(packed, spans, routes=routes)
        assert self.norm is not None and self.norm_moe_gen is not None
        return partition.gather(hidden.apply(text=self.norm, flow=self.norm_moe_gen).packed(spans))
