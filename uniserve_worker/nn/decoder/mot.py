"""Stateless Mixture-of-Transformers decoder composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from ...execution.device_transfer import tensor_to_device
from ...execution.forward_batch import (
    AttentionMode,
    ExpertRoute,
    ForwardBatch,
    RouteSpan,
)
from ..attention import RadixAttention
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
from ..parallel_pipeline import LayerPipeline
from ..parallel_sequence import SequencePartition
from ..rope import apply_rotary_emb, get_rope
from ..row_pipeline import (
    RowStage,
    RowTensors,
    RowTensorSegments,
    independent_linear_rows,
    packed_row_stage,
    run_row_pipeline,
)
from ..vocab_parallel_embedding import VocabParallelEmbedding

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


@dataclass(frozen=True, slots=True)
class _Expert:
    """Groups one expert’s norms, projections, MLP, and component device."""

    input_norm: nn.Module
    qkv: nn.Module
    output: nn.Module
    query_norm: nn.Module
    key_norm: nn.Module
    post_norm: nn.Module
    mlp: nn.Module
    device: torch.device | None


class MoTDecoderLayer(nn.Module):
    """One decoder layer with text and flow experts over shared attention."""

    def __init__(
        self,
        config: MoTConfig,
        *,
        layer_config: LayerConfig,
        generation_device: torch.device | None = None,
    ) -> None:
        """Build text and flow projections around shared paged attention geometry."""

        super().__init__()
        self.generation_device = generation_device
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
        self.qkv_proj = QKVParallelLinear(
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
        self.q_norm = RMSNorm(head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.mlp = GatedMLP(
            hidden,
            int(config.intermediate_size),
            layer_config=layer_config.child("mlp"),
        )

        self.input_layernorm_moe_gen = RMSNorm(hidden, config.rms_norm_eps)
        self.qkv_proj_moe_gen = QKVParallelLinear(
            hidden,
            head_dim,
            total_heads,
            total_kv_heads,
            layer_config=layer_config.child("self_attn"),
            prefix="qkv_proj_moe_gen",
            packed_names=("q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen"),
            bias=True,
        )
        self.o_proj_moe_gen = RowParallelLinear(
            self.total_query_size,
            hidden,
            layer_config=layer_config.child("self_attn"),
            prefix="o_proj_moe_gen",
            bias=False,
        )
        self.q_norm_moe_gen = RMSNorm(head_dim, config.rms_norm_eps)
        self.k_norm_moe_gen = RMSNorm(head_dim, config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = RMSNorm(hidden, config.rms_norm_eps)
        self.mlp_moe_gen = GatedMLP(
            hidden,
            int(config.intermediate_size),
            layer_config=layer_config.child("mlp_moe_gen"),
        )
        self.attention = RadixAttention(
            self.num_heads, self.num_kv_heads, head_dim, sequence=layer_config.sequence
        )

        self._text = _Expert(
            input_norm=self.input_layernorm,
            qkv=self.qkv_proj,
            output=self.o_proj,
            query_norm=self.q_norm,
            key_norm=self.k_norm,
            post_norm=self.post_attention_layernorm,
            mlp=self.mlp,
            device=None,
        )
        self._flow = _Expert(
            input_norm=self.input_layernorm_moe_gen,
            qkv=self.qkv_proj_moe_gen,
            output=self.o_proj_moe_gen,
            query_norm=self.q_norm_moe_gen,
            key_norm=self.k_norm_moe_gen,
            post_norm=self.post_attention_layernorm_moe_gen,
            mlp=self.mlp_moe_gen,
            device=self.generation_device,
        )

    def _project(
        self,
        expert: _Expert,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        context: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project one expert route into normalized rotary QKV heads."""

        target = hidden.device
        staged = tensor_to_device(hidden, expert.device)
        staged_cos = tensor_to_device(cos, expert.device)
        staged_sin = tensor_to_device(sin, expert.device)
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
            tensor_to_device(query, target),
            tensor_to_device(key, target),
            tensor_to_device(value.to(torch.bfloat16), target),
        )

    def row_stage(
        self,
        layer: int,
        *,
        cos: RoutedTensor,
        sin: RoutedTensor,
        context: ForwardBatch,
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
                text=self._text.input_norm,
                flow=self._flow.input_norm,
                generation_device=self.generation_device,
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
                text=self._text.output,
                flow=self._flow.output,
                generation_device=self.generation_device,
            )
            residual = hidden.add(projected)
            normalized = residual.apply(
                text=self._text.post_norm,
                flow=self._flow.post_norm,
                generation_device=self.generation_device,
            ).map(
                lambda item: item.to(torch.bfloat16),
                lambda item: item.to(torch.bfloat16),
            )
            feed_forward = normalized.apply(
                text=self._text.mlp, flow=self._flow.mlp, generation_device=self.generation_device
            )
            return (residual.add(feed_forward).packed(local_spans),)

        return packed_row_stage(
            project,
            self.attention,
            finish,
            context=context,
            partition=partition,
            causal=causal,
            scale=self.scale,
            independent_input=independent_linear_rows(self.qkv_proj, self.qkv_proj_moe_gen),
            independent_output=independent_output,
        )


class MoTModel(nn.Module):
    """Packed text/flow decoder with no request or runtime state."""

    def __init__(
        self,
        config: MoTConfig,
        *,
        layer_config: LayerConfig,
        generation_device: torch.device | None = None,
    ) -> None:
        """Build the routed decoder and bind flow normalization to its tower coordinate."""

        super().__init__()
        self.generation_device = generation_device
        self.pipeline = LayerPipeline(layer_config.pipeline, config.num_hidden_layers)
        self.sequence = layer_config.sequence
        self.hidden_size = config.hidden_size
        self.embed_tokens = (
            VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                layer_config=layer_config,
            )
            if self.pipeline.first
            else None
        )
        self.layers = nn.ModuleDict(
            {
                str(index): MoTDecoderLayer(
                    config,
                    layer_config=layer_config.child(f"layers.{index}"),
                    generation_device=generation_device,
                )
                for index in self.pipeline.layers
            }
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps) if self.pipeline.last else None
        self.norm_moe_gen = (
            RMSNorm(config.hidden_size, config.rms_norm_eps) if self.pipeline.last else None
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
        context: ForwardBatch,
        *,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one decoder sweep described by the explicit attention plan."""

        if inputs_embeds is not None:
            token_count = int(inputs_embeds.shape[0])
        elif context.forward_mode is AttentionMode.PACKED and context.attention_indexes is not None:
            token_count = int(context.attention_indexes.shape[1])
        elif context.forward_mode is AttentionMode.PAGED_DECODE and positions is not None:
            token_count = positions.numel()
        else:
            raise ValueError("pipeline input requires packed or decode row geometry")
        partition = SequencePartition(token_count, self.sequence)
        if inputs_embeds is None:
            if self.pipeline.first:
                raise ValueError("the first decoder stage requires input embeddings")
            first = cast(MoTDecoderLayer, next(iter(self.layers.values())))
            inputs_embeds = first.input_layernorm.weight.new_empty(
                (partition.count, self.hidden_size)
            )
        else:
            if inputs_embeds.ndim != 2:
                raise ValueError("MoT inputs must have shape [tokens, hidden]")
            inputs_embeds = partition.local(inputs_embeds)
        self.pipeline.receive_activation(inputs_embeds)
        spans: tuple[RouteSpan, ...]
        temporal_positions: torch.Tensor
        causal: bool
        if context.forward_mode is AttentionMode.PACKED:
            indexes = context.attention_indexes
            if indexes is None or indexes.ndim != 2 or int(indexes.shape[1]) != token_count:
                raise ValueError("MoT positions must align with input tokens")
            spans = context.route_spans
            temporal_positions = indexes[0].reshape(-1)
            causal = False
        elif context.forward_mode is AttentionMode.PAGED_DECODE:
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
        (packed,) = run_row_pipeline(
            RowTensorSegments.complete((inputs_embeds,)),
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
        ).materialize()
        if not self.pipeline.last:
            self.pipeline.send_activation(packed)
            return packed
        hidden = RoutedTensor.from_packed(packed, spans, routes=routes)
        assert self.norm is not None and self.norm_moe_gen is not None
        return partition.gather(
            hidden.apply(
                text=self.norm, flow=self.norm_moe_gen, generation_device=self.generation_device
            ).packed(spans)
        )
